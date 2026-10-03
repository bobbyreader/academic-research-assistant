from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import ClassVar

import pytest

from core.artifact_store import ArtifactStore, resolve_paths
from core.config_loader import DEFAULTS
from core.config_validation import ConfigValidationError
from core.research_pipeline import ResearchPipelineCancelled
from core.research_service import ResearchService
from core.state_manager import ProjectState, StateManager, WorkflowStage
from orchestrator import Orchestrator, configure_logging
from web_app import DEFAULT_WEB_PORT, QUEUE_POSITION_RUNNING, create_app


def test_web_home_and_health_are_available(tmp_path: Path) -> None:
    client = create_app(tmp_path).test_client()

    home = client.get("/")
    health = client.get("/api/health")

    assert home.status_code == 200
    assert "研究工作台" in home.get_data(as_text=True)
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"


def test_web_home_offers_local_codex_engine(tmp_path: Path) -> None:
    page = create_app(tmp_path).test_client().get("/").get_data(as_text=True)

    assert 'name="provider"' in page
    assert 'value="codex_cli"' in page
    assert "本机 Codex" in page


def test_template_has_direct_file_guidance_and_launcher() -> None:
    project_root = Path(__file__).resolve().parents[1]
    template = (project_root / "web" / "templates" / "index.html").read_text(
        encoding="utf-8"
    )
    launcher = project_root / "启动研究助手.command"

    assert "direct-file-preview" in template
    assert "启动研究助手.command" in template
    assert launcher.is_file()
    launcher_text = launcher.read_text(encoding="utf-8")
    assert DEFAULT_WEB_PORT == 5050
    assert "CANDIDATE_PORTS=(5050" in launcher_text
    assert "SELECTED_PORT" in launcher_text


def test_research_endpoint_validates_topic_and_sources(tmp_path: Path) -> None:
    client = create_app(tmp_path).test_client()

    missing_topic = client.post("/api/research", json={})
    bad_source = client.post(
        "/api/research",
        json={"topic": "A useful topic", "sources": ["unknown"]},
    )

    assert missing_topic.status_code == 400
    assert "研究主题" in missing_topic.get_json()["error"]
    assert bad_source.status_code == 400
    assert "检索源" in bad_source.get_json()["error"]


def test_projects_endpoint_reads_existing_project(tmp_path: Path) -> None:
    orchestrator = Orchestrator(tmp_path)
    orchestrator.init_project("demo", "hybrid")

    response = create_app(tmp_path).test_client().get("/api/projects")

    assert response.status_code == 200
    assert response.get_json()["projects"][0]["name"] == "demo"


def test_research_job_runs_in_background_with_injected_runner(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, base_dir: Path) -> None:
            self.base_dir = base_dir
            self.delegate = Orchestrator(base_dir)

        @property
        def state_manager(self):
            return self.delegate.state_manager

        def init_project(self, name: str, mode: str) -> None:
            self.delegate.init_project(name, mode)

        def run_real_research(self, project_name: str, topic: str, **options: object) -> None:
            captured.update(options)
            callback = options["on_progress"]
            callback("search", "completed")
            callback("lit_review", "completed")
            callback("writing", "completed")

        def export(self, project_name: str, format: str) -> Path:
            output = self.base_dir / "projects" / project_name / "exports" / f"{project_name}_final.{format}"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("test export", encoding="utf-8")
            return output

    app = create_app(tmp_path, orchestrator_factory=FakeRunner)
    client = app.test_client()
    response = client.post(
        "/api/research",
        json={
            "topic": "A useful topic",
            "project_name": "demo_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
    )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    for _ in range(20):
        snapshot = client.get(f"/api/jobs/{job_id}").get_json()
        if snapshot["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)

    assert snapshot["status"] == "completed"
    assert snapshot["project_name"] == "demo_job"
    assert snapshot["artifacts"][0]["format"] == "md"
    assert captured["provider"] == "codex_cli"


def test_multipart_request_keeps_multiple_sources(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, base_dir: Path) -> None:
            self.delegate = Orchestrator(base_dir)

        @property
        def state_manager(self):
            return self.delegate.state_manager

        def init_project(self, name: str, mode: str) -> None:
            self.delegate.init_project(name, mode)

        def run_real_research(self, project_name: str, topic: str, **options: object) -> None:
            captured.update(options)

        def export(self, project_name: str, format: str) -> Path:
            output = self.delegate.base_dir / "projects" / project_name / "exports" / f"{project_name}_final.{format}"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("test export", encoding="utf-8")
            return output

    client = create_app(tmp_path, orchestrator_factory=FakeRunner).test_client()
    response = client.post(
        "/api/research",
        data={
            "topic": "A multipart topic",
            "project_name": "multipart_job",
            "sources": ["crossref", "arxiv"],
            "exports": ["md", "pptx"],
        },
    )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    for _ in range(20):
        snapshot = client.get(f"/api/jobs/{job_id}").get_json()
        if snapshot["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)

    assert snapshot["status"] == "completed"
    assert captured["sources"] == ["crossref", "arxiv"]
    assert {artifact["format"] for artifact in snapshot["artifacts"]} == {"md", "pptx"}


class _ResumeRunner:
    """Fake runner whose result reports which stages were reused."""

    reused_steps: ClassVar[list[str]] = [
        "search",
        "analysis",
        "claims",
        "writing",
        "review",
    ]
    resume_plan: ClassVar[str] = (
        "判定可复用 5 个阶段（检索、分析与统计、论断核验、撰写、同行评审）："
        "全部阶段的产物齐备且与当前输入一致。"
    )
    resume_note: ClassVar[str] = (
        "本次复用了全部 5 个阶段（检索、分析与统计、论断核验、撰写、同行评审），"
        "没有阶段被重新执行。"
    )

    def __init__(self, base_dir: Path) -> None:
        self.delegate = Orchestrator(base_dir)

    @property
    def state_manager(self):
        return self.delegate.state_manager

    def init_project(self, name: str, mode: str) -> None:
        self.delegate.init_project(name, mode)

    def run_real_research(self, project_name: str, topic: str, **options: object):
        callback = options["on_progress"]
        callback("search", "completed")
        callback("lit_review", "completed")
        callback("writing", "completed")
        return _FakeResult(list(self.reused_steps), self.resume_plan, self.resume_note)

    def export(self, project_name: str, format: str) -> Path:
        output = (
            self.delegate.base_dir
            / "projects"
            / project_name
            / "exports"
            / f"{project_name}_final.{format}"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("test export", encoding="utf-8")
        return output


class _FakeResult:
    def __init__(
        self, reused_steps: list[str], resume_plan: str, resume_note: str
    ) -> None:
        self.reused_steps = reused_steps
        self.resume_plan = resume_plan
        self.resume_note = resume_note


def _run_job(
    client, payload: dict[str, object], job_id_holder: dict[str, str]
) -> dict[str, object]:
    response = client.post("/api/research", json=payload)
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    job_id_holder["job_id"] = job_id
    for _ in range(25):
        snapshot = client.get(f"/api/jobs/{job_id}").get_json()
        if snapshot["status"] in {"completed", "failed"}:
            return snapshot
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


def test_research_defaults_to_resume_and_reports_reused_steps(tmp_path: Path) -> None:
    """Resume is ON by default, and the job must report exactly what was reused."""
    captured: dict[str, object] = {}

    class Runner(_ResumeRunner):
        def run_real_research(self, project_name: str, topic: str, **options: object):
            captured.update(options)
            return super().run_real_research(project_name, topic, **options)

    client = create_app(tmp_path, orchestrator_factory=Runner).test_client()
    holder: dict[str, str] = {}
    snapshot = _run_job(
        client,
        {
            "topic": "A useful topic",
            "project_name": "resume_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
        holder,
    )

    assert captured["resume"] is True, "resume must default to True"
    assert snapshot["reused_steps"] == _ResumeRunner.reused_steps
    assert snapshot["resume_plan"] == _ResumeRunner.resume_plan
    assert snapshot["resume_note"] == _ResumeRunner.resume_note
    joined = "\n".join(snapshot["logs"])
    # The plan and the actual fact must appear side by side, each labelled.
    assert "判定：" in joined and _ResumeRunner.resume_plan in joined
    assert "实际：" in joined and _ResumeRunner.resume_note in joined
    assert "没有重新调用模型" in joined


def test_resume_log_is_truthful_when_only_some_stages_are_reused(
    tmp_path: Path,
) -> None:
    """A partially-resumed run re-calls the model; the log must say so.

    Reuse is a contiguous prefix, so ``search``/``analysis`` can be reused while
    ``writing`` is redone. In that case the literature was not re-searched, but
    the model *was* re-called — claiming otherwise would mislead the user about
    the run's cost.
    """

    class PartialRunner(_ResumeRunner):
        def run_real_research(self, project_name: str, topic: str, **options: object):
            callback = options["on_progress"]
            callback("search", "completed")
            return _FakeResult(
                ["search", "analysis"],
                "判定可复用 2 个阶段（检索、分析与统计）：缺少 writing/manuscript.md，需重新撰写。",
                "本次复用了 2 个阶段（检索、分析与统计），重新执行了 3 个阶段"
                "（论断核验、撰写、同行评审）。",
            )

    client = create_app(tmp_path, orchestrator_factory=PartialRunner).test_client()
    holder: dict[str, str] = {}
    snapshot = _run_job(
        client,
        {
            "topic": "A useful topic",
            "project_name": "partial_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
        holder,
    )

    joined = "\n".join(snapshot["logs"])
    assert "没有重新检索" in joined, "search was reused, so the log must say so"
    assert "重新调用了模型" in joined, (
        "writing/claims/review were redone, so the model WAS re-called"
    )
    assert "没有重新调用模型" not in joined, (
        "the log must not claim 'no model call' when a model stage was redone"
    )


def test_log_shows_why_when_changed_input_prevents_reuse(tmp_path: Path) -> None:
    """A changed topic must show *why* nothing was reused, next to the fact.

    Without the reason, a correct "reused nothing" looks like resume is broken.
    The plan line (a prediction) and the actual line (the fact) must both appear,
    labelled, and must not contradict each other.
    """

    class ChangedInputRunner(_ResumeRunner):
        def run_real_research(self, project_name: str, topic: str, **options: object):
            callback = options["on_progress"]
            callback("search", "completed")
            return _FakeResult(
                [],
                "判定不复用任何旧产物：运行输入已变化（研究主题），"
                "阶段「检索」及其后续阶段的旧产物不能用于回答当前问题，需重新执行。",
                "本次未复用任何旧产物，全部阶段均已重新执行。",
            )

    client = create_app(
        tmp_path, orchestrator_factory=ChangedInputRunner
    ).test_client()
    holder: dict[str, str] = {}
    snapshot = _run_job(
        client,
        {
            "topic": "A brand new topic",
            "project_name": "changed_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
        holder,
    )

    joined = "\n".join(snapshot["logs"])
    assert "判定：" in joined and "运行输入已变化" in joined, (
        "the log must explain that the input changed"
    )
    assert "实际：" in joined and "未复用任何旧产物" in joined, (
        "the log must still state the fact that nothing was reused"
    )


def test_research_respects_resume_false(tmp_path: Path) -> None:
    """An explicit resume=false must reach the service unchanged."""
    captured: dict[str, object] = {}

    class Runner(_ResumeRunner):
        def run_real_research(self, project_name: str, topic: str, **options: object):
            captured.update(options)
            return super().run_real_research(project_name, topic, **options)

    client = create_app(tmp_path, orchestrator_factory=Runner).test_client()
    holder: dict[str, str] = {}
    _run_job(
        client,
        {
            "topic": "A useful topic",
            "project_name": "fresh_job",
            "sources": ["crossref"],
            "exports": ["md"],
            "resume": False,
        },
        holder,
    )

    assert captured["resume"] is False


def test_job_status_exposes_resume_fields_for_backwards_compatibility(
    tmp_path: Path,
) -> None:
    """Existing fields must be untouched; the resume fields are additive."""
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, base_dir: Path) -> None:
            self.delegate = Orchestrator(base_dir)

        @property
        def state_manager(self):
            return self.delegate.state_manager

        def init_project(self, name: str, mode: str) -> None:
            self.delegate.init_project(name, mode)

        def run_real_research(self, project_name: str, topic: str, **options: object) -> None:
            captured.update(options)

        def export(self, project_name: str, format: str) -> Path:
            output = self.delegate.base_dir / "projects" / project_name / "exports" / f"{project_name}_final.{format}"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("test export", encoding="utf-8")
            return output

    client = create_app(tmp_path, orchestrator_factory=FakeRunner).test_client()
    holder: dict[str, str] = {}
    snapshot = _run_job(
        client,
        {
            "topic": "A useful topic",
            "project_name": "compat_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
        holder,
    )

    # Pre-existing fields still present and typed as before.
    for key in (
        "job_id",
        "project_name",
        "topic",
        "status",
        "current_stage",
        "stage_status",
        "stages",
        "artifacts",
        "warnings",
        "error",
        "created_at",
        "finished_at",
    ):
        assert key in snapshot, f"backwards-compatible field {key} disappeared"

    # Resume fields exist even when the runner returns no result object.
    assert snapshot["reused_steps"] == []
    assert snapshot["resume_note"] == ""
    # Usage field exists and honestly defaults to "no usage" (empty), never 0.
    assert snapshot["usage_note"] == ""


# ==========================================================================
# Queue visibility + cancellation + usage (P5.6)
# ==========================================================================


class _BlockingRunner:
    """A runner that blocks until released, so jobs pile up in the queue.

    The executor runs one worker at a time, so the first job occupies the single
    worker (``running``) while every later job stays ``queued`` — exactly the
    condition ``queue_position`` is meant to describe.
    """

    def __init__(self, base_dir: Path, release: threading.Event) -> None:
        self.delegate = Orchestrator(base_dir)
        self.release = release
        self.started = threading.Event()
        self.should_continue_seen: dict[str, object] = {}

    @property
    def state_manager(self):
        return self.delegate.state_manager

    def init_project(self, name: str, mode: str) -> None:
        self.delegate.init_project(name, mode)

    def run_real_research(self, project_name: str, topic: str, **options: object):
        self.should_continue_seen["value"] = options.get("should_continue")
        self.started.set()
        self.release.wait(timeout=5)

        class _Result:
            reused_steps: ClassVar[list[str]] = []
            resume_plan: ClassVar[str] = ""
            resume_note: ClassVar[str] = ""
            usage_note: ClassVar[str] = "已上报 2 次调用；1 次未获得用量。"

        return _Result()

    def export(self, project_name: str, format: str) -> Path:
        output = (
            self.delegate.base_dir
            / "projects"
            / project_name
            / "exports"
            / f"{project_name}_final.{format}"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("test export", encoding="utf-8")
        return output


def test_jobs_list_shows_every_job_with_queue_positions(tmp_path: Path) -> None:
    """GET /api/jobs must show all jobs; queued positions are 1-based, running=0."""
    release = threading.Event()
    runner = _BlockingRunner(tmp_path, release)
    client = create_app(tmp_path, orchestrator_factory=lambda base: runner).test_client()

    first = client.post(
        "/api/research",
        json={
            "topic": "First topic",
            "project_name": "first_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
    ).get_json()["job_id"]
    # Wait until the first job is actually running (it holds the only worker).
    assert runner.started.wait(timeout=5)

    second = client.post(
        "/api/research",
        json={
            "topic": "Second topic",
            "project_name": "second_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
    ).get_json()["job_id"]
    third = client.post(
        "/api/research",
        json={
            "topic": "Third topic",
            "project_name": "third_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
    ).get_json()["job_id"]

    listing = client.get("/api/jobs")
    assert listing.status_code == 200
    jobs = {job["job_id"]: job for job in listing.get_json()["jobs"]}

    assert set(jobs) == {first, second, third}, "the list must include every job"
    assert jobs[first]["status"] == "running"
    assert jobs[first]["queue_position"] == QUEUE_POSITION_RUNNING
    # 1-based queue order, in creation order.
    assert jobs[second]["status"] == "queued"
    assert jobs[second]["queue_position"] == 1
    assert jobs[third]["status"] == "queued"
    assert jobs[third]["queue_position"] == 2

    release.set()


def test_cancel_queued_job_is_immediate_and_not_reported_as_failed(
    tmp_path: Path,
) -> None:
    """A queued job cannot have started, so cancellation is immediate."""
    release = threading.Event()
    runner = _BlockingRunner(tmp_path, release)
    client = create_app(tmp_path, orchestrator_factory=lambda base: runner).test_client()

    client.post(
        "/api/research",
        json={
            "topic": "Running topic",
            "project_name": "running_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
    )
    assert runner.started.wait(timeout=5)

    queued_id = client.post(
        "/api/research",
        json={
            "topic": "Queued topic",
            "project_name": "queued_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
    ).get_json()["job_id"]

    response = client.post(f"/api/jobs/{queued_id}/cancel")
    assert response.status_code == 202

    snapshot = client.get(f"/api/jobs/{queued_id}").get_json()
    assert snapshot["status"] == "cancelled", "a queued job is cancelled, not failed"

    release.set()


def test_cancel_running_job_sets_flag_and_wakes_should_continue(
    tmp_path: Path,
) -> None:
    """Cancelling a running job must flip the flag the worker's callback reads."""
    release = threading.Event()
    runner = _BlockingRunner(tmp_path, release)
    client = create_app(tmp_path, orchestrator_factory=lambda base: runner).test_client()

    job_id = client.post(
        "/api/research",
        json={
            "topic": "Running topic",
            "project_name": "running_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
    ).get_json()["job_id"]
    assert runner.started.wait(timeout=5)

    should_continue = runner.should_continue_seen["value"]
    assert callable(should_continue), "worker must pass should_continue to the runner"
    assert should_continue() is True, "an uncancelled run must continue"

    response = client.post(f"/api/jobs/{job_id}/cancel")
    assert response.status_code == 202
    assert should_continue() is False, (
        "after a cancel request the cooperative flag must tell the pipeline to stop"
    )

    snapshot = client.get(f"/api/jobs/{job_id}").get_json()
    assert snapshot["cancel_requested"] is True

    release.set()


def test_cancel_unknown_and_finished_jobs_are_rejected(tmp_path: Path) -> None:
    """Unknown → 404; already-finished → 409. Neither mutates any state."""
    client = create_app(tmp_path, orchestrator_factory=_InstantRunner).test_client()

    missing = client.post("/api/jobs/does-not-exist/cancel")
    assert missing.status_code == 404

    holder: dict[str, str] = {}
    _run_job(
        client,
        {
            "topic": "A useful topic",
            "project_name": "finished_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
        holder,
    )
    finished = client.post(f"/api/jobs/{holder['job_id']}/cancel")
    assert finished.status_code == 409


class _InstantRunner:
    """Completes immediately with no artifacts and no usage — a fully finished job."""

    def __init__(self, base_dir: Path) -> None:
        self.delegate = Orchestrator(base_dir)

    @property
    def state_manager(self):
        return self.delegate.state_manager

    def init_project(self, name: str, mode: str) -> None:
        self.delegate.init_project(name, mode)

    def run_real_research(self, project_name: str, topic: str, **options: object):
        callback = options["on_progress"]
        callback("search", "completed")

        class _Result:
            reused_steps: ClassVar[list[str]] = []
            resume_plan: ClassVar[str] = ""
            resume_note: ClassVar[str] = ""
            usage_note: ClassVar[str] = ""

        return _Result()

    def export(self, project_name: str, format: str) -> Path:
        output = (
            self.delegate.base_dir
            / "projects"
            / project_name
            / "exports"
            / f"{project_name}_final.{format}"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("test export", encoding="utf-8")
        return output


class _CancellingRunner:
    """Raises ResearchPipelineCancelled like the real pipeline does at a boundary."""

    def __init__(self, base_dir: Path) -> None:
        self.delegate = Orchestrator(base_dir)

    @property
    def state_manager(self):
        return self.delegate.state_manager

    def init_project(self, name: str, mode: str) -> None:
        self.delegate.init_project(name, mode)

    def run_real_research(self, project_name: str, topic: str, **options: object):
        callback = options["on_progress"]
        callback("search", "completed")
        raise ResearchPipelineCancelled("用户已请求取消：运行在阶段边界停止。")

    def export(self, project_name: str, format: str) -> Path:
        raise AssertionError("a cancelled run must not export anything")


def _wait_for_terminal(client, job_id: str) -> dict[str, object]:
    """Poll until a job reaches any terminal state, including 'cancelled'."""
    snapshot: dict[str, object] = {}
    for _ in range(25):
        snapshot = client.get(f"/api/jobs/{job_id}").get_json()
        if snapshot["status"] in {"completed", "failed", "cancelled"}:
            return snapshot
        time.sleep(0.02)
    raise AssertionError(f"job did not finish in time: {snapshot}")


def test_cancelled_job_status_is_cancelled_and_log_is_truthful(
    tmp_path: Path,
) -> None:
    """A cancelled run is 'cancelled' (not 'failed') and its log must not overclaim.

    The wording must state the delay (stage boundary), that artifacts are kept,
    and that the next run can resume — and must never claim it stopped instantly.
    """
    client = create_app(tmp_path, orchestrator_factory=_CancellingRunner).test_client()
    response = client.post(
        "/api/research",
        json={
            "topic": "A useful topic",
            "project_name": "cancel_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
    )
    assert response.status_code == 202
    snapshot = _wait_for_terminal(client, response.get_json()["job_id"])

    assert snapshot["status"] == "cancelled", "cancellation is not a failure"
    assert snapshot["error"] is None, "a cancelled run has no error message"
    assert snapshot["artifacts"] == [], "a cancelled run produced no download"

    joined = "\n".join(snapshot["logs"])
    assert "阶段边界" in joined, "the log must say cancellation takes effect at a boundary"
    assert "续跑" in joined, "the log must say the partial artifacts can be resumed"
    assert "立即" not in joined, "the log must NOT claim the run stopped instantly"


def test_usage_note_reaches_the_snapshot_verbatim(tmp_path: Path) -> None:
    """usage_note must be exposed exactly as reported — including 'not reported'."""
    release = threading.Event()
    runner = _BlockingRunner(tmp_path, release)
    client = create_app(tmp_path, orchestrator_factory=lambda base: runner).test_client()

    response = client.post(
        "/api/research",
        json={
            "topic": "Usage topic",
            "project_name": "usage_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
    )
    job_id = response.get_json()["job_id"]
    assert runner.started.wait(timeout=5)
    release.set()

    snapshot = None
    for _ in range(25):
        snapshot = client.get(f"/api/jobs/{job_id}").get_json()
        if snapshot["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)

    assert snapshot is not None and snapshot["status"] == "completed"
    assert snapshot["usage_note"] == "已上报 2 次调用；1 次未获得用量。"
    joined = "\n".join(snapshot["logs"])
    assert "1 次未获得用量" in joined, (
        "the log must carry the honest 'not reported' wording, not a fabricated cost"
    )


# ==========================================================================
# Phase 8: paths.* / logging.* / 配置接线枢纽
# ==========================================================================


def _write_settings(base_dir: Path, body: str) -> None:
    config_dir = base_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "settings.yaml").write_text(body, encoding="utf-8")


# ---------------------------------------------------------------------------
# paths.* 锚点语义
# ---------------------------------------------------------------------------
def test_paths_projects_dir_follows_base_dir_but_scripts_dir_follows_repo_root(
    tmp_path: Path,
) -> None:
    """`projects_dir` 跟随运行根；`scripts_dir` 锚定仓库根（代码资产随代码走）。

    这条测试钉死"双锚点"语义：用户数据（projects/output）跟随 base_dir，代码资产
    （scripts/templates）必须无论从哪个 cwd 启动都能找到，因此锚定代码位置。
    """
    orchestrator = Orchestrator(tmp_path)

    assert orchestrator.projects_dir == tmp_path / "projects", (
        "projects_dir 必须跟随运行根（否则既有 e2e 断言与 web 注入会破裂）"
    )
    repo_root = Path(__file__).resolve().parents[1]
    assert orchestrator.scripts_dir == repo_root / "scripts", (
        "scripts_dir 必须锚定仓库根，而不是运行根"
    )
    assert (orchestrator.scripts_dir / "export_pptx.py").is_file()


def test_paths_projects_dir_absolute_is_used_as_is(tmp_path: Path) -> None:
    """绝对路径一律原样使用，不受锚点影响。"""
    absolute = tmp_path / "custom_projects"
    settings = {"paths": {"projects_dir": str(absolute)}}

    resolved = resolve_paths(tmp_path / "unused", settings)

    assert resolved["projects_dir"] == absolute


def test_configured_projects_dir_receives_the_artifacts(tmp_path: Path) -> None:
    """把 `paths.projects_dir` 指到临时目录后，产物**确实落在那里**（端到端）。

    这是 `paths.*` 真实生效的证据：只断言解析函数不够，必须在真实链路上看到
    产物写进配置指定的目录。
    """
    custom = tmp_path / "custom_store"
    _write_settings(
        tmp_path,
        f"paths:\n  projects_dir: {json.dumps(str(custom))}\n",
    )

    orchestrator = Orchestrator(tmp_path)
    assert orchestrator.projects_dir == custom

    # 走真实 Orchestrator→StateManager/ArtifactStore 落盘一个产物。
    orchestrator.init_project("moved", "hybrid")
    store = ArtifactStore(orchestrator.projects_dir)
    saved = store.save_artifact("moved", "search", "literal.txt", "hello")

    assert saved.is_file()
    assert custom in saved.parents, "产物必须落在配置指定的 projects_dir 下"
    assert (custom / "moved" / "artifacts" / "search").is_dir()


# ---------------------------------------------------------------------------
# logging.*
# ---------------------------------------------------------------------------
def test_logging_section_sets_level_and_file(tmp_path: Path) -> None:
    """`logging.*` 生效：级别与格式可断言，文件输出写到指定路径。"""
    log_file = tmp_path / "logs" / "run.log"
    _write_settings(
        tmp_path,
        "logging:\n"
        "  level: DEBUG\n"
        "  format: '%(levelname)s|%(name)s|%(message)s'\n"
        f"  file: {json.dumps(str(log_file))}\n",
    )

    configure_logging(tmp_path)
    root = logging.getLogger()
    try:
        assert root.level == logging.DEBUG, "logging.level 必须生效"
        logger = logging.getLogger("phase8.probe")
        logger.debug("probe-message")
        for handler in root.handlers:
            handler.flush()
        assert log_file.is_file(), "logging.file 必须创建文件处理器并写入"
        content = log_file.read_text(encoding="utf-8")
        assert "DEBUG|phase8.probe|probe-message" in content, (
            "logging.format 必须生效（含级别|logger 名|消息）"
        )
    finally:
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)
        logging.getLogger().setLevel(logging.WARNING)


def test_logging_unconfigured_defaults_to_info(tmp_path: Path) -> None:
    """未配置时日志级别回退到 DEFAULTS 的 INFO（不改变既有日志行为）。"""
    configure_logging(tmp_path)
    root = logging.getLogger()
    try:
        assert root.level == logging.INFO
    finally:
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)
        logging.getLogger().setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# 配置校验：快速失败 + 警告可见
# ---------------------------------------------------------------------------
def test_invalid_config_still_fails_fast_and_names_the_key(tmp_path: Path) -> None:
    """配置非法 → 抛 ConfigValidationError，且消息点名叫错的键。"""
    _write_settings(tmp_path, "review:\n  reviewer_count: three\n")
    projects_dir = tmp_path / "projects"
    state_manager = StateManager(projects_dir)
    state_manager.save(
        ProjectState(
            name="badcfg", mode="hybrid", current_stage=WorkflowStage.BRAINSTORMING
        )
    )

    with pytest.raises(ConfigValidationError, match="reviewer_count"):
        ResearchService(
            projects_dir, state_manager, ArtifactStore(projects_dir)
        ).run("badcfg", "a topic", sources=["crossref"], max_results=1)


def test_config_warnings_are_visible_in_the_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`validation.warnings` 必须可见（进日志），不得被静默吞掉。"""
    # 未知顶层键会产生 warning（非 error）。
    _write_settings(tmp_path, "unexpected_section:\n  foo: 1\n")
    projects_dir = tmp_path / "projects"
    state_manager = StateManager(projects_dir)
    state_manager.save(
        ProjectState(
            name="warny", mode="hybrid", current_stage=WorkflowStage.BRAINSTORMING
        )
    )

    # 会在保存状态后、构造管线前记录警告；随后因缺少外部桩而失败——这里只关心
    # 警告已记录，故吞掉运行期异常。
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(Exception),  # noqa: B017 - 运行不关心后续是否失败
    ):
        ResearchService(
            projects_dir, state_manager, ArtifactStore(projects_dir)
        ).run("warny", "a topic", sources=["crossref"], max_results=1)

    assert any("配置警告" in record.message for record in caplog.records), (
        "配置警告必须进入日志，绝不能静默"
    )


# ---------------------------------------------------------------------------
# 续跑指纹：未配置时与改动前一致（尤其 consensus_threshold 必须是 0.6，不是 0）
# ---------------------------------------------------------------------------
def test_fingerprint_defaults_include_float_consensus_threshold(tmp_path: Path) -> None:
    """未配置时，指纹里的 `review_consensus_threshold` 必须是 0.6（不是被 get_int 抹成的 0）。

    这是 `get_float` 修复的核心证据：直接断言**指纹字段值**，而不是只看读取代码。
    """
    state_manager = StateManager(tmp_path / "projects")
    state_manager.save(
        ProjectState(
            name="fp", mode="hybrid", current_stage=WorkflowStage.BRAINSTORMING
        )
    )
    service = ResearchService(
        tmp_path / "projects", state_manager, ArtifactStore(tmp_path / "projects")
    )

    captured: dict[str, object] = {}

    import core.research_service as rs

    real_build = rs.RunFingerprint.build

    def spy_build(**kwargs: object):
        captured.update(kwargs)
        return real_build(**kwargs)

    original = rs.RunFingerprint.build
    rs.RunFingerprint.build = staticmethod(spy_build)  # type: ignore[assignment]
    try:
        with pytest.raises(Exception):  # noqa: B017 - 管线后续会因缺少桩而失败
            service.run("fp", "a topic", sources=["crossref"], max_results=1)
    finally:
        rs.RunFingerprint.build = original  # type: ignore[assignment]

    # 引用 DEFAULTS 而非字面量：默认值再调整时测试自动跟随，不会出现"默认值改了、
    # 测试没跟上"的假红/假绿。spy 包住 RunFingerprint.build，断言的是**指纹字段的
    # 实际值**，不是读取代码。
    assert captured["review_consensus_threshold"] == pytest.approx(
        float(DEFAULTS["review.consensus_threshold"])
    )
    # float 语义：即便默认值是整数型 0.0，也必须以 float 传入（get_float 保证）。
    assert isinstance(captured["review_consensus_threshold"], float)
    assert captured["writing_language"] == DEFAULTS["writing.default_language"]
    assert captured["search_year_range"] == tuple(DEFAULTS["search.year_range"])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 核心证据：只改一项配置 → writing 不再被复用，且原因点名"撰写语言"
# ---------------------------------------------------------------------------
class _Phase8Searcher:
    """最小检索桩：每次调用都返回同一篇文献。

    接受 ``year_range`` 关键字（Phase 8 检索层新增参数），以免因下游签名演进而
    使本测试假阳性失败。
    """

    def search(
        self,
        query: str,
        sources: list[str],
        max_results: int,
        **kwargs: object,
    ):
        from core.external_clients import PaperRecord, SearchReport

        return SearchReport(
            papers=[
                PaperRecord(
                    title="Evidence",
                    authors=["A Author"],
                    year=2024,
                    journal="J",
                    doi="10.1234/x",
                    abstract="A finding.",
                    source="crossref",
                )
            ],
            errors=[],
            sources_attempted=list(sources),
            counts_by_source={name: 1 for name in sources},
        )


class _Phase8LLM:
    """最小模型桩：满足分析 / 论断核验 / 写作 / 评审四个关口。"""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        if "事实核查员" in system_prompt:
            return {
                "verdicts": [
                    {
                        "claim_index": 0,
                        "claim": "Finding holds",
                        "citation_id": "P1",
                        "verdict": "supports",
                        "quote": "A finding.",
                        "rationale": "ok",
                    }
                ]
            }
        if "审稿人" in system_prompt:
            return {
                "summary": "ok",
                "strengths": ["clear"],
                "concerns": [
                    {
                        "category": "methodology",
                        "severity": "minor",
                        "statement": "small n",
                        "evidence": "x",
                    }
                ],
                "recommendation": "minor_revision",
                "score": 60,
            }
        return {
            "research_question": "q",
            "key_findings": [{"claim": "Finding holds", "citation_ids": ["P1"]}],
            "research_gaps": ["more"],
            "proposed_methods": ["cohort"],
        }

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "# Draft\n\nFinding holds [P1].\n"


class _Phase8Resolver:
    def resolve(self, doi: str) -> tuple[bool, str | None]:
        return True, None


def _phase8_run(orchestrator: Orchestrator, project: str) -> object:
    return orchestrator.run_real_research(
        project,
        "a topic",
        sources=["crossref"],
        max_results=2,
    )


def test_changing_writing_language_invalidates_the_writing_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**本阶段核心证据**：只改 `writing.default_language` → 重跑 → writing 不再复用。

    若新配置不进指纹，用户改了撰写语言后系统会复用旧语言产出的手稿，却报告
    "复用了全部阶段"——那是一句谎。这条测试确保 `writing.*` 真的参与续跑判定，
    且 `resume_plan` 的原因里点名"撰写语言"。
    """
    monkeypatch.setattr(
        "core.research_service.LiteratureSearcher.from_config",
        lambda **kwargs: _Phase8Searcher(),
    )
    monkeypatch.setattr(
        "core.research_service.build_llm_client", lambda **kwargs: _Phase8LLM()
    )
    monkeypatch.setattr("core.citation_verifier.CrossrefDoiResolver", _Phase8Resolver)

    # 首跑：默认配置（语言 zh）。
    _write_settings(tmp_path, "writing:\n  default_language: zh\n")
    orchestrator = Orchestrator(tmp_path)
    orchestrator.init_project("lang", "hybrid")
    first = _phase8_run(orchestrator, "lang")
    assert first.reused_steps == [], "首跑不应复用任何阶段"

    # 只改一项：撰写语言 zh -> en。
    _write_settings(tmp_path, "writing:\n  default_language: en\n")
    second = _phase8_run(orchestrator, "lang")

    assert "writing" not in second.reused_steps, (
        "改了撰写语言后 writing 必须重新执行，绝不能复用旧语言的手稿"
    )
    assert "review" not in second.reused_steps, (
        "writing 不可复用 ⇒ 其后阶段（review）也不可复用（前缀性）"
    )
    # search 的输入未变，仍可复用——证明这条不是"什么都不复用"的假阳性。
    assert "search" in second.reused_steps, (
        "仅改撰写语言不应让检索阶段重跑（否则每年都要重新联网检索）"
    )
    assert "撰写语言" in second.resume_plan, (
        "续跑判定必须点名变化的输入（撰写语言），否则用户不知道为何没复用"
    )


def test_scripts_dir_asset_resolves_and_pptx_export_succeeds(tmp_path: Path) -> None:
    """`Orchestrator(tmp_path)` 下 `scripts_dir` 指向仓库根，且 pptx 导出成功。

    钉死"代码资产锚定仓库根"：即使运行根是临时目录，`scripts/export_pptx.py`
    仍能被找到，导出为合法 zip（`PK` 开头）。
    """
    orchestrator = Orchestrator(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    assert orchestrator.scripts_dir == repo_root / "scripts"

    state_manager = orchestrator.state_manager
    state_manager.save(
        ProjectState(
            name="deck", mode="hybrid", current_stage=WorkflowStage.WRITING
        )
    )
    orchestrator.artifact_store.save_artifact(
        "deck",
        "writing",
        "manuscript.md",
        "# Title\n\n- point one\n- point two\n",
    )

    out = orchestrator.export("deck", "pptx")

    assert out.read_bytes()[:2] == b"PK", "pptx 必须是合法 zip 容器"


# ==========================================================================
# search.* 真实生效：配置作默认值，命令行显式参数优先
# ==========================================================================
def _capture_cli_research(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> dict[str, object]:
    """驱动 CLI `main()`，捕获传给 run_real_research 的 sources / max_results。

    通过替换 `Orchestrator.run_real_research` 捕获参数并直接返回一个假结果，
    避免真正联网检索；其余（参数解析、settings 读取、项目初始化）走真实代码。
    """
    import orchestrator as orch_module

    captured: dict[str, object] = {}

    class _Result:
        reused_steps: ClassVar[list[str]] = []
        resume_plan: ClassVar[str] = ""
        resume_note: ClassVar[str] = ""
        usage_note: ClassVar[str] = ""

    def fake_run(self, project_name, topic, **options):  # type: ignore[no-untyped-def]
        captured["sources"] = options.get("sources")
        captured["max_results"] = options.get("max_results")
        return _Result()

    monkeypatch.setattr(orch_module.Orchestrator, "run_real_research", fake_run)
    monkeypatch.setattr(
        orch_module.Orchestrator, "export", lambda self, *a, **k: Path("x")
    )
    monkeypatch.setattr(orch_module.Orchestrator, "report_resume", lambda self, r: None)
    monkeypatch.setattr(orch_module.Orchestrator, "report_usage", lambda self, r: None)
    monkeypatch.setattr(
        orch_module.Orchestrator, "report_literature_limit", lambda self, n: None
    )
    monkeypatch.setattr("sys.argv", ["orchestrator.py", *argv])
    orch_module.main()
    return captured


def test_cli_sources_fall_back_to_config_when_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """配置 `search.default_sources: [pubmed]` 且 CLI 不传 `--sources` → 只检索 pubmed。

    这是 `search.default_sources` 真实生效的证据（此前是死配置）。
    """
    _write_settings(tmp_path, "search:\n  default_sources: [pubmed]\n")

    captured = _capture_cli_research(
        tmp_path,
        monkeypatch,
        ["--base-dir", str(tmp_path), "research", "cli_job", "--topic", "a topic"],
    )

    assert captured["sources"] == ["pubmed"], (
        "未传 --sources 时必须回退到配置的 search.default_sources"
    )


def test_cli_max_results_falls_back_to_config_when_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """配置 `search.max_results: 7` 且 CLI 不传 `--max-results` → 用 7。"""
    _write_settings(tmp_path, "search:\n  max_results: 7\n")

    captured = _capture_cli_research(
        tmp_path,
        monkeypatch,
        ["--base-dir", str(tmp_path), "research", "cli_job", "--topic", "a topic"],
    )

    assert captured["max_results"] == 7


def test_cli_explicit_sources_override_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI 显式传 `--sources` 时**覆盖**配置（显式参数优先）。

    `--max-results` 同理：显式值胜过配置。
    """
    _write_settings(
        tmp_path,
        "search:\n  default_sources: [pubmed]\n  max_results: 7\n",
    )

    captured = _capture_cli_research(
        tmp_path,
        monkeypatch,
        [
            "--base-dir",
            str(tmp_path),
            "research",
            "cli_job",
            "--topic",
            "a topic",
            "--sources",
            "crossref,arxiv",
            "--max-results",
            "3",
        ],
    )

    assert captured["sources"] == ["crossref", "arxiv"], (
        "显式 --sources 必须覆盖配置默认值"
    )
    assert captured["max_results"] == 3, "显式 --max-results 必须覆盖配置默认值"


def test_web_research_falls_back_to_config_sources(tmp_path: Path) -> None:
    """Web 未传 sources 时回退到配置 `search.default_sources`（与 CLI 一致）。"""
    _write_settings(tmp_path, "search:\n  default_sources: [pubmed]\n")
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, base_dir: Path) -> None:
            self.delegate = Orchestrator(base_dir)

        @property
        def state_manager(self):  # type: ignore[no-untyped-def]
            return self.delegate.state_manager

        def init_project(self, name: str, mode: str) -> None:
            self.delegate.init_project(name, mode)

        def run_real_research(self, project_name: str, topic: str, **options: object) -> None:
            captured.update(options)

        def export(self, project_name: str, format: str) -> Path:
            output = self.delegate.base_dir / "projects" / project_name / "exports" / f"{project_name}_final.{format}"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("x", encoding="utf-8")
            return output

    client = create_app(tmp_path, orchestrator_factory=FakeRunner).test_client()
    response = client.post(
        "/api/research",
        json={"topic": "A useful topic", "project_name": "cfg_job", "exports": ["md"]},
    )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    # 任务在后台线程执行，轮询到终态后再断言 captured。
    for _ in range(25):
        if client.get(f"/api/jobs/{job_id}").get_json()["status"] in {
            "completed",
            "failed",
        }:
            break
        time.sleep(0.02)

    # 未显式提供 sources → 配置默认值被送达 runner。
    assert captured["sources"] == ["pubmed"], (
        "Web 未传 sources 时必须回退到配置的 search.default_sources"
    )


# ==========================================================================
# citation.export_formats 真实生效（导出白名单，本阶段最后一个未读取键）
# ==========================================================================
def test_export_whitelist_blocks_format_outside_citation_export_formats(
    tmp_path: Path,
) -> None:
    """`citation.export_formats: [md]` 时请求 pdf → 抛错且点名叫出该配置。

    这是 `citation.export_formats` 真实生效的证据（此前是死配置）。
    """
    from core.export_service import ExportError

    _write_settings(tmp_path, "citation:\n  export_formats: [md]\n")
    orchestrator = Orchestrator(tmp_path)
    orchestrator.state_manager.save(
        ProjectState(name="wl", mode="hybrid", current_stage=WorkflowStage.WRITING)
    )
    orchestrator.artifact_store.save_artifact(
        "wl", "writing", "manuscript.md", "# Title\n\n- point\n"
    )

    # 白名单内的格式正常导出。
    md_out = orchestrator.export("wl", "md")
    assert md_out.is_file()

    # 白名单外的格式被拒绝，且消息点名配置键。
    with pytest.raises(ExportError) as excinfo:
        orchestrator.export("wl", "pdf")
    assert "citation.export_formats" in str(excinfo.value)


def test_export_default_format_from_config_applies_when_unspecified(
    tmp_path: Path,
) -> None:
    """`export.default_format: pptx` 且未显式指定格式 → 用 pptx。"""
    _write_settings(tmp_path, "citation:\n  export_formats: [md, pdf, pptx]\nexport:\n  default_format: pptx\n")
    orchestrator = Orchestrator(tmp_path)
    orchestrator.state_manager.save(
        ProjectState(name="df", mode="hybrid", current_stage=WorkflowStage.WRITING)
    )
    orchestrator.artifact_store.save_artifact(
        "df", "writing", "manuscript.md", "# Title\n\n- point\n"
    )

    out = orchestrator.export("df")

    assert out.name == "df_final.pptx", (
        "未显式指定格式时必须回退到配置的 export.default_format"
    )
    assert out.read_bytes()[:2] == b"PK"


def test_web_export_respects_citation_export_formats_whitelist(tmp_path: Path) -> None:
    """Web 请求白名单外的导出格式时，该格式被过滤掉（不生成下载项）。"""
    _write_settings(tmp_path, "citation:\n  export_formats: [md]\n")
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, base_dir: Path) -> None:
            self.delegate = Orchestrator(base_dir)

        @property
        def state_manager(self):  # type: ignore[no-untyped-def]
            return self.delegate.state_manager

        def init_project(self, name: str, mode: str) -> None:
            self.delegate.init_project(name, mode)

        def run_real_research(self, project_name: str, topic: str, **options: object) -> None:
            captured.update(options)

        def export(self, project_name: str, format: str) -> Path:
            captured.setdefault("exported", []).append(format)  # type: ignore[union-attr]
            output = self.delegate.base_dir / "projects" / project_name / "exports" / f"{project_name}_final.{format}"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("x", encoding="utf-8")
            return output

    client = create_app(tmp_path, orchestrator_factory=FakeRunner).test_client()
    response = client.post(
        "/api/research",
        json={
            "topic": "A useful topic",
            "project_name": "wl_job",
            "sources": ["crossref"],
            "exports": ["md", "pdf"],
        },
    )
    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    for _ in range(25):
        if client.get(f"/api/jobs/{job_id}").get_json()["status"] in {
            "completed",
            "failed",
        }:
            break
        time.sleep(0.02)

    assert captured.get("exported") == ["md"], (
        "白名单只含 md 时，pdf 必须被过滤掉，绝不能导出"
    )
