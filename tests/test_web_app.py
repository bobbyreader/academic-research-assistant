from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

from orchestrator import Orchestrator
from web_app import DEFAULT_WEB_PORT, create_app


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
