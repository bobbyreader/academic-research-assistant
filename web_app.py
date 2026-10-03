"""Local, browser-based interface for the research assistant."""

from __future__ import annotations

import argparse
import os
import re
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from core.artifact_store import ArtifactStore
from core.resume import RESUME_STEPS
from core.state_manager import ProjectState, StateManager
from orchestrator import Orchestrator

#: 会产生 LLM 调用的阶段：用于如实判断"本次是否调用过模型"。
_LLM_STEPS = ("analysis", "claims", "writing", "review")

SOURCES = {"crossref", "pubmed", "semantic_scholar", "arxiv"}
EXPORTS = {"md", "pdf", "pptx"}
LLM_PROVIDERS = {"codex_cli", "gemini", "openai_compatible"}
DEFAULT_WEB_PORT = 5050
STAGES = {
    "search": "真实检索",
    "lit_review": "证据分析",
    "writing": "研究写作",
    "export": "整理下载",
}
PROJECT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,48}$")


@dataclass
class WebJob:
    job_id: str
    project_name: str
    topic: str
    status: str = "queued"
    current_stage: str = "search"
    stage_status: dict[str, str] = field(
        default_factory=lambda: {stage: "pending" for stage in STAGES}
    )
    artifacts: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    reused_steps: list[str] = field(default_factory=list)
    resume_plan: str = ""
    resume_note: str = ""
    logs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None


class JobManager:
    """Thread-safe in-memory job registry for the local UI."""

    def __init__(self) -> None:
        self._jobs: dict[str, WebJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-ui")

    def create(self, job: WebJob, worker: Callable[[str], None]) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
        self._executor.submit(worker, job.job_id)

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in changes.items():
                if key == "stage_status" and isinstance(value, dict):
                    job.stage_status.update(value)
                else:
                    setattr(job, key, value)

    def log(self, job_id: str, message: str) -> None:
        """Append a human-readable line to the job's visible log."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.logs.append(message)

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "job_id": job.job_id,
                "project_name": job.project_name,
                "topic": job.topic,
                "status": job.status,
                "current_stage": job.current_stage,
                "stage_status": dict(job.stage_status),
                "stages": [
                    {"id": stage, "label": label, "status": job.stage_status[stage]}
                    for stage, label in STAGES.items()
                ],
                "artifacts": list(job.artifacts),
                "warnings": list(job.warnings),
                "error": job.error,
                "reused_steps": list(job.reused_steps),
                "resume_plan": job.resume_plan,
                "resume_note": job.resume_note,
                "logs": list(job.logs),
                "created_at": job.created_at,
                "finished_at": job.finished_at,
            }


def _validate_project_name(value: str) -> str:
    name = value.strip()
    if not PROJECT_NAME_RE.fullmatch(name):
        raise ValueError("项目名称只能包含英文字母、数字、下划线和短横线，长度为 2–49 个字符")
    return name


def _make_project_name(topic: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", topic.lower()).strip("-")[:24]
    slug = slug or "research"
    return f"{slug}-{datetime.now(UTC).strftime('%m%d%H%M')}-{uuid.uuid4().hex[:4]}"


def _list_value(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _bool_value(value: Any, default: bool) -> bool:
    """Interpret a form/JSON boolean leniently.

    JSON sends a real bool, but an HTML form sends the string ``"false"`` or
    ``"on"``. Anything unrecognised falls back to ``default`` (resume is ON by
    default, matching the CLI).
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return default


def _resume_log_lines(
    reused_steps: list[str], resume_plan: str, resume_note: str
) -> list[str]:
    """Build truthful log lines about reuse, derived from the actual stages.

    Two scopes are kept separate and each is labelled, mirroring the CLI:

    * **判定 (plan)** — ``resume_plan``: what the pre-run decision planned to skip
      and why it could not skip more. A prediction, possibly more optimistic than
      reality, so it is never merged with the fact line.
    * **实际 (actual)** — ``resume_note``: the stage-level fact, single source.

    The cost verdict is *computed* from ``reused_steps``: the web log is the user's
    only view of what a run cost, so it must not claim "no model call" when a
    model-calling stage was redone. ``reused_steps`` is a contiguous prefix, so a
    run can reuse ``search`` yet still re-call the model.
    """
    lines = [f"判定：{resume_plan}" if resume_plan else "判定：本次未提供续跑判定"]
    lines.append(f"实际：{resume_note}" if resume_note else "实际：本次未提供续跑说明")
    lines.append(
        "本次没有重新检索文献（search 已复用）。"
        if "search" in reused_steps
        else "本次重新检索了文献（search 未复用）。"
    )
    redone = [step for step in RESUME_STEPS if step not in reused_steps]
    llm_redone = [step for step in _LLM_STEPS if step in redone]
    if llm_redone:
        lines.append(f"本次重新调用了模型（重做阶段：{'、'.join(llm_redone)}）。")
    else:
        lines.append("本次没有重新调用模型（会产生模型调用的阶段全部已复用）。")
    return lines


def _state_payload(state: ProjectState, artifacts: list[str]) -> dict[str, Any]:
    return {
        "name": state.name,
        "mode": state.mode,
        "current_stage": state.current_stage.value,
        "stage_status": {stage.value: status for stage, status in state.stage_status.items()},
        "topic": state.metadata.get("research_topic", ""),
        "updated_at": state.updated_at,
        "artifacts": artifacts,
    }


def create_app(
    base_dir: Path | None = None,
    orchestrator_factory: Callable[[Path], Any] | None = None,
) -> Flask:
    """Create the local Web UI application."""
    project_root = (base_dir or Path.cwd()).resolve()
    source_root = Path(__file__).parent.resolve()
    app = Flask(
        __name__,
        template_folder=str(source_root / "web" / "templates"),
        static_folder=str(source_root / "web" / "static"),
    )
    app.config.update(
        BASE_DIR=project_root,
        ORCHESTRATOR_FACTORY=orchestrator_factory or Orchestrator,
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    )
    jobs = JobManager()
    app.extensions["research_jobs"] = jobs

    def state_manager() -> StateManager:
        return StateManager(project_root / "projects")

    def artifact_store() -> ArtifactStore:
        return ArtifactStore(project_root / "projects")

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.get("/api/health")
    def health() -> Any:
        return jsonify(
            {
                "status": "ok",
                "llm_configured": bool(
                    os.getenv("GEMINI_API_KEY")
                    or os.getenv("ARS_LLM_API_KEY")
                    or os.getenv("OPENAI_API_KEY")
                ),
            }
        )

    @app.get("/api/projects")
    def projects() -> Any:
        manager = state_manager()
        result = []
        projects_dir = project_root / "projects"
        if projects_dir.exists():
            for directory in sorted(projects_dir.iterdir(), key=lambda path: path.name, reverse=True):
                state = manager.load(directory.name) if directory.is_dir() else None
                if state:
                    result.append(
                        _state_payload(state, artifact_store().list_artifacts(state.name))
                    )
        return jsonify({"projects": result})

    @app.get("/api/projects/<project_name>")
    def project_detail(project_name: str) -> Any:
        try:
            project_name = _validate_project_name(project_name)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        state = state_manager().load(project_name)
        if state is None:
            return jsonify(error="项目不存在"), 404
        return jsonify(
            _state_payload(state, artifact_store().list_artifacts(project_name))
        )

    @app.get("/api/projects/<project_name>/download/<format>")
    def download(project_name: str, format: str) -> Any:
        try:
            project_name = _validate_project_name(project_name)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        if format not in EXPORTS:
            return jsonify(error="不支持的导出格式"), 400
        path = project_root / "projects" / project_name / "exports" / f"{project_name}_final.{format}"
        if not path.is_file():
            return jsonify(error="该格式尚未生成"), 404
        return send_file(path, as_attachment=True, download_name=path.name)

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str) -> Any:
        snapshot = jobs.snapshot(job_id)
        if snapshot is None:
            return jsonify(error="任务不存在"), 404
        return jsonify(snapshot)

    @app.post("/api/research")
    def start_research() -> Any:
        try:
            if request.is_json:
                payload: dict[str, Any] = request.get_json(silent=True) or {}
            else:
                payload = dict(request.form.to_dict())
                payload["sources"] = request.form.getlist("sources")
                payload["exports"] = request.form.getlist("exports")
            topic = str(payload.get("topic", "")).strip()
            if len(topic) < 3:
                raise ValueError("请先输入至少 3 个字符的研究主题")

            requested_name = str(payload.get("project_name", "")).strip()
            project_name = _validate_project_name(requested_name) if requested_name else _make_project_name(topic)
            sources = [source for source in _list_value(payload.get("sources"), ["crossref", "pubmed", "semantic_scholar"]) if source in SOURCES]
            if not sources:
                raise ValueError("至少选择一个检索源")
            max_results = int(payload.get("max_results", 10))
            if not 1 <= max_results <= 50:
                raise ValueError("每个检索源的文献数量应为 1–50")
            exports = [fmt for fmt in _list_value(payload.get("exports"), ["md"]) if fmt in EXPORTS]
            if not exports:
                raise ValueError("至少选择一种导出格式")
            provider = str(payload.get("provider", "codex_cli")).strip().lower()
            if provider not in LLM_PROVIDERS:
                raise ValueError("不支持的内容生成引擎")
            resume = _bool_value(payload.get("resume"), default=True)
        except (TypeError, ValueError) as exc:
            return jsonify(error=str(exc)), 400

        job_id = uuid.uuid4().hex
        upload_path: Path | None = None
        upload = request.files.get("data_file")
        if upload and upload.filename:
            filename = secure_filename(upload.filename) or "data.csv"
            if not filename.lower().endswith(".csv"):
                return jsonify(error="实验数据必须是 CSV 文件"), 400
            upload_dir = project_root / "web_data" / job_id
            upload_dir.mkdir(parents=True, exist_ok=True)
            upload_path = upload_dir / filename
            upload.save(upload_path)

        job = WebJob(job_id=job_id, project_name=project_name, topic=topic)

        def worker(current_job_id: str) -> None:
            jobs.update(current_job_id, status="running", stage_status={**job.stage_status, "search": "in_progress"})
            jobs.log(
                current_job_id,
                "开始执行研究任务"
                + ("" if resume else "（已选择全新运行，不复用旧产物）"),
            )
            try:
                runner = app.config["ORCHESTRATOR_FACTORY"](project_root)
                if runner.state_manager.load(project_name) is None:
                    runner.init_project(project_name, "hybrid")

                def on_progress(stage: str, status: str) -> None:
                    jobs.update(current_job_id, current_stage=stage, stage_status={**job.stage_status, stage: status})

                result = runner.run_real_research(
                    project_name,
                    topic,
                    sources=sources,
                    max_results=max_results,
                    data_path=upload_path,
                    provider=provider,
                    on_progress=on_progress,
                    resume=resume,
                )
                reused_steps = list(getattr(result, "reused_steps", []) or [])
                resume_plan = getattr(result, "resume_plan", "") or ""
                resume_note = getattr(result, "resume_note", "") or ""
                for line in _resume_log_lines(reused_steps, resume_plan, resume_note):
                    jobs.log(current_job_id, line)
                artifacts: list[dict[str, str]] = []
                warnings: list[str] = []
                for export_format in exports:
                    try:
                        runner.export(project_name, export_format)
                    except RuntimeError as exc:
                        warnings.append(str(exc))
                    else:
                        artifacts.append(
                            {
                                "format": export_format,
                                "label": export_format.upper(),
                                "url": f"/api/projects/{project_name}/download/{export_format}",
                            }
                        )
                jobs.update(
                    current_job_id,
                    status="completed",
                    current_stage="export",
                    stage_status={**job.stage_status, "export": "completed"},
                    artifacts=artifacts,
                    warnings=warnings,
                    reused_steps=reused_steps,
                    resume_plan=resume_plan,
                    resume_note=resume_note,
                    finished_at=datetime.now(UTC).isoformat(),
                )
            except Exception as exc:
                app.logger.exception("Research job %s failed", current_job_id)
                jobs.update(
                    current_job_id,
                    status="failed",
                    error=str(exc),
                    finished_at=datetime.now(UTC).isoformat(),
                )

        jobs.create(job, worker)
        return jsonify({"job_id": job_id, "project_name": project_name, "status": "queued"}), 202

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="科研研究助手 Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机访问")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_WEB_PORT,
        help=f"端口，默认 {DEFAULT_WEB_PORT}",
    )
    parser.add_argument("--base-dir", type=Path, default=Path.cwd(), help="项目根目录")
    args = parser.parse_args()
    app = create_app(args.base_dir)
    print(f"研究助手 Web UI: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
