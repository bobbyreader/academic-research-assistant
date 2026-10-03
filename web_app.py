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

from core.artifact_store import ArtifactStore, resolve_paths
from core.config_loader import get_int, get_str_list, load_settings
from core.research_pipeline import ResearchPipelineCancelled
from core.resume import RESUME_STEPS
from core.state_manager import ProjectState, StateManager
from orchestrator import Orchestrator, configure_logging

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
    #: 本次运行的真实用量摘要（`PipelineResult.usage_note` 的逐字透传）。
    #: 可能为空串；空串表示"未获得用量"，**绝不**据此估算花费。
    usage_note: str = ""
    logs: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None
    #: 协作式取消标志。worker 通过 should_continue 读取它；置位后在**阶段边界**
    #: 生效，不会立即杀掉正在写产物的阶段。
    cancel_requested: bool = False


#: 排队位置语义：新建任务按创建顺序获得从 1 开始的序号；0 明确表示"正在运行，
#: 不属于队列"。使用 0 而不是 None，前端/JSON 消费方不必区分 null 与缺省。
QUEUE_POSITION_RUNNING = 0

#: 已结束（不可再取消，也不占用队列位置）的状态。
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


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

    def list_snapshots(self) -> list[dict[str, Any]]:
        """Return a compact snapshot of **every** job, queue order preserved.

        ``queue_position`` is 1-based for jobs still ``queued`` (creation order),
        and ``QUEUE_POSITION_RUNNING`` (0) for the job currently ``running``.
        Finished jobs report ``None`` — they are neither queued nor running.
        """
        with self._lock:
            created_order = sorted(
                self._jobs.values(), key=lambda job: job.created_at
            )
            positions: dict[str, int] = {}
            next_position = 1
            for job in created_order:
                if job.status == "queued":
                    positions[job.job_id] = next_position
                    next_position += 1
            running_id = next(
                (job.job_id for job in created_order if job.status == "running"),
                None,
            )
            snapshots: list[dict[str, Any]] = []
            for job in created_order:
                if job.job_id in positions:
                    position: int | None = positions[job.job_id]
                elif job.job_id == running_id:
                    position = QUEUE_POSITION_RUNNING
                else:
                    position = None
                snapshots.append(
                    {
                        "job_id": job.job_id,
                        "project_name": job.project_name,
                        "topic": job.topic,
                        "status": job.status,
                        "created_at": job.created_at,
                        "finished_at": job.finished_at,
                        "queue_position": position,
                        "cancel_requested": job.cancel_requested,
                    }
                )
            return snapshots

    def cancel(self, job_id: str) -> bool:
        """Request cancellation for a job. Returns False if it cannot be cancelled.

        * ``queued`` — the job has not started, so it is marked ``cancelled``
          immediately. The worker will still run, see the flag, and return
          without doing any work.
        * ``running`` — a cooperative flag is set. The pipeline checks it at each
          **stage boundary**, so cancellation takes effect after the current
          stage finishes, never mid-stage. A ``running`` job stays ``running``
          until the worker observes the flag and marks it ``cancelled``.
        * unknown job or already terminal — returns False (no-op).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL_STATUSES:
                return False
            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.finished_at = datetime.now(UTC).isoformat()
                job.logs.append(
                    "取消请求已受理：任务尚未开始，已直接标记为「已取消」。"
                )
            return True

    def is_cancelled(self, job_id: str) -> bool:
        """Whether a cancel has been requested for this job (thread-safe).

        Used by the worker as its ``should_continue`` predicate: returning the
        negation at each stage boundary is exactly the cooperative signal the
        pipeline expects.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            return job is not None and job.cancel_requested

    def status_of(self, job_id: str) -> str | None:
        """Current status of a job, or ``None`` if unknown (thread-safe)."""
        with self._lock:
            job = self._jobs.get(job_id)
            return job.status if job is not None else None

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
                "usage_note": job.usage_note,
                "cancel_requested": job.cancel_requested,
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
    # 项目根目录由 ``paths.projects_dir`` 决定（唯一入口 ``resolve_paths``：相对
    # 仓库根解析、绝对路径原样）。未配置时回退 DEFAULTS "projects"，与改动前一致。
    # 用 ``load_settings``（非校验版）：Web 启动不应因配置非法而崩溃，配置校验的
    # 快速失败仍由 ``ResearchService.run`` 负责。
    settings = load_settings(project_root)
    resolved_paths = resolve_paths(project_root, settings)
    projects_dir = resolved_paths["projects_dir"]
    # 检索默认值来自 config/settings.yaml 的 search.*：POST /api/research 未显式传
    # sources / max_results 时回退到它们（与 CLI 的"参数优先、配置作默认"语义一致）。
    default_sources = get_str_list(settings, "search.default_sources")
    default_max_results = get_int(settings, "search.max_results")
    # 导出白名单来自 citation.export_formats：Web 选择导出格式时同受其约束
    # （Orchestrator.export 内部也会再次校验，双重保证配置真实生效）。
    configured_export_formats = get_str_list(settings, "citation.export_formats")
    app = Flask(
        __name__,
        template_folder=str(source_root / "web" / "templates"),
        static_folder=str(source_root / "web" / "static"),
    )
    app.config.update(
        BASE_DIR=project_root,
        PROJECTS_DIR=projects_dir,
        ORCHESTRATOR_FACTORY=orchestrator_factory or Orchestrator,
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,
    )
    jobs = JobManager()
    app.extensions["research_jobs"] = jobs

    def state_manager() -> StateManager:
        return StateManager(projects_dir)

    def artifact_store() -> ArtifactStore:
        return ArtifactStore(projects_dir)

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
        path = projects_dir / project_name / "exports" / f"{project_name}_final.{format}"
        if not path.is_file():
            return jsonify(error="该格式尚未生成"), 404
        return send_file(path, as_attachment=True, download_name=path.name)

    @app.get("/api/jobs")
    def job_list() -> Any:
        return jsonify({"jobs": jobs.list_snapshots()})

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id: str) -> Any:
        snapshot = jobs.snapshot(job_id)
        if snapshot is None:
            return jsonify(error="任务不存在"), 404
        return jsonify(snapshot)

    @app.post("/api/jobs/<job_id>/cancel")
    def cancel_job(job_id: str) -> Any:
        if jobs.snapshot(job_id) is None:
            return jsonify(error="任务不存在"), 404
        if not jobs.cancel(job_id):
            return jsonify(error="任务已完成或无法取消"), 409
        return jsonify({"job_id": job_id, "status": "cancel_requested"}), 202

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
            # 未显式传 sources 时回退到配置默认值（search.default_sources）。空列表
            # 同样视为"未提供"（HTML 表单未勾选任何复选框时即如此），因此先取
            # `_list_value` 再判空回退，避免把"没勾选"误报成"至少选择一个检索源"。
            selected_sources = _list_value(payload.get("sources"), [])
            if not selected_sources:
                selected_sources = default_sources
            sources = [source for source in selected_sources if source in SOURCES]
            if not sources:
                raise ValueError("至少选择一个检索源")
            max_results = int(payload.get("max_results") or default_max_results)
            if not 1 <= max_results <= 50:
                raise ValueError("每个检索源的文献数量应为 1–50")
            # 导出格式须同时是全局合法格式（EXPORTS）与 citation.export_formats
            # 白名单成员；白名单为空时视为不限制（与 export_service 语义一致）。
            allowed_exports = (
                set(configured_export_formats) if configured_export_formats else EXPORTS
            )
            exports = [
                fmt
                for fmt in _list_value(payload.get("exports"), ["md"])
                if fmt in EXPORTS and fmt in allowed_exports
            ]
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
            # queued 期间被取消：任务从未开始，直接返回，不进入研究流程。
            if jobs.status_of(current_job_id) == "cancelled":
                return
            jobs.update(current_job_id, status="running", stage_status={**job.stage_status, "search": "in_progress"})
            jobs.log(
                current_job_id,
                "开始执行研究任务"
                + ("" if resume else "（已选择全新运行，不复用旧产物）"),
            )

            def should_continue() -> bool:
                """阶段边界检查：只要用户请求取消即返回 False。"""
                return not jobs.is_cancelled(current_job_id)

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
                    should_continue=should_continue,
                )
                reused_steps = list(getattr(result, "reused_steps", []) or [])
                resume_plan = getattr(result, "resume_plan", "") or ""
                resume_note = getattr(result, "resume_note", "") or ""
                for line in _resume_log_lines(reused_steps, resume_plan, resume_note):
                    jobs.log(current_job_id, line)
                usage_note = getattr(result, "usage_note", "") or ""
                if usage_note:
                    jobs.log(current_job_id, f"用量：{usage_note}")
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
                    usage_note=usage_note,
                    finished_at=datetime.now(UTC).isoformat(),
                )
            except ResearchPipelineCancelled:
                # 协作式取消在**阶段边界**生效：已开始的阶段已完整落盘，可续跑。
                jobs.log(
                    current_job_id,
                    "已停止：取消在阶段边界生效（未打断正在执行的阶段），"
                    "已产出的产物会保留，下次运行可续跑。",
                )
                jobs.update(
                    current_job_id,
                    status="cancelled",
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
    # 应用入口按 logging.* 初始化日志（级别/格式/可选文件），与 CLI 走同一入口。
    configure_logging(args.base_dir)
    app = create_app(args.base_dir)
    print(f"研究助手 Web UI: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
