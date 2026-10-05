"""科研工作流主调度器。

提供 CLI 接口用于项目管理、真实研究执行、状态查询和产出物导出。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

from core.artifact_store import ArtifactStore, resolve_paths, resolve_repo_path
from core.config_loader import (
    get_bool,
    get_int,
    get_str,
    get_str_list,
    load_settings,
)
from core.export_service import export_pdf, export_pptx, resolve_export_format
from core.research_pipeline import PipelineResult
from core.research_service import ResearchService
from core.resume import RESUME_STEPS
from core.state_manager import ProjectState, StateManager, WorkflowStage

# 类型别名
ExportFormat = Literal["md", "pdf", "pptx"]
WorkflowMode = Literal["lightweight", "heavyweight", "hybrid"]


def configure_logging(base_dir: Path) -> None:
    """按 ``logging.*`` 节初始化根 logger（应用入口调用，**唯一入口**）。

    未配置时用 ``DEFAULTS``（level=INFO、既定 format、空 file），因此**空 settings
    下的日志级别与格式与引入本机制之前一致**；``logging.file`` 为空时不添加文件
    处理器，保持仅控制台输出。

    本函数只配置根 logger 的级别/格式与（可选）文件处理器，不改动任何既有处理器
    的类型，也不在 ``file`` 为空时新增处理器——避免"未配置却改变日志行为"。
    """
    settings = load_settings(base_dir)
    level_name = get_str(settings, "logging.level")
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        level = logging.INFO
    log_format = get_str(settings, "logging.format")
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = get_str(settings, "logging.file")
    if log_file:
        file_path = resolve_repo_path(base_dir, log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(file_path, encoding="utf-8"))
    logging.basicConfig(level=level, format=log_format, handlers=handlers, force=True)


class Orchestrator:
    """科研工作流主调度器。

    负责协调各个核心组件（状态管理、产出物存储、研究管线），
    提供统一的 CLI 接口。
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        """初始化调度器。

        Args:
            base_dir: 项目根目录，默认为当前工作目录。
        """
        self.base_dir = base_dir or Path.cwd()
        # 路径由 ``paths.*`` 决定，经**唯一入口** ``resolve_paths`` 解析（相对仓库
        # 根目录，绝对路径原样）。未配置时逐项回退 DEFAULTS，故与改动前一致。
        # 这里用 ``load_settings`` 而非 ``load_validated_settings``：构造调度器不应
        # 因配置非法而崩溃——配置校验的快速失败仍由 ``ResearchService.run`` 负责。
        settings = load_settings(self.base_dir)
        self.paths = resolve_paths(self.base_dir, settings)
        self.projects_dir = self.paths["projects_dir"]
        self.output_dir = self.paths["output_dir"]
        self.scripts_dir = self.paths["scripts_dir"]
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        # 导出配置：``export.default_format`` 作未指定时的默认；
        # ``citation.export_formats`` 作导出白名单（真实生效于 ``export()``）；
        # ``export.pdf_engine`` 决定 PDF 后端（真实生效于 ``_export_pdf``）——此前
        # 该键只被读到、从未传给 ``export_pdf``，是「改了配置却毫无影响」的死键。
        self.export_default_format = get_str(settings, "export.default_format")
        self.export_formats = get_str_list(settings, "citation.export_formats")
        self.export_pdf_engine = get_str(settings, "export.pdf_engine")
        # PPTX 导出参数（真实生效于 ``_export_pptx``）：
        # ``export.pptx_template`` 仅在非空时作为 ``--template`` 传给导出脚本；
        # ``export.include_speaker_notes`` 决定是否补 ``Notes:`` 演讲者备注。二者此前
        # 分别被读到/存进配置载体却**从未被使用**——删除它们曾骗过 Phase 8 门禁，
        # 这里让真正的消费者（``_export_pptx``）读取它们，使"配置 → 产物"成立。
        self.export_pptx_template = get_str(settings, "export.pptx_template")
        self.export_include_speaker_notes = get_bool(
            settings, "export.include_speaker_notes"
        )

        self.state_manager = StateManager(self.projects_dir)
        self.artifact_store = ArtifactStore(self.projects_dir)
        # 显式把运行根传给服务层：用户提供的 ``--data`` 相对**运行根**解析（而非
        # 调用者进程 cwd）。这与 ``paths.projects_dir`` / ``paths.output_dir``
        # 锚定同一根，是 Phase 8 双锚点原则在"用户数据路径"上的延伸。
        self.research_service = ResearchService(
            self.projects_dir,
            self.state_manager,
            self.artifact_store,
            base_dir=self.base_dir,
        )

    def init_project(self, project_name: str, mode: WorkflowMode = "lightweight") -> Path:
        """初始化新项目。

        Args:
            project_name: 项目名称（唯一标识）。
            mode: 工作流模式，默认为 lightweight。

        Returns:
            项目目录路径。

        Raises:
            FileExistsError: 项目已存在。
        """
        project_dir = self.projects_dir / project_name
        if project_dir.exists():
            raise FileExistsError(f"项目 '{project_name}' 已存在")

        # 创建项目目录结构
        project_dir.mkdir(parents=True)
        (project_dir / "artifacts").mkdir()

        # 初始化项目状态
        state = ProjectState(
            name=project_name,
            mode=mode,
            current_stage=WorkflowStage.BRAINSTORMING,
        )
        self.state_manager.save(state)

        print(f"[OK] 项目 '{project_name}' 初始化完成 (模式: {mode})")
        print(f"     路径: {project_dir}")
        return project_dir

    def run_real_research(
        self, project_name: str, topic: str, **options: Any
    ) -> PipelineResult:
        """Run real search, LLM analysis, drafting, and artifact persistence."""
        return self.research_service.run(project_name, topic, **options)

    #: 会产生 LLM 调用的阶段。用于判定"本次是否调用过模型"，而不是凭感觉断言。
    _LLM_STEPS: ClassVar[tuple[str, ...]] = ("analysis", "claims", "writing", "review")

    def report_resume(self, result: PipelineResult) -> None:
        """Print the truth about what was reused, derived from actual data.

        Two scopes are kept strictly separate and each is labelled:

        * **判定 (plan)** — ``resume_plan``: what the pre-run decision *planned* to
          skip, and why it could not skip more. This is a prediction and may be
          more optimistic than reality.
        * **实际 (actual)** — ``resume_note``: what actually got reused, generated
          from the real ``reused_steps``. The stage-level fact has this single
          source; we must not re-list the stages ourselves.

        Merging the two would reintroduce the "prediction reported as fact" defect,
        so they are never combined. The cost verdict below is *computed* from
        ``reused_steps``: reuse is only a contiguous prefix, so a run may reuse
        ``search``/``analysis`` while redoing ``writing`` — the literature was then
        *not* re-searched but the model *was* re-called.
        """
        reused = list(getattr(result, "reused_steps", []) or [])
        plan = getattr(result, "resume_plan", "") or ""
        note = getattr(result, "resume_note", "") or ""

        print("\n[续跑] 判定：" + (plan or "本次未提供续跑判定"))
        print("[续跑] 实际：" + (note or "本次未提供续跑说明"))

        if "search" in reused:
            print("[续跑] 本次没有重新检索文献（search 已复用）。")
        else:
            print("[续跑] 本次重新检索了文献（search 未复用）。")

        redone = [step for step in RESUME_STEPS if step not in reused]
        llm_redone = [step for step in self._LLM_STEPS if step in redone]
        if llm_redone:
            print(
                "[续跑] 本次重新调用了模型（重做阶段："
                f"{'、'.join(llm_redone)}）。"
            )
        else:
            print("[续跑] 本次没有重新调用模型（全部会产生模型调用的阶段都已复用）。")

    def report_usage(self, result: PipelineResult) -> None:
        """Print the run's usage summary verbatim, or nothing when unavailable.

        ``usage_note`` is the pipeline's honest description of what it actually
        measured. We pass it through **unchanged** — no adding up, no cost
        estimate — and print nothing when it is empty, so an empty line is never
        mistaken for "this run cost nothing".
        """
        note = getattr(result, "usage_note", "") or ""
        if note:
            print(f"[用量] {note}")

    def report_warnings(self, result: PipelineResult) -> None:
        """逐条、原样打印本次运行的 ``result.warnings``（无警告则什么都不打印）。

        这些警告是管线**如实汇总**的事实（检索源报错、零结果源、重水化失败、缺数据、
        用量异常等）；CLI 此前从不打印它们，导致用户对「只从 crossref 拿到 3 篇」这类
        情况完全不知情。这里只读、不加工：不排序、不改写、不合并、不臆造，与
        :meth:`report_usage` / :meth:`report_literature_limit` 同构。

        为空时**不输出任何内容**（连空行也不输出），避免被误读成「本次没有警告」以外的
        含义，也避免在成功路径上新增任何字节。
        """
        warnings = getattr(result, "warnings", None) or []
        for message in warnings:
            print(f"[警告] {message}")

    def report_failure(self, project_name: str) -> None:
        """失败路径的可理解结论：失败于哪个阶段、失败前已有哪些阶段完成。

        成功路径根本不调用本方法（只在 ``except`` 中调用），因此成功路径输出逐字节不变。

        事实来源只有 ``state.json``（由 ``ResearchService.run`` 在失败时落盘）：不再臆造
        任何未发生的复用。Run 2 曾出现「用户只看到一行超时，既不知道复用了什么、也不知道
        哪一步失败」，本方法即为修补该缺口。

        只打印**可核对**的两件事：

        * 失败阶段：``current_stage`` 且其状态为 ``blocked``/``cancelled``；
        * 失败前已完成的阶段（其产物已保留在项目中），来自 ``stage_status == completed``。

        刻意**不**把它称作「已复用」：阶段是否复用由运行前的输入指纹判定，失败时无从证明，
        故只陈述「已完成且产物已保留」这一可核对事实。
        """
        try:
            state = self.state_manager.load(project_name)
        except (OSError, ValueError, UnicodeDecodeError):
            return
        if state is None:
            return

        failed_stage = state.current_stage
        outcome = state.stage_status.get(failed_stage)
        if outcome in {"blocked", "cancelled"}:
            label = "用户取消" if outcome == "cancelled" else "失败"
            print(
                f"[续跑] 运行未完成，{label}于「{failed_stage.value}」阶段"
                f"（状态: {outcome}）。"
            )

        done = [
            stage.value
            for stage in WorkflowStage
            if state.stage_status.get(stage) == "completed"
        ]
        if done:
            print(f"[续跑] 失败前已完成且产物已保留的阶段：{'、'.join(done)}。")
        else:
            print("[续跑] 失败前没有任何阶段完成（无可沿用的旧产物）。")

    def report_literature_limit(self, project_name: str) -> None:
        """若有文献因总量上限被裁掉，打印一行说明（数量 + 排序依据摘要）。

        裁掉不是错误，但**必须可见**：静默丢弃会让用户误以为这就是全部检索结果。
        这些数字来自 `search_report.json`（`core.research_pipeline`
        如实落盘），这里只读不改、不重算——没有产物时什么也不打印，绝不臆造数字。
        """
        report = self.artifact_store.get_artifact(
            project_name, "search", "search_report.json"
        )
        if report is None:
            return
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            return
        if not isinstance(payload, dict):
            return
        dropped = payload.get("dropped_by_limit")
        if not isinstance(dropped, int) or isinstance(dropped, bool) or dropped <= 0:
            return
        reasons = payload.get("ranking_reasons")
        summary = (
            "；".join(str(item) for item in reasons[:3])
            if isinstance(reasons, list)
            else ""
        )
        print(
            f"[检索] 有 {dropped} 篇文献因总量上限被裁掉"
            f"（保留 {payload.get('result_count', '?')} 篇）。"
            + (f"排序依据：{summary}。" if summary else "")
        )

    def show_status(self, project_name: str) -> None:
        """显示项目状态。

        Args:
            project_name: 项目名称。

        Raises:
            FileNotFoundError: 项目不存在。
        """
        state = self.state_manager.load(project_name)
        if state is None:
            raise FileNotFoundError(f"项目 '{project_name}' 不存在")

        print(f"\n{'='*50}")
        print(f"项目: {state.name}")
        print(f"模式: {state.mode}")
        print(f"当前阶段: {state.current_stage.value}")
        print(f"{'='*50}")

        for stage, status in state.stage_status.items():
            icon = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[OK]",
                "blocked": "[X]",
            }.get(status, "[?]")
            print(f"  {icon} {stage.value:<20} {status}")

        # 显示产出物统计
        artifacts = self.artifact_store.list_artifacts(project_name)
        if artifacts:
            print(f"\n产出物 ({len(artifacts)} 项):")
            for art in artifacts[:10]:  # 最多显示10条
                print(f"  - {art}")
            if len(artifacts) > 10:
                print(f"  ... 还有 {len(artifacts)-10} 项")

    def list_projects(self) -> None:
        """列出所有项目。"""
        projects = [
            d.name
            for d in self.projects_dir.iterdir()
            if d.is_dir() and (d / "state.json").exists()
        ]

        if not projects:
            print("暂无项目")
            return

        print(f"\n共 {len(projects)} 个项目:")
        for name in sorted(projects):
            state = self.state_manager.load(name)
            if state:
                status = state.stage_status.get(state.current_stage, "unknown")
                print(f"  - {name:<30} [{state.mode}] {state.current_stage.value} ({status})")

    def export(
        self,
        project_name: str,
        format: ExportFormat | None = None,
        output_path: Path | None = None,
    ) -> Path:
        """导出项目产出物。

        Args:
            project_name: 项目名称。
            format: 导出格式 (md/pdf/pptx)。为 None 时依次回退到
                ``export.default_format``（settings.yaml）与历史默认 ``md``。
            output_path: 输出路径。为 None 时用默认位置（``projects/<项目>/exports``，
                与改动前一致，Web 下载端点依赖它）。相对路径以 ``paths.output_dir``
                为根解析；绝对路径原样使用。

        Returns:
            导出文件路径。

        Raises:
            FileNotFoundError: 项目不存在。
            ValueError: 不支持的导出格式，或格式不在 ``citation.export_formats`` 白名单内。
        """
        state = self.state_manager.load(project_name)
        if state is None:
            raise FileNotFoundError(f"项目 '{project_name}' 不存在")

        # 解析实际格式：显式请求 > export.default_format > "md"；并把
        # citation.export_formats 作为白名单（真实生效点）。未配置时二者都回退
        # DEFAULTS，行为与改动前一致（md 默认、md/pdf/pptx 全部允许）。
        resolved = resolve_export_format(
            format,
            configured_default=self.export_default_format,
            allowed_formats=self.export_formats,
        )

        if output_path is None:
            output_path = self.projects_dir / project_name / "exports"
        elif not output_path.is_absolute():
            # 相对输出目录以配置的 ``paths.output_dir`` 为根解析；未配置时回退到
            # DEFAULTS 的 "output"（相对仓库根）。绝对路径原样使用。
            output_path = self.output_dir / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        # 收集所有产出物
        artifacts = self.artifact_store.get_all_artifacts(project_name)

        # 根据格式生成导出文件
        export_file = output_path / f"{project_name}_final.{resolved}"

        if resolved == "md":
            self._export_markdown(export_file, state, artifacts)
        elif resolved == "pdf":
            self._export_pdf(export_file, state, artifacts)
        elif resolved == "pptx":
            self._export_pptx(export_file, state, artifacts)
        else:  # pragma: no cover - resolve_export_format 已保证只返回已知格式
            raise ValueError(f"不支持的导出格式: {resolved}")

        print(f"[OK] 导出完成: {export_file}")
        return export_file

    def _export_markdown(
        self,
        output_path: Path,
        state: ProjectState,
        artifacts: dict[str, list[Path]],
    ) -> None:
        """导出为 Markdown 格式。"""
        manuscript = self.artifact_store.get_artifact(
            state.name, "writing", "manuscript.md"
        )
        if manuscript:
            output_path.write_text(
                manuscript.read_text(encoding="utf-8")
                + "\n\n---\n\n## 项目产出物\n\n"
                + "\n".join(
                    f"- [{f.name}]({os.path.relpath(f, output_path.parent)})"
                    for files in artifacts.values()
                    for f in files
                )
                + "\n",
                encoding="utf-8",
            )
            return
        lines = [
            f"# {state.name} - 研究报告",
            "",
            f"- 工作流模式: {state.mode}",
            f"- 导出时间: {state.updated_at}",
            "",
            "---",
            "",
        ]

        for stage, files in artifacts.items():
            lines.append(f"## {stage}")
            lines.append("")
            for f in files:
                lines.append(f"- [{f.name}]({os.path.relpath(f, output_path.parent)})")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")

    def _export_pdf(
        self,
        output_path: Path,
        state: ProjectState,
        artifacts: dict[str, list[Path]],
    ) -> None:
        """将最新手稿导出为 PDF。

        引擎取自 ``export.pdf_engine``（``auto``/``pandoc``/``reportlab``），缺省时
        回退 ``auto``（与改动前逐字节一致）。用户显式选择 ``pandoc`` 而工具缺失时，
        ``export_pdf`` 会**显式失败且不回退**——这正是让该配置「真实生效」的证据。
        """
        manuscript = self.artifact_store.get_artifact(
            state.name, "writing", "manuscript.md"
        )
        if manuscript is None:
            raise ValueError("没有可导出的 writing/manuscript.md")
        export_pdf(manuscript, output_path, engine=self.export_pdf_engine or "auto")

    def _export_pptx(
        self,
        output_path: Path,
        state: ProjectState,
        artifacts: dict[str, list[Path]],
    ) -> None:
        """将最新手稿转换为 PPTX 大纲。"""
        manuscript = self.artifact_store.get_artifact(
            state.name, "writing", "manuscript.md"
        )
        outline = self.artifact_store.get_artifact(
            state.name, "communication", "presentation_outline.md"
        )
        if manuscript is None:
            raise ValueError("没有可导出的 writing/manuscript.md")
        # ``export.pptx_template`` 非空时作为模板传给导出脚本；为空（缺省）时不传，
        # 与改动前逐字节一致（脚本回落到内置空白模板）。
        template_path = (
            Path(self.export_pptx_template) if self.export_pptx_template else None
        )
        export_pptx(
            outline or manuscript,
            output_path,
            # 脚本路径由 ``paths.scripts_dir`` 决定（未经配置时回退 DEFAULTS
            # "scripts"，相对仓库根，解析结果与改动前 ``Path(__file__).parent /
            # "scripts"`` 一致）。
            self.scripts_dir / "export_pptx.py",
            template_path=template_path,
            include_speaker_notes=self.export_include_speaker_notes,
        )


def main() -> None:
    """CLI 入口函数。"""
    parser = argparse.ArgumentParser(
        description="科研工作流调度器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python orchestrator.py research my_paper --topic "你的研究主题" --export md
  python orchestrator.py init my_project --mode hybrid
  python orchestrator.py status my_project
  python orchestrator.py list
  python orchestrator.py export my_project --format pdf
        """,
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="项目根目录（默认: 当前目录）",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # init 命令
    init_parser = subparsers.add_parser("init", help="初始化新项目")
    init_parser.add_argument("project_name", help="项目名称")
    init_parser.add_argument(
        "--mode",
        choices=["lightweight", "heavyweight", "hybrid"],
        default="lightweight",
        help="工作流模式（默认: lightweight）",
    )

    # research 命令
    research_parser = subparsers.add_parser(
        "research", help="从研究主题自动检索、分析、写作并导出"
    )
    research_parser.add_argument("project_name", help="项目名称")
    research_parser.add_argument("--topic", required=True, help="研究主题")
    research_parser.add_argument(
        "--mode", choices=["lightweight", "heavyweight", "hybrid"], default="hybrid"
    )
    research_parser.add_argument(
        "--sources",
        default=None,
        help=(
            "检索源，逗号分隔。未指定时读取 config/settings.yaml 的 "
            "search.default_sources"
        ),
    )
    research_parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help=(
            "检索结果**总量上限**（不是每个检索源的上限）：各源合计去重后最多保留"
            "该数量的文献，超出部分按排序依据裁掉。未指定时读取 config/settings.yaml "
            "的 search.max_results"
        ),
    )
    research_parser.add_argument("--data", type=Path, help="可选实验数据 CSV")
    research_parser.add_argument(
        "--provider",
        choices=["codex_cli", "gemini", "openai_compatible"],
        help="LLM 提供商（默认读取 settings.yaml 或环境变量）",
    )
    research_parser.add_argument("--model", help="LLM 模型名")
    research_parser.add_argument("--base-url", help="OpenAI 兼容 API 地址")
    research_parser.add_argument(
        "--export", default="md", help="导出格式：md、pdf、pptx 或 all"
    )
    research_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="强制全新运行：不复用该项目已有的产物，重新执行全部阶段",
    )

    # status 命令
    status_parser = subparsers.add_parser("status", help="查看项目状态")
    status_parser.add_argument("project_name", help="项目名称")

    # list 命令
    subparsers.add_parser("list", help="列出所有项目")

    # export 命令
    export_parser = subparsers.add_parser("export", help="导出产出物")
    export_parser.add_argument("project_name", help="项目名称")
    export_parser.add_argument(
        "--format",
        choices=["md", "pdf", "pptx"],
        default=None,
        help="导出格式。未指定时读取 config/settings.yaml 的 export.default_format",
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        help="输出目录（默认: 项目目录/exports/）",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 应用入口按 logging.* 初始化日志（级别/格式/可选文件）。
    configure_logging(args.base_dir)

    orchestrator = Orchestrator(base_dir=args.base_dir)

    try:
        if args.command == "init":
            orchestrator.init_project(args.project_name, args.mode)
        elif args.command == "research":
            if orchestrator.state_manager.load(args.project_name) is None:
                orchestrator.init_project(args.project_name, args.mode)
            else:
                print(f"[INFO] 复用已有项目 '{args.project_name}' 并创建新版本产物")
            # 显式命令行参数优先，缺省时回退到 config/settings.yaml 的 search.*。
            # （argparse 的 default 留 None，因为解析参数时还没读 settings。）
            settings = load_settings(args.base_dir)
            sources = (
                [item.strip() for item in args.sources.split(",") if item.strip()]
                if args.sources is not None
                else get_str_list(settings, "search.default_sources")
            )
            max_results = (
                args.max_results
                if args.max_results is not None
                else get_int(settings, "search.max_results")
            )
            result = orchestrator.run_real_research(
                args.project_name,
                args.topic,
                sources=sources,
                max_results=max_results,
                data_path=args.data,
                provider=args.provider,
                model=args.model,
                base_url=args.base_url,
                resume=not args.no_resume,
            )
            orchestrator.report_resume(result)
            orchestrator.report_usage(result)
            orchestrator.report_warnings(result)
            orchestrator.report_literature_limit(args.project_name)
            # argparse cannot narrow a free-form string to a Literal, and
            # --export accepts arbitrary values; Orchestrator.export validates.
            formats: list[ExportFormat] = (
                ["md", "pdf", "pptx"]
                if args.export == "all"
                else [cast(ExportFormat, args.export)]
            )
            for export_format in formats:
                orchestrator.export(args.project_name, export_format)
            print("[DONE] 真实研究工作流执行完成")
        elif args.command == "status":
            orchestrator.show_status(args.project_name)
        elif args.command == "list":
            orchestrator.list_projects()
        elif args.command == "export":
            # ``--format`` 未指定时为 None，交由 export() 回退到
            # export.default_format（settings.yaml）或历史默认 md。
            requested_format = (
                cast(ExportFormat, args.format) if args.format is not None else None
            )
            orchestrator.export(
                args.project_name,
                requested_format,
                args.output,
            )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as e:
        # 失败路径：先给出可核对的结论（失败于哪一阶段、已有哪些产物保留），再报错。
        # 只在 `research` 命令上做——此时才有「一次运行」失败可言；`status`/`export`
        # 等命令的参数名不同或无项目名，不应误报。成功路径不经过这里，输出不变。
        if args.command == "research":
            orchestrator.report_failure(args.project_name)
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[中断] 用户取消操作")
        sys.exit(130)


if __name__ == "__main__":
    main()
