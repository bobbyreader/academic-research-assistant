"""科研工作流主调度器。

提供 CLI 接口用于项目管理、真实研究执行、状态查询和产出物导出。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Literal, cast

from core.artifact_store import ArtifactStore
from core.export_service import export_pdf, export_pptx
from core.research_pipeline import PipelineResult
from core.research_service import ResearchService
from core.state_manager import ProjectState, StateManager, WorkflowStage

# 类型别名
ExportFormat = Literal["md", "pdf", "pptx"]
WorkflowMode = Literal["lightweight", "heavyweight", "hybrid"]


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
        self.projects_dir = self.base_dir / "projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)

        self.state_manager = StateManager(self.projects_dir)
        self.artifact_store = ArtifactStore(self.projects_dir)
        self.research_service = ResearchService(
            self.projects_dir, self.state_manager, self.artifact_store
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
        (project_dir / "checkpoints").mkdir()

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
        format: ExportFormat = "md",
        output_path: Path | None = None,
    ) -> Path:
        """导出项目产出物。

        Args:
            project_name: 项目名称。
            format: 导出格式 (md/pdf/pptx)。
            output_path: 输出路径，默认为项目目录下的 exports/。

        Returns:
            导出文件路径。

        Raises:
            FileNotFoundError: 项目不存在。
            ValueError: 不支持的导出格式。
        """
        state = self.state_manager.load(project_name)
        if state is None:
            raise FileNotFoundError(f"项目 '{project_name}' 不存在")

        if output_path is None:
            output_path = self.projects_dir / project_name / "exports"
        output_path.mkdir(parents=True, exist_ok=True)

        # 收集所有产出物
        artifacts = self.artifact_store.get_all_artifacts(project_name)

        # 根据格式生成导出文件
        export_file = output_path / f"{project_name}_final.{format}"

        if format == "md":
            self._export_markdown(export_file, state, artifacts)
        elif format == "pdf":
            self._export_pdf(export_file, state, artifacts)
        elif format == "pptx":
            self._export_pptx(export_file, state, artifacts)
        else:
            raise ValueError(f"不支持的导出格式: {format}")

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
        """将最新手稿导出为 PDF。"""
        manuscript = self.artifact_store.get_artifact(
            state.name, "writing", "manuscript.md"
        )
        if manuscript is None:
            raise ValueError("没有可导出的 writing/manuscript.md")
        export_pdf(manuscript, output_path)

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
        export_pptx(
            outline or manuscript,
            output_path,
            Path(__file__).parent / "scripts/export_pptx.py",
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
        default="crossref,pubmed,semantic_scholar",
        help="检索源，逗号分隔",
    )
    research_parser.add_argument("--max-results", type=int, default=10)
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
        default="md",
        help="导出格式（默认: md）",
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

    orchestrator = Orchestrator(base_dir=args.base_dir)

    try:
        if args.command == "init":
            orchestrator.init_project(args.project_name, args.mode)
        elif args.command == "research":
            if orchestrator.state_manager.load(args.project_name) is None:
                orchestrator.init_project(args.project_name, args.mode)
            else:
                print(f"[INFO] 复用已有项目 '{args.project_name}' 并创建新版本产物")
            orchestrator.run_real_research(
                args.project_name,
                args.topic,
                sources=[item.strip() for item in args.sources.split(",") if item.strip()],
                max_results=args.max_results,
                data_path=args.data,
                provider=args.provider,
                model=args.model,
                base_url=args.base_url,
            )
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
            orchestrator.export(
                args.project_name,
                cast(ExportFormat, args.format),
                args.output,
            )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[中断] 用户取消操作")
        sys.exit(130)


if __name__ == "__main__":
    main()
