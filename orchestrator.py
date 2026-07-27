"""科研工作流主调度器。

提供 CLI 接口用于项目管理、工作流执行、状态查询和产出物导出。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

from core.artifact_store import ArtifactStore
from core.skill_bridge import SkillBridge
from core.state_manager import ProjectState, StateManager, WorkflowStage

# 类型别名
ExportFormat = Literal["md", "pdf", "pptx"]
WorkflowMode = Literal["lightweight", "heavyweight", "hybrid"]


class Orchestrator:
    """科研工作流主调度器。

    负责协调各个核心组件（状态管理、产出物存储、Skill 桥接），
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
        self.skill_bridge = SkillBridge()

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

    def run_workflow(
        self,
        project_name: str,
        workflow_name: str | None = None,
        resume_from_checkpoint: bool = False,
    ) -> None:
        """运行项目工作流。

        Args:
            project_name: 项目名称。
            workflow_name: 工作流名称，默认使用项目初始化时的模式。
            resume_from_checkpoint: 是否从最近的 checkpoint 恢复。

        Raises:
            FileNotFoundError: 项目不存在。
        """
        state = self.state_manager.load(project_name)
        if state is None:
            raise FileNotFoundError(f"项目 '{project_name}' 不存在")

        mode = workflow_name or state.mode
        print(f"[RUN] 项目: {project_name} | 模式: {mode}")
        print(f"      当前阶段: {state.current_stage.value}")

        if resume_from_checkpoint:
            checkpoint = self.state_manager.load_latest_checkpoint(project_name)
            if checkpoint:
                state = checkpoint
                print(f"      从 checkpoint 恢复: {state.current_stage.value}")

        # 按阶段顺序执行
        stages = self._get_stage_sequence(mode)
        current_idx = stages.index(state.current_stage)

        for stage in stages[current_idx:]:
            state.current_stage = stage
            state.stage_status[stage] = "in_progress"
            self.state_manager.save(state)
            self.state_manager.save_checkpoint(project_name, state)

            print(f"  [STAGE] {stage.value} ...")

            # 模拟阶段执行（实际由具体 Skill 实现）
            try:
                self._execute_stage(project_name, stage, state)
                state.stage_status[stage] = "completed"
                print(f"  [OK] {stage.value} 完成")
            except Exception as e:
                state.stage_status[stage] = "blocked"
                self.state_manager.save(state)
                print(f"  [ERROR] {stage.value} 失败: {e}")
                raise

            self.state_manager.save(state)

        print(f"[DONE] 工作流执行完成")

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

    def _get_stage_sequence(self, mode: WorkflowMode) -> list[WorkflowStage]:
        """获取指定模式的阶段执行顺序。

        Args:
            mode: 工作流模式。

        Returns:
            阶段列表（按执行顺序）。
        """
        sequences: dict[WorkflowMode, list[WorkflowStage]] = {
            "lightweight": [
                WorkflowStage.BRAINSTORMING,
                WorkflowStage.SEARCH,
                WorkflowStage.LIT_REVIEW,
                WorkflowStage.WRITING,
                WorkflowStage.EXPORT,
            ],
            "heavyweight": [
                WorkflowStage.BRAINSTORMING,
                WorkflowStage.SEARCH,
                WorkflowStage.LIT_REVIEW,
                WorkflowStage.STATISTICS,
                WorkflowStage.VISUALIZATION,
                WorkflowStage.WRITING,
                WorkflowStage.POLISHING,
                WorkflowStage.REVIEW,
                WorkflowStage.EXPORT,
            ],
            "hybrid": [
                WorkflowStage.BRAINSTORMING,
                WorkflowStage.SEARCH,
                WorkflowStage.LIT_REVIEW,
                WorkflowStage.STATISTICS,
                WorkflowStage.VISUALIZATION,
                WorkflowStage.WRITING,
                WorkflowStage.POLISHING,
                WorkflowStage.EXPORT,
            ],
        }
        return sequences.get(mode, sequences["lightweight"])

    def _execute_stage(
        self,
        project_name: str,
        stage: WorkflowStage,
        state: ProjectState,
    ) -> None:
        """执行单个工作流阶段（占位实现）。

        实际执行逻辑由具体的 Skill 模块提供，此处仅做状态记录。

        Args:
            project_name: 项目名称。
            stage: 当前阶段。
            state: 项目状态对象。
        """
        # TODO: 接入具体 Skill 实现
        pass

    def _export_markdown(
        self,
        output_path: Path,
        state: ProjectState,
        artifacts: dict[str, list[Path]],
    ) -> None:
        """导出为 Markdown 格式。"""
        lines = [
            f"# {state.name} - 研究报告",
            f"",
            f"- 工作流模式: {state.mode}",
            f"- 导出时间: {state.updated_at}",
            f"",
            f"---",
            f"",
        ]

        for stage, files in artifacts.items():
            lines.append(f"## {stage}")
            lines.append("")
            for f in files:
                lines.append(f"- [{f.name}]({f.relative_to(output_path.parent)})")
            lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")

    def _export_pdf(
        self,
        output_path: Path,
        state: ProjectState,
        artifacts: dict[str, list[Path]],
    ) -> None:
        """导出为 PDF 格式（占位实现）。"""
        # TODO: 接入 PDF 生成库（如 reportlab, weasyprint）
        output_path.write_text(
            f"PDF 导出占位符\n项目: {state.name}\n",
            encoding="utf-8",
        )

    def _export_pptx(
        self,
        output_path: Path,
        state: ProjectState,
        artifacts: dict[str, list[Path]],
    ) -> None:
        """导出为 PPTX 格式（占位实现）。"""
        # TODO: 接入 python-pptx
        output_path.write_text(
            f"PPTX 导出占位符\n项目: {state.name}\n",
            encoding="utf-8",
        )


def main() -> None:
    """CLI 入口函数。"""
    parser = argparse.ArgumentParser(
        description="科研工作流调度器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python orchestrator.py init my_project --mode heavyweight
  python orchestrator.py run my_project --workflow hybrid
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

    # run 命令
    run_parser = subparsers.add_parser("run", help="运行工作流")
    run_parser.add_argument("project_name", help="项目名称")
    run_parser.add_argument("--workflow", help="工作流名称（覆盖默认模式）")
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="从最近的 checkpoint 恢复",
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
        elif args.command == "run":
            orchestrator.run_workflow(
                args.project_name,
                args.workflow,
                args.resume,
            )
        elif args.command == "status":
            orchestrator.show_status(args.project_name)
        elif args.command == "list":
            orchestrator.list_projects()
        elif args.command == "export":
            orchestrator.export(
                args.project_name,
                args.format,
                args.output,
            )
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[中断] 用户取消操作")
        sys.exit(130)


if __name__ == "__main__":
    main()
