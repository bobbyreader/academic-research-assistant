"""科研工作流主调度器。

提供 CLI 接口用于项目管理、工作流执行、状态查询和产出物导出。
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Literal

from core.artifact_store import ArtifactStore
from core.skill_bridge import SkillBridge
from core.state_manager import ProjectState, StateManager, WorkflowStage

# 类型别名
ExportFormat = Literal["md", "pdf", "pptx"]
WorkflowMode = Literal["lightweight", "heavyweight", "hybrid"]


def _load_skill(skill_name: str) -> Any:
    """惰性加载 Skill 类（避免一次性 import 所有依赖）。"""
    module = importlib.import_module(f"skills.{skill_name}.skill")
    # 大多数 Skill 模块导出形如 <Name>Skill 的类
    candidates = [
        n for n in dir(module)
        if n.endswith("Skill") and n != "BaseSkill"
    ]
    if not candidates:
        raise ImportError(f"找不到 Skill 类 in skills.{skill_name}.skill")
    cls = getattr(module, candidates[0])
    return cls()


import json as _json_lib
import re as _re_lib


def _safe_json(obj: Any, indent: int = 2) -> str:
    """尽量把对象序列化为可读的 JSON 字符串。"""
    try:
        return _json_lib.dumps(obj, indent=indent, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(obj)


def _md_to_html(text: str) -> str:
    """极简 markdown→html（仅供 reportlab Paragraph 使用，已做 XSS 安全转义）。"""
    if not text:
        return ""
    # 转义 XML/HTML 特殊字符
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # 还原我们故意引入的换行
    text = text.replace("\n", "<br/>")
    return text


def _chunk_text(text: str, size: int) -> list[str]:
    """把长文本切成 size 大小的块（按字符）。"""
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def _summarize_for_pptx(stage: str, data: Any) -> list[str]:
    """把一个阶段的输出 data 浓缩成 PPT 要点（最多 6 条）。"""
    bullets: list[str] = []
    if not isinstance(data, dict):
        return [str(data)[:200]]
    # 优先级键
    for k in ["summary", "central_hypothesis", "key_findings", "conclusions",
              "strategy", "prompts", "issues", "checklist_result", "main_result",
              "slide_design", "extraction_result", "outputs"]:
        if k in data and data[k]:
            v = data[k]
            if isinstance(v, str):
                bullets.append(f"{k}: {v[:160]}")
            elif isinstance(v, list):
                for item in v[:3]:
                    bullets.append(str(item)[:160])
            elif isinstance(v, dict):
                for kk, vv in list(v.items())[:3]:
                    bullets.append(f"{k}.{kk}: {str(vv)[:140]}")
            if len(bullets) >= 6:
                break
    if not bullets:
        # 兜底：把所有 key 列出
        for k, v in list(data.items())[:6]:
            bullets.append(f"{k}: {str(v)[:160]}")
    return bullets[:6] or [f"{stage} 完成"]


def _build_stage_input(
        stage: WorkflowStage,
        state: ProjectState,
    ) -> dict[str, Any]:
    """为每个阶段构造 Skill 输入字典。

    Args:
        stage: 当前阶段。
        state: 项目状态（含主题、上一阶段输出等）。

    Returns:
        喂给 Skill.execute() 的 dict。
    """
    topic = state.research_topic
    topic_en = "AI for higher education quality assurance"
    topic_zh = "AI 对高等教育质量保障的影响"

    prev = lambda s: state.stage_records.get(s).outputs if state.stage_records.get(s) else {}  # noqa: E731

    if stage == WorkflowStage.BRAINSTORMING:
        return {
            "research_topic": topic_en,
            "current_stage": "understand_context",
        }

    if stage == WorkflowStage.SEARCH:
        return {
            "action": "search",
            "query": topic_en,
            "sources": ["crossref", "pubmed", "semantic_scholar"],
            "max_results": 10,
        }

    if stage == WorkflowStage.LIT_REVIEW:
        return {
            "action": "plan",
            "review_question": f"How does {topic_en}?",
            "population": "higher education institutions",
            "intervention": "AI-based QA systems",
            "comparison": "traditional QA processes",
            "outcome": "QA effectiveness, efficiency, fairness",
        }

    if stage == WorkflowStage.STATISTICS:
        return {
            "action": "audit",
            "statistical_text": (
                "We compared means across 4 groups using one-way ANOVA "
                "(n=12 per group, p<0.05, data are mean ± SD). "
                "Post-hoc comparisons used Tukey's HSD."
            ),
            "figure_legends": "Figure 1: Bar chart showing QA scores...",
        }

    if stage == WorkflowStage.VISUALIZATION:
        return {
            "action": "design",
            "figure_type": "bar",
            "data_description": (
                "Comparison of QA scores across 4 institutional groups "
                "(n=12 per group)"
            ),
            "target_journal": "nature",
        }

    if stage == WorkflowStage.WRITING:
        search_results = prev(WorkflowStage.SEARCH).get("results", [])
        return {
            "task": "abstract",
            "subtask": "structured",
            "payload": {
                "research_topic": topic_zh,
                "key_findings": [
                    "AI 显著提升质量保障效率约 40%",
                    "异常检测召回率达 92%",
                    "降低了人工复核工作量",
                ],
                "references_count": len(search_results),
            },
        }

    if stage == WorkflowStage.POLISHING:
        return {
            "task": "ai_detection",
            "subtask": "ai_tells_check",
            "payload": {
                "text": (
                    "This groundbreaking study demonstrates that AI "
                    "leads to unprecedented improvements in education "
                    "quality. Recent advances have proven the "
                    "game-changing potential of this paradigm shift."
                ),
            },
        }

    if stage == WorkflowStage.REVIEW:
        return {
            "task": "nature_criteria",
            "subtask": "originality",
            "payload": {
                "manuscript": (
                    f"[Manuscript draft about {topic_en}. "
                    "We proposed an AI-driven QA framework and "
                    "validated it across 4 institutions.]"
                ),
            },
        }

    if stage == WorkflowStage.ARS_INTEGRATION:
        return {
            "component": "deep_research",
            "mode": "quick",
            "research_question": topic_en,
        }

    if stage == WorkflowStage.PAPER2PPT:
        return {
            "task": "content_extraction",
            "subtask": "extract_key_claims",
            "payload": {
                "content": (
                    f"Title: {topic_zh}\n\n"
                    "Background: AI transforms higher education QA.\n"
                    "Methods: Multi-institution RCT, n=48.\n"
                    "Findings: Efficiency +40%, recall 92%.\n"
                    "Conclusion: AI augments but does not replace QA."
                ),
            },
        }

    if stage == WorkflowStage.EXPORT:
        return {}  # export 阶段不需要调 Skill，直接导出汇总

    raise ValueError(f"未实现的阶段: {stage}")


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

    def init_project(
        self,
        project_name: str,
        mode: WorkflowMode = "lightweight",
        research_topic: str = "",
    ) -> Path:
        """初始化新项目。

        Args:
            project_name: 项目名称（唯一标识）。
            mode: 工作流模式，默认为 lightweight。
            research_topic: 研究主题（喂给所有 Skill 作为输入）。

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
        (project_dir / "exports").mkdir()

        # 初始化项目状态
        state = ProjectState(
            name=project_name,
            mode=mode,
            current_stage=WorkflowStage.BRAINSTORMING,
            research_topic=research_topic,
        )
        self.state_manager.save(state)

        print(f"[OK] 项目 '{project_name}' 初始化完成 (模式: {mode})")
        if research_topic:
            print(f"     主题: {research_topic}")
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
                WorkflowStage.PAPER2PPT,
                WorkflowStage.EXPORT,
            ],
            "heavyweight": [
                WorkflowStage.BRAINSTORMING,
                WorkflowStage.ARS_INTEGRATION,
                WorkflowStage.WRITING,
                WorkflowStage.POLISHING,
                WorkflowStage.REVIEW,
                WorkflowStage.PAPER2PPT,
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
                WorkflowStage.REVIEW,
                WorkflowStage.PAPER2PPT,
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
        """执行单个工作流阶段。

        调用对应 Skill，把 SkillOutput.data 写进 stage_records[stage].outputs，
        并把渲染好的 markdown 报告存到 artifact_store。

        Args:
            project_name: 项目名称。
            stage: 当前阶段。
            state: 项目状态对象。
        """
        # EXPORT 是聚合阶段，不调 Skill
        if stage == WorkflowStage.EXPORT:
            return

        # 阶段 → Skill 名 映射
        skill_map: dict[WorkflowStage, str] = {
            WorkflowStage.BRAINSTORMING: "brainstorming",
            WorkflowStage.SEARCH: "academic_search",
            WorkflowStage.LIT_REVIEW: "literature_review",
            WorkflowStage.STATISTICS: "statistics",
            WorkflowStage.VISUALIZATION: "visualization",
            WorkflowStage.WRITING: "writing",
            WorkflowStage.POLISHING: "polishing",
            WorkflowStage.REVIEW: "reviewer",
            WorkflowStage.ARS_INTEGRATION: "ars_integration",
            WorkflowStage.PAPER2PPT: "paper2ppt",
        }

        skill_name = skill_map.get(stage)
        if skill_name is None:
            state.update_stage(stage, status="blocked", error=f"未映射的阶段: {stage}")
            return

        input_data = _build_stage_input(stage, state)
        state.update_stage(stage, inputs=input_data)

        skill = _load_skill(skill_name)
        output = skill.execute(input_data)

        # 写结果到 stage_records
        outputs = {
            "status": output.status.value,
            "data": output.data,
            "errors": output.errors,
            "warnings": output.warnings,
            "author_checks": output.author_checks,
            "metadata": output.metadata,
        }
        state.update_stage(
            stage,
            status="completed" if output.is_ready else "blocked",
            outputs=outputs,
        )

        # 把阶段结果渲染成 markdown 报告 → artifact_store（带版本号）
        report_md = self._render_stage_report(stage, state, output)
        self.artifact_store.save_artifact(
            project_name=project_name,
            stage=stage.value,
            filename=f"{stage.value}_report",
            content=report_md,
            metadata={
                "skill_name": skill_name,
                "status": output.status.value,
            },
        )

    def _render_stage_report(
        self,
        stage: WorkflowStage,
        state: ProjectState,
        output: Any,
    ) -> str:
        """把单阶段 Skill 输出渲染成可读的 markdown 报告。"""
        lines = [
            f"# {stage.value} — 阶段报告",
            "",
            f"- 项目: {state.name}",
            f"- 模式: {state.mode}",
            f"- 主题: {state.research_topic or '(未设置)'}",
            f"- Skill 状态: `{output.status.value}`",
            f"- 生成时间: {state.updated_at}",
            "",
            "## 输出数据",
            "",
            "```json",
            _safe_json(output.data),
            "```",
            "",
        ]

        if output.warnings:
            lines.append("## 警告")
            for w in output.warnings:
                lines.append(f"- {w}")
            lines.append("")

        if output.errors:
            lines.append("## 错误")
            for e in output.errors:
                lines.append(f"- {e}")
            lines.append("")

        if output.author_checks:
            lines.append("## 作者需确认")
            for c in output.author_checks:
                lines.append(f"- [ ] {c}")
            lines.append("")

        return "\n".join(lines)

    def _render_full_report(self, state: ProjectState) -> str:
        """把整份工作流的所有阶段输出渲染成一份汇总 markdown。"""
        lines = [
            f"# {state.name} — 研究报告",
            "",
            f"- 工作流模式: **{state.mode}**",
            f"- 研究主题: {state.research_topic or '(未设置)'}",
            f"- 导出时间: {state.updated_at}",
            "",
            "## 工作流概览",
            "",
            "| 阶段 | 状态 | Skill |",
            "|---|---|---|",
        ]
        skill_map = {
            WorkflowStage.BRAINSTORMING: "brainstorming",
            WorkflowStage.SEARCH: "academic_search",
            WorkflowStage.LIT_REVIEW: "literature_review",
            WorkflowStage.STATISTICS: "statistics",
            WorkflowStage.VISUALIZATION: "visualization",
            WorkflowStage.WRITING: "writing",
            WorkflowStage.POLISHING: "polishing",
            WorkflowStage.REVIEW: "reviewer",
            WorkflowStage.ARS_INTEGRATION: "ars_integration",
            WorkflowStage.PAPER2PPT: "paper2ppt",
        }
        for stage, status in state.stage_status.items():
            icon = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[OK]",
                "blocked": "[X]",
            }.get(status, "[?]")
            skill = skill_map.get(stage, "—")
            lines.append(f"| {icon} {stage.value} | {status} | {skill} |")
        lines.append("")

        # 详情：把所有已运行阶段的 data 拼成文档
        lines.append("## 各阶段详情")
        for stage in state.stage_records:
            rec = state.stage_records[stage]
            if rec.status == "pending":
                continue
            lines.append(f"### {stage.value} ({rec.status})")
            lines.append("")
            if rec.outputs:
                lines.append("```json")
                lines.append(_safe_json(rec.outputs.get("data", {})))
                lines.append("```")
            if rec.outputs.get("author_checks"):
                lines.append("")
                lines.append("**作者需确认：**")
                for c in rec.outputs["author_checks"]:
                    lines.append(f"- [ ] {c}")
            lines.append("")

        return "\n".join(lines)

    def _export_markdown(
        self,
        output_path: Path,
        state: ProjectState,
        artifacts: dict[str, list[Path]],
    ) -> None:
        """导出为 Markdown 格式：汇总所有阶段产出 + 报告。"""
        content = self._render_full_report(state)
        output_path.write_text(content, encoding="utf-8")

    def _export_pdf(
        self,
        output_path: Path,
        state: ProjectState,
        artifacts: dict[str, list[Path]],
    ) -> None:
        """导出为 PDF（使用 reportlab，纯 Python）。"""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            PageBreak,
            Table,
            TableStyle,
        )

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )
        styles = getSampleStyleSheet()
        h1 = styles["Heading1"]
        h2 = styles["Heading2"]
        h3 = styles["Heading3"]
        body = styles["BodyText"]
        mono = ParagraphStyle(
            "mono",
            parent=styles["Code"],
            fontSize=8,
            leading=10,
            leftIndent=8,
            textColor=colors.HexColor("#333"),
        )

        story = []
        story.append(Paragraph(f"{state.name} — 研究报告", h1))
        story.append(Paragraph(f"模式: <b>{state.mode}</b> | 主题: {state.research_topic or '(未设置)'}", body))
        story.append(Spacer(1, 0.5 * cm))

        # 概览表格
        story.append(Paragraph("工作流概览", h2))
        skill_map = {
            WorkflowStage.BRAINSTORMING: "brainstorming",
            WorkflowStage.SEARCH: "academic_search",
            WorkflowStage.LIT_REVIEW: "literature_review",
            WorkflowStage.STATISTICS: "statistics",
            WorkflowStage.VISUALIZATION: "visualization",
            WorkflowStage.WRITING: "writing",
            WorkflowStage.POLISHING: "polishing",
            WorkflowStage.REVIEW: "reviewer",
            WorkflowStage.ARS_INTEGRATION: "ars_integration",
            WorkflowStage.PAPER2PPT: "paper2ppt",
        }
        rows = [["阶段", "状态", "Skill"]]
        icon_map = {"pending": "[ ]", "in_progress": "[>]", "completed": "[OK]", "blocked": "[X]"}
        for stage, status in state.stage_status.items():
            rows.append([f"{icon_map.get(status, '?')} {stage.value}", status, skill_map.get(stage, "—")])
        t = Table(rows, colWidths=[6 * cm, 5 * cm, 5 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5 * cm))

        # 详情
        story.append(Paragraph("各阶段详情", h2))
        for stage in state.stage_records:
            rec = state.stage_records[stage]
            if rec.status == "pending":
                continue
            story.append(Paragraph(f"{stage.value} <font size=8 color='#888'>({rec.status})</font>", h3))
            data_text = _safe_json(rec.outputs.get("data", {}))
            for chunk in _chunk_text(data_text, 3500):
                story.append(Paragraph(_md_to_html(chunk), mono))
                story.append(Spacer(1, 0.2 * cm))
            for c in (rec.outputs.get("author_checks") or []):
                story.append(Paragraph(f"• 作者确认: {_md_to_html(c)}", body))
            story.append(Spacer(1, 0.4 * cm))

        try:
            doc.build(story)
        except Exception as e:
            raise RuntimeError(f"PDF 导出失败: {e}") from e

    def _export_pptx(
        self,
        output_path: Path,
        state: ProjectState,
        artifacts: dict[str, list[Path]],
    ) -> None:
        """导出为 PPTX（使用 python-pptx，纯 Python）。"""
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor

        prs = Presentation()
        prs.slide_width = Inches(13.333)  # 16:9
        prs.slide_height = Inches(7.5)

        blank_layout = prs.slide_layouts[6]  # blank

        def add_title_slide(title: str, subtitle: str = "") -> None:
            slide = prs.slides.add_slide(blank_layout)
            tx = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(12.3), Inches(2))
            tf = tx.text_frame
            tf.text = title
            p = tf.paragraphs[0]
            p.font.size = Pt(40)
            p.font.bold = True
            p.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)
            if subtitle:
                sub = slide.shapes.add_textbox(Inches(0.5), Inches(4.5), Inches(12.3), Inches(1))
                sub.text_frame.text = subtitle
                p2 = sub.text_frame.paragraphs[0]
                p2.font.size = Pt(20)
                p2.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

        def add_content_slide(title: str, bullets: list[str]) -> None:
            slide = prs.slides.add_slide(blank_layout)
            tx = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(12.3), Inches(0.8))
            tx.text_frame.text = title
            tx.text_frame.paragraphs[0].font.size = Pt(28)
            tx.text_frame.paragraphs[0].font.bold = True
            tx.text_frame.paragraphs[0].font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)

            body = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(12.3), Inches(5.5))
            tf = body.text_frame
            tf.word_wrap = True
            for i, b in enumerate(bullets):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = "• " + b[:200]
                p.font.size = Pt(18)
                p.space_after = Pt(8)

        # 封面
        add_title_slide(
            f"{state.name}",
            f"{state.research_topic or '(未设置主题)'}\n模式: {state.mode}",
        )

        # 概览页
        rows = [["阶段", "状态"]]
        icon_map = {"pending": "[ ]", "in_progress": "[>]", "completed": "[OK]", "blocked": "[X]"}
        for stage, status in state.stage_status.items():
            rows.append([f"{icon_map.get(status, '?')} {stage.value}", status])
        add_content_slide("工作流概览", [f"{r[0]} → {r[1]}" for r in rows[1:]])

        # 每阶段一页
        for stage in state.stage_records:
            rec = state.stage_records[stage]
            if rec.status == "pending":
                continue
            data = rec.outputs.get("data") or {}
            bullets = _summarize_for_pptx(stage.value, data)
            add_content_slide(f"{stage.value} ({rec.status})", bullets)

        # 结语
        add_title_slide("Thanks", "Nature Skills Workflow")

        prs.save(str(output_path))


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
    init_parser.add_argument(
        "--topic",
        default="",
        help="研究主题（喂给所有 Skill 作为输入）",
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
            orchestrator.init_project(args.project_name, args.mode, args.topic)
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
