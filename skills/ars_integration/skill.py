"""Academic Research Suite (ARS) 集成技能模块。

封装 ARS 的四大组件：Deep Research / Academic Paper /
Academic Paper Reviewer / Academic Pipeline。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.skill_bridge import BaseSkill, SkillOutput, SkillStatus


class DeepResearchMode(Enum):
    """Deep Research 模式。"""

    FULL = "full"  # 完整研究
    QUICK = "quick"  # 快速简报
    SYSTEMATIC_REVIEW = "systematic_review"  # 系统性文献回顾
    SOCRATIC = "socratic"  # 苏格拉底引导
    FACT_CHECK = "fact_check"  # 事实核查
    LIT_REVIEW = "lit_review"  # 文献回顾
    REVIEW = "review"  # 论文审查
    THREE_WAY_SCAN = "three_way_scan"  # 轻量 WHY/HOW/WHAT 论文比较


class AcademicPaperMode(Enum):
    """Academic Paper 模式。"""

    FULL = "full"  # 完整撰写
    PLAN = "plan"  # 引导规划
    OUTLINE_ONLY = "outline_only"  # 只做大纲
    REVISION = "revision"  # 修订
    REVISION_COACH = "revision_coach"  # 修订路线图
    ABSTRACT_ONLY = "abstract_only"  # 摘要
    LIT_REVIEW = "lit_review"  # 文献回顾论文
    FORMAT_CONVERT = "format_convert"  # 格式转换
    CITATION_CHECK = "citation_check"  # 引用检查
    DISCLOSURE = "disclosure"  # AI 使用声明
    REBUTTAL_AUDIT = "rebuttal_audit"  # 审稿回复审计


class ReviewerMode(Enum):
    """Reviewer 模式。"""

    FULL = "full"  # 完整评审
    QUICK = "quick"  # 快速评审
    GUIDED = "guided"  # 引导式
    METHODOLOGY_FOCUS = "methodology_focus"  # 方法论聚焦
    RE_REVIEW = "re_review"  # 再审验收
    CALIBRATION = "calibration"  # 校准


class PipelineStage(Enum):
    """Pipeline 阶段。"""

    STAGE_1_RESEARCH = "stage_1_research"
    STAGE_2_WRITING = "stage_2_writing"
    STAGE_2_5_INTEGRITY_PRE = "stage_2_5_integrity_pre"  # 强制
    STAGE_3_REVIEW_R1 = "stage_3_review_r1"
    STAGE_3_PRIME_REVIEW = "stage_3_prime_review"
    STAGE_4_REVISION = "stage_4_revision"
    STAGE_4_5_INTEGRITY_FINAL = "stage_4_5_integrity_final"  # 强制
    STAGE_5_FINALIZE = "stage_5_finalize"
    STAGE_6_PROCESS_RECORD = "stage_6_process_record"
    POST_PUBLICATION_AUDIT = "post_publication_audit"


@dataclass
class IntegrityCheckResult:
    """学术诚信检查结果。"""

    stage: str
    citation_verified: int = 0
    citation_failed: int = 0
    fabrication_detected: int = 0
    statistical_errors: int = 0
    temporal_inconsistencies: int = 0
    passed: bool = False
    report_path: str = ""


@dataclass
class PipelineCheckpoint:
    """Pipeline checkpoint。"""

    stage: PipelineStage
    timestamp: str
    user_confirmed: bool = False
    artifacts: list[str] = field(default_factory=list)
    next_stage: PipelineStage | None = None


class ARSIntegrationSkill(BaseSkill):
    """ARS 集成技能。

    提供完整的研究-写作-审查 pipeline。
    """

    name: str = "ars_integration"
    version: str = "3.19.0"
    description: str = "Academic Research Suite 集成：Deep Research / Academic Paper / Reviewer / Pipeline"

    # 不可跳过的阶段
    MANDATORY_STAGES = [
        PipelineStage.STAGE_2_5_INTEGRITY_PRE,
        PipelineStage.STAGE_4_5_INTEGRITY_FINAL,
    ]

    def execute(self, input_data: dict[str, Any]) -> SkillOutput:
        """执行 ARS 任务。

        Args:
            input_data: 包含以下键的字典：
                - component: 组件 (deep_research/academic_paper/reviewer/pipeline)
                - mode: 模式
                - research_question: 研究问题
                - manuscript_draft: 手稿草稿（revision/review 时必填）
                - pipeline_state: Pipeline 状态（pipeline 时必填）

        Returns:
            SkillOutput 包含研究结果/论文草稿/审稿报告/诚信验证。
        """
        component = input_data.get("component", "")

        handlers = {
            "deep_research": self._handle_deep_research,
            "academic_paper": self._handle_academic_paper,
            "reviewer": self._handle_reviewer,
            "pipeline": self._handle_pipeline,
        }

        handler = handlers.get(component)
        if handler is None:
            output = self._create_output()
            output.add_error(
                f"未知组件: {component}。支持: {list(handlers.keys())}"
            )
            return output

        return handler(input_data)

    def _handle_deep_research(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理 Deep Research。"""
        output = self._create_output()

        mode = input_data.get("mode", "full")
        question = input_data.get("research_question", "")

        if not question:
            output.add_error("缺少研究问题 (research_question)")
            return output

        research = {
            "mode": mode,
            "research_question": question,
            "agent_count": 13,
            "outputs": {
                "research_brief": "[研究简报]",
                "methodology_blueprint": "[方法论蓝图]",
                "key_findings": ["[关键发现]"],
                "gaps_identified": ["[研究空白]"],
                "recommended_citations": [],
            },
            "validation": {
                "semantic_scholar_verified": True,
                "cross_model_check": False,  # 需设置 ARS_CROSS_MODEL
            },
        }

        output.data = research
        output.add_author_check("请确认研究简报和方法论蓝图是否符合研究意图")
        return output

    def _handle_academic_paper(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理 Academic Paper。"""
        output = self._create_output()

        mode = input_data.get("mode", "full")
        research_brief = input_data.get("research_brief", {})
        manuscript_draft = input_data.get("manuscript_draft", "")

        paper = {
            "mode": mode,
            "agent_count": 12,
            "input_sources": {
                "research_brief_provided": bool(research_brief),
                "manuscript_draft_provided": bool(manuscript_draft),
            },
            "outputs": {
                "outline": "[论文大纲]",
                "draft_sections": {},
                "abstract_zh": "[中文摘要]",
                "abstract_en": "[English Abstract]",
                "references": [],
                "latex_project": "[LaTeX 项目路径]",
            },
            "quality_checks": {
                "style_calibration": "待执行",
                "writing_quality": "待执行",
                "citation_format": "待执行",
                "ai_flavor_check": "待执行",
            },
        }

        output.data = paper
        output.add_author_check("请审阅论文大纲和草稿，确认研究方向和论证逻辑")
        return output

    def _handle_reviewer(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理 Academic Paper Reviewer。"""
        output = self._create_output()

        mode = input_data.get("mode", "full")
        manuscript = input_data.get("manuscript_draft", "")

        if not manuscript:
            output.add_error("缺少手稿草稿 (manuscript_draft)")
            return output

        review = {
            "mode": mode,
            "agent_count": 7,
            "reviewers": [
                {"role": "editor_in_chief", "focus": "整体评估"},
                {"role": "reviewer_1", "focus": "方法论"},
                {"role": "reviewer_2", "focus": "创新性"},
                {"role": "reviewer_3", "focus": "技术细节"},
                {"role": "devil_advocate", "focus": "挑战假设"},
            ],
            "scores": {
                "overall": 0,
                "originality": 0,
                "methodology": 0,
                "significance": 0,
                "clarity": 0,
            },
            "decision_guide": {
                "80_100": "接受",
                "65_79": "小修",
                "50_64": "大修",
                "below_50": "退稿",
            },
            "reports": [],
            "devil_advocate_critique": "[魔鬼代言人 critique]",
        }

        output.data = review
        output.add_author_check("请审阅各审稿人意见和魔鬼代言人 critique")
        return output

    def _handle_pipeline(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理 Academic Pipeline。"""
        output = self._create_output()

        pipeline_state = input_data.get("pipeline_state", {})
        current_stage = pipeline_state.get("current_stage", "stage_1_research")

        # 检查是否在强制阶段
        current_enum = PipelineStage(current_stage) if current_stage in [s.value for s in PipelineStage] else None
        is_mandatory = current_enum in self.MANDATORY_STAGES

        pipeline = {
            "current_stage": current_stage,
            "is_mandatory_stage": is_mandatory,
            "mandatory_stages": [s.value for s in self.MANDATORY_STAGES],
            "checkpoints": [
                {
                    "stage": "stage_1_research",
                    "confirmed": pipeline_state.get("stage_1_confirmed", False),
                    "artifacts": ["research_brief.md"],
                },
                {
                    "stage": "stage_2_writing",
                    "confirmed": pipeline_state.get("stage_2_confirmed", False),
                    "artifacts": ["draft.md", "outline.md"],
                },
                {
                    "stage": "stage_2_5_integrity_pre",
                    "confirmed": pipeline_state.get("stage_2_5_confirmed", False),
                    "artifacts": ["integrity_report_pre.md"],
                    "mandatory": True,
                },
                {
                    "stage": "stage_3_review",
                    "confirmed": pipeline_state.get("stage_3_confirmed", False),
                    "artifacts": ["review_r1.md", "review_r2.md", "review_r3.md"],
                },
                {
                    "stage": "stage_4_revision",
                    "confirmed": pipeline_state.get("stage_4_confirmed", False),
                    "artifacts": ["revised_draft.md", "response_to_reviewers.md"],
                },
                {
                    "stage": "stage_4_5_integrity_final",
                    "confirmed": pipeline_state.get("stage_4_5_confirmed", False),
                    "artifacts": ["integrity_report_final.md"],
                    "mandatory": True,
                },
                {
                    "stage": "stage_5_finalize",
                    "confirmed": pipeline_state.get("stage_5_confirmed", False),
                    "artifacts": ["final_manuscript.pdf"],
                },
            ],
            "integrity_checks": {
                "pre_review": {
                    "citation_verified": 0,
                    "citation_failed": 0,
                    "fabrication_detected": 0,
                    "passed": False,
                },
                "final": {
                    "citation_verified": 0,
                    "citation_failed": 0,
                    "fabrication_detected": 0,
                    "statistical_errors": 0,
                    "passed": False,
                },
            },
            "next_action": self._get_next_action(current_stage),
        }

        output.data = pipeline

        if is_mandatory and not pipeline_state.get(f"{current_stage}_confirmed"):
            output.status = SkillStatus.BLOCKED
            output.add_error(f"当前为强制阶段 {current_stage}，必须完成学术诚信验证后才能继续")

        return output

    def _get_next_action(self, current_stage: str) -> str:
        """获取下一步行动建议。"""
        actions = {
            "stage_1_research": "确认研究简报，进入 Stage 2 写作",
            "stage_2_writing": "确认论文草稿，进入 Stage 2.5 学术诚信审查",
            "stage_2_5_integrity_pre": "完成诚信审查（不可跳过），进入 Stage 3 同行评审",
            "stage_3_review": "审阅审稿意见，进入 Stage 4 修订",
            "stage_4_revision": "确认修订稿，进入 Stage 4.5 最终诚信验证",
            "stage_4_5_integrity_final": "完成最终诚信验证（不可跳过），进入 Stage 5 定稿",
            "stage_5_finalize": "定稿完成，进入 Stage 6 过程记录",
            "stage_6_process_record": "流程完成，可选出版后审计",
        }
        return actions.get(current_stage, "未知阶段")

    def run_integrity_check(
        self,
        manuscript_path: str,
        stage: str = "pre",
    ) -> IntegrityCheckResult:
        """运行学术诚信检查（占位实现）。"""
        return IntegrityCheckResult(
            stage=stage,
            citation_verified=0,
            citation_failed=0,
            fabrication_detected=0,
            statistical_errors=0,
            passed=False,
            report_path=f"{manuscript_path}.integrity_report.md",
        )
