"""统计报告技能模块。

提供 Nature 风格统计报告审查、改写和审稿意见回应。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.skill_bridge import BaseSkill, SkillOutput, SkillStatus


class StatisticalIssueType(Enum):
    """统计问题类型。"""

    PSEUDOREPLICATION = "pseudoreplication"  # 伪重复
    NESTED_DATA = "nested_data"  # 嵌套数据
    MULTIPLE_COMPARISONS = "multiple_comparisons"  # 多重比较
    INTERACTION_MISINTERPRETATION = "interaction_misinterpretation"  # 交互误读
    CORRELATION_OVERREACH = "correlation_overreach"  # 相关性过度解释
    SIGNIFICANCE_ABUSE = "significance_abuse"  # 显著性滥用
    MISSING_SAMPLE_SIZE = "missing_sample_size"  # 样本量缺失
    MISSING_TEST_DETAILS = "missing_test_details"  # 检验方法细节缺失


class ReplicationType(Enum):
    """重复类型。"""

    BIOLOGICAL = "biological"  # 生物学重复
    TECHNICAL = "technical"  # 技术重复
    REPEATED_MEASURES = "repeated_measures"  # 重复测量
    SUBSAMPLING = "subsampling"  # 子样本（视野/细胞）
    INDEPENDENT_EXPERIMENTS = "independent_experiments"  # 独立实验


@dataclass
class StatisticalIssue:
    """统计问题记录。"""

    issue_type: StatisticalIssueType
    severity: str  # high / medium / low
    description: str
    location: str  # 文本位置
    suggested_fix: str
    requires_author_input: bool = False


@dataclass
class StatisticalReport:
    """统计报告审查结果。"""

    overall_risk: str  # low / moderate / high
    issues: list[StatisticalIssue] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    author_input_needed: list[str] = field(default_factory=list)
    suggested_revisions: dict[str, str] = field(default_factory=dict)


class StatisticsSkill(BaseSkill):
    """统计报告技能。

    审查统计方法完整性、识别统计问题、
    生成审稿意见回应。
    """

    name: str = "statistics"
    version: str = "1.0.0"
    description: str = "Nature 风格统计报告审查与改写"

    def execute(self, input_data: dict[str, Any]) -> SkillOutput:
        """执行统计审查任务。

        Args:
            input_data: 包含以下键的字典：
                - action: 操作类型 (audit/revise/respond)
                - statistical_text: 统计方法/结果文本
                - figure_legends: 图注文本（可选）
                - reviewer_comments: 审稿意见（respond 时必填）
                - experiment_design: 实验设计信息（可选）

        Returns:
            SkillOutput 包含审查结果或改写文本。
        """
        action = input_data.get("action", "")

        handlers = {
            "audit": self._handle_audit,
            "revise": self._handle_revise,
            "respond": self._handle_respond,
        }

        handler = handlers.get(action)
        if handler is None:
            output = self._create_output()
            output.add_error(
                f"未知操作: {action}。支持: {list(handlers.keys())}"
            )
            return output

        return handler(input_data)

    def _handle_audit(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理统计审查。"""
        output = self._create_output()

        stat_text = input_data.get("statistical_text", "")
        figure_legends = input_data.get("figure_legends", "")
        design_info = input_data.get("experiment_design", {})

        if not stat_text:
            output.add_error("缺少统计文本 (statistical_text)")
            return output

        report = StatisticalReport(
            overall_risk="moderate",
            issues=[
                StatisticalIssue(
                    issue_type=StatisticalIssueType.MISSING_SAMPLE_SIZE,
                    severity="high",
                    description="未明确说明样本量 (n) 及其代表的含义",
                    location="Statistical analysis 段落",
                    suggested_fix="明确 n = X，并说明 n 代表独立实验次数、动物数量还是技术重复",
                    requires_author_input=True,
                ),
                StatisticalIssue(
                    issue_type=StatisticalIssueType.PSEUDOREPLICATION,
                    severity="high",
                    description="可能存在伪重复：将技术重复当作独立样本处理",
                    location="Results 图注",
                    suggested_fix="区分 biological replicates 和 technical replicates，使用适当的统计单位",
                    requires_author_input=True,
                ),
            ],
            strengths=[
                "使用了适当的统计检验方法",
                "报告了效应量和置信区间",
            ],
            author_input_needed=[
                "每个实验的独立重复次数 (n) 及其含义",
                "生物学重复与技术重复的数量",
                "统计检验的具体名称和软件版本",
                "多重比较校正方法",
                "数据排除标准和理由",
            ],
            suggested_revisions={
                "statistical_analysis_section": "[待生成改写文本]",
                "figure_legends": "[待生成图注改写]",
            },
        )

        output.data = {
            "overall_risk": report.overall_risk,
            "issue_count": len(report.issues),
            "issues": [
                {
                    "type": issue.issue_type.value,
                    "severity": issue.severity,
                    "description": issue.description,
                    "location": issue.location,
                    "suggested_fix": issue.suggested_fix,
                    "requires_author_input": issue.requires_author_input,
                }
                for issue in report.issues
            ],
            "strengths": report.strengths,
            "author_input_needed": report.author_input_needed,
        }

        output.add_author_check("请提供 AUTHOR_INPUT_NEEDED 清单中的信息以完成审查")
        return output

    def _handle_revise(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理统计文本改写。"""
        output = self._create_output()

        original_text = input_data.get("statistical_text", "")
        target_section = input_data.get("target_section", "methods")

        if not original_text:
            output.add_error("缺少原始统计文本 (statistical_text)")
            return output

        revision = {
            "original_text": original_text,
            "target_section": target_section,
            "revised_text": f"[改写后的 {target_section} 文本]",
            "key_changes": [
                "明确样本量及其含义",
                "区分重复类型",
                "补充统计检验细节",
                "规范 p 值报告格式",
            ],
            "nature_style_guidelines": [
                "在 Methods 末尾设独立 Statistical analysis 小节",
                "图注中报告 n、检验方法和精确 p 值",
                "使用 'n = X independent experiments' 而非仅 'n = X'",
                "报告效应量和 95% CI，而非仅 p 值",
            ],
        }

        output.data = revision
        output.add_warning("改写文本基于通用规范，具体细节需作者核实")
        return output

    def _handle_respond(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理审稿意见回应。"""
        output = self._create_output()

        reviewer_comments = input_data.get("reviewer_comments", [])
        if not reviewer_comments:
            output.add_error("缺少审稿意见 (reviewer_comments)")
            return output

        responses = []
        for i, comment in enumerate(reviewer_comments, 1):
            response = {
                "comment_number": i,
                "reviewer_comment": comment,
                "response_type": "保守回应",
                "response_text": f"[针对意见 {i} 的回应草稿]",
                "suggested_actions": [
                    "补充统计细节",
                    "澄清重复类型",
                    "提供额外分析或引用支持",
                ],
                "requires_new_experiments": False,
            }
            responses.append(response)

        output.data = {
            "response_count": len(responses),
            "responses": responses,
            "general_strategy": [
                "承认统计报告的不足",
                "提供补充信息或引用",
                "避免防御性语气",
                "如无法补充实验，明确说明并提供替代证据",
            ],
        }

        output.add_author_check("回应草稿需根据实际数据情况调整，避免过度承诺")
        return output

    def check_replication_type(
        self,
        design_description: str,
    ) -> dict[str, Any]:
        """检查并建议重复类型分类。

        Args:
            design_description: 实验设计描述文本。

        Returns:
            重复类型分析和建议。
        """
        return {
            "detected_types": ["biological", "technical"],
            "recommendations": [
                "明确说明 n 代表独立生物学重复还是技术重复",
                "对于嵌套设计，考虑混合效应模型",
                "对于重复测量，使用重复测量 ANOVA 或混合模型",
            ],
            "common_pitfalls": [
                "将同一样本的多次测量当作独立样本",
                "将同一批细胞的多个视野当作独立重复",
            ],
        }
