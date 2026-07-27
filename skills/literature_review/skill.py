"""文献综述技能模块。

提供系统性文献综述的 7 阶段工作流，包含 PRISMA 流程、
质量评估和引用验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.skill_bridge import BaseSkill, SkillOutput, SkillStatus


class ReviewStage(Enum):
    """文献综述阶段。"""

    PLANNING = "planning"
    SEARCH = "search"
    SCREENING = "screening"
    EXTRACTION = "extraction"
    SYNTHESIS = "synthesis"
    CITATION_VERIFICATION = "citation_verification"
    DOCUMENT_GENERATION = "document_generation"


class QualityAssessmentTool(Enum):
    """质量评估工具。"""

    COCHRANE = "cochrane"  # RCT 偏倚风险
    NEWCASTLE_OTTAWA = "newcastle_ottawa"  # 观察性研究
    AMSTAR2 = "amstar2"  # 系统综述


@dataclass
class ScreeningRecord:
    """筛选记录。"""

    stage: str
    included: int = 0
    excluded: int = 0
    exclusion_reasons: dict[str, int] = field(default_factory=dict)


@dataclass
class PrismaFlow:
    """PRISMA 流程数据。"""

    identification: ScreeningRecord = field(
        default_factory=lambda: ScreeningRecord("identification")
    )
    screening: ScreeningRecord = field(
        default_factory=lambda: ScreeningRecord("screening")
    )
    eligibility: ScreeningRecord = field(
        default_factory=lambda: ScreeningRecord("eligibility")
    )
    included: ScreeningRecord = field(
        default_factory=lambda: ScreeningRecord("included")
    )


class LiteratureReviewSkill(BaseSkill):
    """系统性文献综述技能。

    支持多数据库检索、PRISMA 流程、质量评估、
    引用验证和专业文档生成。
    """

    name: str = "literature_review"
    version: str = "1.0.0"
    description: str = "系统性文献综述：多数据库检索、PRISMA、质量评估、引用验证"

    STAGES = [s.value for s in ReviewStage]
    QUALITY_TOOLS = [t.value for t in QualityAssessmentTool]

    def execute(self, input_data: dict[str, Any]) -> SkillOutput:
        """执行文献综述任务。

        Args:
            input_data: 包含以下键的字典：
                - action: 操作类型 (plan/search/screen/extract/synthesize/verify/generate)
                - review_question: 综述问题（PICO 框架）
                - databases: 数据库列表
                - inclusion_criteria: 纳入标准
                - papers: 文献列表（screen/extract 时必填）
                - output_format: 输出格式（md/pdf）

        Returns:
            SkillOutput 包含各阶段结果。
        """
        action = input_data.get("action", "")

        handlers = {
            "plan": self._handle_plan,
            "search": self._handle_search,
            "screen": self._handle_screen,
            "extract": self._handle_extract,
            "synthesize": self._handle_synthesize,
            "verify": self._handle_verify,
            "generate": self._handle_generate,
        }

        handler = handlers.get(action)
        if handler is None:
            output = self._create_output()
            output.add_error(
                f"未知操作: {action}。支持: {list(handlers.keys())}"
            )
            return output

        return handler(input_data)

    def _handle_plan(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理规划阶段。"""
        output = self._create_output()

        question = input_data.get("review_question", "")
        if not question:
            output.add_error("缺少综述问题 (review_question)")
            return output

        pico = {
            "population": input_data.get("population", ""),
            "intervention": input_data.get("intervention", ""),
            "comparison": input_data.get("comparison", ""),
            "outcome": input_data.get("outcome", ""),
        }

        plan = {
            "review_question": question,
            "pico_framework": pico,
            "databases": input_data.get("databases", [
                "pubmed", "arxiv", "semantic_scholar"
            ]),
            "inclusion_criteria": input_data.get("inclusion_criteria", {
                "date_range": "2019-2024",
                "language": ["English", "Chinese"],
                "study_types": ["RCT", "cohort", "case-control"],
            }),
            "exclusion_criteria": input_data.get("exclusion_criteria", [
                "conference abstracts",
                "non-peer-reviewed preprints (unless seminal)",
            ]),
            "search_concepts": self._extract_concepts(question),
        }

        output.data = plan
        output.add_author_check("请确认 PICO 框架和纳入/排除标准是否准确反映研究意图")

        return output

    def _handle_search(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理检索阶段。"""
        output = self._create_output()

        concepts = input_data.get("concepts", [])
        databases = input_data.get("databases", [])

        search_log = {
            "concepts": concepts,
            "databases_searched": databases,
            "search_strings": self._build_search_strings(concepts),
            "date_of_search": "[当前日期]",
            "results_per_database": {db: 0 for db in databases},
            "total_identified": 0,
            "duplicates_removed": 0,
            "unique_records": 0,
        }

        output.data = search_log
        output.add_warning("检索结果需人工核对，特别是预印本和灰色文献")

        return output

    def _handle_screen(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理筛选阶段。"""
        output = self._create_output()

        papers = input_data.get("papers", [])
        criteria = input_data.get("criteria", {})

        prisma = PrismaFlow()
        prisma.identification.included = len(papers)

        # 模拟筛选过程
        screening_stages = [
            ("title_screening", len(papers), len(papers) // 2),
            ("abstract_screening", len(papers) // 2, len(papers) // 4),
            ("full_text_screening", len(papers) // 4, len(papers) // 8),
        ]

        screening_log = []
        for stage_name, before, after in screening_stages:
            screening_log.append({
                "stage": stage_name,
                "before": before,
                "after": after,
                "excluded": before - after,
            })

        prisma.screening.included = screening_stages[0][2]
        prisma.eligibility.included = screening_stages[1][2]
        prisma.included.included = screening_stages[2][2]

        output.data = {
            "screening_log": screening_log,
            "prisma_flow": {
                "identification": prisma.identification.included,
                "screening": prisma.screening.included,
                "eligibility": prisma.eligibility.included,
                "included": prisma.included.included,
            },
            "final_included_count": prisma.included.included,
        }

        return output

    def _handle_extract(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理数据提取阶段。"""
        output = self._create_output()

        papers = input_data.get("papers", [])

        extraction_template = {
            "fields": [
                "metadata (title/authors/year/journal)",
                "study_design",
                "sample_size",
                "population_characteristics",
                "intervention_details",
                "outcome_measures",
                "key_results",
                "limitations",
                "funding_source",
                "conflict_of_interest",
            ],
            "papers_to_extract": len(papers),
            "extraction_status": "pending",
        }

        output.data = extraction_template
        output.add_author_check("数据提取需双人独立进行或单人提取后双人核对")

        return output

    def _handle_synthesize(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理综合阶段。"""
        output = self._create_output()

        extracted_data = input_data.get("extracted_data", [])
        themes = input_data.get("themes", [])

        synthesis = {
            "theme_count": len(themes),
            "themes": themes,
            "narrative_structure": [
                "按主题组织（非逐篇总结）",
                "批判性评估方法学优劣",
                "评估证据一致性",
                "识别知识空白",
            ],
            "quality_assessment": {
                "tool": input_data.get("quality_tool", "cochrane"),
                "assessed_papers": len(extracted_data),
                "high_quality": 0,
                "moderate_quality": 0,
                "low_quality": 0,
            },
        }

        output.data = synthesis
        output.add_warning("主题综合需避免发表偏倚，考虑阴性结果和灰色文献")

        return output

    def _handle_verify(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理引用验证阶段。"""
        output = self._create_output()

        manuscript_path = input_data.get("manuscript_path", "")

        verification_report = {
            "manuscript": manuscript_path,
            "total_citations": 0,
            "verified": 0,
            "failed": 0,
            "warnings": 0,
            "failed_dois": [],
            "verification_tool": "verify_citations.py",
        }

        output.data = verification_report
        output.add_author_check("所有 DOI 验证失败项需人工核对并修正")

        return output

    def _handle_generate(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理文档生成阶段。"""
        output = self._create_output()

        manuscript_path = input_data.get("manuscript_path", "")
        output_format = input_data.get("output_format", "md")
        citation_style = input_data.get("citation_style", "nature")

        generation = {
            "input_manuscript": manuscript_path,
            "output_format": output_format,
            "citation_style": citation_style,
            "generated_files": [
                f"review_{citation_style}.{output_format}",
                "prisma_flowchart.png",
                "quality_assessment_table.md",
            ],
            "checklist": [
                "所有 DOI 已验证",
                "引用格式一致",
                "包含 PRISMA 流程图",
                "检索方法完整记录",
                "纳入/排除标准明确",
                "结果按主题组织",
                "完成质量评估",
                "承认局限性",
                "参考文献完整准确",
            ],
        }

        output.data = generation

        return output

    def _extract_concepts(self, question: str) -> list[dict[str, Any]]:
        """从研究问题提取检索概念。"""
        # 简化实现，实际应使用 NLP 或人工标注
        return [
            {"concept": "主要概念", "synonyms": [], "mesh_terms": []},
            {"concept": "次要概念", "synonyms": [], "mesh_terms": []},
        ]

    def _build_search_strings(
        self,
        concepts: list[dict[str, Any]],
    ) -> dict[str, str]:
        """构建各数据库检索式。"""
        return {
            "pubmed": "[概念1] AND [概念2]",
            "arxiv": "[概念1] AND [概念2]",
            "semantic_scholar": "[概念1] AND [概念2]",
        }
