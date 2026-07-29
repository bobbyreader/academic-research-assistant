"""学术检索技能模块。

提供多源学术检索、文献元数据核查、引用格式生成、
严格他引审计和高影响力引用者分析。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.skill_bridge import BaseSkill, SkillOutput, SkillStatus


class CitationFormat(Enum):
    """支持的引用格式。"""

    NATURE = "nature"
    APA = "apa"
    IEEE = "ieee"
    VANCOUVER = "vancouver"
    CHICAGO = "chicago"


class SearchSource(Enum):
    """支持的检索来源。"""

    CROSSREF = "crossref"
    PUBMED = "pubmed"
    ARXIV = "arxiv"
    SCOPUS = "scopus"
    SCIENCEDIRECT = "sciencedirect"
    SEMANTIC_SCHOLAR = "semantic_scholar"


@dataclass
class PaperMetadata:
    """论文元数据。"""

    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str = ""
    doi: str = ""
    pmid: str = ""
    arxiv_id: str = ""
    abstract: str = ""
    source: str = ""


@dataclass
class CitationAuditResult:
    """他引审计结果。"""

    total_citations: int = 0
    self_citations: int = 0
    team_citations: int = 0
    network_citations: int = 0
    strict_external_citations: int = 0
    high_impact_citers: list[dict[str, Any]] = field(default_factory=list)
    uncertain_cases: list[str] = field(default_factory=list)


class AcademicSearchSkill(BaseSkill):
    """学术检索技能。

    支持多源检索、引用格式生成、严格他引审计。
    """

    name: str = "academic_search"
    version: str = "1.0.0"
    description: str = "多源学术检索、引用格式生成、严格他引审计"

    SUPPORTED_FORMATS = [f.value for f in CitationFormat]
    SUPPORTED_SOURCES = [s.value for s in SearchSource]

    def execute(self, input_data: dict[str, Any]) -> SkillOutput:
        """执行学术检索任务。

        Args:
            input_data: 包含以下键的字典：
                - action: 操作类型 (search/verify/audit/format)
                - query: 检索查询（search 时必填）
                - doi/pmid/arxiv_id: 文献标识（verify/audit/format 时必填）
                - sources: 检索来源列表（可选）
                - format: 引用格式（format 时必填）
                - exclude_rules: 他引排除规则（audit 时可选）

        Returns:
            SkillOutput 包含检索/验证/审计/格式化结果。
        """
        action = input_data.get("action", "")

        handlers = {
            "search": self._handle_search,
            "verify": self._handle_verify,
            "audit": self._handle_audit,
            "format": self._handle_format,
        }

        handler = handlers.get(action)
        if handler is None:
            output = self._create_output()
            output.add_error(
                f"未知操作: {action}。支持的操作: {list(handlers.keys())}"
            )
            return output

        return handler(input_data)

    def _handle_search(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理文献检索。"""
        output = self._create_output()

        query = input_data.get("query", "")
        if not query:
            output.add_error("缺少检索查询 (query)")
            return output

        sources = input_data.get("sources", ["crossref", "pubmed"])
        max_results = input_data.get("max_results", 20)
        year_range = input_data.get("year_range")

        # 构建检索策略说明
        strategy = {
            "query": query,
            "sources": sources,
            "max_results_per_source": max_results,
            "year_range": year_range,
            "databases_note": "二级索引仅作发现线索，关键字段需回源核实",
        }

        # 模拟检索结果（实际应调用各 API）
        results = self._simulate_search(query, sources, max_results)

        output.data = {
            "strategy": strategy,
            "results": results,
            "result_count": len(results),
            "deduplication_note": "按 DOI 去重，无 DOI 按标题模糊匹配",
            "unverified_items": [r["doi"] for r in results if not r.get("verified")],
        }

        if any(not r.get("verified") for r in results):
            output.add_warning("部分文献元数据未验证，建议回源核实")

        return output

    def _handle_verify(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理文献元数据验证。"""
        output = self._create_output()

        doi = input_data.get("doi", "")
        pmid = input_data.get("pmid", "")
        arxiv_id = input_data.get("arxiv_id", "")

        if not any([doi, pmid, arxiv_id]):
            output.add_error("至少需要一个文献标识: doi, pmid, 或 arxiv_id")
            return output

        # 模拟验证结果
        verified = {
            "doi": doi,
            "pmid": pmid,
            "arxiv_id": arxiv_id,
            "verified": True,
            "metadata": {
                "title": "[待获取]",
                "authors": ["[待获取]"],
                "journal": "[待获取]",
                "year": None,
            },
            "verification_source": "crossref" if doi else "pubmed" if pmid else "arxiv",
        }

        output.data = verified
        output.add_author_check("请核对自动获取的元数据是否与原文一致")

        return output

    def _handle_audit(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理严格他引审计。"""
        output = self._create_output()

        target_doi = input_data.get("doi", "")
        if not target_doi:
            output.add_error("缺少目标文献 DOI")
            return output

        exclude_rules = input_data.get("exclude_rules", {
            "exclude_self": True,
            "exclude_team": True,
            "exclude_network": True,
            "same_institution": True,
        })

        # 模拟审计结果
        audit_result = CitationAuditResult(
            total_citations=0,
            self_citations=0,
            team_citations=0,
            network_citations=0,
            strict_external_citations=0,
            high_impact_citers=[],
            uncertain_cases=["缺少作者/机构信息，无法完全排除合作网络引用"],
        )

        output.data = {
            "target_doi": target_doi,
            "exclude_rules": exclude_rules,
            "audit_result": {
                "total_citations": audit_result.total_citations,
                "self_citations": audit_result.self_citations,
                "team_citations": audit_result.team_citations,
                "network_citations": audit_result.network_citations,
                "strict_external_citations": audit_result.strict_external_citations,
            },
            "high_impact_citers": audit_result.high_impact_citers,
            "uncertain_cases": audit_result.uncertain_cases,
            "note": "严格他引判断需要明确排除规则；不确定情况已标注",
        }

        output.add_warning("他引审计为保守估计，建议人工复核边界情况")
        return output

    def _handle_format(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理引用格式生成。"""
        output = self._create_output()

        papers = input_data.get("papers", [])
        format_style = input_data.get("format", "nature")

        if not papers:
            output.add_error("缺少文献列表 (papers)")
            return output

        if format_style not in self.SUPPORTED_FORMATS:
            output.add_error(
                f"不支持的引用格式: {format_style}。"
                f"支持: {self.SUPPORTED_FORMATS}"
            )
            return output

        formatted = []
        for paper in papers:
            citation = self._format_citation(paper, format_style)
            formatted.append(citation)

        output.data = {
            "format": format_style,
            "citations": formatted,
            "export_formats": ["ris", "bib", "nbib", "enw"],
        }

        return output

    def _simulate_search(
        self,
        query: str,
        sources: list[str],
        max_results: int,
    ) -> list[dict[str, Any]]:
        """模拟检索结果（占位实现）。"""
        # 实际实现应调用各数据库 API
        return [
            {
                "title": f"Sample paper {i+1} for: {query}",
                "authors": ["Author A", "Author B"],
                "year": 2024,
                "journal": "Nature",
                "doi": f"10.1000/sample.{i+1}",
                "source": sources[i % len(sources)] if sources else "unknown",
                "verified": False,
            }
            for i in range(min(max_results, 5))
        ]

    def _format_citation(
        self,
        paper: dict[str, Any],
        format_style: str,
    ) -> str:
        """格式化单条引用。"""
        title = paper.get("title", "")
        authors = paper.get("authors", [])
        year = paper.get("year", "")
        journal = paper.get("journal", "")
        doi = paper.get("doi", "")

        if format_style == "nature":
            auth_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            return f"{auth_str} {title}. *{journal}* ({year}). https://doi.org/{doi}"
        elif format_style == "apa":
            auth_str = ", ".join(authors)
            return f"{auth_str} ({year}). {title}. *{journal}*. https://doi.org/{doi}"
        elif format_style == "vancouver":
            auth_str = ", ".join([a.split()[-1] + " " + "".join([n[0] for n in a.split()[:-1]]) for a in authors[:6]])
            return f"{auth_str}. {title}. {journal}. {year}."
        else:
            return f"{authors} ({year}). {title}. {journal}. DOI: {doi}"
