"""Post-draft citation integrity checks.

This module is the trust gate of the research pipeline. It answers one
question before any manuscript is exported: *can every citation in the draft
be traced back to a real, retrieved record?*

Two independent checks are performed:

1. **Marker integrity** (deterministic, always on). Every ``[Pn]`` marker in
   the manuscript body must resolve to a reference that was actually
   retrieved. An unknown marker means the model invented a citation, which
   blocks the pipeline.
2. **DOI resolvability** (network, advisory). Every reference that carries a
   DOI is checked against Crossref. Unresolved DOIs downgrade the run to
   "author checks required" but do not block, because transient network
   failures must not be confused with fabrication.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

from core.http_client import HttpClientError, UrllibTransport
from core.research_models import ArtifactDecodeError, PaperRecord

#: Matches the ``[P1]``-style citation markers the writing prompt mandates.
MARKER_PATTERN = re.compile(r"\[P(\d+)\]")

CROSSREF_WORKS_URL = "https://api.crossref.org/works"


def _marker_sort_key(marker: str) -> int:
    return int(marker[1:])


class DoiResolver(Protocol):
    """Resolve a DOI to a boolean ``(resolved, error)`` outcome."""

    def resolve(self, doi: str) -> tuple[bool, str | None]: ...


# --------------------------------------------------------------------------- #
# 反序列化校验：结构非法一律抛 ValueError（绝不静默构造半个对象）
# --------------------------------------------------------------------------- #
def _require_object(value: object, owner: str) -> dict:
    """要求是映射（dict）；否则抛 ``ValueError``。"""
    if not isinstance(value, dict):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 期望 payload 为 dict，实际为 {type(value).__name__}"
        )
    return value


def _require_str(data: dict, field: str, owner: str) -> str:
    value = data.get(field)
    if not isinstance(value, str):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field}' 期望 str，"
            f"实际为 {type(value).__name__}"
        )
    return value


def _require_optional_str(data: dict, field: str, owner: str) -> str | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field}' 期望 str 或 null，"
            f"实际为 {type(value).__name__}"
        )
    return value


def _require_int(data: dict, field: str, owner: str) -> int:
    value = data.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field}' 期望 int，"
            f"实际为 {type(value).__name__}"
        )
    return value


def _require_bool(data: dict, field: str, owner: str) -> bool:
    value = data.get(field)
    if not isinstance(value, bool):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field}' 期望 bool，"
            f"实际为 {type(value).__name__}"
        )
    return value


def _require_str_list(data: dict, field: str, owner: str) -> list[str]:
    value = data.get(field)
    if not isinstance(value, list):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field}' 期望 list[str]，"
            f"实际为 {type(value).__name__}"
        )
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ArtifactDecodeError(
                f"{owner}.from_dict 字段 '{field}' 的第 {index} 个元素期望 str，"
                f"实际为 {type(item).__name__}"
            )
    return list(value)


class CrossrefDoiResolver:
    """Check DOI resolvability against the Crossref REST API."""

    def __init__(self, transport: UrllibTransport | None = None) -> None:
        self.transport = transport or UrllibTransport(timeout=15, max_retries=1)

    def resolve(self, doi: str) -> tuple[bool, str | None]:
        doi = doi.strip()
        if not doi:
            return False, "缺少 DOI"
        try:
            payload = self.transport.get_json(f"{CROSSREF_WORKS_URL}/{doi}")
        except HttpClientError as exc:
            return False, str(exc)
        if isinstance(payload, dict) and payload.get("message"):
            return True, None
        return False, "Crossref 未返回该 DOI 的元数据"


@dataclass(frozen=True)
class DoiCheck:
    """Outcome of verifying a single reference's DOI."""

    citation_id: str
    doi: str
    resolved: bool
    error: str | None = None

    @classmethod
    def from_dict(cls, payload: object) -> DoiCheck:
        """从 :meth:`to_dict` 的输出还原；结构不合法时抛 ``ValueError``。"""
        data = _require_object(payload, cls.__name__)
        return cls(
            citation_id=_require_str(data, "citation_id", cls.__name__),
            doi=_require_str(data, "doi", cls.__name__),
            resolved=_require_bool(data, "resolved", cls.__name__),
            error=_require_optional_str(data, "error", cls.__name__),
        )


@dataclass
class CitationVerificationReport:
    """Aggregated citation integrity result for one manuscript."""

    total_references: int
    cited_markers: list[str] = field(default_factory=list)
    unknown_markers: list[str] = field(default_factory=list)
    unused_references: list[str] = field(default_factory=list)
    references_without_doi: list[str] = field(default_factory=list)
    doi_checks: list[DoiCheck] = field(default_factory=list)

    @property
    def verified_doi_count(self) -> int:
        return sum(1 for check in self.doi_checks if check.resolved)

    @property
    def unresolved_doi_count(self) -> int:
        return sum(1 for check in self.doi_checks if not check.resolved)

    @property
    def passed(self) -> bool:
        """A run passes only when no fabricated marker was found."""
        return not self.unknown_markers

    def warnings(self) -> list[str]:
        items: list[str] = []
        if self.unresolved_doi_count:
            items.append(
                f"{self.unresolved_doi_count} 条文献的 DOI 未能通过 Crossref 校验，请人工核对。"
            )
        if self.references_without_doi:
            items.append(
                f"{len(self.references_without_doi)} 条文献没有 DOI（如 arXiv 预印本），未做解析校验。"
            )
        if self.unused_references:
            items.append(
                f"{len(self.unused_references)} 条参考文献未被正文引用。"
            )
        return items

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "total_references": self.total_references,
            "cited_markers": self.cited_markers,
            "unknown_markers": self.unknown_markers,
            "unused_references": self.unused_references,
            "references_without_doi": self.references_without_doi,
            "doi_verified": self.verified_doi_count,
            "doi_unresolved": self.unresolved_doi_count,
            "doi_checks": [asdict(check) for check in self.doi_checks],
        }

    @classmethod
    def from_dict(cls, payload: object) -> CitationVerificationReport:
        """从 :meth:`to_dict` 的输出还原；结构不合法时抛 ``ValueError``。

        ``passed``、``doi_verified``、``doi_unresolved`` 均为派生字段，重建时由
        属性自动重算。
        """
        data = _require_object(payload, cls.__name__)
        doi_checks = data.get("doi_checks")
        if not isinstance(doi_checks, list):
            raise ArtifactDecodeError(
                f"{cls.__name__}.from_dict 字段 'doi_checks' 期望 list，"
                f"实际为 {type(doi_checks).__name__}"
            )
        decoded: list[DoiCheck] = []
        for index, item in enumerate(doi_checks):
            if not isinstance(item, dict):
                raise ArtifactDecodeError(
                    f"{cls.__name__}.from_dict 字段 'doi_checks' 的第 {index} 个元素"
                    f"期望 dict，实际为 {type(item).__name__}"
                )
            decoded.append(DoiCheck.from_dict(item))
        return cls(
            total_references=_require_int(data, "total_references", cls.__name__),
            cited_markers=_require_str_list(data, "cited_markers", cls.__name__),
            unknown_markers=_require_str_list(data, "unknown_markers", cls.__name__),
            unused_references=_require_str_list(
                data, "unused_references", cls.__name__
            ),
            references_without_doi=_require_str_list(
                data, "references_without_doi", cls.__name__
            ),
            doi_checks=decoded,
        )


def verify_citations(
    manuscript_body: str,
    papers: list[PaperRecord],
    resolver: DoiResolver | None = None,
) -> CitationVerificationReport:
    """Check a manuscript body against the records it was allowed to cite.

    Args:
        manuscript_body: Draft text *before* the reference list is appended.
        papers: Retrieved records, in the same order used to build ``[Pn]`` ids.
        resolver: DOI resolver; defaults to the live Crossref resolver.

    Returns:
        A report whose ``passed`` flag is False when the draft cites a
        reference id that does not exist.
    """
    resolver = resolver or CrossrefDoiResolver()
    valid_ids = {f"P{index}" for index in range(1, len(papers) + 1)}

    cited = sorted(
        {f"P{number}" for number in MARKER_PATTERN.findall(manuscript_body)},
        key=_marker_sort_key,
    )
    unknown = [marker for marker in cited if marker not in valid_ids]
    unused = sorted(valid_ids - set(cited), key=_marker_sort_key)

    doi_checks: list[DoiCheck] = []
    references_without_doi: list[str] = []
    for index, paper in enumerate(papers, 1):
        citation_id = f"P{index}"
        if not paper.doi:
            references_without_doi.append(citation_id)
            continue
        resolved, error = resolver.resolve(paper.doi)
        doi_checks.append(
            DoiCheck(citation_id=citation_id, doi=paper.doi, resolved=resolved, error=error)
        )

    return CitationVerificationReport(
        total_references=len(papers),
        cited_markers=cited,
        unknown_markers=unknown,
        unused_references=unused,
        references_without_doi=references_without_doi,
        doi_checks=doi_checks,
    )


def render_verification_markdown(report: CitationVerificationReport) -> str:
    """Render the verification report as a human-readable Markdown artifact."""
    lines = [
        "# 引用验证报告",
        "",
        f"- 结论: {'通过' if report.passed else '未通过（存在无法追溯的引用）'}",
        f"- 参考文献总数: {report.total_references}",
        f"- 正文引用标识: {', '.join(report.cited_markers) or '（无）'}",
        f"- DOI 校验: {report.verified_doi_count} 通过 / {report.unresolved_doi_count} 未解析",
        "",
    ]

    if report.unknown_markers:
        lines.extend(["## 无法追溯的引用（阻断）", ""])
        lines.extend(f"- {marker}" for marker in report.unknown_markers)
        lines.append("")

    if report.unused_references:
        lines.extend(["## 未被正文引用的文献", ""])
        lines.extend(f"- {marker}" for marker in report.unused_references)
        lines.append("")

    unresolved = [check for check in report.doi_checks if not check.resolved]
    if unresolved:
        lines.extend(["## 未解析 DOI（需人工核对）", ""])
        lines.extend(
            f"- {check.citation_id}: {check.doi} — {check.error or '未知原因'}"
            for check in unresolved
        )
        lines.append("")

    if report.references_without_doi:
        lines.extend(["## 无 DOI 文献（未做解析校验）", ""])
        lines.extend(f"- {marker}" for marker in report.references_without_doi)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
