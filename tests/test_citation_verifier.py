from __future__ import annotations

from core.citation_verifier import (
    render_verification_markdown,
    verify_citations,
)
from core.research_models import PaperRecord


class FakeResolver:
    def __init__(self, resolved: bool = True) -> None:
        self.resolved = resolved
        self.checked: list[str] = []

    def resolve(self, doi: str) -> tuple[bool, str | None]:
        self.checked.append(doi)
        return (self.resolved, None if self.resolved else "unresolved")


def _papers() -> list[PaperRecord]:
    return [
        PaperRecord(title="With DOI", doi="10.1/alpha", source="crossref"),
        PaperRecord(title="Preprint without DOI", doi="", source="arxiv"),
    ]


def test_marker_integrity_flags_unknown_citation() -> None:
    report = verify_citations("See [P1] and also [P7].", _papers(), FakeResolver())

    assert report.unknown_markers == ["P7"]
    assert report.passed is False
    assert report.cited_markers == ["P1", "P7"]
    assert report.unused_references == ["P2"]


def test_valid_draft_passes_and_records_doi_checks() -> None:
    resolver = FakeResolver()
    report = verify_citations("Only [P1] is cited here.", _papers(), resolver)

    assert report.passed is True
    assert report.unknown_markers == []
    assert report.verified_doi_count == 1
    assert report.unresolved_doi_count == 0
    assert report.references_without_doi == ["P2"]
    assert resolver.checked == ["10.1/alpha"]


def test_unresolved_doi_is_a_warning_not_a_block() -> None:
    report = verify_citations("Only [P1].", _papers(), FakeResolver(resolved=False))

    assert report.passed is True
    assert report.unresolved_doi_count == 1
    assert any("Crossref" in warning for warning in report.warnings())


def test_to_dict_and_markdown_render_are_serializable() -> None:
    report = verify_citations("Only [P1].", _papers(), FakeResolver(resolved=False))

    payload = report.to_dict()
    assert payload["passed"] is True
    assert payload["doi_verified"] == 0
    assert payload["doi_unresolved"] == 1

    markdown = render_verification_markdown(report)
    assert "引用验证报告" in markdown
    assert "10.1/alpha" in markdown
