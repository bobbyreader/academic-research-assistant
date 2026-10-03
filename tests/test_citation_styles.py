"""core.citation_styles 的单元测试。

覆盖：
* 两种样式的内联标记与参考文献条目；
* 作者/年份缺失时的确定性降级；
* 未知样式**抛错**（绝不静默回退）；
* ``numeric`` 与改动前的既有格式逐字符一致（回归基线）；
* ``citation.export_formats`` 的查询/校验入口。
"""

from __future__ import annotations

import pytest

from core.citation_styles import (
    KNOWN_EXPORT_FORMATS,
    KNOWN_STYLES,
    UnknownCitationStyleError,
    known_export_formats,
    known_styles,
    render_inline_marker,
    render_reference,
    render_references,
    validate_export_formats,
)
from core.research_models import PaperRecord


def _paper(
    *,
    authors: list[str] | None = None,
    title: str = "A Study",
    year: int | None = 2024,
    journal: str = "Journal of Tests",
    doi: str = "",
    url: str = "",
) -> PaperRecord:
    return PaperRecord(
        title=title,
        authors=authors if authors is not None else ["Jane Smith"],
        year=year,
        journal=journal,
        doi=doi,
        url=url,
    )


# --------------------------------------------------------------------------- #
# numeric：必须与改动前的既有格式逐字符一致（回归基线）
# --------------------------------------------------------------------------- #
def test_numeric_reference_matches_legacy_format() -> None:
    """含 DOI 的完整 numeric 条目应与既有装配器逐字符一致。"""
    paper = _paper(
        authors=["Jane Smith", "John Doe"],
        title="A Study",
        year=2024,
        journal="Journal of Tests",
        doi="10.1/alpha",
    )

    # 既有基线（research_pipeline._attach_references 的口径）。
    legacy = "[P1] Jane Smith, John Doe. A Study. *Journal of Tests*. (2024). https://doi.org/10.1/alpha"

    assert render_reference(paper, style="numeric", index=1) == legacy


def test_numeric_reference_without_doi_falls_back_to_url() -> None:
    """无 DOI 但有 URL 时，numeric 条目以 URL 结尾（与现状一致）。"""
    paper = _paper(doi="", url="https://example.org/paper")

    assert render_reference(paper, style="numeric", index=2).endswith(
        " https://example.org/paper"
    )


def test_numeric_reference_omits_optional_segments() -> None:
    """期刊/年份缺失时对应段整段省略（与现状一致）。"""
    paper = _paper(authors=["Jane Smith"], title="Untitled", year=None, journal="", doi="")

    rendered = render_reference(paper, style="numeric", index=3)

    assert rendered == "[P3] Jane Smith. Untitled."


def test_numeric_reference_without_authors_keeps_empty_author_segment() -> None:
    """作者缺失时 numeric **保留** ``[Pn] . Title.``（现状字节基线，不得改良）。"""
    paper = _paper(authors=[], title="Lost authors paper", year=2021, journal="", doi="")

    rendered = render_reference(paper, style="numeric", index=3)

    assert rendered == "[P3] . Lost authors paper. (2021)."


def test_numeric_reference_more_than_five_authors_uses_et_al() -> None:
    """作者超过 5 位时，numeric 只保留前 5 位并追加 et al.。"""
    authors = [f"Author {i}" for i in range(1, 8)]
    paper = _paper(authors=authors)

    rendered = render_reference(paper, style="numeric", index=1)

    assert "Author 1, Author 2, Author 3, Author 4, Author 5 et al." in rendered
    assert "Author 6" not in rendered


# --------------------------------------------------------------------------- #
# numeric：内联标记
# --------------------------------------------------------------------------- #
def test_numeric_inline_marker_is_bracketed_index() -> None:
    assert render_inline_marker(_paper(), style="numeric", index=7) == "[P7]"


# --------------------------------------------------------------------------- #
# author_year：内联标记与参考文献条目
# --------------------------------------------------------------------------- #
def test_author_year_inline_marker_single_author() -> None:
    paper = _paper(authors=["Jane Smith"], year=2024)

    assert render_inline_marker(paper, style="author_year", index=1) == "(Smith, 2024)"


def test_author_year_inline_marker_multiple_authors_uses_et_al() -> None:
    paper = _paper(authors=["Jane Smith", "John Doe"], year=2024)

    assert render_inline_marker(paper, style="author_year", index=1) == "(Smith et al., 2024)"


def test_author_year_reference_orders_surname_first() -> None:
    paper = _paper(
        authors=["Jane Smith", "John Doe"],
        title="A Study",
        year=2024,
        journal="Journal of Tests",
        doi="10.1/alpha",
    )

    rendered = render_reference(paper, style="author_year", index=1)

    assert rendered == (
        "Smith, Jane, Doe, John (2024). A Study. *Journal of Tests*. "
        "https://doi.org/10.1/alpha"
    )


def test_author_year_reference_preserves_surname_first_input() -> None:
    """已带逗号的「姓, 名」输入不被二次改写。"""
    paper = _paper(authors=["Smith, Jane"])

    assert render_reference(paper, style="author_year", index=1).startswith("Smith, Jane (")


# --------------------------------------------------------------------------- #
# 降级规则：作者/年份缺失
# --------------------------------------------------------------------------- #
def test_author_year_inline_missing_year_uses_n_d() -> None:
    paper = _paper(authors=["Jane Smith"], year=None)

    assert render_inline_marker(paper, style="author_year", index=1) == "(Smith, n.d.)"


def test_author_year_inline_missing_author_uses_anon() -> None:
    paper = _paper(authors=[], year=2024)

    assert render_inline_marker(paper, style="author_year", index=1) == "(Anon., 2024)"


def test_author_year_inline_missing_both_uses_anon_n_d() -> None:
    paper = _paper(authors=[], year=None)

    assert render_inline_marker(paper, style="author_year", index=1) == "(Anon., n.d.)"


def test_author_year_reference_missing_author_and_year() -> None:
    paper = _paper(authors=[], title="Untitled", year=None, journal="", doi="")

    assert render_reference(paper, style="author_year", index=1) == "Anon. (n.d.). Untitled."


# --------------------------------------------------------------------------- #
# render_references：批量、顺序、样式透传
# --------------------------------------------------------------------------- #
def test_render_references_preserves_order() -> None:
    papers = [
        _paper(authors=["First P"], year=2020),
        _paper(authors=["Second P"], year=2021),
    ]

    rendered = render_references(papers, style="numeric")

    assert rendered[0].startswith("[P1] First P.")
    assert rendered[1].startswith("[P2] Second P.")


def test_render_references_empty_returns_empty() -> None:
    assert render_references([], style="numeric") == []


# --------------------------------------------------------------------------- #
# 未知样式：必须抛错，绝不静默回退
# --------------------------------------------------------------------------- #
def test_unknown_style_raises_for_reference() -> None:
    with pytest.raises(UnknownCitationStyleError):
        render_reference(_paper(), style="mla", index=1)


def test_unknown_style_raises_for_inline_marker() -> None:
    with pytest.raises(UnknownCitationStyleError):
        render_inline_marker(_paper(), style="ieee", index=1)


def test_unknown_style_raises_for_batch() -> None:
    with pytest.raises(UnknownCitationStyleError):
        render_references([_paper()], style="chicago")


def test_unknown_style_error_names_the_bad_style() -> None:
    with pytest.raises(UnknownCitationStyleError, match="mla"):
        render_reference(_paper(), style="mla", index=1)


# --------------------------------------------------------------------------- #
# known_styles / export formats 查询入口
# --------------------------------------------------------------------------- #
def test_known_styles_contains_frozen_values() -> None:
    assert known_styles() == ("numeric", "author_year")
    assert set(KNOWN_STYLES) == {"numeric", "author_year"}


def test_known_export_formats_contains_frozen_values() -> None:
    assert known_export_formats() == ("md", "pdf", "pptx")
    assert KNOWN_EXPORT_FORMATS == ("md", "pdf", "pptx")


def test_validate_export_formats_returns_only_illegal_entries() -> None:
    assert validate_export_formats(["md", "docx", "pdf", "tex"]) == ["docx", "tex"]


def test_validate_export_formats_all_legal_is_empty() -> None:
    assert validate_export_formats(["md", "pdf", "pptx"]) == []


def test_validate_export_formats_empty_is_empty() -> None:
    assert validate_export_formats([]) == []
