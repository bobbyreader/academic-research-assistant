"""引用样式的渲染器：把 ``PaperRecord`` 渲染成内联标记与参考文献条目。

职责边界（刻意与 :mod:`core.citation_verifier` 分离）：
  * **核验**（``citation_verifier``）回答"正文里的每个 ``[Pn]`` 能否追溯到一条
    真实检索到的记录"——这是可信性关口；
  * **渲染**（本模块）只负责把已经核验过的记录**排版**成不同引用样式。

两者是不同的问题，**不得**把渲染逻辑塞进核验模块。

支持的样式
----------
``numeric``
    现状格式（编号制）：内联标记 ``[P1]``；参考文献条目形如
    ``[P1] A. Author, B. Author. Title. *Journal*. (2024). https://doi.org/...``。
    本模块的 ``numeric`` 输出与装配器 :meth:`ResearchPipeline._attach_references`
    **逐字符一致**——这是硬性回归要求，因为既有手稿产物与测试依赖该格式。

``author_year``
    作者—年份制：内联标记 ``(Smith, 2024)``；参考文献条目形如
    ``Smith, J., Doe, A. (2024). Title. *Journal*. https://doi.org/...``。

降级规则（确定性，绝不静默）
--------------------------
* **作者缺失**（``authors`` 为空）：
  - 内联：``numeric`` 仍用 ``[Pn]``（编号与作者无关）；``author_year`` 用 ``Anon.``。
  - 参考文献：``numeric`` 的作者段为空串（沿用现状行为）；``author_year`` 用 ``Anon.``。
* **年份缺失**（``year`` 为 ``None``）：
  - 内联 ``author_year`` 与参考文献 ``author_year`` 均用 ``n.d.``（no date）。
  - 参考文献 ``numeric`` 与现状一致：直接省略 ``(year)`` 段。

未知样式**必须抛错**（:class:`UnknownCitationStyleError`，``ValueError`` 子类），
因为静默回退到 ``numeric`` 会让用户以为自己的配置生效了——而那正是本仓库反复
清理的"看起来配置了其实什么都没发生"这一类 bug。
"""

from __future__ import annotations

from collections.abc import Sequence

from core.research_models import PaperRecord

#: 已知引用样式；``citation.supported_styles`` 的默认值必须与此一致。
KNOWN_STYLES: tuple[str, ...] = ("numeric", "author_year")

#: 允许的引用/文献表导出格式；``citation.export_formats`` 的默认值必须与此一致。
KNOWN_EXPORT_FORMATS: tuple[str, ...] = ("md", "pdf", "pptx")

#: 作者缺失时使用的确定性占位（参考文献的 author_year 条）。
_ANON = "Anon."
#: 年份缺失时使用的确定性占位（author_year 内联标记与条目）。
_NO_DATE = "n.d."


class UnknownCitationStyleError(ValueError):
    """请求了未知的引用样式时抛出。

    刻意不复用 :class:`ValueError` 之外的基类，也**绝不回退**到默认样式：静默
    回退会让配置错误看起来像配置生效。
    """


def known_styles() -> tuple[str, ...]:
    """返回本模块支持的引用样式名（有序、可复现）。"""
    return KNOWN_STYLES


def known_export_formats() -> tuple[str, ...]:
    """返回引用/文献表允许的导出格式（有序、可复现）。

    这是 ``citation.export_formats`` 的**单一事实来源**：导出层与
    ``config_validation`` 都应通过本函数取用，而不是各自硬编码一份。
    """
    return KNOWN_EXPORT_FORMATS


def validate_export_formats(requested: Sequence[str]) -> list[str]:
    """返回 ``requested`` 中**不属于** :data:`KNOWN_EXPORT_FORMATS` 的项（按输入顺序）。

    空列表即表示全部合法。保留重复项，以便调用方能如实地把每一项都报告给用户。
    """
    allowed = set(KNOWN_EXPORT_FORMATS)
    return [item for item in requested if item not in allowed]


def _require_known_style(style: str) -> None:
    """样式未知时抛 :class:`UnknownCitationStyleError`。"""
    if style not in KNOWN_STYLES:
        raise UnknownCitationStyleError(
            f"未知引用样式 {style!r}；支持的样式为 {list(KNOWN_STYLES)}"
        )


def _author_list(paper: PaperRecord) -> list[str]:
    """返回作者列表（保持原顺序；可能为空）。"""
    return [author for author in paper.authors if author]


def _numeric_authors(authors: list[str]) -> str:
    """编号制的作者段：前 5 位逗号分隔，超过 5 位加 ``et al.``（与现状一致）。"""
    text = ", ".join(authors[:5])
    if len(authors) > 5:
        text += " et al."
    return text


def _author_year_names(paper: PaperRecord) -> str:
    """作者—年份制的作者段。

    形式为 ``Smith, J., Doe, A.``：把 ``"J. Smith"`` 这类"名前姓后"的字符串
    规范化为"姓在前"。无法安全拆分时原样保留该段，绝不猜测或丢弃。
    作者缺失时返回 ``Anon.``。
    """
    authors = _author_list(paper)
    if not authors:
        return _ANON
    rendered = [_normalize_author(name) for name in authors]
    return ", ".join(rendered)


def _normalize_author(raw: str) -> str:
    """把单个作者字符串规范化为"姓[ 首字母]."。

    仅在作者字符串**恰好是两段**（``"First Last"`` / ``"J. Last"``）时才改写为
    ``"Last, First"``；其余情况（含已带逗号的 ``"Last, First"``、多段姓名、
    单段字符串）一律原样返回——宁可保守，也不猜测姓名结构。
    """
    text = raw.strip()
    if not text or "," in text:
        return text
    parts = text.split()
    if len(parts) != 2:
        return text
    first, last = parts
    return f"{last}, {first}"


def _style_suffix(paper: PaperRecord) -> str:
    """参考文献条目末尾的 DOI/URL 段（两种样式共用，与现状逻辑一致）。"""
    if paper.doi:
        return f" https://doi.org/{paper.doi}"
    if paper.url:
        return f" {paper.url}"
    return ""


def _render_numeric_reference(paper: PaperRecord, *, index: int) -> str:
    """编号制参考文献条目，**与既有装配器逐字符一致**。

    格式：``[P1] Authors. Title. *Journal*. (2024). https://doi.org/...``

    **空作者必须保留 ``". "``**：既有装配器在作者缺失时产出
    ``[P3] . Title.``（作者段为空串但点号与空格仍在）。这是缺省（numeric）路径
    的逐字节基线，不得"改良"——因此作者段**始终**写入，不做条件省略。
    """
    authors = _numeric_authors(_author_list(paper))
    citation = f"[P{index}] {authors}. {paper.title}."
    if paper.journal:
        citation += f" *{paper.journal}*."
    if paper.year:
        citation += f" ({paper.year})."
    citation += _style_suffix(paper)
    return citation


def _render_author_year_reference(paper: PaperRecord) -> str:
    """作者—年份制参考文献条目。

    格式：``Smith, J. (2024). Title. *Journal*. https://doi.org/...``
    作者缺失用 ``Anon.``，年份缺失用 ``n.d.``。
    """
    names = _author_year_names(paper)
    year = paper.year if paper.year else _NO_DATE
    citation = f"{names} ({year}). {paper.title}."
    if paper.journal:
        citation += f" *{paper.journal}*."
    citation += _style_suffix(paper)
    return citation


def render_reference(paper: PaperRecord, *, style: str, index: int) -> str:
    """把一条记录渲染为参考文献条目。

    Args:
        paper: 已检索到的规范化记录。
        style: 引用样式，必须是 :func:`known_styles` 之一。
        index: 从 1 开始的序号（``numeric`` 用于生成 ``[Pn]``；``author_year``
            不使用该序号，但为统一签名仍要求传入）。

    Raises:
        UnknownCitationStyleError: ``style`` 不在 :func:`known_styles` 中。
    """
    _require_known_style(style)
    if style == "numeric":
        return _render_numeric_reference(paper, index=index)
    return _render_author_year_reference(paper)


def render_references(papers: Sequence[PaperRecord], *, style: str) -> list[str]:
    """把一组记录按给定样式渲染成参考文献条目列表（顺序与输入一致）。

    序号从 1 开始，与 ``[Pn]`` 编号口径一致。

    Raises:
        UnknownCitationStyleError: ``style`` 不在 :func:`known_styles` 中。
    """
    _require_known_style(style)
    return [
        render_reference(paper, style=style, index=index)
        for index, paper in enumerate(papers, 1)
    ]


def _author_year_inline(paper: PaperRecord) -> str:
    """作者—年份制内联标记的主体：``(Smith, 2024)``。

    作者缺失用 ``Anon.``，年份缺失用 ``n.d.``。多位作者时取第一位（后用
    ``et al.`` 提示还有更多作者），与常见 author-year 惯例一致。
    """
    authors = _author_list(paper)
    if not authors:
        surname = _ANON
    else:
        surname = _surname(_normalize_author(authors[0]))
        if len(authors) > 1:
            surname += " et al."
    year = paper.year if paper.year else _NO_DATE
    return f"({surname}, {year})"


def _surname(normalized: str) -> str:
    """从规范化后的作者串中取姓氏部分（``"Last, First"`` -> ``"Last"``）。"""
    return normalized.split(",", 1)[0].strip() if "," in normalized else normalized


def render_inline_marker(paper: PaperRecord, *, style: str, index: int) -> str:
    """渲染正文内联引用标记。

    * ``numeric``：``[Pn]``（编号制，与现状一致）；
    * ``author_year``：``(Smith, 2024)``。

    Raises:
        UnknownCitationStyleError: ``style`` 不在 :func:`known_styles` 中。
    """
    _require_known_style(style)
    if style == "numeric":
        return f"[P{index}]"
    return _author_year_inline(paper)
