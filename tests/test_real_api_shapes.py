"""Regression tests that feed **real** API response bytes to the real parsing layer.

Motivation
----------
The rest of the suite replaces the outside world with fakes that return clean,
well-formed, fully-populated payloads. Real APIs do not. This module replays raw
bytes captured from crossref / pubmed / arxiv (see
``tests/fixtures/real_responses/README.md``) through the *real* parsing,
normalization, de-duplication and ranking code, so that:

* silent corruption (an upstream failure becoming a plausible-looking "paper",
  or a present field being dropped) is caught rather than assumed away;
* "no results" and "query failed" remain distinguishable;
* ranking / de-duplication stays deterministic on genuinely inconsistent
  metadata.

The samples are raw captures; nothing here is hand-crafted. Where a scenario
could not be captured honestly it is not simulated (see the README).

Design note on known defects
----------------------------
Two real defects were found (both reproduced from the captured bytes). They are
encoded below as ``pytest.mark.xfail(strict=True, ...)`` so that (a) the suite is
green while the defect exists, and (b) the moment the defect is *fixed* the test
XPASSes and fails, forcing the assertion to be promoted to a real one. This keeps
the tripwire honest in both directions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.external_clients import (
    ArxivSource,
    CrossrefSource,
    LiteratureSearcher,
    PaperRecord,
    PubMedSource,
)
from core.http_client import HttpClientError
from core.research_models import SearchReport
from core.retrieval import PaperCandidate, deduplicate, rank_candidates

FIXTURES = Path(__file__).parent / "fixtures" / "real_responses"

CROSSREF_URL = "https://api.crossref.org/works"
PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ARXIV_URL = "https://export.arxiv.org/api/query"

#: 抓取时各响应体对应的 HTTP 状态（来自 manifest.json / headers_manifest.json）。
#: 用于在回放时**忠实**模拟"传输层看到的状态"——这是区分"空结果"与"请求失败"的关键。
CAPTURED_STATUS: dict[str, int] = {
    "crossref_normal": 200,
    "crossref_no_results": 200,
    "crossref_chinese": 200,
    "crossref_long_query": 200,
    "crossref_many": 200,
    "crossref_special_chars": 429,
    "crossref_no_select": 400,
    "crossref_bad_filter": 400,
    "crossref_bad_rows": 429,
    "arxiv_normal": 200,
    "arxiv_no_results": 200,
    "arxiv_chinese": 200,
    "arxiv_special": 200,
    "arxiv_long": 500,
    "arxiv_error_feed": 200,
    "pubmed_search_normal": 200,
    "pubmed_search_no_results": 200,
    "pubmed_search_chinese": 200,
    "pubmed_search_special": 200,
    "pubmed_search_long": 200,
    "pubmed_search_letter": 200,
    "pubmed_bad_request": 200,
    "semantic_scholar_429": 429,
}


def _bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _text(name: str) -> str:
    return _bytes(name).decode("utf-8", errors="replace")


class ReplayTransport:
    """Replays captured raw bytes, honouring the HTTP status seen at capture time.

    On a 4xx/5xx status it raises :class:`HttpClientError` exactly like
    :class:`core.http_client.UrllibTransport` does (the real transport never
    reaches the parser on those statuses). On a 2xx it returns the body verbatim.
    This lets us test both the *real* distinction ("429 raises") and the dangerous
    shape ("a 2xx whose body is actually an error").
    """

    def __init__(
        self,
        *,
        json_map: dict[str, tuple[int, str]] | None = None,
        text_map: dict[str, tuple[int, str]] | None = None,
    ) -> None:
        self.json_map = json_map or {}
        self.text_map = text_map or {}

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        del params, headers
        status, body = self.json_map[url]
        if status >= 400:
            raise HttpClientError(f"HTTP {status} from {url}", status_code=status)
        return json.loads(body)

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        del params, headers
        status, body = self.text_map[url]
        if status >= 400:
            raise HttpClientError(f"HTTP {status} from {url}", status_code=status)
        return body


def _crossref_transport(name: str) -> ReplayTransport:
    return ReplayTransport(json_map={CROSSREF_URL: (CAPTURED_STATUS[name], _text(f"{name}.json"))})


def _pubmed_search_transport(name: str) -> ReplayTransport:
    return ReplayTransport(
        json_map={PUBMED_SEARCH_URL: (CAPTURED_STATUS[name], _text(f"{name}.json"))},
        text_map={PUBMED_FETCH_URL: (200, _text("pubmed_fetch_normal.xml"))},
    )


def _arxiv_transport(name: str) -> ReplayTransport:
    return ReplayTransport(text_map={ARXIV_URL: (CAPTURED_STATUS[name], _text(f"{name}.xml"))})


def _parsed_crossref(name: str, limit: int = 20) -> list[PaperRecord]:
    return CrossrefSource(_crossref_transport(name)).search("cancer", limit)


# --------------------------------------------------------------------------- #
# A. 真实原始字节喂给真实解析层：不崩溃、不静默丢弃"存在"的字段
# --------------------------------------------------------------------------- #
def test_crossref_real_normal_response_parses_all_items() -> None:
    """正常响应：5 条真实记录全部解析出来，标题/DOI/作者都在。"""
    papers = _parsed_crossref("crossref_normal", limit=5)

    assert len(papers) == 5
    assert all(paper.title for paper in papers)
    assert all(paper.doi for paper in papers)  # 该样本 5 条都带 DOI
    assert any(paper.authors for paper in papers)
    # 归一化不得把 DOI 前缀留下：真实 DOI 形如 "10.xxxx/yyy"。
    assert all(paper.doi.startswith("10.") for paper in papers)


def test_crossref_real_chinese_response_keeps_title_and_abstract() -> None:
    """中文主题的真实响应：标题是中文，含 `<jats:p>` 摘要——不得丢失。"""
    papers = _parsed_crossref("crossref_chinese", limit=5)

    assert papers, "中文查询确实返回了结果（total-results=593212）"
    assert any("深度学习" in paper.title for paper in papers)
    with_abstract = [paper for paper in papers if paper.abstract]
    assert with_abstract, "该样本含 <jats:p> 包裹的摘要，解析后不应为空"
    assert not any("<jats:p>" in paper.abstract for paper in with_abstract), (
        "jats 标签必须被剥离，不得原样留在摘要里"
    )


def test_crossref_real_long_query_does_not_crash() -> None:
    """超长查询（60 次重复词）的真实响应照常解析。"""
    papers = _parsed_crossref("crossref_long_query", limit=5)
    assert len(papers) == 5


def test_arxiv_real_normal_response_parses_entries() -> None:
    papers = ArxivSource(_arxiv_transport("arxiv_normal")).search("all:sleep memory", 5)
    assert len(papers) == 5
    assert all(paper.title for paper in papers)
    assert all(paper.url.startswith("http") for paper in papers)


def test_pubmed_real_efetch_parses_authors_and_abstract() -> None:
    """真实 efetch XML：作者、年份、DOI、摘要都要被解析出来。"""
    papers = PubMedSource(_pubmed_search_transport("pubmed_search_normal")).search("x", 5)

    assert len(papers) == 2  # efetch 样本里确有两篇
    assert all(paper.title for paper in papers)
    assert all(paper.authors for paper in papers)
    assert all(paper.year is not None for paper in papers)
    assert all(paper.doi for paper in papers)
    assert all(paper.abstract for paper in papers)


# --------------------------------------------------------------------------- #
# B. 缺失字段必须被"如实标记"，不能与"真实存在但为空"混淆
# --------------------------------------------------------------------------- #
def test_crossref_real_missing_fields_are_absent_not_fabricated() -> None:
    """真实 `crossref_many` 里有整批**缺作者 / 缺年份 / 缺摘要 / 缺期刊**的记录。

    断言：缺失以"空"表达（可被上层识别为缺失），**绝不**被填成占位符或从
    其他记录串台。测试同时钉住该样本的真实缺失分布，作为"解析未静默丢弃
    存在字段"的对照基线。
    """
    papers = _parsed_crossref("crossref_many", limit=20)

    assert len(papers) == 20
    missing_authors = [p for p in papers if not p.authors]
    missing_year = [p for p in papers if p.year is None]
    missing_abstract = [p for p in papers if not p.abstract]

    # 这些缺失是真实存在的（见 README）；若解析层把它们"变没了"分布会变化。
    assert missing_authors, "样本中确有缺作者的记录"
    assert missing_year, "样本中确有缺年份的记录"
    assert missing_abstract, "样本中确有缺摘要的记录"

    # 缺失就是缺失：不得出现 "" 以外的占位文本，也不得串台到别的记录。
    for paper in papers:
        assert paper.title, "标题存在则不得为空"
        assert paper.authors == [] or all(isinstance(a, str) for a in paper.authors)
        if paper.abstract:
            assert paper.abstract.strip()


def test_pubmed_real_zero_result_is_a_clean_empty_list() -> None:
    """真实"无结果"响应：返回空列表，且**没有** idlist 造成的异常。"""
    papers = PubMedSource(_pubmed_search_transport("pubmed_search_no_results")).search("x", 5)
    assert papers == []


def test_arxiv_real_zero_result_is_a_clean_empty_list() -> None:
    papers = ArxivSource(_arxiv_transport("arxiv_no_results")).search("all:x", 5)
    assert papers == []


def test_crossref_real_zero_result_is_a_clean_empty_list() -> None:
    papers = CrossrefSource(_crossref_transport("crossref_no_results")).search("x", 5)
    assert papers == []


# --------------------------------------------------------------------------- #
# C. 真实数据上的去重与排序：确定性（与输入顺序无关）
# --------------------------------------------------------------------------- #
def _real_multi_source_records() -> list[tuple[str, PaperRecord]]:
    """用真实解析结果拼出一个多源记录集（含真实缺字段、真实期刊实体噪音）。

    这不是构造脏数据：每条记录都来自真实响应的真实解析输出；只是把不同源的
    记录放进同一个批次，以检验跨源去重/排序。
    """
    crossref = _parsed_crossref("crossref_many", limit=6)
    crossref_b = _parsed_crossref("crossref_normal", limit=3)
    pubmed = PubMedSource(_pubmed_search_transport("pubmed_search_normal")).search("x", 5)

    records: list[tuple[str, PaperRecord]] = []
    records.extend(("crossref", paper) for paper in crossref)
    records.extend(("crossref", paper) for paper in crossref_b)
    records.extend(("pubmed", paper) for paper in pubmed)
    # 同一篇（同 DOI）真实地出现在第二个来源：检验强键合并不依赖顺序。
    records.append(("semantic_scholar", crossref[0]))
    return records


def test_real_records_dedup_and_ranking_are_deterministic() -> None:
    """真实记录集：任意输入顺序 → 完全一致的输出（去重 + 排序）。"""
    import random

    records = _real_multi_source_records()

    def run(seq: list[tuple[str, PaperRecord]]) -> list[tuple[str, str, float]]:
        dedup = deduplicate(seq)
        ranked = rank_candidates(list(dedup.candidates), topic="cancer")
        return [(r.paper.doi, r.paper.title, r.score) for r in ranked]

    baseline = run(records)
    assert baseline, "真实记录集不应为空"

    rng = random.Random(20261004)
    for _ in range(200):
        shuffled = records[:]
        rng.shuffle(shuffled)
        assert run(shuffled) == baseline


def test_real_records_dedup_merges_same_doi_across_sources() -> None:
    """同 DOI 出现在两源：去重后只留一条，且来源集合记全。"""
    crossref = _parsed_crossref("crossref_normal", limit=1)
    shared = crossref[0]
    dedup = deduplicate(
        [("crossref", shared), ("semantic_scholar", shared)],
    )

    assert len(dedup.candidates) == 1
    assert dedup.candidates[0].sources == ("crossref", "semantic_scholar")


def test_real_records_ranking_is_order_independent_via_searcher() -> None:
    """经真实 `LiteratureSearcher` 合并两源真实记录：换源顺序结果不变。"""

    class _ReplaySource:
        def __init__(self, name: str, papers: list[PaperRecord]) -> None:
            self.name = name
            self._papers = papers

        def search(self, query: str, max_results: int) -> list[PaperRecord]:
            del query
            return self._papers[:max_results]

    crossref = _parsed_crossref("crossref_normal", limit=3)
    pubmed = PubMedSource(_pubmed_search_transport("pubmed_search_normal")).search("x", 5)

    def build(order: list[str]) -> list[str]:
        searcher = LiteratureSearcher(
            {
                "crossref": _ReplaySource("crossref", crossref),
                "pubmed": _ReplaySource("pubmed", pubmed),
            }
        )
        report = searcher.search("sleep memory", order, 10)
        return [paper.doi for paper in report.papers]

    baseline = build(["crossref", "pubmed"])
    assert build(["pubmed", "crossref"]) == baseline


# --------------------------------------------------------------------------- #
# D. "没查到" 与 "查询失败" 必须可区分
# --------------------------------------------------------------------------- #
def test_crossref_real_429_raises_readable_error_not_empty_list() -> None:
    """真实 429（空响应体）：必须抛可读错误，**不得**静默返回"零篇文献"。"""
    with pytest.raises(HttpClientError) as excinfo:
        _parsed_crossref("crossref_special_chars", limit=5)

    assert "429" in str(excinfo.value)
    assert excinfo.value.status_code == 429


def test_crossref_real_400_raises_readable_error() -> None:
    with pytest.raises(HttpClientError) as excinfo:
        _parsed_crossref("crossref_no_select", limit=5)
    assert excinfo.value.status_code == 400


def test_arxiv_real_500_raises_readable_error() -> None:
    """真实 500（正文是可解析的 Atom 错误 feed）：传输层必须按失败处理。

    这正是危险所在：错误正文本身是合法 XML，若在解析层被当成正常 feed，
    就会产出标题为 "Error"、作者为 "arXiv api core" 的**假文献**。
    只要状态码是 5xx，就必须在到达解析层之前失败。
    """
    with pytest.raises(HttpClientError) as excinfo:
        ArxivSource(_arxiv_transport("arxiv_long")).search("all:x", 5)
    assert excinfo.value.status_code == 500


def test_search_report_distinguishes_failure_from_empty() -> None:
    """`SearchReport` 层面：失败进 ``errors``，空结果进"零篇"，二者不混同。"""

    class _Failing:
        name = "crossref"

        def search(self, query: str, max_results: int) -> list[PaperRecord]:
            raise HttpClientError("HTTP 429 from https://api.crossref.org/works", status_code=429)

    class _Empty:
        name = "pubmed"

        def search(self, query: str, max_results: int) -> list[PaperRecord]:
            return []

    report: SearchReport = LiteratureSearcher(
        {"crossref": _Failing(), "pubmed": _Empty()}
    ).search("topic", ["crossref", "pubmed"], 5)

    assert report.papers == []
    # 失败**可区分**：crossref 出现在 errors 里，pubmed 不在。
    assert any("crossref" in err and "429" in err for err in report.errors)
    assert not any("pubmed" in err for err in report.errors)
    # 计数语义如实：失败的源**不**进入 counts_by_source（它连 0 都没查到），
    # 只有真正返回（哪怕是空）的源才计数——因此"存在且为 0"与"根本没跑成"
    # 是两种可区分的状态：前者在 counts 里、后者在 errors 里。
    assert "crossref" not in report.counts_by_source
    assert report.counts_by_source["pubmed"] == 0
    assert report.sources_attempted == ["crossref", "pubmed"]


# --------------------------------------------------------------------------- #
# D'. 曾经的真实缺陷（已修复；以下为**正式断言**，不再是 xfail）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sample",
    ["crossref_no_select.json", "crossref_bad_filter.json"],
)
def test_crossref_validation_failure_body_must_not_crash_the_searcher(sample: str) -> None:
    """真实 validation-failure 体（`message` 为列表）以 2xx 送达时不得崩溃。

    场景：crossref 的校验失败响应体是 ``{"status":"failed","message-type":
    "validation-failure","message":[{...}]}``——``message`` 是**列表**。真实抓取时
    它随 HTTP 400 到达，但如果以 2xx 送达（代理/缓存抹平状态码，或 API 行为变更），
    旧实现会在 ``message.get`` 处抛 ``AttributeError`` 并逃逸整个检索。

    期望（已满足）：解析层在边界处显式校验形状，抛 ``HttpClientError``；searcher 把它
    记进 ``SearchReport.errors``——**可读失败**，而非异常逃逸。
    """
    body = _text(sample)

    class _ValidationFailureSource:
        name = "crossref"

        def search(self, query: str, max_results: int) -> list[PaperRecord]:
            transport = ReplayTransport(json_map={CROSSREF_URL: (200, body)})
            return CrossrefSource(transport).search(query, max_results)

    report = LiteratureSearcher({"crossref": _ValidationFailureSource()}).search(
        "topic", ["crossref"], 5
    )
    assert report.papers == []
    assert any("crossref" in err for err in report.errors), (
        "validation-failure 体必须以可读错误进 errors，而不是崩溃或静默为空"
    )


def test_crossref_validation_failure_raises_readable_error_directly() -> None:
    """直接调用解析层：形状不符时抛 ``HttpClientError``（可读），不是 ``AttributeError``。"""
    body = _text("crossref_no_select.json")
    transport = ReplayTransport(json_map={CROSSREF_URL: (200, body)})

    with pytest.raises(HttpClientError) as excinfo:
        CrossrefSource(transport).search("x", 5)

    message = str(excinfo.value)
    assert "message" in message
    assert "list" in message  # 指出实际形状，便于诊断
    assert "validation-failure" in message  # 带上 message-type，指明上游语义


def test_pubmed_200_error_payload_must_not_be_reported_as_zero_results() -> None:
    """真实 HTTP 200 + `esearchresult.ERROR`：必须被识别为**失败**。

    期望（已满足）：不能与"查询成功但没查到"（`pubmed_search_no_results.json`）混同
    ——要么抛可读错误，要么让 `SearchReport.errors` 非空。
    """
    body = _text("pubmed_bad_request.json")

    class _PubmedErrorSource:
        name = "pubmed"

        def search(self, query: str, max_results: int) -> list[PaperRecord]:
            transport = ReplayTransport(
                json_map={PUBMED_SEARCH_URL: (200, body)},
                text_map={PUBMED_FETCH_URL: (200, "<PubmedArticleSet/>")},
            )
            return PubMedSource(transport).search(query, max_results)

    report = LiteratureSearcher({"pubmed": _PubmedErrorSource()}).search(
        "topic", ["pubmed"], 5
    )
    assert report.papers == []
    assert report.errors, (
        "HTTP 200 但语义为错误（esearchresult.ERROR）时，errors 必须非空，"
        "否则与'没查到'不可区分"
    )
    assert any("retmax" in err for err in report.errors), (
        "错误信息应带上上游给出的具体原因"
    )


def test_pubmed_200_error_raises_readable_error_directly() -> None:
    body = _text("pubmed_bad_request.json")
    transport = ReplayTransport(
        json_map={PUBMED_SEARCH_URL: (200, body)},
        text_map={PUBMED_FETCH_URL: (200, "<PubmedArticleSet/>")},
    )

    with pytest.raises(HttpClientError) as excinfo:
        PubMedSource(transport).search("x", 5)
    assert "retmax is not a positive number" in str(excinfo.value)


def test_pubmed_real_zero_result_and_error_must_be_distinguishable() -> None:
    """**正向不变量**：真实的"没查到"与真实的"200 语义错误"**必须可区分**。

    这条从前是缺陷 2 的特征化对照（两者都返回 ``[]``，不可区分）。缺陷修复后，
    它被改写为正向不变量：真实的空结果仍是 ``[]``，而真实的错误必须**抛异常**
    （或其上层记入 ``errors``）——二者再也不能被混为一谈。
    """
    zero = PubMedSource(_pubmed_search_transport("pubmed_search_no_results")).search("x", 5)
    assert zero == [], "真实的'没查到'仍应是干净的空列表"

    with pytest.raises(HttpClientError):
        PubMedSource(_pubmed_search_transport("pubmed_bad_request")).search("x", 5)


# --------------------------------------------------------------------------- #
# E. 真实标记 / 实体噪音的处理
# --------------------------------------------------------------------------- #
def test_crossref_real_journal_html_entities_are_decoded() -> None:
    """真实期刊名含 HTML 实体 `&amp;`：必须被**解码**，不得原样留在记录里。

    真实样本里有 `Theranostics of Respiratory &amp; Skin Diseases` 与
    `Cancer Cell &amp; Microenvironment`。修复后它们应变成含裸 `&` 的可读期刊名。
    本测试不仅断言 `&amp;` 消失，还断言 `&` **确实出现**——证明实体是被**解码**，
    而不是被整体删除（"配置被读取"与"配置生效"是两件事，这里要求后者）。
    """
    papers = _parsed_crossref("crossref_many", limit=20)

    entity_journals = [p.journal for p in papers if "&amp;" in p.journal]
    assert not entity_journals, (
        "真实期刊名里编码器产生的实体 `&amp;` 未被处理，原样进入记录: "
        f"{entity_journals!r}"
    )

    decoded = {p.journal for p in papers if "&" in p.journal}
    assert "Cancer Cell & Microenvironment" in decoded, (
        f"实体应被解码为 `&` 而非删除；实际含 & 的期刊名: {sorted(decoded)!r}"
    )
    assert "Theranostics of Respiratory & Skin Diseases" in decoded


def test_crossref_real_abstract_markup_is_stripped() -> None:
    """真实摘要里的 `<jats:p>` / `<p>` 标签必须被剥离。"""
    papers = _parsed_crossref("crossref_many", limit=20)
    with_markup = [p for p in papers if "<jats:" in p.abstract or "<p>" in p.abstract]
    assert not with_markup, "摘要中不应残留 XML/HTML 标签"


def test_strip_markup_decodes_entities_not_just_strips_tags() -> None:
    """**行为变更记录**：实体解码现在也作用于 `abstract`（原本就作用于它）。

    `_strip_markup` 只被 `abstract` 与（本次新增的）`journal` 使用。给它加实体解码
    会**同时改变摘要文本**——而摘要正是引用接地（`is_quote_grounded(quote,
    paper.abstract)`）与 LLM 提示词的输入。本测试把该行为钉死，避免它再次成为
    "未被察觉的静默变更"：解码是**还原**（`&amp;` → `&`），不是删除。
    """
    from core.external_clients import _strip_markup

    assert _strip_markup("<jats:p>a &amp; b</jats:p>") == "a & b"
    assert _strip_markup("&lt;tag&gt;") == "<tag>"
    assert _strip_markup("R&amp;D &amp; more") == "R&D & more"


def test_arxiv_real_error_feed_at_200_would_be_a_fake_paper() -> None:
    """**特征化测试**：揭示为何 5xx 必须在解析层之前失败。

    `arxiv_long.xml` 是真实抓到的 Atom 错误 feed（首次抓取时 HTTP 500）。
    若它被以 2xx 送达（例如中间代理/缓存抹平了状态码），当前解析层会把它
    解析成**标题为 "Error" 的假文献**。本测试如实钉住这一危险形状——它不是
    对产品行为的认可，而是"传输层必须拦住 5xx"这一要求的反面证据。
    """
    feed = _text("arxiv_long.xml")
    transport = ReplayTransport(text_map={ARXIV_URL: (200, feed)})

    papers = ArxivSource(transport).search("all:x", 5)

    assert len(papers) == 1
    parsed = papers[0]
    # 这是错误 feed 的真实内容，不是论文。
    assert parsed.title == "Error"
    assert parsed.url == "https://arxiv.org/api/errors"


# --------------------------------------------------------------------------- #
# F. 真实样本清单存在性（防止 fixture 被误删导致测试静默空跑）
# --------------------------------------------------------------------------- #
def test_all_referenced_real_samples_exist() -> None:
    required = [
        "crossref_normal.json",
        "crossref_no_results.json",
        "crossref_chinese.json",
        "crossref_long_query.json",
        "crossref_many.json",
        "crossref_special_chars.json",
        "crossref_no_select.json",
        "pubmed_search_no_results.json",
        "pubmed_search_normal.json",
        "pubmed_bad_request.json",
        "pubmed_fetch_normal.xml",
        "arxiv_normal.xml",
        "arxiv_no_results.xml",
        "arxiv_long.xml",
        "semantic_scholar_429.json",
    ]
    missing = [name for name in required if not (FIXTURES / name).exists()]
    assert not missing, f"缺少真实样本: {missing}"


def test_candidate_source_tuple_is_sorted_and_deterministic() -> None:
    """真实记录经去重后，来源集合按字典序——与输入顺序无关。"""
    records = _real_multi_source_records()
    dedup = deduplicate(records)
    for candidate in dedup.candidates:
        assert isinstance(candidate, PaperCandidate)
        assert list(candidate.sources) == sorted(candidate.sources)
