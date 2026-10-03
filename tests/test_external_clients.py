from __future__ import annotations

from itertools import permutations
from typing import Any

from core.external_clients import (
    ArxivSource,
    CrossrefSource,
    LiteratureSearcher,
    PaperRecord,
    PubMedSource,
    SemanticScholarSource,
)


class FakeTransport:
    def __init__(self, json_responses: dict[str, Any]) -> None:
        self.json_responses = json_responses

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        del params, headers
        response = self.json_responses[url]
        if isinstance(response, Exception):
            raise response
        return response

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        del url, params, headers
        return ""


class XMLTransport(FakeTransport):
    def __init__(self, json_responses: dict[str, Any], text_responses: dict[str, str]) -> None:
        super().__init__(json_responses)
        self.text_responses = text_responses

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        del params, headers
        return self.text_responses[url]


def test_crossref_source_normalizes_external_metadata() -> None:
    transport = FakeTransport(
        {
            "https://api.crossref.org/works": {
                "message": {
                    "items": [
                        {
                            "title": ["A real paper"],
                            "author": [
                                {"given": "Ada", "family": "Lovelace"},
                            ],
                            "published": {"date-parts": [[2025, 2, 1]]},
                            "container-title": ["Research Journal"],
                            "DOI": "10.1234/example",
                            "abstract": "<jats:p>Evidence summary.</jats:p>",
                            "URL": "https://doi.org/10.1234/example",
                        }
                    ]
                }
            }
        }
    )

    paper = CrossrefSource(transport).search("research topic", 5)[0]

    assert paper.title == "A real paper"
    assert paper.authors == ["Ada Lovelace"]
    assert paper.year == 2025
    assert paper.doi == "10.1234/example"
    assert paper.abstract == "Evidence summary."
    assert paper.source == "crossref"


def test_searcher_deduplicates_by_doi_and_keeps_source_errors() -> None:
    class DuplicateSource:
        name = "duplicate"

        def search(self, query: str, max_results: int) -> list[PaperRecord]:
            del query, max_results
            return [
                PaperRecord(
                    title="Same paper",
                    authors=["A Author"],
                    year=2024,
                    doi="10.1234/same",
                    source="duplicate",
                )
            ]

    class BrokenSource:
        name = "broken"

        def search(self, query: str, max_results: int) -> list[PaperRecord]:
            del query, max_results
            raise RuntimeError("upstream unavailable")

    report = LiteratureSearcher(
        {"duplicate": DuplicateSource(), "broken": BrokenSource()}
    ).search("topic", ["duplicate", "broken"], 5)

    assert len(report.papers) == 1
    assert report.papers[0].doi == "10.1234/same"
    assert report.errors == ["broken: upstream unavailable"]


def test_semantic_scholar_source_reads_external_ids() -> None:
    transport = FakeTransport(
        {
            "https://api.semanticscholar.org/graph/v1/paper/search": {
                "data": [
                    {
                        "paperId": "abc",
                        "title": "Semantic paper",
                        "authors": [{"name": "A Researcher"}],
                        "year": 2023,
                        "venue": "Open Journal",
                        "abstract": "An abstract",
                        "externalIds": {"DOI": "10.5678/semantic"},
                        "url": "https://www.semanticscholar.org/paper/abc",
                        "citationCount": 12,
                    }
                ]
            }
        }
    )

    paper = SemanticScholarSource(transport).search("topic", 3)[0]

    assert paper.doi == "10.5678/semantic"
    assert paper.citation_count == 12
    assert paper.url.endswith("/abc")


def test_pubmed_source_parses_xml_metadata() -> None:
    transport = XMLTransport(
        {
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi": {
                "esearchresult": {"idlist": ["123"]}
            }
        },
        {
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi": """
            <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID>
              <Article><ArticleTitle>PubMed paper</ArticleTitle>
                <Journal><Title>Medical Journal</Title><JournalIssue><PubDate><Year>2022</Year></PubDate></JournalIssue></Journal>
                <AuthorList><Author><ForeName>Grace</ForeName><LastName>Hopper</LastName></Author></AuthorList>
                <Abstract><AbstractText>Clinical evidence.</AbstractText></Abstract>
              </Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1111/pubmed</ArticleId></ArticleIdList></PubmedData>
            </PubmedArticle></PubmedArticleSet>
            """,
        },
    )

    paper = PubMedSource(transport).search("topic", 1)[0]

    assert paper.title == "PubMed paper"
    assert paper.year == 2022
    assert paper.doi == "10.1111/pubmed"
    assert paper.url.endswith("/123/")


def test_arxiv_source_parses_atom_feed() -> None:
    transport = XMLTransport(
        {},
        {
            "https://export.arxiv.org/api/query": """
            <feed xmlns="http://www.w3.org/2005/Atom"><entry>
              <id>http://arxiv.org/abs/1234.5678</id><title>Arxiv paper</title>
              <summary>Summary text.</summary><published>2021-01-01T00:00:00Z</published>
              <author><name>Alan Turing</name></author>
            </entry></feed>
            """,
        },
    )

    paper = ArxivSource(transport).search("topic", 1)[0]

    assert paper.title == "Arxiv paper"
    assert paper.year == 2021
    assert paper.external_id.endswith("1234.5678")


# --------------------------------------------------------------------------- #
# max_results 的总量语义（刻意行为变更）
# --------------------------------------------------------------------------- #
class _StubSource:
    """返回固定条目的检索源；记录被传入的单源预算与查询串。"""

    def __init__(self, name: str, papers: list[PaperRecord]) -> None:
        self.name = name
        self.papers = papers
        self.received_budget: int | None = None
        self.received_query: str | None = None

    def search(self, query: str, max_results: int) -> list[PaperRecord]:
        self.received_query = query
        self.received_budget = max_results
        return list(self.papers)


def _papers(source: str, count: int, prefix: str = "") -> list[PaperRecord]:
    return [
        PaperRecord(
            title=f"{prefix}{source}-paper-{index}",
            doi=f"10.0000/{source}-{index}",
            source=source,
        )
        for index in range(count)
    ]


def test_max_results_is_a_total_cap_across_sources() -> None:
    source_a = _StubSource("a", _papers("a", 10))
    source_b = _StubSource("b", _papers("b", 10))
    source_c = _StubSource("c", _papers("c", 10))
    searcher = LiteratureSearcher({"a": source_a, "b": source_b, "c": source_c})

    report = searcher.search("topic", ["a", "b", "c"], 10)

    # 旧行为会给到最多 30 篇；新行为是总量上限 10。
    assert len(report.papers) == 10
    # 单源预算向上取整：ceil(10 / 3) == 4。
    assert source_a.received_budget == 4
    assert source_b.received_budget == 4
    assert source_c.received_budget == 4


def test_max_results_smaller_than_source_count_gives_budget_of_one() -> None:
    source_a = _StubSource("a", _papers("a", 10))
    source_b = _StubSource("b", _papers("b", 10))
    source_c = _StubSource("c", _papers("c", 10))
    searcher = LiteratureSearcher({"a": source_a, "b": source_b, "c": source_c})

    report = searcher.search("topic", ["a", "b", "c"], 3)

    assert source_a.received_budget == 1
    assert source_b.received_budget == 1
    assert source_c.received_budget == 1
    assert len(report.papers) <= 3


def test_search_multi_source_hit_ranks_first_and_records_sources() -> None:
    shared = PaperRecord(title="Shared paper", doi="10.5/shared", source="a")
    only_a = PaperRecord(title="Only A", doi="10.5/only-a", source="a")
    only_b = PaperRecord(title="Only B", doi="10.5/only-b", source="b")
    only_c = PaperRecord(title="Only C", doi="10.5/only-c", source="c")
    searcher = LiteratureSearcher(
        {
            "a": _StubSource("a", [only_a, shared]),
            "b": _StubSource("b", [only_b, shared]),
            "c": _StubSource("c", [only_c, shared]),
        }
    )

    report = searcher.search("shared", ["a", "b", "c"], 10)

    assert report.papers[0].doi == "10.5/shared"
    assert any("3 个检索源" in reason for reason in report.ranking_reasons)


def test_search_accounts_for_dropped_by_limit_without_silent_loss() -> None:
    searcher = LiteratureSearcher(
        {
            "a": _StubSource("a", _papers("a", 10)),
            "b": _StubSource("b", _papers("b", 10)),
            "c": _StubSource("c", _papers("c", 10)),
        }
    )

    report = searcher.search("topic", ["a", "b", "c"], 10)

    # 三源共返回 30 条、零重复；最终保留 10 条，理论上限裁掉 20 条。
    assert report.total_found == 30
    assert report.deduplicated_count == 30
    assert report.dropped_by_limit == 30 - len(report.papers)
    assert report.deduplicated_count - len(report.papers) == report.dropped_by_limit
    assert report.deduplicated_count == report.total_found  # 无重复


def test_search_dedup_accounting_is_arithmetically_consistent() -> None:
    shared = PaperRecord(title="Shared", doi="10.9/shared", source="a")
    searcher = LiteratureSearcher(
        {
            "a": _StubSource("a", [*_papers("a", 5), shared]),
            "b": _StubSource("b", [*_papers("b", 5), shared]),
        }
    )

    report = searcher.search("topic", ["a", "b"], 100)

    assert report.total_found == 12            # 5 + 5 + 2（共享在两源各出现一次）
    assert report.deduplicated_count == 11     # 共享去重后只算一次
    assert report.total_found - report.deduplicated_count == 1
    assert report.dropped_by_limit == 0        # 上限 100，未触发裁剪
    assert report.deduplicated_count - len(report.papers) == report.dropped_by_limit


def test_search_uses_source_specific_query_syntax() -> None:
    source = _StubSource("pubmed", [])
    arxiv = _StubSource("arxiv", [])
    searcher = LiteratureSearcher({"pubmed": source, "arxiv": arxiv})

    searcher.search("climate change", ["pubmed", "arxiv"], 10)

    assert source.received_query is not None
    assert " AND " in source.received_query
    assert arxiv.received_query is not None
    assert " AND " in arxiv.received_query


def test_search_falls_back_to_original_topic_query() -> None:
    source = _StubSource("crossref", [])
    searcher = LiteratureSearcher({"crossref": source})

    searcher.search("the impact of research", ["crossref"], 5)

    # 词项全被剔除 -> 回退为原主题，绝不构造空查询。
    assert source.received_query == "the impact of research"


def test_search_rejects_empty_sources_list() -> None:
    searcher = LiteratureSearcher({})

    try:
        searcher.search("topic", [], 5)
    except ValueError as exc:
        assert "sources" in str(exc)
    else:
        raise AssertionError("空 sources 应当抛 ValueError")


def test_search_ranking_is_deterministic_for_identical_inputs() -> None:
    """相同输入连续调用必须逐条一致。"""

    def build(order: list[str]) -> list[str]:
        sources = {
            "a": _StubSource("a", _papers("a", 4)),
            "b": _StubSource("b", _papers("b", 4)),
            "c": _StubSource("c", _papers("c", 4)),
        }
        report = LiteratureSearcher(sources).search("topic", order, 12)
        return [paper.doi for paper in report.papers]

    baseline = build(["a", "b", "c"])
    assert len(baseline) == 12
    for _ in range(5):
        assert build(["a", "b", "c"]) == baseline


def test_search_result_is_order_independent_of_source_ordering() -> None:
    """**换 sources 顺序不得改变结果**——否则用户改源顺序就会改变进入手稿的证据集。

    构造完全并列的文献（同分、同源数、同年、无引用），并让它们分散在不同源里，
    这样 `sources` 的遍历顺序就是唯一的顺序输入。排序必须是文献集合的纯函数。
    """

    def tied(doi: str, title: str) -> PaperRecord:
        return PaperRecord(title=title, doi=doi, year=2024)

    def build(order: list[str]) -> list[str]:
        sources = {
            "a": _StubSource("a", [tied("10.1/a", "sleep memory"), tied("10.1/d", "x")]),
            "b": _StubSource("b", [tied("10.1/b", "sleep memory"), tied("10.1/e", "y")]),
            "c": _StubSource("c", [tied("10.1/c", "sleep and memory")]),
        }
        report = LiteratureSearcher(sources).search("sleep memory", order, 10)
        return [paper.doi for paper in report.papers]

    baseline = build(["a", "b", "c"])
    assert len(baseline) == 5
    for order in permutations(["a", "b", "c"]):
        assert build(list(order)) == baseline


# --------------------------------------------------------------------------- #
# 缺陷 A：同一 DOI 元数据不同 -> 得分也必须与源顺序无关
# --------------------------------------------------------------------------- #
def test_search_score_is_order_independent_for_duplicate_metadata() -> None:
    def rich(source: str) -> _StubSource:
        return _StubSource(
            source,
            [
                PaperRecord(
                    title="generic",
                    doi="10.t",
                    abstract="coffee climate abstract",
                    source=source,
                ),
                PaperRecord(title="other", doi="10.r", source=source),
            ],
        )

    def sparse(source: str) -> _StubSource:
        return _StubSource(
            source,
            [
                PaperRecord(title="generic", doi="10.t", source=source),
                PaperRecord(title="other", doi="10.r", source=source),
            ],
        )

    def build(order: list[str]) -> tuple[list[str], str]:
        sources = {"a": rich("a"), "b": sparse("b")}
        report = LiteratureSearcher(sources).search("coffee climate", order, 10)
        generic = next(
            reason
            for reason in report.ranking_reasons
            if "generic" in reason
        )
        return [paper.doi for paper in report.papers], generic

    forward, f_reason = build(["a", "b"])
    backward, b_reason = build(["b", "a"])

    assert forward == backward
    # 得分与命中词项解释也必须一致（合并后摘要取最长，与顺序无关）。
    assert f_reason == b_reason
    assert "主题词项命中 2 个" in f_reason


# --------------------------------------------------------------------------- #
# 缺陷 B：无 DOI 的不同论文不得因标点被静默合并
# --------------------------------------------------------------------------- #
def test_search_does_not_silently_merge_distinct_no_doi_papers() -> None:
    p1 = PaperRecord(title="Sleep: memory & learning!", year=2020, authors=["A"])
    p2 = PaperRecord(title="Sleep memory learning", year=2021, authors=["B"])

    report = LiteratureSearcher({"x": _StubSource("x", [p1, p2])}).search(
        "sleep", ["x"], 10
    )

    assert report.total_found == 2
    assert report.deduplicated_count == 2
    assert len(report.papers) == 2
    # 该决策必须**可见**，不得静默。
    assert any("不足以判定为同一篇" in reason for reason in report.ranking_reasons)


def test_search_still_merges_true_duplicates() -> None:
    p1 = PaperRecord(title="Same", doi="10.1/same", source="a")
    p2 = PaperRecord(title="Same", doi="10.1/same", source="b")

    report = LiteratureSearcher(
        {"a": _StubSource("a", [p1]), "b": _StubSource("b", [p2])}
    ).search("same", ["a", "b"], 10)

    assert report.total_found == 2
    assert report.deduplicated_count == 1
    assert len(report.papers) == 1


# --------------------------------------------------------------------------- #
# 缺陷 D：重名来源 -> counts_by_source 与 total_found 自洽
# --------------------------------------------------------------------------- #
def test_search_deduplicates_repeated_source_names() -> None:
    source = _StubSource("ok", [PaperRecord(title="p1", doi="10.x")])
    report = LiteratureSearcher({"ok": source}).search("t", ["ok", "ok", "ok"], 10)

    assert report.sources_attempted == ["ok"]
    assert report.total_found == 1
    assert sum(report.counts_by_source.values()) == report.total_found


# --------------------------------------------------------------------------- #
# search.year_range：年份过滤（真实实现，非"只读配置"）
# --------------------------------------------------------------------------- #
def _year_paper(title: str, year: int | None, doi: str) -> PaperRecord:
    return PaperRecord(title=title, doi=doi, year=year)


def test_year_range_default_and_zero_zero_do_not_filter() -> None:
    """``None`` 与 ``[0, 0]`` 都表示不过滤——行为与 Phase 7 逐字节一致。"""
    papers = [
        _year_paper("old", 1999, "10.y/old"),
        _year_paper("mid", 2015, "10.y/mid"),
        _year_paper("new", 2025, "10.y/new"),
        _year_paper("unknown", None, "10.y/unknown"),
    ]

    def build(year_range: tuple[int, int] | None) -> tuple[list[str], int, list[str]]:
        report = LiteratureSearcher({"a": _StubSource("a", papers)}).search(
            "t", ["a"], 10, year_range
        )
        return (
            [paper.doi for paper in report.papers],
            report.excluded_by_year,
            list(report.ranking_reasons),
        )

    no_arg = build(None)
    zero_zero = build((0, 0))

    assert no_arg == zero_zero
    # 顺序由排序决定（新近优先），这里只断言集合与可见字段一致。
    assert sorted(no_arg[0]) == ["10.y/mid", "10.y/new", "10.y/old", "10.y/unknown"]
    assert no_arg[1] == 0
    # 不过滤时不出现年份过滤说明，排序依据与 Phase 7 一致。
    assert not any("年份范围过滤" in reason for reason in no_arg[2])


def test_year_range_filters_both_bounds_and_reports_excluded() -> None:
    papers = [
        _year_paper("too-old", 1990, "10.y/old"),
        _year_paper("in-1", 2010, "10.y/in1"),
        _year_paper("in-2", 2015, "10.y/in2"),
        _year_paper("too-new", 2025, "10.y/new"),
    ]
    report = LiteratureSearcher({"a": _StubSource("a", papers)}).search(
        "t", ["a"], 10, (2010, 2015)
    )

    assert sorted(paper.doi for paper in report.papers) == ["10.y/in1", "10.y/in2"]
    # 边界包含：起始年与结束年都在范围内。
    assert report.excluded_by_year == 2
    assert any("年份范围过滤" in reason for reason in report.ranking_reasons)
    assert any("2 篇" in reason for reason in report.ranking_reasons)


def test_year_range_single_sided_zero_means_unbounded() -> None:
    papers = [
        _year_paper("1999", 1999, "10.y/a"),
        _year_paper("2010", 2010, "10.y/b"),
        _year_paper("2020", 2020, "10.y/c"),
    ]

    lower_only = LiteratureSearcher({"a": _StubSource("a", papers)}).search(
        "t", ["a"], 10, (2010, 0)
    )
    upper_only = LiteratureSearcher({"a": _StubSource("a", papers)}).search(
        "t", ["a"], 10, (0, 2010)
    )

    assert sorted(p.doi for p in lower_only.papers) == ["10.y/b", "10.y/c"]
    assert lower_only.excluded_by_year == 1
    assert sorted(p.doi for p in upper_only.papers) == ["10.y/a", "10.y/b"]
    assert upper_only.excluded_by_year == 1


def test_year_range_keeps_papers_with_unknown_year_but_makes_it_visible() -> None:
    """**选定语义：``year`` 为 ``None`` 的文献被保留**（宁可多留也不丢证据）。

    年份缺失是上游元数据不全，不是"不在范围内"的证据。保留必须**可见**。
    """
    papers = [
        _year_paper("in", 2015, "10.y/in"),
        _year_paper("unknown-a", None, "10.y/unknown-a"),
        _year_paper("unknown-b", None, "10.y/unknown-b"),
        _year_paper("out", 1990, "10.y/out"),
    ]
    report = LiteratureSearcher({"a": _StubSource("a", papers)}).search(
        "t", ["a"], 10, (2010, 2020)
    )

    assert sorted(paper.doi for paper in report.papers) == [
        "10.y/in",
        "10.y/unknown-a",
        "10.y/unknown-b",
    ]
    # 无年份文献**不**计入排除数。
    assert report.excluded_by_year == 1
    # 保留无年份文献这一事实必须可见，避免用户误以为"范围内只有这些"。
    assert any("年份未知" in reason for reason in report.ranking_reasons)


def test_year_range_exclusion_accounting_is_arithmetically_consistent() -> None:
    papers = [
        _year_paper("a", 2000, "10.y/a"),
        _year_paper("b", 2010, "10.y/b"),
        _year_paper("c", 2020, "10.y/c"),
        _year_paper("d", None, "10.y/d"),
    ]
    report = LiteratureSearcher({"a": _StubSource("a", papers)}).search(
        "t", ["a"], 10, (2005, 2015)
    )

    assert report.deduplicated_count == 4
    assert report.excluded_by_year == 2                 # 2000 与 2020 出界
    assert len(report.papers) == 2                      # 2010（在范围内）+ 无年份
    # 算术自洽：去重候选 = 范围外 + 最终保留 + 总量上限裁掉。
    assert report.deduplicated_count == (
        report.excluded_by_year + len(report.papers) + report.dropped_by_limit
    )
    assert report.dropped_by_limit == 0


def test_year_range_interacts_with_max_results_cap() -> None:
    """过滤先于总量裁剪：上限只作用于"范围内"的候选。"""
    papers = [_year_paper(f"p{i}", 2000 + i, f"10.y/{i}") for i in range(6)]
    report = LiteratureSearcher({"a": _StubSource("a", papers)}).search(
        "t", ["a"], 2, (2002, 2004)  # 范围内 3 篇（2002/2003/2004），上限 2
    )

    assert report.excluded_by_year == 3
    assert len(report.papers) == 2
    assert report.dropped_by_limit == 1
    assert report.deduplicated_count == (
        report.excluded_by_year + len(report.papers) + report.dropped_by_limit
    )


def test_year_range_filter_preserves_determinism_across_source_order() -> None:
    """过滤不得破坏确定性：同一集合 + 同一范围，任意源顺序结果一致。"""

    def build(order: list[str]) -> list[str]:
        sources = {
            "a": _StubSource(
                "a",
                [_year_paper("sleep memory", 2015, "10.y/a"), _year_paper("x", 1990, "10.y/d")],
            ),
            "b": _StubSource(
                "b",
                [_year_paper("sleep memory", 2016, "10.y/b"), _year_paper("y", None, "10.y/e")],
            ),
            "c": _StubSource("c", [_year_paper("sleep and memory", 2015, "10.y/c")]),
        }
        report = LiteratureSearcher(sources).search("sleep memory", order, 10, (2010, 2020))
        return [paper.doi for paper in report.papers]

    baseline = build(["a", "b", "c"])
    for order in permutations(["a", "b", "c"]):
        assert build(list(order)) == baseline


def test_year_range_invalid_shape_raises_before_searching() -> None:
    """非二元范围是编程错误，必须尽早失败（不发起检索）。"""
    source = _StubSource("a", [])

    for bad in ((2020,), (2020, 2021, 2022)):
        try:
            LiteratureSearcher({"a": source}).search("t", ["a"], 5, bad)
        except ValueError as exc:
            assert "year_range" in str(exc)
        else:
            raise AssertionError("非法 year_range 应当抛 ValueError")

    assert source.received_budget is None  # 未发起任何检索


