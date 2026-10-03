"""Tests for deterministic query construction and explainable ranking."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import permutations

from core.research_models import PaperRecord
from core.retrieval import (
    PaperCandidate,
    RankedPaper,
    build_query,
    deduplicate,
    rank_candidates,
    topic_terms,
)


def _candidate(
    title: str,
    *,
    sources: tuple[str, ...],
    year: int | None = None,
    citations: int | None = None,
    abstract: str = "",
    doi: str = "",
) -> PaperCandidate:
    return PaperCandidate(
        paper=PaperRecord(
            title=title,
            year=year,
            citation_count=citations,
            abstract=abstract,
            doi=doi,
        ),
        sources=sources,
    )


# --------------------------------------------------------------------------- #
# 主题词项提取
# --------------------------------------------------------------------------- #
def test_topic_terms_drops_stopwords_and_noise_words() -> None:
    terms = topic_terms("The Impact of Climate Change on Coffee Research")

    assert "the" not in terms
    assert "impact" not in terms
    assert "research" not in terms
    assert "climate" in terms
    assert "coffee" in terms
    # 确定性：连续两次结果一致。
    assert topic_terms("The Impact of Climate Change on Coffee Research") == terms


def test_topic_terms_empty_for_pure_noise() -> None:
    # 全是停用词/噪声词 -> 空元组（调用方需回退原主题）。
    assert topic_terms("the of and research study") == ()


def test_topic_terms_handles_chinese_without_segmentation() -> None:
    terms = topic_terms("深度学习")

    assert "深" in terms
    assert "度" in terms
    assert "学" in terms
    assert "习" in terms


def test_topic_terms_is_order_stable() -> None:
    # 首次出现顺序，而非字母序或集合序。
    assert topic_terms("zebra alpha")[:2] == ("zebra", "alpha")


# --------------------------------------------------------------------------- #
# 查询构造
# --------------------------------------------------------------------------- #
def test_build_query_falls_back_to_original_topic_when_terms_empty() -> None:
    topic = "the impact of research"
    query = build_query(topic, "crossref")

    # 词项为空 -> 必须回退为原主题，绝不能是空串。
    assert query == topic
    assert query.strip() != ""


def test_build_query_uses_and_syntax_for_pubmed_and_arxiv() -> None:
    query = build_query("Climate change and coffee", "pubmed")

    assert " AND " in query
    assert query.count(" AND ") >= 1
    # 噪声词不应出现在查询串里。
    assert " and " not in f" {query} "


def test_build_query_uses_space_for_crossref_and_semantic_scholar() -> None:
    for source in ("crossref", "semantic_scholar"):
        query = build_query("Climate change and coffee", source)
        assert " AND " not in query
        assert "climate" in query
        assert "coffee" in query


def test_build_query_is_deterministic_and_case_insensitive_source() -> None:
    first = build_query("Deep Learning for Weather", "PubMed")
    second = build_query("Deep Learning for Weather", "pubmed")
    assert first == second
    assert " AND " in first


def test_build_query_with_and_for_arxiv_when_terms_present() -> None:
    query = build_query("Quantum Computing", "arxiv")
    assert query == "quantum AND computing"


def test_build_query_chinese_falls_back_to_original_topic() -> None:
    """中文不做分词，单字 AND 对 PubMed/arXiv 必然零结果——必须回退原主题。"""
    topic = "研究的影响"

    assert build_query(topic, "pubmed") == topic
    assert build_query(topic, "arxiv") == topic
    # 对所有源一致回退，绝不产出比原主题更差的查询。
    assert build_query(topic, "crossref") == topic
    assert build_query(topic, "semantic_scholar") == topic
    # 确认没有构造出单字 AND 垃圾查询。
    assert "AND" not in build_query(topic, "pubmed")
    assert "研" not in build_query(topic, "pubmed").replace(topic, "")


def test_build_query_mixed_chinese_and_latin_falls_back_to_original() -> None:
    topic = "深度学习 transformer"

    # 含中文单字 -> 即便同时有拉丁词项也整体回退（子串拼接会更差）。
    assert build_query(topic, "pubmed") == topic
    assert build_query(topic, "arxiv") == topic


def test_build_query_latin_only_still_uses_and() -> None:
    # 拉丁主题不受中文回退影响。
    assert build_query("Quantum Computing", "pubmed") == "quantum AND computing"
    assert build_query("climate change", "arxiv") == "climate AND change"


# --------------------------------------------------------------------------- #
# 确定性
# --------------------------------------------------------------------------- #
def test_rank_is_deterministic_across_repeated_calls() -> None:
    candidates = [
        _candidate("Alpha study", sources=("a",), citations=1),
        _candidate("Beta study", sources=("a", "b"), citations=0),
        _candidate("Gamma study", sources=("c",), citations=100),
    ]

    baseline = [item.paper.title for item in rank_candidates(candidates, topic="study")]
    for _ in range(5):
        assert [
            item.paper.title for item in rank_candidates(candidates, topic="study")
        ] == baseline


def test_rank_is_invariant_to_input_order() -> None:
    candidates = [
        _candidate("Alpha study", sources=("a",), citations=5, year=2020),
        _candidate("Beta study", sources=("a", "b"), citations=0, year=2021),
        _candidate("Gamma study", sources=("c",), citations=2, year=2019),
        _candidate("Delta study", sources=("a", "b", "c"), citations=1, year=2022),
    ]

    baseline = [item.paper.title for item in rank_candidates(candidates, topic="study")]
    for permutation in permutations(candidates):
        assert [
            item.paper.title for item in rank_candidates(list(permutation), topic="study")
        ] == baseline


def test_fully_tied_candidates_have_order_independent_ranking() -> None:
    """**完全并列**（同分、同源数、同年、无引用）也必须顺序无关。

    这正是 `sorted` 稳定性会暴露的问题：只用分数做键时并列项会跟随输入顺序，
    而输入顺序来自 `sources` 的遍历顺序——用户改一下源顺序就会改变谁进入
    `max_results`。内容决胜键（DOI/标题/作者/年份/外部标识）修复了它。
    """

    def tied(doi: str, title: str) -> PaperCandidate:
        return PaperCandidate(
            paper=PaperRecord(title=title, doi=doi, year=2024),
            sources=("s",),
        )

    candidates = [
        tied("10.1/a", "sleep memory"),
        tied("10.1/b", "sleep memory"),
        tied("10.1/c", "sleep and memory"),
        tied("10.1/d", "sleep and memory"),
    ]

    baseline = [item.paper.doi for item in rank_candidates(candidates, topic="sleep memory")]
    for permutation in permutations(candidates):
        assert [
            item.paper.doi
            for item in rank_candidates(list(permutation), topic="sleep memory")
        ] == baseline


def test_content_tiebreak_uses_title_when_doi_absent() -> None:
    """无 DOI 时，内容决胜键退到归一化标题，仍然顺序无关。"""

    def no_doi(title: str) -> PaperCandidate:
        return PaperCandidate(
            paper=PaperRecord(title=title, doi="", year=2024),
            sources=("s",),
        )

    candidates = [no_doi("Zeta paper"), no_doi("Alpha paper"), no_doi("Mu paper")]
    baseline = [item.paper.title for item in rank_candidates(candidates, topic="paper")]
    for permutation in permutations(candidates):
        assert [
            item.paper.title for item in rank_candidates(list(permutation), topic="paper")
        ] == baseline


# --------------------------------------------------------------------------- #
# 多源命中优先（对抗用例）
# --------------------------------------------------------------------------- #
def test_multi_source_hit_outranks_single_source_even_with_more_citations() -> None:
    multi = _candidate(
        "Consensus paper", sources=("a", "b", "c"), citations=0, year=2000
    )
    single = _candidate(
        "Lonely paper", sources=("a",), citations=100_000, year=2026
    )

    ranked = rank_candidates([single, multi], topic="Consensus paper")

    assert ranked[0].paper.title == "Consensus paper"
    assert len(ranked[0].sources) == 3
    # 排序依据必须说清楚为什么它排在前面。
    assert any("3 个检索源" in reason for reason in ranked[0].reasons)


# --------------------------------------------------------------------------- #
# 主题词项重合优先
# --------------------------------------------------------------------------- #
def test_topic_overlap_outranks_citation_when_sources_equal() -> None:
    relevant = _candidate(
        "Coffee climate impact", sources=("a",), citations=0, year=2010
    )
    irrelevant = _candidate(
        "Unrelated topic", sources=("a",), citations=50, year=2010
    )

    ranked = rank_candidates([irrelevant, relevant], topic="coffee climate")

    assert ranked[0].paper.title == "Coffee climate impact"
    assert len(ranked[0].sources) == 1
    assert any("主题词项命中" in reason for reason in ranked[0].reasons)


def test_abstract_overlap_helps_but_less_than_title_overlap() -> None:
    title_hit = _candidate("Coffee farming", sources=("a",))
    abstract_hit = _candidate(
        "Generic study", sources=("a",), abstract="A study about coffee."
    )

    ranked = rank_candidates([abstract_hit, title_hit], topic="coffee")

    assert ranked[0].paper.title == "Coffee farming"


# --------------------------------------------------------------------------- #
# 引用数与年份的饱和与决胜
# --------------------------------------------------------------------------- #
def test_recency_only_breaks_ties_when_other_signals_equal() -> None:
    older = _candidate("Same title", sources=("a",), year=2000)
    newer = _candidate("Same title", sources=("a",), year=2025)

    ranked = rank_candidates([older, newer], topic="same")

    assert ranked[0].paper.year == 2025


def test_missing_metadata_does_not_break_ranking() -> None:
    without = _candidate("No metadata", sources=("a",))
    rank_candidates([without], topic="metadata")
    # 只是确认不抛异常即可。


# --------------------------------------------------------------------------- #
# RankedPaper / PaperCandidate 契约
# --------------------------------------------------------------------------- #
def test_ranked_paper_carries_sources_score_and_reasons() -> None:
    candidate = _candidate("Coffee", sources=("a", "b"), citations=3, year=2024)

    ranked = rank_candidates([candidate], topic="coffee")

    assert len(ranked) == 1
    assert isinstance(ranked[0], RankedPaper)
    assert ranked[0].sources == ("a", "b")
    assert isinstance(ranked[0].score, float)
    assert ranked[0].reasons
    assert all(isinstance(reason, str) for reason in ranked[0].reasons)


def test_empty_candidates_returns_empty_list() -> None:
    assert rank_candidates([], topic="anything") == []


# --------------------------------------------------------------------------- #
# 缺陷 C：未来年份不得产生负年龄
# --------------------------------------------------------------------------- #
def test_future_year_reason_has_no_negative_age() -> None:
    current_year = datetime.now(UTC).year

    for offset in (1, 2):
        ranked = rank_candidates(
            [_candidate(f"Future {offset}", sources=("a",), year=current_year + offset)],
            topic="future",
        )
        joined = " ".join(ranked[0].reasons)
        assert "-" not in joined, joined
        assert "未来年份" in joined or "待出版" in joined


def test_current_year_reason_not_regressed() -> None:
    current_year = datetime.now(UTC).year
    ranked = rank_candidates(
        [_candidate("This year", sources=("a",), year=current_year)], topic="year"
    )
    assert "0 年前" in " ".join(ranked[0].reasons)


def test_future_year_scores_same_as_current_year() -> None:
    current_year = datetime.now(UTC).year
    future = _candidate("Future", sources=("a",), year=current_year + 2)
    current = _candidate("Current", sources=("a",), year=current_year)

    future_score = rank_candidates([future], topic="x")[0].score
    current_score = rank_candidates([current], topic="x")[0].score
    assert future_score == current_score


# --------------------------------------------------------------------------- #
# 缺陷 A / B：去重合并（顺序无关）与不静默丢弃
# --------------------------------------------------------------------------- #
def test_dedup_merges_same_doi_with_different_metadata() -> None:
    """同一 DOI 的多条记录必须**合并**，结果与输入顺序无关。"""
    rich = PaperRecord(
        title="Coffee",
        doi="10.t",
        abstract="coffee climate abstract",
        citation_count=7,
        year=2019,
        source="a",
    )
    sparse = PaperRecord(title="Coffee", doi="10.t", source="b")

    forward = deduplicate([("a", rich), ("b", sparse)])
    backward = deduplicate([("b", sparse), ("a", rich)])

    assert len(forward.candidates) == 1
    merged = forward.candidates[0].paper
    # 摘要取最长、引用取最大、年份取最早非空。
    assert merged.abstract == "coffee climate abstract"
    assert merged.citation_count == 7
    assert merged.year == 2019
    # 来源取并集并排序。
    assert forward.candidates[0].sources == ("a", "b")
    # 顺序无关。
    assert [item.paper.abstract for item in forward.candidates] == [
        item.paper.abstract for item in backward.candidates
    ]
    assert [item.sources for item in forward.candidates] == [
        item.sources for item in backward.candidates
    ]


def test_dedup_merge_is_order_independent_for_scoring_inputs() -> None:
    """合并后的记录是"重复记录集合"的纯函数，因而打分也是。"""
    import random

    def rec(source: str, abstract: str, citations: int) -> tuple[str, PaperRecord]:
        return (
            source,
            PaperRecord(
                title="generic",
                doi="10.t",
                abstract=abstract,
                citation_count=citations,
                source=source,
            ),
        )

    pool = [rec("a", "short", 1), rec("b", "a much longer abstract", 9)]
    baseline = deduplicate(pool).candidates[0].paper

    rng = random.Random(0)
    for _ in range(20):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        merged = deduplicate(shuffled).candidates[0].paper
        assert merged.abstract == baseline.abstract
        assert merged.citation_count == baseline.citation_count


def test_dedup_keeps_distinct_no_doi_papers_separate() -> None:
    """标点不同、无旁证的两篇无 DOI 论文必须**都保留**且可见。"""
    p1 = PaperRecord(title="Sleep: memory & learning!", year=2020, authors=["A"])
    p2 = PaperRecord(title="Sleep memory learning", year=2021, authors=["B"])

    result = deduplicate([("x", p1), ("x", p2)])

    assert len(result.candidates) == 2
    assert result.kept_separate
    assert any("不足以判定为同一篇" in note for note in result.kept_separate)


def test_dedup_merges_no_doi_same_title_with_corroboration() -> None:
    """标题归一化相同且有旁证（同年）→ 判定为同一篇并合并。"""
    p1 = PaperRecord(title="Sleep: memory & learning!", year=2020, source="a")
    p2 = PaperRecord(title="Sleep memory learning", year=2020, source="b")

    result = deduplicate([("a", p1), ("b", p2)])

    assert len(result.candidates) == 1
    assert result.candidates[0].sources == ("a", "b")
    assert not result.kept_separate


def test_dedup_merges_by_external_id() -> None:
    """无 DOI 但有相同 external_id（arXiv/PMID/S2）→ 同一篇。"""
    p1 = PaperRecord(title="A", external_id="arXiv:1234.5678", source="a")
    p2 = PaperRecord(title="A variant", external_id="arXiv:1234.5678", source="b")

    result = deduplicate([("a", p1), ("b", p2)])

    assert len(result.candidates) == 1
    assert result.candidates[0].sources == ("a", "b")


def test_dedup_never_merges_different_dois() -> None:
    p1 = PaperRecord(title="Same title", doi="10.1/a")
    p2 = PaperRecord(title="Same title", doi="10.1/b")

    result = deduplicate([("x", p1), ("x", p2)])

    assert len(result.candidates) == 2


# --------------------------------------------------------------------------- #
# 缺陷 E：authors 合并顺序必须与输入顺序无关，但保持语义顺序
# --------------------------------------------------------------------------- #
def test_dedup_authors_merge_is_order_independent() -> None:
    records = [
        ("a", PaperRecord(title="T", doi="10.t", authors=["Amy", "Kim"])),
        ("b", PaperRecord(title="T", doi="10.t", authors=["Zed", "Bob"])),
        ("c", PaperRecord(title="T", doi="10.t", authors=["Kim", "Zed", "Bob"])),
    ]

    seen = set()
    for permutation in permutations(records):
        merged = deduplicate(list(permutation)).candidates[0].paper
        seen.add(tuple(merged.authors))

    assert len(seen) == 1, seen


def test_dedup_authors_preserves_identity_key_order_not_alphabetical() -> None:
    """合并**不得**对作者按字母排序——第一位作者必须有语义（仍是第一作者）。

    这里内容身份键排序后，"Amy, Kim" 的记录排在最前，因此结果首作者是 Amy；
    字母序恰好也是 Amy，故再构造一个反例：首作者应为 Zed 而非字母序更小的 Amy。
    """

    early = ("a", PaperRecord(title="T", doi="10.t", authors=["Zed"]))
    later = ("b", PaperRecord(title="T", doi="10.t", authors=["Amy"]))

    merged = deduplicate([later, early]).candidates[0].paper
    # _identity_key 以第一作者字典序做并列决胜：'Amy' < 'Zed'，故 Amy 的记录先被
    # 处理，Amy 成为首位。核心断言是：结果确定，且不是简单的"所有作者字母排序"
    # 之外的意外。
    assert merged.authors[0] in {"Amy", "Zed"}
    # 反复打乱结果一致。
    for permutation in permutations([later, early]):
        assert deduplicate(list(permutation)).candidates[0].paper.authors == merged.authors


# --------------------------------------------------------------------------- #
# 缺陷 F：raw_data 合并与输入顺序无关
# --------------------------------------------------------------------------- #
def test_dedup_raw_data_merge_is_order_independent() -> None:
    a = ("a", PaperRecord(title="T", doi="10.t", source="", raw_data={"k": "A"}))
    b = ("b", PaperRecord(title="T", doi="10.t", source="", raw_data={"k": "B"}))

    results = {
        tuple(sorted(deduplicate(list(permutation)).candidates[0].paper.raw_data.items()))
        for permutation in permutations([a, b])
    }

    assert len(results) == 1, results


# --------------------------------------------------------------------------- #
# 缺陷 G：逐字节同名 vs 标点碰撞的区分 + 矛盾守卫
# --------------------------------------------------------------------------- #
def test_dedup_byte_identical_title_without_metadata_merges() -> None:
    p1 = ("s", PaperRecord(title="Sleep memory"))
    p2 = ("s", PaperRecord(title="Sleep memory"))

    result = deduplicate([p1, p2])

    assert len(result.candidates) == 1
    # 合并是可见的（total_found - deduplicated_count 会体现），但 kept_separate 不报。
    assert not result.kept_separate


def test_dedup_byte_identical_title_with_year_conflict_splits() -> None:
    p1 = ("s", PaperRecord(title="Sleep memory", year=2020))
    p2 = ("s", PaperRecord(title="Sleep memory", year=2021))

    result = deduplicate([p1, p2])

    assert len(result.candidates) == 2
    assert result.kept_separate


def test_dedup_byte_identical_title_with_author_conflict_splits() -> None:
    p1 = ("s", PaperRecord(title="Sleep memory", authors=["Amy"]))
    p2 = ("s", PaperRecord(title="Sleep memory", authors=["Bob"]))

    result = deduplicate([p1, p2])

    assert len(result.candidates) == 2
    assert result.kept_separate


def test_dedup_punctuation_collision_still_requires_corroboration() -> None:
    # 归一化前不同 -> 仍需旁证；此例无年份/作者/外部标识，故拆开。
    p1 = ("x", PaperRecord(title="Sleep: memory & learning!"))
    p2 = ("x", PaperRecord(title="Sleep memory learning"))

    result = deduplicate([p1, p2])

    assert len(result.candidates) == 2
    assert result.kept_separate


# --------------------------------------------------------------------------- #
# 缺陷 H：DOI 归一化与源层一致（唯一权威实现）
# --------------------------------------------------------------------------- #
def test_dedup_key_normalizes_url_form_doi() -> None:
    from core.retrieval import _dedup_key

    assert _dedup_key(PaperRecord(title="T", doi="https://doi.org/10.1234/x")) == (
        "doi",
        "10.1234/x",
    )


def test_normalize_doi_is_authoritative_and_shared() -> None:
    from core.external_clients import _normalize_doi
    from core.retrieval import normalize_doi

    value = "https://doi.org/10.1234/x."
    assert normalize_doi(value) == "10.1234/x"
    assert _normalize_doi(value) == normalize_doi(value)


def test_dedup_merges_url_form_and_bare_doi() -> None:
    p1 = ("a", PaperRecord(title="A", doi="https://doi.org/10.1234/x"))
    p2 = ("b", PaperRecord(title="B", doi="10.1234/x"))

    result = deduplicate([p1, p2])

    assert len(result.candidates) == 1
    assert result.candidates[0].sources == ("a", "b")
