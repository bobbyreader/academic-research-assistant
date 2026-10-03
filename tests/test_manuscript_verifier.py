"""手稿正文引用语义追溯的对抗性测试。

所有测试都用假的 LLM 对象，绝不触网。重点覆盖三类失败模式：

* 正文句子与被引文献**错配**必须被捕获（本门存在的理由）；
* 模型调用次数必须精确可控（无引用 0 次、有引用恰好 1 次）；
* ``## References`` 章节不得被当成引用句子（最容易踩的坑）。
"""

from __future__ import annotations

import json

import pytest

from core.claim_verifier import ClaimVerificationReport
from core.manuscript_verifier import (
    extract_citing_claims,
    render_manuscript_claim_markdown,
    verify_manuscript_claims,
)
from core.research_models import PaperRecord

P1_ABSTRACT = "Urban heat islands raise cardiovascular mortality in elderly residents."
P2_ABSTRACT = "Green roofs reduce indoor temperatures during summer heat waves."
P3_ABSTRACT = "A study on unrelated quantum optics in cold atoms."


class CountingLLM:
    """计数桩：记录 ``complete_json`` 的调用，用于断言调用次数。"""

    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover
        raise AssertionError("complete() 不应用于核验器")

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response  # type: ignore[return-value]


def _papers() -> list[PaperRecord]:
    return [
        PaperRecord(title="Urban heat and mortality", abstract=P1_ABSTRACT),
        PaperRecord(title="Green roofs", abstract=P2_ABSTRACT),
        PaperRecord(title="Quantum optics", abstract=P3_ABSTRACT),
    ]


def _verdict(
    claim_index: int,
    citation_id: str,
    verdict: str,
    *,
    claim: str,
    quote: str = "",
    rationale: str = "judged",
) -> dict:
    return {
        "claim_index": claim_index,
        "claim": claim,
        "citation_id": citation_id,
        "verdict": verdict,
        "quote": quote,
        "rationale": rationale,
    }


# --------------------------------------------------------------------------- #
# 1. 错配必须被捕获（本门的核心断言）
# --------------------------------------------------------------------------- #
def test_mismatched_pair_is_captured_as_unsupported() -> None:
    """句子讲热岛效应，却引用了讲量子光学的 P3，且 quote 取自 P3 摘要。

    stub 对「句子—文献不匹配」的配对返回 unsupported；断言该配对为
    unsupported 且整体 ``passed`` 为 False。
    """
    body = "城市热岛会显著提高老年人心血管死亡率 [P3]。"
    sentence = "城市热岛会显著提高老年人心血管死亡率 [P3]"
    llm = CountingLLM(
        {
            "verdicts": [
                _verdict(
                    0,
                    "P3",
                    "unsupported",
                    claim=sentence,
                    quote=P3_ABSTRACT,
                    rationale="P3 讨论量子光学，不涉及热岛。",
                )
            ]
        }
    )

    report = verify_manuscript_claims(llm, body, _papers())

    assert report.passed is False
    assert len(report.unsupported_claims) == 1
    evidence = report.claims[0].evidence[0]
    assert evidence.citation_id == "P3"
    assert evidence.verdict == "unsupported"
    assert any("unsupported" in warning for warning in report.warnings)
    assert len(llm.calls) == 1


def test_matching_pair_with_grounded_quote_passes() -> None:
    """正确配对 + 逐字引用 → supports，报告通过。"""
    body = "城市热岛会提高死亡率 [P1]。"
    sentence = "城市热岛会提高死亡率 [P1]"
    llm = CountingLLM(
        {
            "verdicts": [
                _verdict(
                    0,
                    "P1",
                    "supports",
                    claim=sentence,
                    quote=P1_ABSTRACT,
                )
            ]
        }
    )

    report = verify_manuscript_claims(llm, body, _papers())

    assert report.passed is True
    assert report.claims[0].overall == "supports"
    assert report.claims[0].evidence[0].verdict == "supports"


# --------------------------------------------------------------------------- #
# 2. 调用次数精确可控
# --------------------------------------------------------------------------- #
def test_no_citations_returns_empty_report_and_zero_model_calls() -> None:
    llm = CountingLLM({"verdicts": []})

    report = verify_manuscript_claims(llm, "这是一段没有任何引用标识的正文。", _papers())

    assert isinstance(report, ClaimVerificationReport)
    assert report.claims == []
    assert llm.calls == []
    assert any("未执行正文级核验" in warning for warning in report.warnings)


def test_many_sentences_still_exactly_one_model_call() -> None:
    """20 个带引用的句子也必须只触发一次 complete_json（批量）。"""
    body = " ".join(f"第{i}条论断 [P1]。" for i in range(20))
    llm = CountingLLM({"verdicts": []})

    report = verify_manuscript_claims(llm, body, _papers())

    assert len(llm.calls) == 1
    assert len(report.claims) == 20


def test_abstract_body_never_calls_model_when_no_valid_ids() -> None:
    """正文里只有未知标识（[P99]）→ 不可核验 → 0 次调用。"""
    llm = CountingLLM({"verdicts": []})

    report = verify_manuscript_claims(llm, "某条论断 [P99]。", _papers())

    assert report.claims == []
    assert llm.calls == []
    assert any("未执行正文级核验" in warning for warning in report.warnings)


# --------------------------------------------------------------------------- #
# 3. 抽取：句子切分、一多标识、去重
# --------------------------------------------------------------------------- #
def test_extract_splits_on_chinese_and_english_punctuation() -> None:
    body = "第一条论断 [P1]。第二条论断 [P2]！第三条 [P3]? Fourth one [P1]."
    claims = extract_citing_claims(body)

    assert [item["claim"] for item in claims] == [
        "第一条论断 [P1]",
        "第二条论断 [P2]",
        "第三条 [P3]",
        "Fourth one [P1]",
    ]


def test_extract_handles_newlines_as_sentence_boundaries() -> None:
    body = "论断甲 [P1]\n论断乙 [P2]"
    claims = extract_citing_claims(body)

    assert [item["claim"] for item in claims] == ["论断甲 [P1]", "论断乙 [P2]"]


def test_extract_one_sentence_may_cite_multiple_papers_in_order() -> None:
    body = "两个来源共同支持该结论 [P2][P1] 且与 [P3] 一致。"
    claims = extract_citing_claims(body)

    assert len(claims) == 1
    assert claims[0]["citation_ids"] == ["P2", "P1", "P3"]
    assert claims[0]["claim"] == "两个来源共同支持该结论 [P2][P1] 且与 [P3] 一致"


def test_extract_dedupes_repeated_markers_keeping_first_seen_order() -> None:
    body = "重复引用 [P3] 与 [P1] 以及再次 [P3]。"
    claims = extract_citing_claims(body)

    assert claims[0]["citation_ids"] == ["P3", "P1"]


def test_extract_strips_sentences() -> None:
    body = "   带空格的论断 [P1]   。   "
    claims = extract_citing_claims(body)

    assert claims[0]["claim"] == "带空格的论断 [P1]"


def test_extract_ignores_plain_markers_like_bare_p() -> None:
    body = "这不是引用 [P] 也不是 [Px]，只是普通文字 [P1]。"
    claims = extract_citing_claims(body)

    assert len(claims) == 1
    assert claims[0]["citation_ids"] == ["P1"]


def test_extract_returns_empty_for_body_without_markers() -> None:
    assert extract_citing_claims("没有任何引用。只是普通正文。") == []


# --------------------------------------------------------------------------- #
# 4. References 章节防御（回归）
# --------------------------------------------------------------------------- #
_REFERENCES_BODY = (
    "城市热岛会提高死亡率 [P1]。\n\n"
    "## References\n\n"
    "[P1] Zhang, L., Wang, H. (2021). Urban heat and mortality. *Nature Climate Change*."
    " https://doi.org/10.1000/xyz\n"
    "[P2] Smith, J. et al. (2019). Green roofs. *Building and Environment*.\n"
)


def test_references_section_lines_are_not_treated_as_citing_sentences() -> None:
    claims = extract_citing_claims(_REFERENCES_BODY)

    assert len(claims) == 1
    assert claims[0]["claim"] == "城市热岛会提高死亡率 [P1]"
    assert claims[0]["citation_ids"] == ["P1"]

    # 参考文献行绝不出现在抽取结果里。
    joined = " ".join(item["claim"] for item in claims)
    assert "Zhang" not in joined
    assert "doi.org" not in joined


def test_references_section_alone_yields_no_citations() -> None:
    """只有 References 章节 → 没有可核验正文 → 0 次模型调用。"""
    llm = CountingLLM({"verdicts": []})

    report = verify_manuscript_claims(llm, _REFERENCES_BODY.split("\n\n", 1)[1], _papers())

    assert report.claims == []
    assert llm.calls == []
    assert any("未执行正文级核验" in warning for warning in report.warnings)


def test_reference_entry_styled_line_without_heading_is_defended() -> None:
    """没有 ``## References`` 标题，但整行以 ``[P1]`` 开头且带作者特征 → 防御。"""
    body = (
        "正文论断 [P1]。\n"
        "[P1] Li, W. (2020). A paper title. *Journal of Testing*. doi.org/10.1/2\n"
    )
    claims = extract_citing_claims(body)

    assert len(claims) == 1
    assert claims[0]["claim"] == "正文论断 [P1]"


def test_legitimate_sentence_starting_with_marker_is_kept() -> None:
    """以标识开头但没有作者特征的正常正文句，不应被误杀。"""
    body = "[P1] 提供了支持该结论的关键证据。"
    claims = extract_citing_claims(body)

    assert len(claims) == 1
    assert claims[0]["citation_ids"] == ["P1"]


def test_code_fence_ends_body_extraction() -> None:
    body = "正文论断 [P1]。\n\n```\n[P2] fabricated reference line\n```\n"
    claims = extract_citing_claims(body)

    assert len(claims) == 1
    assert claims[0]["citation_ids"] == ["P1"]


# --------------------------------------------------------------------------- #
# 5. 未知标识不得进入 citation_ids / 不得提交核验
# --------------------------------------------------------------------------- #
def test_unknown_marker_is_excluded_from_extraction() -> None:
    claims = extract_citing_claims("论断 [P99]。")

    # 抽取阶段保留原样（供调用方观察），但 —— 见下一个测试 —— 不会提交核验。
    assert claims[0]["citation_ids"] == ["P99"]


def test_unknown_marker_is_not_sent_to_model() -> None:
    body = "论断 [P99] 与真实引用 [P1]。"
    sentence = "论断 [P99] 与真实引用 [P1]"
    llm = CountingLLM(
        {"verdicts": [_verdict(0, "P1", "supports", claim=sentence, quote=P1_ABSTRACT)]}
    )

    report = verify_manuscript_claims(llm, body, _papers())

    assert llm.calls, "应因存在真实标识而调用一次模型"
    prompt = llm.calls[0][1]
    # 未知标识绝不能被当作「可核验的文献」提交：它不得出现在证据行的
    # `- P99 《标题》摘要: ...` 位置上（论断原文中保留 [P99] 是允许的）。
    assert "- P99 《" not in prompt
    assert "- P1 《" in prompt
    ids = [item.citation_id for item in report.claims[0].evidence]
    assert ids == ["P1"]


def test_sentence_with_only_unknown_markers_is_dropped() -> None:
    """一句中的标识全部不可解析时，该句不进入核验。"""
    body = "只有未知标识 [P99]。另一句 [P1]。"
    sentence = "另一句 [P1]"
    llm = CountingLLM(
        {"verdicts": [_verdict(0, "P1", "supports", claim=sentence, quote=P1_ABSTRACT)]}
    )

    report = verify_manuscript_claims(llm, body, _papers())

    assert len(report.claims) == 1
    assert "P99" not in llm.calls[0][1]
    assert report.claims[0].citation_ids == ["P1"]


def test_out_of_range_marker_is_dropped() -> None:
    body = "越界标识 [P7] 与合法 [P2]。"
    sentence = "越界标识 [P7] 与合法 [P2]"
    llm = CountingLLM(
        {"verdicts": [_verdict(0, "P2", "supports", claim=sentence, quote=P2_ABSTRACT)]}
    )

    report = verify_manuscript_claims(llm, body, _papers())

    assert len(report.claims) == 1
    assert report.claims[0].citation_ids == ["P2"]
    assert "- P7 《" not in llm.calls[0][1]
    assert "- P2 《" in llm.calls[0][1]


# --------------------------------------------------------------------------- #
# 6. 模型失败降级，绝不上抛
# --------------------------------------------------------------------------- #
def test_model_failure_degrades_without_raising() -> None:
    llm = CountingLLM(RuntimeError("upstream boom"))

    report = verify_manuscript_claims(llm, "论断 [P1]。", _papers())

    assert isinstance(report, ClaimVerificationReport)
    assert report.claims[0].evidence[0].verdict == "unclear"
    assert any("模型调用失败" in warning for warning in report.warnings)
    assert len(llm.calls) == 1


def test_empty_body_returns_empty_report_and_no_calls() -> None:
    llm = CountingLLM({"verdicts": []})

    report = verify_manuscript_claims(llm, "", _papers())

    assert report.claims == []
    assert llm.calls == []


# --------------------------------------------------------------------------- #
# 7. 渲染
# --------------------------------------------------------------------------- #
def test_render_markdown_includes_sentence_verdict_and_warnings() -> None:
    body = "城市热岛会提高死亡率 [P1]。"
    sentence = "城市热岛会提高死亡率 [P1]"
    llm = CountingLLM(
        {"verdicts": [_verdict(0, "P1", "supports", claim=sentence, quote=P1_ABSTRACT)]}
    )

    report = verify_manuscript_claims(llm, body, _papers())
    markdown = render_manuscript_claim_markdown(report)

    assert "手稿正文引用语义核验报告" in markdown
    assert sentence in markdown
    assert "P1" in markdown
    assert "supports" in markdown
    assert markdown.endswith("\n")


def test_render_markdown_for_empty_report_mentions_zero() -> None:
    llm = CountingLLM({"verdicts": []})
    report = verify_manuscript_claims(llm, "无引用正文。", _papers())

    markdown = render_manuscript_claim_markdown(report)

    assert "带引用标识的句子总数: 0" in markdown
    assert "## 作者注意事项" in markdown
    assert "## 系统警告" in markdown


def test_render_markdown_is_deterministic() -> None:
    body = "论断甲 [P1]。论断乙 [P2]。"
    llm = CountingLLM({"verdicts": []})
    report = verify_manuscript_claims(llm, body, _papers())

    assert render_manuscript_claim_markdown(report) == render_manuscript_claim_markdown(
        report
    )


# --------------------------------------------------------------------------- #
# 8. 序列化（供管线落盘）
# --------------------------------------------------------------------------- #
def test_report_is_json_serializable_and_exposes_integration_keys() -> None:
    body = "论断甲 [P1]。"
    sentence = "论断甲 [P1]"
    llm = CountingLLM(
        {"verdicts": [_verdict(0, "P1", "supports", claim=sentence, quote=P1_ABSTRACT)]}
    )

    report = verify_manuscript_claims(llm, body, _papers())
    dumped = json.dumps(report.to_dict(), ensure_ascii=False)

    assert sentence in dumped
    payload = report.to_dict()
    for key in ("passed", "claim_count", "unsupported_count", "claims", "warnings"):
        assert key in payload


# --------------------------------------------------------------------------- #
# 9. 不修改入参
# --------------------------------------------------------------------------- #
def test_input_papers_sequence_is_not_mutated() -> None:
    papers = _papers()
    snapshot = [paper.to_dict() for paper in papers]
    llm = CountingLLM({"verdicts": []})

    verify_manuscript_claims(llm, "无引用正文。", papers)

    assert [paper.to_dict() for paper in papers] == snapshot


@pytest.mark.parametrize(
    "body",
    [
        "第一句 [P1]。第二句 [P2]！",
        "English [P1]. Another [P2]?",
        "混排 mixed 标点 [P3].",
    ],
)
def test_extract_never_returns_empty_claim_strings(body: str) -> None:
    for item in extract_citing_claims(body):
        assert item["claim"].strip() == item["claim"]
        assert item["claim"]
        assert item["citation_ids"]
