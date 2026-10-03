"""Tests for the evidence-bound claim verifier.

Every test drives :func:`verify_claims` with a fake LLM object so the suite
never touches the network.
"""

from __future__ import annotations

import json

from core.claim_verifier import (
    VERDICTS,
    ClaimVerificationReport,
    verify_claims,
)
from core.research_models import PaperRecord

P1_ABSTRACT = "Urban heat islands raise cardiovascular mortality in elderly residents."
P2_ABSTRACT = "Green roofs reduce indoor temperatures during summer heat waves."


class FakeLLM:
    """Deterministic stand-in for an LLM client (no network)."""

    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover
        raise AssertionError("complete() must not be used by the verifier")

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response  # type: ignore[return-value]


def _papers() -> list[PaperRecord]:
    return [
        PaperRecord(title="Urban heat and mortality", abstract=P1_ABSTRACT),
        PaperRecord(title="Green roofs", abstract=P2_ABSTRACT),
    ]


def _verdict(
    claim_index: int,
    citation_id: str,
    verdict: str,
    *,
    quote: str = "quoted evidence",
    rationale: str = "looks right",
) -> dict:
    return {
        "claim_index": claim_index,
        "citation_id": citation_id,
        "verdict": verdict,
        "quote": quote,
        "rationale": rationale,
    }


# --------------------------------------------------------------------------- #
# 1. 全部支撑
# --------------------------------------------------------------------------- #
def test_all_supported_passes_without_warnings() -> None:
    llm = FakeLLM({"verdicts": [_verdict(0, "P1", "supports", quote=P1_ABSTRACT)]})

    report = verify_claims(
        llm,
        claims=[{"claim": "Heat raises mortality.", "citation_ids": ["P1"]}],
        papers=_papers(),
    )

    assert report.passed is True
    assert report.warnings == []
    assert report.claims[0].overall == "supports"


# --------------------------------------------------------------------------- #
# 2. 一条 unsupported
# --------------------------------------------------------------------------- #
def test_unsupported_claim_is_surfaced_and_fails_report() -> None:
    llm = FakeLLM({"verdicts": [_verdict(0, "P1", "unsupported", quote=P2_ABSTRACT)]})

    report = verify_claims(
        llm,
        claims=[{"claim": "Green roofs kill people.", "citation_ids": ["P1"]}],
        papers=_papers(),
    )

    assert report.passed is False
    assert len(report.unsupported_claims) == 1
    assert any("unsupported" in warning for warning in report.warnings)


# --------------------------------------------------------------------------- #
# 3. 证据约束：空引用 → 强制 unclear
# --------------------------------------------------------------------------- #
def test_whitespace_quote_forces_unclear() -> None:
    llm = FakeLLM(
        {"verdicts": [_verdict(0, "P1", "supports", quote="   ", rationale="模型声称支持")]}
    )

    report = verify_claims(
        llm,
        claims=[{"claim": "Heat raises mortality.", "citation_ids": ["P1"]}],
        papers=_papers(),
    )

    evidence = report.claims[0].evidence[0]
    assert evidence.verdict == "unclear"
    assert "未提供原文引用" in evidence.rationale
    assert "模型声称支持" in evidence.rationale
    assert report.claims[0].overall == "unclear"


# --------------------------------------------------------------------------- #
# 4. 防编造：模型引用未引用的标识 → 丢弃
# --------------------------------------------------------------------------- #
def test_verdict_for_uncited_id_is_dropped() -> None:
    llm = FakeLLM(
        {
            "verdicts": [
                _verdict(0, "P1", "supports", quote=P1_ABSTRACT),
                _verdict(0, "P7", "supports", quote="fabricated"),
            ]
        }
    )

    report = verify_claims(
        llm,
        claims=[{"claim": "Heat raises mortality.", "citation_ids": ["P1"]}],
        papers=_papers(),
    )

    ids = [item.citation_id for item in report.claims[0].evidence]
    assert ids == ["P1"]
    assert all(item.citation_id != "P7" for item in report.claims[0].evidence)


# --------------------------------------------------------------------------- #
# 5. 覆盖保证：模型漏答 → 补齐 unclear
# --------------------------------------------------------------------------- #
def test_missing_pair_is_filled_with_unclear() -> None:
    llm = FakeLLM({"verdicts": [_verdict(0, "P1", "supports", quote=P1_ABSTRACT)]})

    report = verify_claims(
        llm,
        claims=[{"claim": "Two sources support this.", "citation_ids": ["P1", "P2"]}],
        papers=_papers(),
    )

    by_id = {item.citation_id: item for item in report.claims[0].evidence}
    assert set(by_id) == {"P1", "P2"}
    assert by_id["P2"].verdict == "unclear"
    assert "模型未对该文献给出判定" in by_id["P2"].rationale


# --------------------------------------------------------------------------- #
# 6. 无引用论断
# --------------------------------------------------------------------------- #
def test_claim_without_citations_gets_note_and_warning() -> None:
    llm = FakeLLM({"verdicts": []})

    report = verify_claims(
        llm,
        claims=[{"claim": "An unsupported assertion.", "citation_ids": []}],
        papers=_papers(),
    )

    claim = report.claims[0]
    assert claim.note == "该论断未引用任何文献"
    assert claim in report.claims_without_evidence
    assert report.passed is False
    assert any("未引用任何文献" in warning for warning in report.warnings)
    # 没有可核验的配对 → 不应调用模型。
    assert llm.calls == []


# --------------------------------------------------------------------------- #
# 7. 不存在的引用标识
# --------------------------------------------------------------------------- #
def test_unknown_citation_id_is_unclear_without_asking_model() -> None:
    llm = FakeLLM({"verdicts": [_verdict(0, "P1", "supports", quote=P1_ABSTRACT)]})

    report = verify_claims(
        llm,
        claims=[{"claim": "Heat raises mortality.", "citation_ids": ["P1", "P9"]}],
        papers=_papers(),
    )

    by_id = {item.citation_id: item for item in report.claims[0].evidence}
    assert by_id["P9"].verdict == "unclear"
    assert by_id["P9"].rationale == "引用了不存在的文献标识"
    assert any("P9" in warning for warning in report.warnings)

    # 模型从未被问及 P9。
    assert llm.calls, "expected one batch call for the resolvable paper"
    prompt = llm.calls[0][1]
    assert "P9" not in prompt


# --------------------------------------------------------------------------- #
# 8. 摘要缺失
# --------------------------------------------------------------------------- #
def test_missing_abstract_is_unclear_without_asking_model() -> None:
    papers = [
        PaperRecord(title="Has abstract", abstract=P1_ABSTRACT),
        PaperRecord(title="No abstract", abstract="   "),
    ]
    llm = FakeLLM({"verdicts": [_verdict(0, "P1", "supports", quote=P1_ABSTRACT)]})

    report = verify_claims(
        llm,
        claims=[{"claim": "Two sources support this.", "citation_ids": ["P1", "P2"]}],
        papers=papers,
    )

    by_id = {item.citation_id: item for item in report.claims[0].evidence}
    assert by_id["P2"].verdict == "unclear"
    assert by_id["P2"].rationale == "摘要缺失，无法核验"
    assert any("缺少摘要" in warning for warning in report.warnings)

    prompt = llm.calls[0][1]
    assert "No abstract" not in prompt


# --------------------------------------------------------------------------- #
# 9. 模型抛异常 → 降级，不逃逸
# --------------------------------------------------------------------------- #
def test_model_failure_degrades_to_unclear_without_raising() -> None:
    llm = FakeLLM(RuntimeError("upstream boom"))

    report = verify_claims(
        llm,
        claims=[{"claim": "Heat raises mortality.", "citation_ids": ["P1"]}],
        papers=_papers(),
    )

    assert isinstance(report, ClaimVerificationReport)
    assert report.claims[0].evidence[0].verdict == "unclear"
    assert any("模型调用失败" in warning for warning in report.warnings)


# --------------------------------------------------------------------------- #
# 10. 空论断 → 不调用模型
# --------------------------------------------------------------------------- #
def test_empty_claims_returns_empty_report_and_never_calls_model() -> None:
    llm = FakeLLM({"verdicts": []})

    report = verify_claims(llm, claims=[], papers=_papers())

    assert report.claims == []
    assert llm.calls == []


# --------------------------------------------------------------------------- #
# 11. overall 最坏判定
# --------------------------------------------------------------------------- #
def test_overall_is_worst_verdict() -> None:
    llm = FakeLLM(
        {
            "verdicts": [
                _verdict(0, "P1", "supports", quote=P1_ABSTRACT),
                _verdict(0, "P2", "unsupported", quote=P2_ABSTRACT),
            ]
        }
    )

    report = verify_claims(
        llm,
        claims=[{"claim": "Mixed evidence.", "citation_ids": ["P1", "P2"]}],
        papers=_papers(),
    )

    assert report.claims[0].overall == "unsupported"


def test_overall_without_evidence_is_unclear() -> None:
    llm = FakeLLM({"verdicts": []})

    report = verify_claims(
        llm,
        claims=[{"claim": "No evidence here.", "citation_ids": []}],
        papers=_papers(),
    )

    assert report.claims[0].overall == "unclear"


# --------------------------------------------------------------------------- #
# 12. 确定性
# --------------------------------------------------------------------------- #
def test_identical_inputs_and_replies_are_deterministic() -> None:
    def run() -> dict:
        llm = FakeLLM(
            {
                "verdicts": [
                    _verdict(0, "P1", "partially_supports", quote=P1_ABSTRACT),
                    _verdict(1, "P1", "supports", quote=P1_ABSTRACT),
                ]
            }
        )
        return verify_claims(
            llm,
            claims=[
                {"claim": "First claim.", "citation_ids": ["P1"]},
                {"claim": "Second claim.", "citation_ids": ["P1"]},
            ],
            papers=_papers(),
        ).to_dict()

    assert run() == run()


# --------------------------------------------------------------------------- #
# 13. 序列化与 Markdown
# --------------------------------------------------------------------------- #
def test_report_is_serializable_and_renders_markdown() -> None:
    llm = FakeLLM({"verdicts": [_verdict(0, "P1", "supports", quote=P1_ABSTRACT)]})

    report = verify_claims(
        llm,
        claims=[{"claim": "Heat raises mortality.", "citation_ids": ["P1"]}],
        papers=_papers(),
    )

    dumped = json.dumps(report.to_dict(), ensure_ascii=False)
    assert "Heat raises mortality." in dumped

    markdown = report.to_markdown()
    assert "Heat raises mortality." in markdown
    assert "建议" in markdown
    assert "结论" in markdown


# --------------------------------------------------------------------------- #
# 契约健全性
# --------------------------------------------------------------------------- #
def test_verdicts_vocabulary_is_exposed() -> None:
    assert VERDICTS == ("supports", "partially_supports", "unsupported", "unclear")


def test_to_dict_exposes_keys_the_peer_reviewer_reads() -> None:
    # core/peer_reviewer.py injects deterministic concerns by reading these
    # counts off the dict, so missing keys would silently drop a cross-check.
    llm = FakeLLM({"verdicts": [_verdict(0, "P1", "unsupported", quote=P2_ABSTRACT)]})

    report = verify_claims(
        llm,
        claims=[
            {"claim": "Unsupported.", "citation_ids": ["P1"]},
            {"claim": "No citations.", "citation_ids": []},
        ],
        papers=_papers(),
    )

    payload = report.to_dict()
    for key in (
        "passed",
        "claim_count",
        "unsupported_count",
        "claims_without_evidence_count",
        "claims",
        "warnings",
        "author_checks",
    ):
        assert key in payload, f"missing integration key: {key}"

    assert payload["unsupported_count"] == len(report.unsupported_claims) == 1
    assert (
        payload["claims_without_evidence_count"]
        == len(report.claims_without_evidence)
        == 1
    )

    claim_payload = payload["claims"][0]
    assert set(claim_payload) == {
        "index",
        "claim",
        "citation_ids",
        "overall",
        "note",
        "evidence",
    }
    assert set(claim_payload["evidence"][0]) == {
        "citation_id",
        "paper_title",
        "verdict",
        "quote",
        "rationale",
    }

    markdown = report.to_markdown()
    assert markdown.index("论断总数") < markdown.index("## 论断 1")


def test_author_checks_state_advisory_and_abstract_only() -> None:
    llm = FakeLLM({"verdicts": [_verdict(0, "P1", "supports", quote=P1_ABSTRACT)]})

    report = verify_claims(
        llm,
        claims=[{"claim": "Heat raises mortality.", "citation_ids": ["P1"]}],
        papers=_papers(),
    )

    joined = "".join(report.author_checks)
    assert "建议" in joined
    assert "摘要" in joined
    assert report.author_checks
