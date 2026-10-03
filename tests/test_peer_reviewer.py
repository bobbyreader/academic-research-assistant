"""Tests for the evidence-bound pre-submission peer reviewer.

Every test drives :func:`review_manuscript` with a fake LLM object so the suite
never touches the network.
"""

from __future__ import annotations

import json

import pytest

from core.peer_reviewer import (
    PeerReviewBundle,
    PeerReviewError,
    _has_tests,
    review_manuscript,
)

MANUSCRIPT = (
    "We show that sleep improves memory. Attention correlated with recall "
    "(p = 0.001), and prior work supports this claim [P7]."
)


def _payload(
    *,
    summary: str = "A clear but under-powered study.",
    strengths: list[str] | None = None,
    concerns: list[dict] | None = None,
    recommendation: str = "minor_revision",
    score: int = 72,
) -> dict:
    return {
        "summary": summary,
        "strengths": strengths if strengths is not None else ["Novel topic."],
        "concerns": concerns
        if concerns is not None
        else [
            {
                "category": "methodology",
                "severity": "major",
                "statement": "Sample size is not justified.",
                "evidence": "We show that sleep improves memory.",
            }
        ],
        "recommendation": recommendation,
        "score": score,
    }


class FakeLLM:
    """Deterministic stand-in for an LLM client (no network)."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover
        raise AssertionError("complete() must not be used by the reviewer")

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append((system_prompt, user_prompt))
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _review(llm: FakeLLM, **kwargs) -> PeerReviewBundle:
    defaults = {
        "topic": "Sleep and memory",
        "manuscript_body": MANUSCRIPT,
        "reviewer_count": 3,
    }
    defaults.update(kwargs)
    return review_manuscript(llm, **defaults)


def test_happy_path_three_reviewers_is_serializable_and_renders_markdown() -> None:
    llm = FakeLLM([_payload(), _payload(), _payload()])

    bundle = _review(llm)

    assert len(bundle.reports) == 3
    assert [r.reviewer_role for r in bundle.reports] == [
        "methodology",
        "statistics",
        "novelty",
    ]

    # to_dict() must be JSON-serializable.
    dumped = json.dumps(bundle.to_dict(), ensure_ascii=False)
    assert "methodology" in dumped

    markdown = bundle.to_markdown()
    for role in ("methodology", "statistics", "novelty"):
        assert role in markdown


def test_concern_with_empty_evidence_is_dropped() -> None:
    payload = _payload(
        concerns=[
            {
                "category": "clarity",
                "severity": "minor",
                "statement": "Vague wording somewhere.",
                "evidence": "   ",  # whitespace-only => must be dropped
            },
            {
                "category": "methodology",
                "severity": "major",
                "statement": "No sample-size justification.",
                "evidence": "We show that sleep improves memory.",
            },
        ]
    )
    llm = FakeLLM([payload, payload, payload])

    bundle = _review(llm)

    for report in bundle.reports:
        assert all(concern.evidence.strip() for concern in report.concerns)
        assert all("Vague wording" not in c.statement for c in report.concerns)


def test_unknown_markers_injected_as_major_claim_traceability_concern() -> None:
    llm = FakeLLM([_payload(), _payload(), _payload()])

    bundle = _review(llm, citation_verification={"unknown_markers": ["P7", "P9"]})

    injected = [
        c
        for report in bundle.reports
        for c in report.concerns
        if c.category == "claim_traceability"
    ]
    assert injected, "expected an injected claim_traceability concern"
    assert all(c.severity == "major" for c in injected)
    assert any("P7" in c.evidence for c in injected)


def test_unmatched_statistics_injected_as_major_statistics_concern() -> None:
    llm = FakeLLM([_payload(), _payload(), _payload()])

    bundle = _review(llm, statistics_verification={"unmatched_count": 2})

    injected = [
        c
        for report in bundle.reports
        for c in report.concerns
        if c.category == "statistics" and c.severity == "major"
    ]
    assert injected
    assert all("unmatched_count = 2" in c.evidence for c in injected)


def test_malformed_json_for_one_reviewer_is_skipped_with_warning() -> None:
    exc = ValueError("模型返回的内容不是有效 JSON")
    llm = FakeLLM([exc, _payload(), _payload()])

    bundle = _review(llm)

    assert len(bundle.reports) == 2
    assert len(bundle.warnings) == 1
    assert "methodology" in bundle.warnings[0]


def test_all_reviewers_malformed_raises() -> None:
    exc = ValueError("模型返回的内容不是有效 JSON")
    llm = FakeLLM([exc, exc, exc])

    with pytest.raises(PeerReviewError):
        _review(llm)


@pytest.mark.parametrize("bad_score", [150, -20, "abc", None])
def test_out_of_range_score_is_clamped(bad_score) -> None:
    llm = FakeLLM(
        [
            _payload(score=bad_score),
            _payload(score=bad_score),
            _payload(score=bad_score),
        ]
    )

    bundle = _review(llm)

    assert all(0 <= report.score <= 100 for report in bundle.reports)


def test_decision_rule_is_deterministic_across_calls() -> None:
    def build() -> FakeLLM:
        return FakeLLM(
            [
                _payload(recommendation="accept"),
                _payload(recommendation="reject"),
                _payload(recommendation="minor_revision"),
            ]
        )

    first = _review(build()).decision
    second = _review(build()).decision

    assert first == second
    # reject is the worst-case recommendation => worst-case wins.
    assert first == "reject"


def test_reviewer_count_one_yields_exactly_one_report() -> None:
    llm = FakeLLM([_payload()])

    bundle = _review(llm, reviewer_count=1)

    assert len(bundle.reports) == 1
    assert bundle.reports[0].reviewer_role == "methodology"


def test_synthesis_counts_distinct_concerns_not_per_reviewer_copies() -> None:
    # 3 reviewers, each with the SAME single model concern, plus identical
    # injected traceability concerns. Distinct problems must not be tripled.
    llm = FakeLLM([_payload(), _payload(), _payload()])

    bundle = _review(
        llm,
        citation_verification={"unknown_markers": ["P7"], "doi_unresolved": 1},
        statistics_verification={"unmatched_count": 2},
        statistics_report={"tests": []},
    )

    # Work out the expected DISTINCT count independently of the synthesis text.
    distinct = {
        (c.category, c.statement, c.evidence)
        for report in bundle.reports
        for c in report.concerns
    }
    per_reviewer_sum = sum(len(report.concerns) for report in bundle.reports)

    assert len(bundle.reports) == 3
    # Sanity: there really are duplicated concerns across reviewers.
    assert per_reviewer_sum > len(distinct)

    assert f"合计提出 {len(distinct)} 处独立问题" in bundle.synthesis
    assert f"{per_reviewer_sum} 处" not in bundle.synthesis


@pytest.mark.parametrize(
    ("tests_value", "expected"),
    [
        (0, False),
        (3, True),
        ([], False),
        (["t"], True),
        ("", False),
        ("non-empty", True),
    ],
)
def test_has_tests_handles_int_list_and_string(tests_value, expected) -> None:
    assert _has_tests({"tests": tests_value}) is expected
