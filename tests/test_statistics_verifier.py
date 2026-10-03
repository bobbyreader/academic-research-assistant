from __future__ import annotations

from core.statistics_verifier import (
    render_statistics_verification_markdown,
    verify_statistics,
)


def test_exact_p_value_matches_computed_statistic() -> None:
    report = verify_statistics("The effect was reliable (p = 0.031).", [0.031])

    assert report.passed is True
    assert report.claims[0].matched is True


def test_reported_p_value_absent_from_analysis_is_flagged() -> None:
    report = verify_statistics("A strong effect emerged (p = 0.001).", [0.412])

    assert report.passed is False
    assert report.unmatched[0].value == 0.001
    assert "未产生" in report.unmatched[0].note


def test_significance_claim_without_any_significant_test_is_flagged() -> None:
    report = verify_statistics("Groups differed (p < 0.05).", [0.42, 0.77])

    assert report.passed is False
    assert "阈值" in report.unmatched[0].note


def test_significance_claim_is_honest_when_a_test_reaches_threshold() -> None:
    report = verify_statistics("Groups differed (p < 0.05).", [0.01, 0.42])

    assert report.passed is True


def test_tolerance_accepts_rounding_differences() -> None:
    report = verify_statistics("Reported as p = 0.03.", [0.031])

    assert report.passed is True


def test_no_claims_is_a_pass() -> None:
    report = verify_statistics("No statistics were reported here.", [0.2])

    assert report.passed is True
    assert report.claims == []


def test_duplicate_claims_are_reported_once() -> None:
    report = verify_statistics("First p < 0.05. Then again p < 0.05.", [0.9])

    assert len(report.claims) == 1


def test_unrelated_text_is_not_parsed_as_a_p_value() -> None:
    report = verify_statistics("pH was 7.4 and Figure 1 shows the trend.", [0.5])

    assert report.claims == []
    assert report.passed is True


def test_variants_of_p_value_notation_are_recognised() -> None:
    report = verify_statistics("p-value = 0.02; P <= 0.01; p=0.5", [0.02, 0.005, 0.5])

    assert len(report.claims) == 3
    assert report.passed is True


def test_report_is_serializable_and_renders_markdown() -> None:
    report = verify_statistics("Effect was present (p = 0.001).", [0.4])

    payload = report.to_dict()
    assert payload["passed"] is False
    assert payload["unmatched_count"] == 1

    markdown = render_statistics_verification_markdown(report)
    assert "统计陈述追溯报告" in markdown
    assert "p = 0.001" in markdown


def test_warnings_and_author_checks_are_empty_when_everything_traces() -> None:
    report = verify_statistics("p = 0.02", [0.02])

    assert report.warnings() == []
    assert report.author_checks() == []


def test_small_p_value_is_not_swallowed_by_a_fixed_tolerance() -> None:
    # An absolute tolerance of 0.005 would wrongly match 0.0001 against 0.001.
    report = verify_statistics("A decisive effect (p = 0.0001).", [0.001053])

    assert report.passed is False
    assert report.unmatched[0].value == 0.0001


def test_matching_is_precision_aware() -> None:
    assert verify_statistics("p = 0.03", [0.031]).passed is True
    assert verify_statistics("p = 0.031", [0.031]).passed is True
    assert verify_statistics("p = 0.03", [0.039]).passed is False
