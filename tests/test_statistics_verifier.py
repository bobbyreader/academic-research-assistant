from __future__ import annotations

from dataclasses import asdict

import pytest

from core.research_models import ArtifactDecodeError
from core.statistics_engine import StatTestResult
from core.statistics_verifier import (
    StatisticsClaim,
    StatisticsVerificationReport,
    _reported_decimals,
    render_statistics_verification_markdown,
    verify_statistics,
)


def _welch(p_value: float, *, n: int = 10, d: float = 0.3) -> StatTestResult:
    """A Welch t-test row carrying a Cohen's d effect size and a sample size."""
    return StatTestResult(
        test_name="Welch t-test",
        variables=["score"],
        groups=["a", "b"],
        n=n,
        statistic=1.0,
        p_value=p_value,
        p_value_adjusted=p_value,
        effect_size=d,
        effect_size_name="Cohen's d",
    )


def _pearson(p_value: float, *, n: int = 30, r: float = 0.5) -> StatTestResult:
    """A Pearson correlation row whose effect size is an ``r``."""
    return StatTestResult(
        test_name="Pearson correlation",
        variables=["x", "y"],
        groups=[],
        n=n,
        statistic=r,
        p_value=p_value,
        p_value_adjusted=p_value,
        effect_size=r,
        effect_size_name="Pearson r",
    )


# --------------------------------------------------------------------------- #
# p 值：行为未回归（原断言一律不放宽）
# --------------------------------------------------------------------------- #
def test_exact_p_value_matches_computed_statistic() -> None:
    report = verify_statistics("The effect was reliable (p = 0.031).", [_welch(0.031)])

    assert report.passed is True
    assert report.claims[0].matched is True


def test_reported_p_value_absent_from_analysis_is_flagged() -> None:
    report = verify_statistics("A strong effect emerged (p = 0.001).", [_welch(0.412)])

    assert report.passed is False
    assert report.unmatched[0].value == 0.001
    assert "未产生" in report.unmatched[0].note


def test_significance_claim_without_any_significant_test_is_flagged() -> None:
    report = verify_statistics(
        "Groups differed (p < 0.05).", [_welch(0.42), _welch(0.77)]
    )

    assert report.passed is False
    assert "阈值" in report.unmatched[0].note


def test_significance_claim_is_honest_when_a_test_reaches_threshold() -> None:
    report = verify_statistics("Groups differed (p < 0.05).", [_welch(0.01), _welch(0.42)])

    assert report.passed is True


def test_tolerance_accepts_rounding_differences() -> None:
    report = verify_statistics("Reported as p = 0.03.", [_welch(0.031)])

    assert report.passed is True


def test_no_claims_is_a_pass() -> None:
    report = verify_statistics("No statistics were reported here.", [_welch(0.2)])

    assert report.passed is True
    assert report.claims == []


def test_duplicate_claims_are_reported_once() -> None:
    report = verify_statistics("First p < 0.05. Then again p < 0.05.", [_welch(0.9)])

    assert len(report.claims) == 1


def test_unrelated_text_is_not_parsed_as_a_p_value() -> None:
    report = verify_statistics("pH was 7.4 and Figure 1 shows the trend.", [_welch(0.5)])

    assert report.claims == []
    assert report.passed is True


def test_variants_of_p_value_notation_are_recognised() -> None:
    report = verify_statistics(
        "p-value = 0.02; P <= 0.01; p=0.5",
        [_welch(0.02), _welch(0.005), _welch(0.5)],
    )

    assert len(report.claims) == 3
    assert report.passed is True


def test_report_is_serializable_and_renders_markdown() -> None:
    report = verify_statistics("Effect was present (p = 0.001).", [_welch(0.4)])

    payload = report.to_dict()
    assert payload["passed"] is False
    assert payload["unmatched_count"] == 1

    markdown = render_statistics_verification_markdown(report)
    assert "统计陈述追溯报告" in markdown
    assert "p = 0.001" in markdown


def test_warnings_and_author_checks_are_empty_when_everything_traces() -> None:
    report = verify_statistics("p = 0.02", [_welch(0.02)])

    assert report.warnings() == []
    assert report.author_checks() == []


def test_small_p_value_is_not_swallowed_by_a_fixed_tolerance() -> None:
    # An absolute tolerance of 0.005 would wrongly match 0.0001 against 0.001.
    report = verify_statistics("A decisive effect (p = 0.0001).", [_welch(0.001053)])

    assert report.passed is False
    assert report.unmatched[0].value == 0.0001


def test_matching_is_precision_aware() -> None:
    assert verify_statistics("p = 0.03", [_welch(0.031)]).passed is True
    assert verify_statistics("p = 0.031", [_welch(0.031)]).passed is True
    assert verify_statistics("p = 0.03", [_welch(0.039)]).passed is False


# --------------------------------------------------------------------------- #
# 效应量：伪造必须被捕获（本次修复核心）
# --------------------------------------------------------------------------- #
def test_fabricated_cohens_d_is_caught() -> None:
    report = verify_statistics(
        "A large effect emerged, Cohen's d = 1.2.",
        [_welch(0.03, d=0.3)],
    )

    assert report.passed is False
    effect = [c for c in report.claims if c.kind == "effect_size"]
    assert len(effect) == 1
    assert effect[0].value == 1.2
    assert effect[0].matched is False


def test_reported_bare_d_matches_computed_effect_size() -> None:
    report = verify_statistics("The effect was moderate (d = 0.3).", [_welch(0.03, d=0.3)])

    assert report.passed is True
    assert report.claims[0].kind == "effect_size"


def test_effect_size_is_matched_at_the_reported_precision() -> None:
    assert verify_statistics("d = 0.4", [_welch(0.03, d=0.42)]).passed is True
    assert verify_statistics("d = 0.42", [_welch(0.03, d=0.42)]).passed is True


def test_effect_size_notation_variants_are_captured_raw() -> None:
    report = verify_statistics(
        "d=0.3 and Cohen's d = 0.3 and r = 0.5",
        [_welch(0.03, d=0.3), _pearson(0.01, r=0.5)],
    )

    raws = {claim.raw for claim in report.claims}
    assert "d=0.3" in raws
    assert "Cohen's d = 0.3" in raws
    assert "r = 0.5" in raws
    assert report.passed is True


def test_d_symbol_never_matches_a_correlation_effect_size() -> None:
    # The run only produced a Pearson r; a reported ``d`` must NOT be satisfied
    # by it, even when the numbers happen to agree.
    report = verify_statistics("The association was d = 0.5.", [_pearson(0.01, r=0.5)])

    effect = [c for c in report.claims if c.kind == "effect_size"]
    assert len(effect) == 1
    assert effect[0].matched is False
    assert report.passed is False


def test_r_symbol_never_matches_a_cohens_d_effect_size() -> None:
    report = verify_statistics("The association was r = 0.5.", [_welch(0.03, d=0.5)])

    effect = [c for c in report.claims if c.kind == "effect_size"]
    assert len(effect) == 1
    assert effect[0].matched is False


def test_eta_squared_has_no_symbol_so_no_effect_claim_is_extracted() -> None:
    eta = StatTestResult(
        test_name="One-way ANOVA",
        variables=["score"],
        groups=["a", "b", "c"],
        n=30,
        statistic=4.0,
        p_value=0.01,
        p_value_adjusted=0.01,
        effect_size=0.14,
        effect_size_name="eta squared",
    )
    report = verify_statistics("There were no effect-size symbols here.", [eta])

    assert [c for c in report.claims if c.kind == "effect_size"] == []


# --------------------------------------------------------------------------- #
# 覆盖率：真实手稿的四种形态必须被识别（本次修复核心）
# --------------------------------------------------------------------------- #
def _eta(p_value: float, *, n: int = 30, eta: float = 0.14) -> StatTestResult:
    return StatTestResult(
        test_name="One-way ANOVA",
        variables=["score"],
        groups=["a", "b", "c"],
        n=n,
        statistic=4.0,
        p_value=p_value,
        p_value_adjusted=p_value,
        effect_size=eta,
        effect_size_name="eta squared",
    )


def test_scientific_notation_p_value_is_recognised_and_matched() -> None:
    report = verify_statistics("The effect was decisive (p = 3.57e-16).", [_welch(3.57e-16)])

    assert report.claim_count == 1
    assert report.claims[0].matched is True
    assert report.passed is True


def test_scientific_notation_p_value_keeps_full_mantissa_precision() -> None:
    # ``3.5745385179030436e-16`` must match at the *reported* magnitude, not at a
    # bogus 16 + 16 decimals that can never round-trip.
    report = verify_statistics(
        "p = 3.5745385179030436e-16", [_welch(3.5745385179030436e-16)]
    )

    assert report.passed is True


def test_scientific_notation_at_reported_precision_still_rejects_different_value() -> None:
    report = verify_statistics("p = 3.57e-16", [_welch(3.57e-12)])

    assert report.claim_count == 1
    assert report.passed is False


def test_reported_decimals_handles_scientific_and_plain_notation() -> None:
    assert _reported_decimals("0.03") == 2
    assert _reported_decimals("0.006677181726293035") == 18
    assert _reported_decimals("30") == 0
    # 3.5745385179030436e-16 -> last digit at 10^-(16+16) = 10^-32.
    assert _reported_decimals("3.5745385179030436e-16") == 32
    assert _reported_decimals("1e-5") == 5


def test_negative_correlation_is_recognised() -> None:
    report = verify_statistics("Spearman r=-0.9021134593993325", [_pearson(0.01, r=-0.9021134593993325)])

    assert report.claim_count == 1
    assert report.claims[0].kind == "effect_size"
    assert report.claims[0].value == -0.9021134593993325
    assert report.passed is True


def test_eta_squared_statement_is_now_recognised_and_matched() -> None:
    report = verify_statistics(
        "eta squared=0.9371719993842926", [_eta(0.01, eta=0.9371719993842926)]
    )

    assert report.claim_count == 1
    assert report.claims[0].kind == "effect_size"
    assert report.claims[0].matched is True


def test_eta_squared_statement_not_satisfied_by_cohens_d() -> None:
    # Family isolation: an eta squared claim must never be satisfied by a d.
    report = verify_statistics("eta squared = 0.5", [_welch(0.03, d=0.5)])

    claims = [c for c in report.claims if c.kind == "effect_size"]
    assert len(claims) == 1
    assert claims[0].matched is False
    assert report.passed is False


def test_cjk_adjacent_p_value_is_recognised() -> None:
    # ``Holm校正p=…``: the CJK ``正`` binds to ``p`` so ``\bp`` never matched.
    report = verify_statistics("Holm校正p=0.006677181726293035", [_welch(0.006677181726293035)])

    assert report.claim_count == 1
    assert report.claims[0].kind == "p_value"
    assert report.passed is True


def test_cjk_adjacent_sample_size_is_recognised() -> None:
    report = verify_statistics("单因素方差分析报告n=30", [_welch(0.03, n=30)])

    assert report.claim_count == 1
    assert report.claims[0].kind == "sample_size"
    assert report.passed is True


def test_ascii_prefixed_p_is_still_not_matched() -> None:
    # The lookbehind must still reject ``ap = 0.5`` / ``pH``-like false positives.
    report = verify_statistics("The gap = 0.5 is irrelevant.", [_welch(0.03)])

    assert report.claim_count == 0
    assert report.passed is True


def test_holm_adjusted_p_value_is_traceable() -> None:
    # A manuscript legitimately quotes the multi-comparison-corrected p-value.
    test = StatTestResult(
        test_name="One-way ANOVA",
        variables=["score"],
        groups=["a", "b", "c"],
        n=30,
        statistic=4.0,
        p_value=5.957564196505072e-17,
        p_value_adjusted=3.5745385179030436e-16,
        effect_size=0.94,
        effect_size_name="eta squared",
    )
    report = verify_statistics("p = 3.5745385179030436e-16", [test])

    assert report.passed is True


# --------------------------------------------------------------------------- #
# 覆盖率可见性：识别到 0 条必须浮出 warning
# --------------------------------------------------------------------------- #
def test_zero_claims_emits_a_coverage_warning() -> None:
    report = verify_statistics("No statistics were reported here.", [_welch(0.2)])

    assert report.claim_count == 0
    assert report.coverage_warning is not None
    assert any("识别到任何统计陈述" in w for w in report.warnings())


def test_nonzero_claims_have_no_coverage_warning() -> None:
    report = verify_statistics("p = 0.02", [_welch(0.02)])

    assert report.coverage_warning is None
    assert report.warnings() == []


def test_claim_count_is_visible_in_dict_and_markdown() -> None:
    report = verify_statistics("p = 0.03, d = 0.3, n = 10", [_welch(0.03, d=0.3, n=10)])

    assert report.to_dict()["claim_count"] == 3
    markdown = render_statistics_verification_markdown(report)
    assert "正文识别到的统计陈述条数（覆盖率）: 3 处" in markdown


def test_ran_false_never_passes_and_is_visible() -> None:
    report = StatisticsVerificationReport(
        computed_p_values=[0.03],
        ran=False,
        not_run_reason="上游未提供统计检验结果",
    )

    assert report.passed is False
    assert any("未执行" in w for w in report.warnings())
    markdown = render_statistics_verification_markdown(report)
    assert "是否执行: 否" in markdown


def test_legacy_payload_defaults_ran_true() -> None:
    legacy = {
        "passed": True,
        "computed_p_values": [0.03],
        "claims": [],
    }
    report = StatisticsVerificationReport.from_dict(legacy)

    assert report.ran is True
    assert report.not_run_reason == ""


# --------------------------------------------------------------------------- #
# 样本量
# --------------------------------------------------------------------------- #
def test_fabricated_sample_size_is_caught() -> None:
    report = verify_statistics(
        "A total of n = 500 participants were analysed.",
        [_welch(0.03, n=10)],
    )

    assert report.passed is False
    size = [c for c in report.claims if c.kind == "sample_size"]
    assert len(size) == 1
    assert size[0].value == 500.0
    assert size[0].matched is False


def test_uppercase_n_is_recognised_and_matches() -> None:
    report = verify_statistics("The full sample was N = 10.", [_welch(0.03, n=10)])

    size = [c for c in report.claims if c.kind == "sample_size"]
    assert len(size) == 1
    assert size[0].matched is True


def test_sample_size_notation_variants_are_recognised() -> None:
    # ``n=10`` and ``N = 10`` describe the same sample, so they are deduplicated
    # into a single claim (same semantics as the existing p-value dedup).
    report = verify_statistics("n=10; N = 10", [_welch(0.03, n=10)])

    sizes = [c for c in report.claims if c.kind == "sample_size"]
    assert len(sizes) == 1
    assert sizes[0].value == 10.0
    assert sizes[0].matched is True


def test_uppercase_and_lowercase_notations_are_both_parsed() -> None:
    report = verify_statistics("N = 12 but n = 10", [_welch(0.03, n=10)])

    sizes = [c for c in report.claims if c.kind == "sample_size"]
    assert len(sizes) == 2
    assert [claim.value for claim in sizes] == [12.0, 10.0]
    assert [claim.matched for claim in sizes] == [False, True]


def test_sample_size_must_equal_a_computed_n_exactly() -> None:
    report = verify_statistics("Only n = 11 were analysed.", [_welch(0.03, n=10)])

    assert report.passed is False


# --------------------------------------------------------------------------- #
# 宁可漏报也不要噪声：保守提取
# --------------------------------------------------------------------------- #
def test_r_squared_notation_is_not_treated_as_an_effect_size() -> None:
    # ``R² = 0.85`` is overwhelmingly a coefficient of determination; we do not
    # extract it, so it cannot generate a false positive.
    report = verify_statistics("The model explained variance (R² = 0.85).", [_welch(0.03)])

    assert [c for c in report.claims if c.kind == "effect_size"] == []
    assert report.passed is True


def test_count_of_literature_items_is_not_treated_as_a_sample_size() -> None:
    report = verify_statistics("We pooled n = 3 studies.", [_welch(0.03, n=10)])

    assert [c for c in report.claims if c.kind == "sample_size"] == []
    assert report.passed is True


def test_bare_english_word_r_is_not_parsed_as_an_effect_size() -> None:
    report = verify_statistics(
        "The result revealed r nothing quantitative.", [_welch(0.03)]
    )

    assert [c for c in report.claims if c.kind == "effect_size"] == []


def test_literature_effect_size_is_reported_as_unmatched_not_blocking() -> None:
    # Known and accepted: a statistic quoted from the literature registers as
    # unmatched. It must be advisory only — never an exception.
    report = verify_statistics(
        "Smith reported d = 0.8 [P1].", [_welch(0.03, d=0.3)]
    )

    assert report.passed is False
    assert "人工核对" in report.warnings()[0]
    assert "d = 0.8" in report.author_checks()[0]


# --------------------------------------------------------------------------- #
# 无数据集 / 空 tests
# --------------------------------------------------------------------------- #
def test_empty_tests_flags_every_kind_without_raising() -> None:
    report = verify_statistics(
        "Cohen's d = 0.3, n = 10 and p = 0.03.", []
    )

    assert report.passed is False
    assert len(report.claims) == 3
    assert all(claim.matched is False for claim in report.claims)


def test_no_dataset_yields_empty_computed_collections() -> None:
    report = verify_statistics("No statistics here.", [])

    assert report.computed_p_values == []
    assert report.computed_effect_sizes == []
    assert report.computed_sample_sizes == []
    assert report.passed is True


def test_non_finite_computed_values_are_excluded() -> None:
    nan_test = StatTestResult(
        test_name="Welch t-test",
        variables=["x"],
        groups=["a", "b"],
        n=10,
        p_value=float("nan"),
        effect_size=float("nan"),
        effect_size_name="Cohen's d",
    )
    report = verify_statistics("p = 0.5", [nan_test])

    assert report.computed_p_values == []
    assert report.computed_effect_sizes == []
    assert report.computed_sample_sizes == [10]
    assert report.passed is False


# --------------------------------------------------------------------------- #
# 序列化：往返 + 旧格式兼容
# --------------------------------------------------------------------------- #
def test_to_dict_from_dict_roundtrip_is_lossless() -> None:
    report = verify_statistics(
        "Cohen's d = 1.2 (p = 0.03), n = 500.",
        [_welch(0.03, n=10, d=0.3)],
    )
    restored = StatisticsVerificationReport.from_dict(report.to_dict())

    assert restored == report
    assert restored.to_dict() == report.to_dict()


def test_to_dict_keeps_all_legacy_keys_and_adds_new_ones() -> None:
    report = verify_statistics("p = 0.03", [_welch(0.03)])
    payload = report.to_dict()

    for legacy in (
        "passed",
        "computed_p_values",
        "claim_count",
        "unmatched_count",
        "claims",
    ):
        assert legacy in payload
    assert "computed_effect_sizes" in payload
    assert "computed_sample_sizes" in payload
    assert payload["claims"][0]["kind"] == "p_value"


def test_legacy_payload_without_new_keys_is_accepted() -> None:
    legacy = {
        "passed": True,
        "computed_p_values": [0.03],
        "claim_count": 1,
        "unmatched_count": 0,
        "claims": [
            {
                "raw": "p = 0.03",
                "operator": "=",
                "value": 0.03,
                "matched": True,
                "note": "",
            }
        ],
    }
    report = StatisticsVerificationReport.from_dict(legacy)

    assert report.computed_effect_sizes == []
    assert report.computed_sample_sizes == []
    assert report.claims[0].kind == "p_value"
    assert report.passed is True


def test_legacy_claim_without_kind_defaults_to_p_value() -> None:
    claim = StatisticsClaim.from_dict(
        {"raw": "p = 0.03", "operator": "=", "value": 0.03, "matched": True, "note": ""}
    )

    assert claim.kind == "p_value"


def test_claim_kind_is_preserved_through_asdict_roundtrip() -> None:
    report = verify_statistics("Cohen's d = 0.3", [_welch(0.03, d=0.3)])
    claim = report.claims[0]
    assert claim.kind == "effect_size"
    assert StatisticsClaim.from_dict(asdict(claim)) == claim


@pytest.mark.parametrize(
    "payload",
    [
        [],  # not a mapping
        {"claims": "nope"},  # claims not a list
        {"claims": [123]},  # element not a dict
        {"computed_p_values": [0.1]},  # claims missing entirely
        {"claims": [], "computed_sample_sizes": [1.5]},  # sample size not int
        {"claims": [], "computed_effect_sizes": ["x"]},  # effect size not numeric
        {"claims": [], "computed_p_values": [True]},  # bool is not a p-value
    ],
)
def test_from_dict_rejects_malformed_payloads(payload: object) -> None:
    with pytest.raises(ArtifactDecodeError):
        StatisticsVerificationReport.from_dict(payload)


def test_from_dict_rejects_unknown_claim_kind() -> None:
    payload = {
        "computed_p_values": [],
        "claims": [
            {
                "raw": "x",
                "operator": "",
                "value": 1.0,
                "matched": True,
                "note": "",
                "kind": "odds_ratio",
            }
        ],
    }
    with pytest.raises(ArtifactDecodeError):
        StatisticsVerificationReport.from_dict(payload)
