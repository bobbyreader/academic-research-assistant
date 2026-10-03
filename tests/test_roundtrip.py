"""往返（round-trip）契约测试：`X.from_dict(x.to_dict()) == x`。

这批测试锁定"产物写得出、也读得回"这一地基。任何 `from_dict` 的实现都必须：

1. **无损往返** —— `from_dict(to_dict(x))` 与 `x` 相等（dataclass 相等），且再次
   `to_dict()` 也逐键相等（派生字段由属性重算，键集不得改变）；
2. **拒绝坏数据** —— 非 dict / 缺必需键 / 键类型错误 / 嵌套元素非法时抛
   `ValueError`（具体为 `ArtifactDecodeError`，它是 `ValueError` 的子类），
   绝不静默构造一个"半个对象"——那等同于伪造证据。

说明：`to_dict()` 的键集是被现有测试锁定的公共契约，因此这里也断言
`from_dict(...).to_dict() == x.to_dict()`，防止实现者为了好写而增删键。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from core.citation_verifier import CitationVerificationReport, DoiCheck
from core.claim_verifier import (
    ClaimEvidence,
    ClaimVerdict,
    ClaimVerificationReport,
)
from core.figure_builder import FigureBundle, FigureSpec
from core.peer_reviewer import (
    PeerReviewBundle,
    ReviewConcern,
    ReviewerReport,
)
from core.research_models import ArtifactDecodeError, PaperRecord
from core.statistics_engine import StatisticsReport, StatTestResult
from core.statistics_verifier import (
    StatisticsClaim,
    StatisticsVerificationReport,
)

# --------------------------------------------------------------------------- #
# 样例工厂
# --------------------------------------------------------------------------- #


def _paper() -> PaperRecord:
    return PaperRecord(
        title="Climate adaptation",
        authors=["A Author", "B Author"],
        year=2024,
        journal="Research Journal",
        doi="10.1234/climate",
        abstract="A finding about adaptation.",
        url="https://example.org/paper",
        source="crossref",
        citation_count=12,
        external_id="crossref:abc",
        raw_data={"score": 0.9, "tags": ["climate"]},
    )


def _stat_test() -> StatTestResult:
    return StatTestResult(
        test_name="Welch t-test",
        variables=["score"],
        groups=["A", "B"],
        n=10,
        statistic=2.5,
        p_value=0.03,
        p_value_adjusted=0.04,
        effect_size=0.8,
        effect_size_name="Cohen's d",
        ci_low=0.1,
        ci_high=0.9,
        assumptions=["independent observations"],
        warnings=["Shapiro-Wilk rejected normality"],
    )


def _statistics_report() -> StatisticsReport:
    return StatisticsReport(
        rows=10,
        numeric_columns=["score"],
        categorical_columns=["group"],
        group_column="group",
        alpha=0.05,
        correction_method="holm",
        tests=[_stat_test()],
        author_checks=["confirm design"],
        warnings=["small sample"],
    )


def _figure_spec(path: Path) -> FigureSpec:
    return FigureSpec(
        figure_id="Figure 1",
        filename="figure_1_distribution.png",
        title="score 的分布",
        caption="图件说明 n=10",
        figure_type="distribution",
        path=path,
    )


def _claim_verification_report() -> ClaimVerificationReport:
    evidence = ClaimEvidence(
        citation_id="P1",
        paper_title="Climate adaptation",
        verdict="supports",
        quote="A finding about adaptation.",
        rationale="摘要直接支撑该论断",
    )
    verdict = ClaimVerdict(
        index=0,
        claim="Adaptation reduces risk.",
        citation_ids=["P1"],
        evidence=[evidence],
        note="",
    )
    verdict_no_evidence = ClaimVerdict(
        index=1,
        claim="An uncited claim.",
        citation_ids=[],
        evidence=[],
        note="该论断未引用任何文献",
    )
    return ClaimVerificationReport(
        claims=[verdict, verdict_no_evidence],
        warnings=["1 条论断未引用任何文献"],
        author_checks=["请人工复核"],
    )


def _peer_review_bundle() -> PeerReviewBundle:
    concern = ReviewConcern(
        category="methodology",
        severity="major",
        statement="Sample size is not justified.",
        evidence="We show that sleep improves memory.",
    )
    minor = ReviewConcern(
        category="clarity",
        severity="minor",
        statement="Some sentences are long.",
        evidence="Sleep improves memory.",
    )
    report = ReviewerReport(
        reviewer_role="methodology",
        summary="A clear but under-powered study.",
        strengths=["Novel topic."],
        concerns=[concern, minor],
        recommendation="minor_revision",
        score=72,
    )
    return PeerReviewBundle(
        reports=[report],
        synthesis="1 处 major、1 处 minor。",
        decision="minor_revision",
        author_checks=["请人类作者复核"],
        warnings=["reviewer statistics 回复无法解析，已跳过"],
    )


def _statistics_verification() -> StatisticsVerificationReport:
    matched = StatisticsClaim(
        raw="p = 0.03",
        operator="=",
        value=0.03,
        matched=True,
        note="",
    )
    unmatched = StatisticsClaim(
        raw="p < 0.05",
        operator="<",
        value=0.05,
        matched=False,
        note="正文声称 p < 0.05，但没有任何检验达到该阈值",
    )
    return StatisticsVerificationReport(
        computed_p_values=[0.03, 0.4],
        claims=[matched, unmatched],
    )


def _citation_verification() -> CitationVerificationReport:
    return CitationVerificationReport(
        total_references=2,
        cited_markers=["P1"],
        unknown_markers=["P7"],
        unused_references=["P2"],
        references_without_doi=["P2"],
        doi_checks=[
            DoiCheck(citation_id="P1", doi="10.1/alpha", resolved=False, error="unresolved"),
            DoiCheck(citation_id="P2", doi="10.2/beta", resolved=True, error=None),
        ],
    )


ROUNDTRIP_SAMPLES: list[Any] = [
    _paper(),
    _stat_test(),
    _statistics_report(),
    _figure_spec(Path("/nonexistent/tmp/figure_1_distribution.png")),
    FigureBundle(figures=[_figure_spec(Path("/gone/f.png"))], warnings=["skipped"]),
    _claim_verification_report(),
    _peer_review_bundle(),
    _statistics_verification(),
    _citation_verification(),
]


# --------------------------------------------------------------------------- #
# 1. 无损往返
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sample", ROUNDTRIP_SAMPLES, ids=lambda s: type(s).__name__)
def test_roundtrip_is_lossless(sample: Any) -> None:
    """`from_dict(to_dict(x)) == x`，且再次 `to_dict()` 逐键相等。"""
    restored = type(sample).from_dict(sample.to_dict())

    assert restored == sample, f"{type(sample).__name__} 往返后不相等"
    assert restored.to_dict() == sample.to_dict(), (
        f"{type(sample).__name__} 往返后 to_dict 键集/取值发生变化"
    )


def test_nested_types_roundtrip_individually() -> None:
    """嵌套类型本身也必须可独立往返。"""
    for sample in (_stat_test(), _figure_spec(Path("/gone/f.png"))):
        assert type(sample).from_dict(sample.to_dict()) == sample

    for claim in _claim_verification_report().claims:
        assert ClaimVerdict.from_dict(claim.to_dict()) == claim
        for evidence in claim.evidence:
            assert ClaimEvidence.from_dict(evidence.to_dict()) == evidence

    for report in _peer_review_bundle().reports:
        assert ReviewerReport.from_dict(report.to_dict()) == report
        for concern in report.concerns:
            assert ReviewConcern.from_dict(concern.to_dict()) == concern

    # StatisticsClaim / DoiCheck 没有独立的 to_dict（由其父对象的 to_dict 经
    # asdict 序列化），因此这里用 asdict 构造它们的磁盘形态。
    for claim in _statistics_verification().claims:
        assert StatisticsClaim.from_dict(asdict(claim)) == claim

    for check in _citation_verification().doi_checks:
        assert DoiCheck.from_dict(asdict(check)) == check


def test_figure_spec_tolerates_missing_path() -> None:
    """path 指向的临时目录可能已被删除，还原时不得抛异常。"""
    spec = FigureSpec(
        figure_id="Figure 1",
        filename="f.png",
        title="t",
        caption="c",
        figure_type="distribution",
        path=Path("/definitely/not/here/f.png"),
    )

    restored = FigureSpec.from_dict(spec.to_dict())

    assert restored.path == Path("/definitely/not/here/f.png")
    assert isinstance(restored.path, Path)
    assert restored == spec


def test_roundtrip_survives_json_encoding() -> None:
    """经真实 JSON 编解码（json.loads 的结果是 object）后仍可往返。"""
    import json

    for sample in ROUNDTRIP_SAMPLES:
        payload: object = json.loads(json.dumps(sample.to_dict(), ensure_ascii=False))
        assert type(sample).from_dict(payload) == sample


# --------------------------------------------------------------------------- #
# 2. 坏数据一律抛 ValueError（ArtifactDecodeError）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sample", ROUNDTRIP_SAMPLES, ids=lambda s: type(s).__name__)
@pytest.mark.parametrize("bad_payload", [None, 42, "not-a-dict", [1, 2, 3]])
def test_non_dict_payload_raises_value_error(sample: Any, bad_payload: object) -> None:
    with pytest.raises(ValueError):
        type(sample).from_dict(bad_payload)


@pytest.mark.parametrize("sample", ROUNDTRIP_SAMPLES, ids=lambda s: type(s).__name__)
def test_empty_dict_raises_value_error(sample: Any) -> None:
    """缺失必需键必须报错，绝不静默构造半个对象。"""
    with pytest.raises(ValueError):
        type(sample).from_dict({})


def test_error_is_the_dedicated_decode_error() -> None:
    """非法输入抛的是专用异常 `ArtifactDecodeError`（ValueError 的子类）。

    调用方只需捕获 `ValueError` 即可统一退化为"重新执行该阶段"。
    """
    assert issubclass(ArtifactDecodeError, ValueError)
    with pytest.raises(ArtifactDecodeError):
        PaperRecord.from_dict({})
    with pytest.raises(ArtifactDecodeError):
        ClaimVerificationReport.from_dict("nope")
    with pytest.raises(ArtifactDecodeError):
        PeerReviewBundle.from_dict(None)


def test_error_message_names_the_offending_field() -> None:
    """错误消息点名字段、期望与实际，便于定位损坏的产物。"""
    payload = _paper().to_dict()
    payload["title"] = 123  # 类型错误

    with pytest.raises(ValueError) as excinfo:
        PaperRecord.from_dict(payload)

    message = str(excinfo.value)
    assert "title" in message
    assert "str" in message


def test_wrong_scalar_types_raise_value_error() -> None:
    payload = _statistics_report().to_dict()
    payload["rows"] = "ten"  # 应为 int

    with pytest.raises(ValueError):
        StatisticsReport.from_dict(payload)


def test_bool_is_not_accepted_as_int() -> None:
    """bool 是 int 的子类，但不能被当作计数字段接受。"""
    payload = _statistics_report().to_dict()
    payload["rows"] = True

    with pytest.raises(ValueError):
        StatisticsReport.from_dict(payload)


def test_nested_element_must_be_dict() -> None:
    """嵌套列表里混入非 dict 元素必须报错。"""
    payload = _statistics_report().to_dict()
    payload["tests"] = ["not-a-dict"]

    with pytest.raises(ValueError) as excinfo:
        StatisticsReport.from_dict(payload)

    assert "tests" in str(excinfo.value)


def test_missing_nested_required_key_raises_value_error() -> None:
    payload = _claim_verification_report().to_dict()
    # 删除断言中一条 evidence 的必需键
    del payload["claims"][0]["evidence"][0]["verdict"]

    with pytest.raises(ValueError) as excinfo:
        ClaimVerificationReport.from_dict(payload)

    assert "verdict" in str(excinfo.value)


def test_derived_keys_are_recomputed_not_trusted() -> None:
    """派生字段（如 overall/passed）被传递进来也不得被当作事实采纳。

    传入伪造的 `overall=supports`，重建后必须按 evidence 重新计算为 supports
    （此处 evidence 确实为 supports），说明它来自计算而非输入。
    """
    verdict = _claim_verification_report().claims[0]
    payload = verdict.to_dict()
    payload["overall"] = "unsupported"  # 伪造

    restored = ClaimVerdict.from_dict(payload)

    assert restored.overall == "supports"


def test_from_dict_has_no_side_effects(tmp_path: Path) -> None:
    """`from_dict` 不得读写文件/建目录：传入不存在的路径也不得创建它。"""
    ghost = tmp_path / "must-not-exist"
    spec_payload = _figure_spec(ghost / "f.png").to_dict()

    FigureSpec.from_dict(spec_payload)

    assert not ghost.exists()
