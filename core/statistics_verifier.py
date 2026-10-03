"""Traceability gate for quantitative claims in a drafted manuscript.

Phase 1 established one rule: every citation must trace back to a record that was
actually retrieved. This module applies the same rule to numbers. When a run has
analysed a dataset, every p-value asserted in the manuscript body must trace back
to a statistic that was actually computed.

The check is deliberately **advisory** rather than blocking:

* numeric formatting legitimately varies (``0.03`` vs ``0.031``);
* a discussion section may quote p-values from the *literature*.

Hard-blocking on either would produce false positives and make the tool
unusable. Unmatched claims are therefore surfaced as author checks and recorded
in an artifact instead of passing silently.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from core.research_models import ArtifactDecodeError

#: Matches ``p < 0.05``, ``p = 0.031``, ``p-value=0.03``, ``P <= .01`` …
P_VALUE_PATTERN = re.compile(
    r"\bp\s*(?:-?\s*value\s*)?(?P<operator><=|>=|<|>|=)\s*(?P<value>\d*\.?\d+)",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# 反序列化校验：结构非法一律抛 ValueError（绝不静默构造半个对象）
# --------------------------------------------------------------------------- #
def _require_object(value: object, owner: str) -> dict:
    """要求是映射（dict）；否则抛 ``ValueError``。"""
    if not isinstance(value, dict):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 期望 payload 为 dict，实际为 {type(value).__name__}"
        )
    return value


def _require_str(data: dict, field: str, owner: str) -> str:
    value = data.get(field)
    if not isinstance(value, str):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field}' 期望 str，"
            f"实际为 {type(value).__name__}"
        )
    return value


def _require_bool(data: dict, field: str, owner: str) -> bool:
    value = data.get(field)
    if not isinstance(value, bool):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field}' 期望 bool，"
            f"实际为 {type(value).__name__}"
        )
    return value


def _require_float(data: dict, field: str, owner: str) -> float:
    value = data.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field}' 期望 float，"
            f"实际为 {type(value).__name__}"
        )
    return float(value)


def _require_float_list(data: dict, field: str, owner: str) -> list[float]:
    value = data.get(field)
    if not isinstance(value, list):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field}' 期望 list[float]，"
            f"实际为 {type(value).__name__}"
        )
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ArtifactDecodeError(
                f"{owner}.from_dict 字段 '{field}' 的第 {index} 个元素期望数值，"
                f"实际为 {type(item).__name__}"
            )
    return [float(item) for item in value]


def _matches_reported_precision(reported_text: str, value: float, computed: Sequence[float]) -> bool:
    """Match a reported p-value against a computed one at the reported precision.

    A fixed absolute tolerance is wrong here: ``0.005`` would make ``p = 0.0001``
    "match" a computed ``0.001``, silently hiding a fabricated small p-value.
    Comparing at the number of decimals the author actually wrote is both
    stricter and fairer — ``p = 0.03`` still matches a computed ``0.031``.
    """
    decimals = len(reported_text.split(".", 1)[1]) if "." in reported_text else 0
    return any(round(item, decimals) == value for item in computed)


@dataclass(frozen=True)
class StatisticsClaim:
    """A single p-value assertion extracted from the manuscript body."""

    raw: str
    operator: str
    value: float
    matched: bool
    note: str = ""

    @classmethod
    def from_dict(cls, payload: object) -> StatisticsClaim:
        """从 :meth:`to_dict` 的输出还原；结构不合法时抛 ``ValueError``。"""
        data = _require_object(payload, cls.__name__)
        return cls(
            raw=_require_str(data, "raw", cls.__name__),
            operator=_require_str(data, "operator", cls.__name__),
            value=_require_float(data, "value", cls.__name__),
            matched=_require_bool(data, "matched", cls.__name__),
            note=_require_str(data, "note", cls.__name__),
        )


@dataclass
class StatisticsVerificationReport:
    """Outcome of checking manuscript claims against computed statistics."""

    computed_p_values: list[float] = field(default_factory=list)
    claims: list[StatisticsClaim] = field(default_factory=list)

    @property
    def unmatched(self) -> list[StatisticsClaim]:
        return [claim for claim in self.claims if not claim.matched]

    @property
    def passed(self) -> bool:
        """True when every extracted claim traces back to a computed statistic."""
        return not self.unmatched

    def warnings(self) -> list[str]:
        if not self.unmatched:
            return []
        return [
            f"{len(self.unmatched)} 处统计陈述无法追溯到本次分析结果，请人工核对。"
        ]

    def author_checks(self) -> list[str]:
        if not self.unmatched:
            return []
        return [
            "正文中的部分 p 值并非本次数据分析产生（可能引用自文献或由模型推断）："
            + "、".join(f"`{claim.raw}`" for claim in self.unmatched)
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "computed_p_values": self.computed_p_values,
            "claim_count": len(self.claims),
            "unmatched_count": len(self.unmatched),
            "claims": [asdict(claim) for claim in self.claims],
        }

    @classmethod
    def from_dict(cls, payload: object) -> StatisticsVerificationReport:
        """从 :meth:`to_dict` 的输出还原；结构不合法时抛 ``ValueError``。

        ``passed``、``claim_count``、``unmatched_count`` 均为派生字段，重建时由
        属性自动重算。
        """
        data = _require_object(payload, cls.__name__)
        claims = data.get("claims")
        if not isinstance(claims, list):
            raise ArtifactDecodeError(
                f"{cls.__name__}.from_dict 字段 'claims' 期望 list，"
                f"实际为 {type(claims).__name__}"
            )
        decoded: list[StatisticsClaim] = []
        for index, item in enumerate(claims):
            if not isinstance(item, dict):
                raise ArtifactDecodeError(
                    f"{cls.__name__}.from_dict 字段 'claims' 的第 {index} 个元素"
                    f"期望 dict，实际为 {type(item).__name__}"
                )
            decoded.append(StatisticsClaim.from_dict(item))
        return cls(
            computed_p_values=_require_float_list(
                data, "computed_p_values", cls.__name__
            ),
            claims=decoded,
        )


def _is_matched(
    operator: str,
    value: float,
    reported_text: str,
    computed: Sequence[float],
) -> tuple[bool, str]:
    if not computed:
        return False, "本次运行没有产生任何统计检验结果"

    if operator == "=":
        if _matches_reported_precision(reported_text, value, computed):
            return True, ""
        return False, "正文报告了本次分析未产生的 p 值"

    if operator in ("<", "<="):
        if any(item < value for item in computed):
            return True, ""
        return False, f"正文声称 p {operator} {value}，但没有任何检验达到该阈值"

    # ">" or ">="
    if any(item > value for item in computed):
        return True, ""
    return False, f"正文声称 p {operator} {value}，但没有任何检验满足该条件"


def verify_statistics(
    manuscript_body: str,
    p_values: Sequence[float],
) -> StatisticsVerificationReport:
    """Check the p-values asserted in a draft against the computed statistics.

    Args:
        manuscript_body: Draft text *before* the system appends its own
            computed statistics table.
        p_values: Every raw p-value produced by the statistics engine.

    Returns:
        A report whose ``passed`` flag is False when a claim cannot be traced.
    """
    computed = [float(value) for value in p_values]

    seen: set[tuple[str, float]] = set()
    claims: list[StatisticsClaim] = []
    for match in P_VALUE_PATTERN.finditer(manuscript_body):
        operator = match.group("operator")
        reported_text = match.group("value")
        try:
            value = float(reported_text)
        except ValueError:  # pragma: no cover - regex guarantees a numeric group
            continue
        key = (operator, value)
        if key in seen:
            continue
        seen.add(key)

        matched, note = _is_matched(operator, value, reported_text, computed)
        claims.append(
            StatisticsClaim(
                raw=match.group(0).strip(),
                operator=operator,
                value=value,
                matched=matched,
                note=note,
            )
        )

    return StatisticsVerificationReport(computed_p_values=computed, claims=claims)


def render_statistics_verification_markdown(
    report: StatisticsVerificationReport,
) -> str:
    """Render the traceability report as a Markdown artifact."""
    lines = [
        "# 统计陈述追溯报告",
        "",
        f"- 结论: {'通过' if report.passed else '存在无法追溯的统计陈述'}",
        f"- 本次分析产生的 p 值: {len(report.computed_p_values)} 个",
        f"- 正文提取到的统计陈述: {len(report.claims)} 处",
        f"- 无法追溯: {len(report.unmatched)} 处",
        "",
    ]

    if report.unmatched:
        lines.extend(["## 无法追溯的陈述（需人工核对）", ""])
        lines.extend(
            f"- `{claim.raw}` — {claim.note}" for claim in report.unmatched
        )
        lines.append("")

    if report.claims and report.passed:
        lines.extend(["## 已追溯的陈述", ""])
        lines.extend(f"- `{claim.raw}`" for claim in report.claims)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
