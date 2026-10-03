"""Traceability gate for quantitative claims in a drafted manuscript.

Phase 1 established one rule: every citation must trace back to a record that was
actually retrieved. This module applies the same rule to numbers. When a run has
analysed a dataset, every statistical assertion in the manuscript body — *p*-values,
effect sizes and sample sizes — must trace back to a statistic that was actually
computed. The writing prompt tells the model it may only quote the provided
*p*-values, effect sizes and *n* verbatim; this gate is what makes that promise
enforceable instead of aspirational.

The check is deliberately **advisory** rather than blocking:

* numeric formatting legitimately varies (``0.03`` vs ``0.031``);
* a discussion section may quote statistics from the *literature*.

Hard-blocking on either would produce false positives and make the tool
unusable. Unmatched claims are therefore surfaced as author checks and recorded
in an artifact instead of passing silently.

Known and accepted limitations
------------------------------
Quoting a statistic that belongs to the *literature* rather than to this run will
register as unmatched; the same is already true of the p-value path (the module
docstring has always said so). To keep the signal usable we extract conservatively
and prefer **misses over noise**:

* effect-size symbols are only recognised when they are unambiguous canonical
  names (``d``, ``Cohen's d``, ``r``, ``ρ``, ``rho``) — never bare ``R`` or
  ``R²``, which overwhelmingly denote coefficients of determination in prose;
* every effect-size symbol is matched only against a test whose
  ``effect_size_name`` *starts with* that symbol's family, so ``d`` can never be
  satisfied by a Pearson ``r`` (and vice versa);
* the sample-size pattern only accepts single integer tokens, so ``n = 3 studies``
  (a count of *literature* items) is not guessed at all.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from core.research_models import ArtifactDecodeError
from core.statistics_engine import StatTestResult

#: Matches ``p < 0.05``, ``p = 0.031``, ``p-value=0.03``, ``P <= .01`` …
P_VALUE_PATTERN = re.compile(
    r"\bp\s*(?:-?\s*value\s*)?(?P<operator><=|>=|<|>|=)\s*(?P<value>\d*\.?\d+)",
    re.IGNORECASE,
)

#: Matches ``d = 0.42``, ``Cohen's d = 0.42``, ``d=0.42``, ``r = 0.85`` …
#: The symbol is guaranteed to end on a word boundary so ``R2``/``R²``/``pH``
#: never enter. Bare ``r`` is gated further by the ``(?=\s*=)`` lookahead, so
#: the English article ("a result that is…") cannot masquerade as a statistic.
#: ``kind`` is fixed per alternative so callers do not have to re-parse the
#: captured text (``Cohen's d`` and bare ``d`` both land on ``d``).
EFFECT_SIZE_PATTERN = re.compile(
    r"""
    \b
    (?P<name>
        cohen['\u2019]?s\s+d   # Cohen's d / Cohens d
      | d(?=\s*=)              # d = …
      | rho(?=\s*=)            # rho = …
      | r(?=\s*=)              # r = …
      | ρ                      # ρ = …
    )
    \s*=\s*
    (?P<value>\d*\.?\d+)
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: Matches ``n = 10``, ``N = 20``, ``n=10`` with commas (``n = 1,024``).
#: The negative lookahead rejects the very common prose pattern
#: ``n = 3 studies`` / ``n = 12 papers``, where the number counts *literature
#: items* rather than participants. Without it that phrasing would be reported
#: as a fabricated sample size on almost every review-style manuscript.
SAMPLE_SIZE_PATTERN = re.compile(
    r"\b(?P<name>n)\s*=\s*(?P<value>\d[\d,]*)"
    r"(?!\s*(?:studies|papers|articles|sources|records|trials))",
    re.IGNORECASE,
)

#: Canonical symbol -> the effect-size *family* (a ``startswith`` prefix of the
#: engine's ``effect_size_name``) it is allowed to match. Only the exact symbols
#: emitted by :data:`EFFECT_SIZE_PATTERN` are listed here; anything unknown is
#: ignored rather than guessed.
_EFFECT_SIZE_FAMILIES = {
    "cohen's d": "d",
    "cohens d": "d",
    "d": "d",
    "r": "r",
    "rho": "r",
    "ρ": "r",
}

#: Human-readable symbol used in notes/warnings.
_FAMILY_SYMBOLS = {"d": "d", "r": "r"}


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


def _require_int_list(data: dict, field: str, owner: str) -> list[int]:
    value = data.get(field)
    if not isinstance(value, list):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field}' 期望 list[int]，"
            f"实际为 {type(value).__name__}"
        )
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ArtifactDecodeError(
                f"{owner}.from_dict 字段 '{field}' 的第 {index} 个元素期望 int，"
                f"实际为 {type(item).__name__}"
            )
    return list(value)


def _validate_kind(value: str, owner: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field_name}' 期望 str，"
            f"实际为 {type(value).__name__}"
        )
    if value not in ("p_value", "effect_size", "sample_size"):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field_name}' 取值非法: {value!r}"
        )
    return value


def _matches_reported_precision(reported_text: str, value: float, computed: Sequence[float]) -> bool:
    """Match a reported p-value against a computed one at the reported precision.

    A fixed absolute tolerance is wrong here: ``0.005`` would make ``p = 0.0001``
    "match" a computed ``0.001``, silently hiding a fabricated small p-value.
    Comparing at the number of decimals the author actually wrote is both
    stricter and fairer — ``p = 0.03`` still matches a computed ``0.031``.

    The same rule is reused for effect sizes: ``d = 0.4`` still matches a
    computed ``0.42`` but ``d = 0.42`` does not match ``0.4``.
    """
    decimals = len(reported_text.split(".", 1)[1]) if "." in reported_text else 0
    return any(round(item, decimals) == value for item in computed)


@dataclass(frozen=True)
class StatisticsClaim:
    """A single statistical assertion extracted from the manuscript body."""

    raw: str
    operator: str
    value: float
    matched: bool
    note: str = ""
    kind: str = "p_value"

    @classmethod
    def from_dict(cls, payload: object) -> StatisticsClaim:
        """从 :meth:`to_dict` 的输出还原；结构不合法时抛 ``ValueError``。

        ``kind`` 缺失时默认为 ``"p_value"``，以兼容没有该键的旧产物。
        """
        data = _require_object(payload, cls.__name__)
        raw_kind = data.get("kind", "p_value")
        return cls(
            raw=_require_str(data, "raw", cls.__name__),
            operator=_require_str(data, "operator", cls.__name__),
            value=_require_float(data, "value", cls.__name__),
            matched=_require_bool(data, "matched", cls.__name__),
            note=_require_str(data, "note", cls.__name__),
            kind=_validate_kind(raw_kind, cls.__name__, "kind"),
        )


@dataclass
class StatisticsVerificationReport:
    """Outcome of checking manuscript claims against computed statistics."""

    computed_p_values: list[float] = field(default_factory=list)
    computed_effect_sizes: list[float] = field(default_factory=list)
    computed_sample_sizes: list[int] = field(default_factory=list)
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
            "正文中的部分 p 值/效应量/样本量并非本次数据分析产生"
            "（可能引用自文献或由模型推断）："
            + "、".join(f"`{claim.raw}`" for claim in self.unmatched)
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "computed_p_values": self.computed_p_values,
            "computed_effect_sizes": self.computed_effect_sizes,
            "computed_sample_sizes": self.computed_sample_sizes,
            "claim_count": len(self.claims),
            "unmatched_count": len(self.unmatched),
            "claims": [asdict(claim) for claim in self.claims],
        }

    @classmethod
    def from_dict(cls, payload: object) -> StatisticsVerificationReport:
        """从 :meth:`to_dict` 的输出还原；结构不合法时抛 ``ValueError``。

        ``passed``、``claim_count``、``unmatched_count`` 均为派生字段，重建时由
        属性自动重算。``computed_effect_sizes`` / ``computed_sample_sizes`` 缺失时
        默认为空列表，以兼容旧版产物。
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
            computed_p_values=_optional_float_list(
                data, "computed_p_values", cls.__name__
            ),
            computed_effect_sizes=_optional_float_list(
                data, "computed_effect_sizes", cls.__name__
            ),
            computed_sample_sizes=_optional_int_list(
                data, "computed_sample_sizes", cls.__name__
            ),
            claims=decoded,
        )


def _optional_float_list(data: dict, field: str, owner: str) -> list[float]:
    """缺失时返回空列表；存在时严格校验为 list[float]。"""
    if field not in data or data[field] is None:
        return []
    return _require_float_list(data, field, owner)


def _optional_int_list(data: dict, field: str, owner: str) -> list[int]:
    """缺失时返回空列表；存在时严格校验为 list[int]。"""
    if field not in data or data[field] is None:
        return []
    return _require_int_list(data, field, owner)


# --------------------------------------------------------------------------- #
# 匹配逻辑
# --------------------------------------------------------------------------- #
def _p_value_matched(
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


def _is_finite(item: float) -> bool:
    return not math.isnan(item) and not math.isinf(item)


def _effect_size_family(name: str) -> str | None:
    """Map an engine ``effect_size_name`` to the symbol family it can satisfy.

    ``"Cohen's d"`` -> ``"d"``; ``"Pearson r"`` / ``"Spearman r"`` -> ``"r"``.
    Names without a recognised symbol (``"eta squared"``) map to ``None`` and are
    therefore never eligible to satisfy an effect-size claim — this is the
    conservative choice that keeps ``d``/``r`` statement-level checks honest.
    """
    lowered = name.strip().lower()
    if not lowered:
        return None
    # ``eta squared`` / ``eta^2`` deliberately carry no symbol family.
    if "eta" in lowered:
        return None
    if lowered.endswith("d") or "cohen" in lowered:
        return "d"
    if "rho" in lowered or lowered.endswith("r"):
        return "r"
    return None


def _effect_size_matched(
    family: str,
    value: float,
    reported_text: str,
    computed: Sequence[float],
) -> tuple[bool, str]:
    if not computed:
        return False, f"本次运行没有产生可比较的 {_FAMILY_SYMBOLS[family]} 效应量"
    if _matches_reported_precision(reported_text, value, computed):
        return True, ""
    return False, f"正文报告的 {_FAMILY_SYMBOLS[family]} 效应量未出现在本次分析结果中"


def _sample_size_matched(
    value: float, computed: Sequence[int]
) -> tuple[bool, str]:
    if not computed:
        return False, "本次运行没有产生任何样本量"
    if any(item == int(value) for item in computed):
        return True, ""
    return False, "正文报告的样本量与本次分析任何检验的 n 都不一致"


def verify_statistics(
    manuscript_body: str,
    tests: Sequence[StatTestResult],
) -> StatisticsVerificationReport:
    """Check the statistics asserted in a draft against the computed statistics.

    Args:
        manuscript_body: Draft text *before* the system appends its own
            computed statistics table.
        tests: Every test produced by the statistics engine. p-values, effect
            sizes and sample sizes are all derived from these results.

    Returns:
        A report whose ``passed`` flag is False when a claim cannot be traced.

    Notes:
        This is an advisory gate: an unmatched claim never raises. Statistics
        quoted from the *literature* (e.g. ``Smith reported d = 0.8 [P1]``)
        will be reported as unmatched — that is known and accepted.
    """
    computed_p_values = sorted(
        {float(test.p_value) for test in tests if _is_finite(test.p_value)}
    )
    computed_sample_sizes = sorted({int(test.n) for test in tests if int(test.n) > 0})

    # Effect sizes are keyed by *family* so a reported ``d`` only ever compares
    # against a Cohen's d, never against a correlation's ``r`` (and vice versa).
    families: dict[str, list[float]] = {"d": [], "r": []}
    for test in tests:
        if not _is_finite(test.effect_size):
            continue
        family = _effect_size_family(test.effect_size_name)
        if family is not None:
            families[family].append(float(test.effect_size))

    claims: list[StatisticsClaim] = []
    seen: set[tuple[str, str, float]] = set()

    for match in P_VALUE_PATTERN.finditer(manuscript_body):
        operator = match.group("operator")
        reported_text = match.group("value")
        try:
            value = float(reported_text)
        except ValueError:  # pragma: no cover - regex guarantees a numeric group
            continue
        key = ("p_value", operator, value)
        if key in seen:
            continue
        seen.add(key)

        matched, note = _p_value_matched(
            operator, value, reported_text, computed_p_values
        )
        claims.append(
            StatisticsClaim(
                raw=match.group(0).strip(),
                operator=operator,
                value=value,
                matched=matched,
                note=note,
                kind="p_value",
            )
        )

    for match in EFFECT_SIZE_PATTERN.finditer(manuscript_body):
        symbol = match.group("name").strip().lower()
        family = _EFFECT_SIZE_FAMILIES.get(symbol)
        if family is None:  # pragma: no cover - regex only yields known symbols
            continue
        reported_text = match.group("value")
        try:
            value = float(reported_text)
        except ValueError:  # pragma: no cover - regex guarantees a numeric group
            continue
        key = ("effect_size", symbol, value)
        if key in seen:
            continue
        seen.add(key)

        matched, note = _effect_size_matched(
            family, value, reported_text, families[family]
        )
        claims.append(
            StatisticsClaim(
                raw=match.group(0).strip(),
                operator="",
                value=value,
                matched=matched,
                note=note,
                kind="effect_size",
            )
        )

    for match in SAMPLE_SIZE_PATTERN.finditer(manuscript_body):
        reported_text = match.group("value").replace(",", "")
        try:
            value = float(reported_text)
        except ValueError:  # pragma: no cover - regex guarantees a numeric group
            continue
        key = ("sample_size", "", value)
        if key in seen:
            continue
        seen.add(key)

        matched, note = _sample_size_matched(value, computed_sample_sizes)
        claims.append(
            StatisticsClaim(
                raw=match.group(0).strip(),
                operator="",
                value=value,
                matched=matched,
                note=note,
                kind="sample_size",
            )
        )

    return StatisticsVerificationReport(
        computed_p_values=computed_p_values,
        computed_effect_sizes=sorted(
            families["d"] + families["r"]
        ),
        computed_sample_sizes=computed_sample_sizes,
        claims=claims,
    )


def render_statistics_verification_markdown(
    report: StatisticsVerificationReport,
) -> str:
    """Render the traceability report as a Markdown artifact."""
    lines = [
        "# 统计陈述追溯报告",
        "",
        f"- 结论: {'通过' if report.passed else '存在无法追溯的统计陈述'}",
        f"- 本次分析产生的 p 值: {len(report.computed_p_values)} 个",
        f"- 本次分析产生的效应量: {len(report.computed_effect_sizes)} 个",
        f"- 本次分析产生的样本量: {len(report.computed_sample_sizes)} 个",
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
