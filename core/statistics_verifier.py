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

Recognition scope (what counts as a "claim")
--------------------------------------------
The extractor must be able to *see* the statistics a real manuscript writes. The
forms the regexes recognise are:

* p-values — ``p = 0.03``, ``p < 0.05``, ``p-value = 0.02``, ``P <= .01``, plus
  **scientific notation** (``p = 3.57e-16``) and a **CJK-adjacent** ``p``
  (``Holm校正p=…``);
* effect sizes — ``d``, ``Cohen's d``, ``r``, ``rho``, ``ρ`` and
  **``eta squared``**, with an optional leading **minus sign**
  (``r = -0.902…``);
* sample sizes — ``n = 10`` / ``N = 10`` (single integer tokens only).

Every pattern anchors its keyword with ``(?<![A-Za-z0-9])`` rather than ``\\b``:
Python's ``\\w`` matches CJK characters, so in ``校正p=`` there is **no** word
boundary between ``正`` and ``p`` and a ``\\bp`` would silently match nothing.
The lookbehind still rejects ASCII-prefixed false positives (``ap =``, ``pH``).

Known and accepted limitations
------------------------------
Quoting a statistic that belongs to the *literature* rather than to this run will
register as unmatched; the same is already true of the p-value path (the module
docstring has always said so). To keep the signal usable we extract conservatively
and prefer **misses over noise**:

* effect-size symbols are only recognised when they are unambiguous canonical
  names (``d``, ``Cohen's d``, ``r``, ``ρ``, ``rho``, ``eta squared``) — never
  bare ``R`` or ``R²``, which overwhelmingly denote coefficients of determination
  in prose;
* every effect-size symbol is matched only against a test in the *same* family, so
  ``d`` can never be satisfied by a Pearson ``r``, and ``eta squared`` by neither;
* the sample-size pattern only accepts single integer tokens, so ``n = 3 studies``
  (a count of *literature* items) is not guessed at all.

Coverage boundaries — what this gate does *not* cover
-----------------------------------------------------
A gate that only knows what it **can** do invites the reader to assume it did
everything. Writing the boundary down is the honest half of the contract. The
following forms are **deliberately not** extracted (an honest gap list, not a
to-do list of bugs):

* **pure prose without a symbol or label** — ``"效应很大"`` / ``"显著降低"`` carry
  no number to locate, so no claim is produced;
* ``R²`` / ``R2`` coefficients of determination — recognised never (bare ``R`` /
  ``R²`` overwhelmingly denote variance explained in prose, so extracting them
  would be noise);
* **confidence intervals** (``95% CI [a, b]``) — not parsed at all;
* **Greek-letter variants other than** ``ρ`` — e.g. ``η²`` (a common ASCII-less
  spelling of ``eta squared``) and ``χ²`` are not recognised; the symbolic form
  ``eta squared`` *is*;
* **test statistics themselves** (``F = 201.37…``, ``t = …``) — outside the three
  tracked kinds (p-value / effect size / sample size) and therefore unchecked;
* **per-occurrence counting** — claims are **deduped by value**, so ``n = 30``
  appearing five times counts as **one** claim and four distinct Holm-adjusted p
  values count as four. ``claim_count`` is a count of *distinct statistical
  statements*, **not** of textual occurrences.

What ``claim_count`` means
--------------------------
``claim_count`` / ``report.claims`` is the number of **distinct** statistical
statements this gate managed to *recognise* in the manuscript body. Reading it
requires both halves of the contract:

1. the **denominator** is "what the regexes above can see" (see *Recognition
   scope*), not "every statistical statement ever written";
2. the **numerator** is only the recognised subset — the boundary forms above are
   **outside** it and are not counted either way.

So ``claim_count = 10`` means "ten recognised statements", never "the manuscript
contains exactly ten statistical statements".

What ``passed = true`` means (and its residual risk)
-----------------------------------------------------
``passed = true`` means: **every *recognised* claim traces back to a statistic
this run actually computed.** It does **not** mean "every statistical statement in
the manuscript has been traced", and it does **not** assert that the coverage is
adequate.

This gate deliberately does **not** force ``passed = False`` merely because
``claim_count`` is low: with no ground truth for how many statistics a manuscript
"should" contain, any threshold would be invented, and a manuscript that legitimately
contains no statistics would be failed (and :meth:`author_checks` would report
"untraceable claims" that do not exist). Poor coverage is therefore carried by
:meth:`StatisticsVerificationReport.coverage_warning` and by the explicit coverage
line in the rendered artifact — **visibility, not a fabricated pass/fail gate.**

**The residual risk this leaves, stated plainly:** a manuscript quoting statistics
in a form this gate cannot see (a boundary form above) can report ``passed = true``
with a low ``claim_count``. This is exactly the Phase 9 incident: the artifact
already recorded ``claim_count: 1`` — the *number* was visible — yet ``passed=true``
overrode it and nobody found "1" suspicious. **The number being present and the
number being correctly interpreted are two different things**; the interpretation
needs the prose above, which is why it lives here and not only in a chat log.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from core.research_models import ArtifactDecodeError
from core.statistics_engine import StatTestResult

#: 数值字面量：整数/小数，可选科学计数法指数，可选前导负号。
#: 三个模式共用同一份定义，避免各自漂移。
#:
#: 为什么必须同时覆盖这三件事（真实手稿
#: ``projects/phase9-real-urban-heat/artifacts/writing/manuscript.md.v1`` 里
#: 逐条出现过）：
#:
#: * **科学计数法**：``p = 3.5745385179030436e-16`` 里的 ``e-16`` 会被朴素的
#:   ``\d*\.?\d+`` 丢成尾数 ``3.574…``（数值量级改变约 16 个数量级）或整体失配；
#: * **负号**：相关系数 ``r = -0.902…`` 的负号不在旧字符集内，整条直接漏识别；
#: * 单独放开这两个还不够——见下面各模式对词边界的处理。
#:
#: 前导 ``-?`` 只允许真正的负号；``(?P<value>…)`` 捕获完整字面量（含 ``-`` 与
#: ``e``），供 :func:`float` 与精度比对使用。
_NUMBER = r"-?\d*\.?\d+(?:[eE][+-]?\d+)?"

#: Matches ``p < 0.05``, ``p = 0.031``, ``p-value=0.03``, ``P <= .01``,
#: ``p = 3.57e-16``, ``Holm校正p=0.0066`` …
#:
#: 词边界从 ``\b`` 改为 ``(?<![A-Za-z0-9])``。原因：Python 的 ``\w`` 匹配中文，
#: 在 ``Holm校正p=…`` 中 ``正`` 与 ``p`` 之间**没有**词边界，``\bp`` 根本不成立
#: ——真实手稿里 3 条 ``Holm校正p=…`` 全部因此漏识别。负向后行断言只排除 ASCII
#: 字母/数字前缀，既救回中文相邻的 ``p``，又继续挡住 ``ap = 0.5`` / ``file2p =``
#: 这类误命（``pH`` 也照旧不命中，因为紧跟的是 ``H`` 而非运算符）。
P_VALUE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])p\s*(?:-?\s*value\s*)?"
    r"(?P<operator><=|>=|<|>|=)\s*(?P<value>" + _NUMBER + r")",
    re.IGNORECASE,
)

#: Matches ``d = 0.42``, ``Cohen's d = 0.42``, ``d=0.42``, ``r = 0.85``,
#: ``r=-0.902…``, ``eta squared=0.9371…`` …
#:
#: 符号词边界同样从 ``\b`` 改为 ``(?<![A-Za-z0-9])``：中文相邻的
#: ``Spearman r=…`` 里 ``n`` 与 ``r`` 均为 ``\w``、两者之间没有词边界，
#: 旧的 ``\b`` 让这 3 条负相关效应量一条都不命中。
#:
#: ``eta squared`` 现在也是一等的效应量符号（真实手稿 1 条陈述含 2 处，
#: 共 6 处），归入 :data:`_EFFECT_SIZE_FAMILIES` 的 ``"eta squared"`` 家族。
#:
#: 保留一条**防噪声**约束（既有的"宁可漏报不要噪声"原则，非本次放松）：
#: 裸 ``d`` / ``r`` 仍要求等号紧跟在符号后（允许符号与 ``=`` 之间有空格），
#: 这样英文散文里孤立的字母 d/r 不会被当作效应量。注意 ``d(?=\s*=)`` 本身已
#: 保证 ``=`` 紧随，故旧行为不变；新增的只有 ``eta squared`` 与 ``-`` 号。
EFFECT_SIZE_PATTERN = re.compile(
    r"""
    (?<![A-Za-z0-9])
    (?P<name>
        eta\s+squared          # eta squared = 0.9371…
      | cohen['\u2019]?s\s+d   # Cohen's d / Cohens d
      | d(?=\s*=)              # d = …
      | rho(?=\s*=)            # rho = …
      | r(?=\s*=)              # r = …
      | ρ                      # ρ = …
    )
    \s*=\s*
    (?P<value>""" + _NUMBER + r""")
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: Matches ``n = 10``, ``N = 20``, ``n=10`` with commas (``n = 1,024``).
#: The negative lookahead rejects the very common prose pattern
#: ``n = 3 studies`` / ``n = 12 papers``, where the number counts *literature
#: items* rather than participants. Without it that phrasing would be reported
#: as a fabricated sample size on almost every review-style manuscript.
#:
#: 词边界同样从 ``\b`` 改为 ``(?<![A-Za-z0-9])``，救回 ``报告n=30`` 这类中文
#: 紧邻的样本量陈述；``\d[\d,]*`` 保持不变——n 不应有科学计数法或负号。
SAMPLE_SIZE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?P<name>n)\s*=\s*(?P<value>\d[\d,]*)"
    r"(?!\s*(?:studies|papers|articles|sources|records|trials))",
    re.IGNORECASE,
)

#: Canonical symbol -> the effect-size *family* (a ``startswith`` prefix of the
#: engine's ``effect_size_name``) it is allowed to match. Only the exact symbols
#: emitted by :data:`EFFECT_SIZE_PATTERN` are listed here; anything unknown is
#: ignored rather than guessed.
#:
#: ``eta squared`` 单列一个家族：它既不是 Cohen's ``d`` 也不是相关系数 ``r``，
#: 若归入 ``d`` 会让 ``eta squared`` 陈述被一个 Cohen's d 结果"对应上"，
#: 属放松核验强度，故与 ``d``/``r`` 严格区隔。
_EFFECT_SIZE_FAMILIES = {
    "cohen's d": "d",
    "cohens d": "d",
    "d": "d",
    "r": "r",
    "rho": "r",
    "ρ": "r",
    "eta squared": "eta squared",
}

#: Human-readable symbol used in notes/warnings.
_FAMILY_SYMBOLS = {"d": "d", "r": "r", "eta squared": "eta squared"}


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


def _reported_decimals(reported_text: str) -> int:
    """报告的数值字面量"写到小数点后几位"——用于按作者精度比对。

    A fixed absolute tolerance is wrong here: ``0.005`` would make ``p = 0.0001``
    "match" a computed ``0.001``, silently hiding a fabricated small p-value.
    Comparing at the number of decimals the author actually wrote is both
    stricter and fairer — ``p = 0.03`` still matches a computed ``0.031``.

    **科学计数法必须特殊处理**：``3.5745385179030436e-16`` 的尾数有 16 位小数，
    指数为 -16，最后一个有效数字落在 ``10^(16 + 16) = 10^-32`` 位，即**小数点后
    32 位**。若只数字面量里小数点后的字符（``5745385179030436e-16`` → 21 位，把
    ``e-16`` 也当成了数字），``round(item, 21)`` 与真实的 32 位量级错位，科学
    计数法的 p 值会被误判为"未产生"。正确的小数位数 = 尾数小数位数 − 指数
    （指数为负时增大，为正时减小）：``16 - (-16) = 32``。
    """
    normalized = reported_text.lower()
    mantissa, sep, exponent = normalized.partition("e")
    mantissa_decimals = len(mantissa.split(".", 1)[1]) if "." in mantissa else 0
    if not sep:
        return mantissa_decimals
    try:
        return mantissa_decimals - int(exponent)
    except ValueError:  # pragma: no cover - regex guarantees an integer exponent
        return mantissa_decimals


def _matches_reported_precision(reported_text: str, value: float, computed: Sequence[float]) -> bool:
    """Match a reported p-value against a computed one at the reported precision.

    The same rule is reused for effect sizes: ``d = 0.4`` still matches a
    computed ``0.42`` but ``d = 0.42`` does not match ``0.4``.
    """
    decimals = _reported_decimals(reported_text)
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
    """Outcome of checking manuscript claims against computed statistics.

    **覆盖率可见性与"未检查"语义（本次修复的第二条主线）**

    缺陷背景：真实运行里关口只识别到 1 条陈述却报 ``passed=true``——问题不只是
    漏识别，更在于它**从不说自己识别了多少**。「没识别到」被读成了「没有问题」。
    这里通过三点纠正：

    * :attr:`claim_count` / :attr:`unmatched_count` 已进 :meth:`to_dict`，产品
      层面**始终**能看到识别了多少条；
    * :attr:`ran` 记录本关口是否真的执行过。未执行（``ran=False``）时
      :attr:`passed` **强制为 False**——从未检查绝不能被读作「检查通过」，
      与 :class:`core.claim_verifier.ClaimVerificationReport` 同一铁律；
    * :attr:`coverage_warning`：识别条数为 0 时报「未识别到任何统计陈述」，
      在 :meth:`warnings` 里以 warning 浮出。**理由**：手稿里出现 ``p=`` /
      ``r=`` / ``n=`` 这类数值陈述时，识别条数为 0 几乎总是**抽取器失灵**
      而非「正文确实没有统计陈述」；让它以 warning 暴露，可被观测、可被追责，
      但不硬阻断（正文确实可能不含任何统计陈述，硬阻断会误伤）。

    ``passed`` 的语义被精确定义为：**若本关口执行过，则当且仅当每条已识别的
    陈述都能追溯到本次分析结果时为真**。刻意**不**因为「识别条数低」而独立地
    让 ``passed`` 为 False——否则正文确实不含统计陈述时会被误判为失败，且
    :meth:`author_checks` 会声称存在"无法追溯的陈述"（一条都没有）。识别条数
    偏低由 :attr:`coverage_warning` 单独承担可见性职责。本关口在流水线里是
    顾问级（warning 通道），不阻断，故该策略与产品的硬阻断关口不冲突。
    """

    computed_p_values: list[float] = field(default_factory=list)
    computed_effect_sizes: list[float] = field(default_factory=list)
    computed_sample_sizes: list[int] = field(default_factory=list)
    claims: list[StatisticsClaim] = field(default_factory=list)
    #: 本关口是否真的执行过。``False`` 时 :attr:`passed` 强制为 ``False``。
    #: 缺省 ``True`` 以兼容 Phase 9 之前产出的、没有该键的旧产物（那些产物
    #: 都是成功执行后写出的，故默认 ``ran=True`` 与事实一致）。
    ran: bool = True
    #: ``ran=False`` 时说明未执行的原因。
    not_run_reason: str = ""

    @property
    def claim_count(self) -> int:
        """正文中识别到的统计陈述条数（覆盖率的核心指标）。"""
        return len(self.claims)

    @property
    def unmatched(self) -> list[StatisticsClaim]:
        return [claim for claim in self.claims if not claim.matched]

    @property
    def coverage_warning(self) -> str | None:
        """识别条数为 0 时的覆盖率告警文本；否则为 ``None``。

        单独成属性，便于 :meth:`warnings` 与渲染复用同一措辞。
        """
        if self.claim_count != 0:
            return None
        return (
            "统计陈述追溯关口未在手稿正文中识别到任何统计陈述（p 值/效应量/"
            "样本量）。这不等于正文没有问题：「未识别到」可能是抽取失灵，而非"
            "正文确实不含统计陈述；请人工核对正文是否包含未识别出的统计陈述，"
            "以及抽取规则是否需要扩展。"
        )

    @property
    def passed(self) -> bool:
        """见类 docstring 的语义定义。

        * ``ran=False`` → 强制 ``False``（未执行 ≠ 通过）；
        * 已执行 → 当且仅当没有无法追溯的陈述时为真。
        """
        if not self.ran:
            return False
        return not self.unmatched

    def warnings(self) -> list[str]:
        if not self.ran:
            return [
                "统计陈述追溯关口未执行："
                + (self.not_run_reason or "原因未知")
                + "（未执行不等于核验通过）。"
            ]
        messages: list[str] = []
        if self.unmatched:
            messages.append(
                f"{len(self.unmatched)} 处统计陈述无法追溯到本次分析结果，请人工核对。"
            )
        coverage = self.coverage_warning
        if coverage is not None:
            messages.append(coverage)
        return messages

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
            "claim_count": self.claim_count,
            "unmatched_count": len(self.unmatched),
            "claims": [asdict(claim) for claim in self.claims],
            "ran": self.ran,
            "not_run_reason": self.not_run_reason,
        }

    @classmethod
    def from_dict(cls, payload: object) -> StatisticsVerificationReport:
        """从 :meth:`to_dict` 的输出还原；结构不合法时抛 ``ValueError``。

        ``passed``、``claim_count``、``unmatched_count`` 均为派生字段，重建时由
        属性自动重算。``computed_effect_sizes`` / ``computed_sample_sizes`` 缺失时
        默认为空列表，以兼容旧版产物。``ran`` / ``not_run_reason`` 缺失时默认
        ``True`` / ``""``：Phase 9 之前的产物都是由**成功执行**的关口写出的，
        故默认与事实一致。
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
            ran=_optional_bool(data, "ran", cls.__name__, default=True),
            not_run_reason=_optional_str(data, "not_run_reason", cls.__name__),
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


def _optional_bool(
    data: dict, field: str, owner: str, *, default: bool
) -> bool:
    """缺失时返回 ``default``；存在时严格校验为 bool。"""
    if field not in data or data[field] is None:
        return default
    return _require_bool(data, field, owner)


def _optional_str(data: dict, field: str, owner: str) -> str:
    """缺失时返回空串；存在时严格校验为 str。"""
    if field not in data or data[field] is None:
        return ""
    return _require_str(data, field, owner)


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

    ``"Cohen's d"`` -> ``"d"``; ``"Pearson r"`` / ``"Spearman r"`` -> ``"r"``;
    ``"eta squared"`` -> ``"eta squared"`` (its own family). Names without a
    recognised symbol map to ``None`` and are therefore never eligible to satisfy
    an effect-size claim — the conservative choice that keeps the
    statement-level checks honest.
    """
    lowered = name.strip().lower()
    if not lowered:
        return None
    # ``eta squared`` / ``eta^2`` form their own family: they must never be
    # satisfied by a Cohen's d or an r (comparing across families would weaken
    # the check). A reported ``eta squared`` likewise only matches here.
    if "eta" in lowered:
        return "eta squared"
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
        A report whose ``passed`` flag is False when a claim cannot be traced,
        **and** whose ``claim_count`` / ``coverage_warning`` expose how much of
        the manuscript was actually recognised. A count of zero never silently
        passes: the report carries a coverage warning saying "未识别到任何统计
        陈述" (see :class:`StatisticsVerificationReport`).

    Notes:
        This is an advisory gate: an unmatched claim never raises. Statistics
        quoted from the *literature* (e.g. ``Smith reported d = 0.8 [P1]``)
        will be reported as unmatched — that is known and accepted. The returned
        report always has ``ran=True``; ``ran=False`` exists only for a caller
        that could not run the check at all.
    """
    # p 值同时纳入**原始值**与**多重比较校正值**：手稿可以合法地引用任一口径
    # （真实手稿就写成 ``Holm校正p=3.57e-16``，对应 ``p_value_adjusted`` 而非
    # 原始 ``p_value``）。两者都是本次运行**确实产生**的数，纳入识别面不放松强度
    # ——伪造的 p 值仍与两者都不匹配。校正值可能为 ``nan``（未做校正时），非有限
    # 值一律排除。
    computed_p_values = sorted(
        {
            float(value)
            for test in tests
            for value in (test.p_value, test.p_value_adjusted)
            if _is_finite(value)
        }
    )
    computed_sample_sizes = sorted({int(test.n) for test in tests if int(test.n) > 0})

    # Effect sizes are keyed by *family* so a reported ``d`` only ever compares
    # against a Cohen's d, never against a correlation's ``r`` (and vice versa);
    # ``eta squared`` is likewise its own family.
    families: dict[str, list[float]] = {"d": [], "r": [], "eta squared": []}
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
        # 归一化：``eta  squared`` / ``ETA SQUARED`` → ``eta squared``。
        symbol = re.sub(r"\s+", " ", match.group("name").strip().lower())
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
            families["d"] + families["r"] + families["eta squared"]
        ),
        computed_sample_sizes=computed_sample_sizes,
        claims=claims,
    )


def render_statistics_verification_markdown(
    report: StatisticsVerificationReport,
) -> str:
    """Render the traceability report as a Markdown artifact.

    产物**必须让覆盖率可见**：即便一切可追溯，也要写出"正文提取到的统计陈述"
    条数，使「只识别到 1 条却报通过」这类问题在产物里一眼可见。
    """
    lines = [
        "# 统计陈述追溯报告",
        "",
        f"- 是否执行: {'是' if report.ran else '否'}",
        f"- 结论: {'通过' if report.passed else '存在无法追溯的统计陈述'}",
        f"- 本次分析产生的 p 值: {len(report.computed_p_values)} 个",
        f"- 本次分析产生的效应量: {len(report.computed_effect_sizes)} 个",
        f"- 本次分析产生的样本量: {len(report.computed_sample_sizes)} 个",
        f"- 正文识别到的统计陈述条数（覆盖率）: {report.claim_count} 处",
        f"- 无法追溯: {len(report.unmatched)} 处",
        "",
    ]

    if not report.ran:
        lines.extend(
            [
                "## 本关口未执行",
                "",
                f"- {report.not_run_reason or '原因未知'}",
                "",
                "- 未执行不等于核验通过；请人工核对正文统计陈述。",
                "",
            ]
        )

    coverage = report.coverage_warning
    if coverage is not None:
        lines.extend(["## 覆盖率告警", "", f"- {coverage}", ""])

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
