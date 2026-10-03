"""Defensible inferential statistics for optional experimental data.

设计原则（会被审计）：
- 不声称因果；
- 只运行数据结构支持的检验；
- 报告效应量与样本量，绝不只报告 p 值；
- 对多重比较做校正；
- 显式声明需要作者判断的假设。

该模块接收一个 CSV，自动识别数值列与分类列，并在可辩护的前提下运行
Welch t 检验、单因素方差分析（ANOVA）以及 Pearson / Spearman 相关。
所有结果都是确定性的（不含随机性），并同时给出原始 p 值与校正后的 p 值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# 显著性水平与校正方法的允许取值
_ALPHA = 0.05
_CORRECTION_METHODS = {"holm", "none"}

# 自动识别分类列时的取值数量范围
_MIN_CATEGORICAL_LEVELS = 2
_MAX_CATEGORICAL_LEVELS = 12

# 一个数值列至少要有的有效观测数
_MIN_NUMERIC_VALUES = 3

# 单组 / 整体运行检验所需的最小样本量
_MIN_GROUP_N = 3

# 描述 Cohen's d 量级时使用的阈值
_EFFECT_SIZE_THRESHOLDS = {"small": 0.2, "medium": 0.5, "large": 0.8}


class StatisticsError(ValueError):
    """当可选数据文件无法被安全地做统计推断时抛出。"""


@dataclass
class StatTestResult:
    """单个统计检验的结果（含效应量、样本量与假设声明）。"""

    test_name: str
    variables: list[str]
    groups: list[str] = field(default_factory=list)
    n: int = 0
    statistic: float = float("nan")
    p_value: float = float("nan")
    p_value_adjusted: float = float("nan")
    effect_size: float = float("nan")
    effect_size_name: str = ""
    ci_low: float | None = None
    ci_high: float | None = None
    assumptions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """返回可序列化的字典，便于写入报告或做快照比较。"""
        return {
            "test_name": self.test_name,
            "variables": list(self.variables),
            "groups": list(self.groups),
            "n": self.n,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "p_value_adjusted": self.p_value_adjusted,
            "effect_size": self.effect_size,
            "effect_size_name": self.effect_size_name,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "assumptions": list(self.assumptions),
            "warnings": list(self.warnings),
        }

    def to_markdown_row(self) -> str:
        """渲染为 Markdown 表格的一行（不含表头）。"""

        def fmt(value: float) -> str:
            return "NA" if value is None or math.isnan(value) else f"{value:.4g}"

        ci = "NA"
        if self.ci_low is not None and self.ci_high is not None:
            ci = f"[{self.ci_low:.4g}, {self.ci_high:.4g}]"
        variables = ", ".join(self.variables) if self.variables else "NA"
        groups = ", ".join(self.groups) if self.groups else "NA"
        return (
            f"| {self.test_name} | {variables} | {groups} | {self.n} | "
            f"{fmt(self.statistic)} | {fmt(self.p_value)} | {fmt(self.p_value_adjusted)} | "
            f"{fmt(self.effect_size)} ({self.effect_size_name}) | {ci} |"
        )


@dataclass
class StatisticsReport:
    """对一份 CSV 的完整推断统计报告。"""

    rows: int
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    group_column: str | None = None
    alpha: float = _ALPHA
    correction_method: str = "holm"
    tests: list[StatTestResult] = field(default_factory=list)
    author_checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """返回可序列化的字典。"""
        return {
            "rows": self.rows,
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "group_column": self.group_column,
            "alpha": self.alpha,
            "correction_method": self.correction_method,
            "tests": [test.to_dict() for test in self.tests],
            "author_checks": list(self.author_checks),
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        """渲染为 Markdown 报告（含假设列表与作者核对清单）。"""
        lines: list[str] = ["# 推断统计分析报告", ""]
        lines.append(f"- 样本行数: {self.rows}")
        lines.append(f"- 显著性水平 alpha: {self.alpha}")
        lines.append(f"- 多重比较校正方法: {self.correction_method}")
        lines.append(f"- 数值列: {', '.join(self.numeric_columns) or '无'}")
        lines.append(f"- 分类列: {', '.join(self.categorical_columns) or '无'}")
        lines.append(f"- 分组列: {self.group_column or '未指定'}")
        lines.append("")
        lines.append(
            "| 检验 | 变量 | 分组 | n | 统计量 | p | p(校正) | 效应量 | 95% CI |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        if self.tests:
            for test in self.tests:
                lines.append(test.to_markdown_row())
        else:
            lines.append("| （未运行任何检验） | NA | NA | 0 | NA | NA | NA | NA | NA |")
        lines.append("")

        if self.warnings:
            lines.append("## 警告")
            lines.append("")
            for warning in self.warnings:
                lines.append(f"- {warning}")
            lines.append("")

        if self.author_checks:
            lines.append("## 作者需核对的事项")
            lines.append("")
            for check in self.author_checks:
                lines.append(f"- {check}")
            lines.append("")

        return "\n".join(lines)


def analyze_statistics(
    path: Path,
    *,
    group_column: str | None = None,
    alpha: float = _ALPHA,
    correction_method: str = "holm",
) -> StatisticsReport:
    """读取 CSV 并运行可辩护的推断统计。

    参数：
        path: CSV 文件路径。
        group_column: 若给出则强制作为分组列；否则按规则自动识别。
        alpha: 显著性水平（默认 0.05）。
        correction_method: 多重比较校正方法，"holm" 或 "none"。

    返回：
        StatisticsReport —— 含检验结果、假设列表与作者核对清单。
    """
    if correction_method not in _CORRECTION_METHODS:
        raise StatisticsError(
            f"不支持的校正方法: {correction_method}；可选: {sorted(_CORRECTION_METHODS)}"
        )
    if not (0 < alpha < 1):
        raise StatisticsError(f"alpha 必须位于 (0, 1) 之间，当前为 {alpha}")

    dataframe = _load_csv(path)
    rows = len(dataframe)

    numeric_columns = _detect_numeric_columns(dataframe)
    categorical_columns = _detect_categorical_columns(dataframe)
    warnings: list[str] = []

    resolved_group = _resolve_group_column(
        dataframe, numeric_columns, categorical_columns, group_column, warnings
    )

    tests: list[StatTestResult] = []
    if resolved_group is not None:
        for column in numeric_columns:
            result = _run_group_test(dataframe, resolved_group, column, warnings)
            if result is not None:
                tests.append(result)

    for column_a, column_b in _numeric_pairs(numeric_columns):
        result = _run_correlation(dataframe, column_a, column_b, warnings)
        if result is not None:
            tests.append(result)

    if not tests:
        warnings.append("未找到可辩护的检验组合；请检查数据结构或手动指定分组列。")

    _apply_multiple_comparison_correction(tests, correction_method)

    report = StatisticsReport(
        rows=rows,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        group_column=resolved_group,
        alpha=alpha,
        correction_method=correction_method,
        tests=tests,
        author_checks=_author_checks(),
        warnings=warnings,
    )
    return report


# --------------------------------------------------------------------------- #
# 数据读取与列识别
# --------------------------------------------------------------------------- #
def _load_csv(path: Path) -> pd.DataFrame:
    """读取 CSV 并按列尽量转为数值（保留原始列名）。"""
    if not path.exists():
        raise StatisticsError(f"数据文件不存在: {path}")
    if path.suffix.lower() != ".csv":
        raise StatisticsError("当前仅支持 CSV 数据文件")
    try:
        dataframe = pd.read_csv(path, encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        raise StatisticsError(f"无法解析 CSV 文件: {error}") from error
    if dataframe.shape[1] == 0:
        raise StatisticsError("CSV 缺少数据列")
    # 仅对“以数值为主”的列做数值化，避免把分类文本列整体破坏为 NaN。
    for column in dataframe.columns:
        raw = dataframe[column]
        coerced = pd.to_numeric(raw, errors="coerce")
        non_missing = raw.notna()
        if not non_missing.any():
            continue
        numeric_ratio = float(coerced[non_missing].notna().mean())
        if numeric_ratio >= 1.0:
            dataframe[column] = coerced
    return dataframe


def _valid_values(series: pd.Series) -> np.ndarray:
    """返回某列的非缺失有限数值。"""
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return numeric[np.isfinite(numeric)]


def _detect_numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    """数值列：非缺失有限值数量 >= 3。"""
    detected: list[str] = []
    for column in dataframe.columns:
        if _valid_values(dataframe[column]).size >= _MIN_NUMERIC_VALUES:
            detected.append(str(column))
    return detected


def _detect_categorical_columns(dataframe: pd.DataFrame) -> list[str]:
    """分类列：在数值化后仍为非数值、且取值数介于 2..12 的列。"""
    detected: list[str] = []
    for column in dataframe.columns:
        raw = dataframe[column]
        # 如果该列能数值化的比例较高，则视为数值列，不当作分类列。
        numeric_ratio = float(np.mean(np.isfinite(pd.to_numeric(raw, errors="coerce")))) if len(raw) else 0.0
        if numeric_ratio > 0.5:
            continue
        categories = pd.Series(raw).dropna().astype(str)
        levels = categories.nunique()
        if _MIN_CATEGORICAL_LEVELS <= levels <= _MAX_CATEGORICAL_LEVELS:
            detected.append(str(column))
    return detected


def _resolve_group_column(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    group_column: str | None,
    warnings: list[str],
) -> str | None:
    """确定分组列：显式指定优先，否则仅当恰好一个分类列时自动选择。"""
    if group_column is not None:
        if group_column not in dataframe.columns:
            raise StatisticsError(f"指定的分组列不存在: {group_column}")
        return group_column
    candidates = [column for column in categorical_columns if column not in numeric_columns]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        warnings.append(
            f"检测到多个候选分组列 {candidates}，已跳过分组比较；"
            "请通过 group_column 明确指定。"
        )
    return None


def _numeric_pairs(numeric_columns: list[str]) -> list[tuple[str, str]]:
    """返回所有数值列的两两组合。"""
    pairs: list[tuple[str, str]] = []
    for index, column_a in enumerate(numeric_columns):
        for column_b in numeric_columns[index + 1 :]:
            pairs.append((column_a, column_b))
    return pairs


# --------------------------------------------------------------------------- #
# 分组检验
# --------------------------------------------------------------------------- #
def _split_groups(
    dataframe: pd.DataFrame, group_column: str, value_column: str
) -> dict[str, np.ndarray]:
    """按分组列拆分数值列，成对删除缺失值（该检验内完整观测）。"""
    subset = dataframe[[group_column, value_column]].copy()
    subset[value_column] = pd.to_numeric(subset[value_column], errors="coerce")
    subset = subset.dropna(subset=[group_column, value_column])
    groups: dict[str, np.ndarray] = {}
    for level, chunk in subset.groupby(group_column, sort=True):
        groups[str(level)] = chunk[value_column].to_numpy(dtype=float)
    return groups


def _run_group_test(
    dataframe: pd.DataFrame,
    group_column: str,
    value_column: str,
    warnings: list[str],
) -> StatTestResult | None:
    """按可辩护性运行 Welch t 检验或单因素 ANOVA。"""
    groups = _split_groups(dataframe, group_column, value_column)
    groups = {name: values for name, values in groups.items() if values.size > 0}

    if len(groups) < 2:
        warnings.append(
            f"[{group_column} × {value_column}] 可用分组不足 2 个，已跳过检验。"
        )
        return None

    # 丢弃样本量过小的分组，并记录警告。
    kept: dict[str, np.ndarray] = {}
    for name, values in groups.items():
        if values.size < _MIN_GROUP_N:
            warnings.append(
                f"[{group_column} × {value_column}] 分组 '{name}' 仅 n={values.size}，"
                f"低于最小样本量 {_MIN_GROUP_N}，已从该检验中剔除。"
            )
        else:
            kept[name] = values

    if len(kept) < 2:
        warnings.append(
            f"[{group_column} × {value_column}] 剔除小样本分组后可用分组不足 2 个，已跳过检验。"
        )
        return None

    # 常量列无法支持推断检验。
    all_values = np.concatenate(list(kept.values()))
    if all_values.size > 1 and float(np.ptp(all_values)) == 0.0:
        warnings.append(
            f"[{group_column} × {value_column}] 该数值列在可用数据上为常量，已跳过检验。"
        )
        return None

    group_names = list(kept.keys())
    samples = [kept[name] for name in group_names]
    assumptions = _assumption_notes(samples, value_column)

    if len(kept) == 2:
        sample_a, sample_b = samples
        n = int(sample_a.size + sample_b.size)
        statistic, p_value = stats.ttest_ind(sample_a, sample_b, equal_var=False)
        effect_size = _cohens_d(sample_a, sample_b)
        ci_low, ci_high = _mean_difference_ci(sample_a, sample_b)
        result = StatTestResult(
            test_name="Welch t-test",
            variables=[value_column],
            groups=group_names,
            n=n,
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_size_name="Cohen's d",
            ci_low=ci_low,
            ci_high=ci_high,
            assumptions=list(assumptions),
        )
    else:
        statistic, p_value = stats.f_oneway(*samples)
        n = int(all_values.size)
        effect_size = _eta_squared(samples)
        result = StatTestResult(
            test_name="One-way ANOVA",
            variables=[value_column],
            groups=group_names,
            n=n,
            statistic=float(statistic),
            p_value=float(p_value),
            effect_size=float(effect_size),
            effect_size_name="eta squared",
            ci_low=None,
            ci_high=None,
            assumptions=list(assumptions),
        )

    # 不自动改用非参数检验（否则会改变估计目标）；改为显式升级为警告。
    if any(not _passes_shapiro(sample) for sample in samples):
        alternative = "Mann-Whitney U 检验" if len(kept) == 2 else "Kruskal-Wallis 检验"
        result.warnings.append(
            f"至少一个分组的 Shapiro-Wilk 检验拒绝正态性；已按参数检验报告，"
            f"建议考虑非参数替代（{alternative}）。"
        )

    if not math.isfinite(result.statistic):
        result.warnings.append("检验统计量非有限值，结果不可用。")
    return result


def _assumption_notes(samples: list[np.ndarray], value_column: str) -> list[str]:
    """记录该组检验所需的假设，并附上 Shapiro-Wilk 正态性结果。"""
    notes: list[str] = []
    for index, sample in enumerate(samples):
        notes.append(f"分组 {index + 1}（n={sample.size}）：{_shapiro_note(sample)}")
    notes.append("假定各组观测相互独立；数据中的重复测量/聚类结构未被建模。")
    notes.append(f"'{value_column}' 被当作连续数值变量处理，请确认其量纲与测量水平。")
    return notes


def _shapiro_note(sample: np.ndarray) -> str:
    """返回 Shapiro-Wilk 正态性检验的一句话说明。"""
    if sample.size < 3:
        return "样本量不足以做 Shapiro-Wilk 正态性检验。"
    if float(np.ptp(sample)) == 0.0:
        return "数据为常量，Shapiro-Wilk 不可用。"
    try:
        statistic, p_value = stats.shapiro(sample)
    except ValueError:
        return "Shapiro-Wilk 正态性检验无法计算。"
    verdict = "未拒绝正态性 (p>0.05)" if p_value > 0.05 else "拒绝正态性 (p<=0.05)"
    return f"Shapiro-Wilk W={statistic:.4g}, p={p_value:.4g}，{verdict}。"


def _cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """计算独立样本 Cohen's d（使用合并标准差）。"""
    n_a, n_b = sample_a.size, sample_b.size
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = float(np.var(sample_a, ddof=1))
    var_b = float(np.var(sample_b, ddof=1))
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_var <= 0:
        return float("nan")
    return float((np.mean(sample_a) - np.mean(sample_b)) / math.sqrt(pooled_var))


def _mean_difference_ci(
    sample_a: np.ndarray, sample_b: np.ndarray, confidence: float = 0.95
) -> tuple[float | None, float | None]:
    """给出两组均值差的 Welch 置信区间。"""
    n_a, n_b = sample_a.size, sample_b.size
    if n_a < 2 or n_b < 2:
        return None, None
    mean_diff = float(np.mean(sample_a) - np.mean(sample_b))
    se = math.sqrt(float(np.var(sample_a, ddof=1)) / n_a + float(np.var(sample_b, ddof=1)) / n_b)
    if se <= 0:
        return None, None
    # Welch–Satterthwaite 自由度
    term_a = float(np.var(sample_a, ddof=1)) / n_a
    term_b = float(np.var(sample_b, ddof=1)) / n_b
    df = (term_a + term_b) ** 2 / (
        (term_a**2) / (n_a - 1) + (term_b**2) / (n_b - 1)
    )
    critical = stats.t.ppf(0.5 + confidence / 2, df)
    return mean_diff - critical * se, mean_diff + critical * se


def _eta_squared(samples: list[np.ndarray]) -> float:
    """计算单因素 ANOVA 的 eta squared；主要由组间平方和贡献。"""
    all_values = np.concatenate(samples)
    grand_mean = float(np.mean(all_values))
    ss_between = sum(sample.size * (float(np.mean(sample)) - grand_mean) ** 2 for sample in samples)
    ss_total = float(np.sum((all_values - grand_mean) ** 2))
    if ss_total <= 0:
        return float("nan")
    return float(ss_between / ss_total)


# --------------------------------------------------------------------------- #
# 相关分析
# --------------------------------------------------------------------------- #
def _run_correlation(
    dataframe: pd.DataFrame,
    column_a: str,
    column_b: str,
    warnings: list[str],
) -> StatTestResult | None:
    """根据正态性检验结果选择 Pearson 或 Spearman 相关。"""
    subset = dataframe[[column_a, column_b]].apply(pd.to_numeric, errors="coerce")
    subset = subset.dropna()
    n = len(subset)
    if n < 3:
        warnings.append(
            f"[{column_a} × {column_b}] 完整观测仅 {n} 条，少于 3，已跳过相关分析。"
        )
        return None

    values_a = subset[column_a].to_numpy(dtype=float)
    values_b = subset[column_b].to_numpy(dtype=float)
    if float(np.ptp(values_a)) == 0.0 or float(np.ptp(values_b)) == 0.0:
        warnings.append(
            f"[{column_a} × {column_b}] 存在常量列，无法计算相关，已跳过。"
        )
        return None

    note_a = _shapiro_note(values_a)
    note_b = _shapiro_note(values_b)
    normal_a = _passes_shapiro(values_a)
    normal_b = _passes_shapiro(values_b)

    if normal_a and normal_b:
        test_name = "Pearson correlation"
        effect_size_name = "Pearson r"
        statistic, p_value = stats.pearsonr(values_a, values_b)
        # Fisher z 置信区间依赖双变量正态性，仅对 Pearson 有效。
        ci_low, ci_high = _fisher_z_ci(float(statistic), n)
    else:
        test_name = "Spearman correlation"
        effect_size_name = "Spearman r"
        statistic, p_value = stats.spearmanr(values_a, values_b)
        # Spearman rho 不满足 Fisher z 的正态性前提，故不报告置信区间。
        ci_low, ci_high = None, None

    assumptions = [
        f"'{column_a}' 正态性：{note_a}",
        f"'{column_b}' 正态性：{note_b}",
        "假定观测相互独立；相关不等于因果。",
    ]
    return StatTestResult(
        test_name=test_name,
        variables=[column_a, column_b],
        groups=[],
        n=int(n),
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size=float(statistic),
        effect_size_name=effect_size_name,
        ci_low=ci_low,
        ci_high=ci_high,
        assumptions=assumptions,
    )


def _passes_shapiro(sample: np.ndarray) -> bool:
    """样本量足够且 Shapiro-Wilk 未拒绝正态性时返回 True。"""
    if sample.size < 3 or float(np.ptp(sample)) == 0.0:
        return False
    try:
        _, p_value = stats.shapiro(sample)
    except ValueError:
        return False
    return bool(p_value > 0.05)


def _fisher_z_ci(r: float, n: int, confidence: float = 0.95) -> tuple[float | None, float | None]:
    """通过 Fisher z 变换给出相关系数的置信区间。"""
    if n < 4 or not math.isfinite(r) or abs(r) >= 1.0:
        return None, None
    z = math.atanh(r)
    se = 1.0 / math.sqrt(n - 3)
    critical = stats.norm.ppf(0.5 + confidence / 2)
    return float(math.tanh(z - critical * se)), float(math.tanh(z + critical * se))


# --------------------------------------------------------------------------- #
# 多重比较校正
# --------------------------------------------------------------------------- #
def _apply_multiple_comparison_correction(
    tests: list[StatTestResult], method: str
) -> None:
    """就地对所有原始 p 值做多重比较校正，写回 p_value_adjusted。"""
    p_values = [test.p_value for test in tests]
    adjusted = _holm_adjust(p_values) if method == "holm" else list(p_values)
    for test, value in zip(tests, adjusted):
        test.p_value_adjusted = float(value)


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni 逐步下降法；保持输入顺序，输出受 [0, 1] 约束。"""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, index in enumerate(order):
        value = p_values[index] * (m - rank)
        running_max = max(running_max, value)
        adjusted[index] = min(1.0, running_max)
    return adjusted


# --------------------------------------------------------------------------- #
# 作者核对清单
# --------------------------------------------------------------------------- #
def _author_checks() -> list[str]:
    """返回必须由作者根据研究设计确认的事项（中文）。"""
    return [
        "检验是自动选择的，必须结合研究设计（自变量、因变量、分组方式）人工确认后方可引用。",
        "本引擎未对重复测量或聚类数据建模；若同一受试者/单位出现多次，请改用混合效应或重复测量模型。",
        "多重比较校正默认使用 Holm-Bonferroni，并假定各检验相互独立或弱相关；"
        "若检验间高度相关，请重新评估校正方法的适用性。",
        "所有检验均假定观测相互独立；请确认抽样与实验设计满足该假设。",
        "显著性与效应量仅描述样本内的关联，不构成因果证据。",
        "报告的 n 为每个检验实际使用的完整观测数（成对删除缺失值），各检验的 n 可能不同。",
        "未执行任何事后（post-hoc）两两比较：方差分析显著只能说明各组总体均值存在差异，"
        "不能据此推断某一组高于另一组；任何两两排序结论都必须使用专门的事后检验（如 Tukey HSD）。",
        "当 Shapiro-Wilk 检验拒绝正态性时，本引擎仍按参数检验报告（Welch t / ANOVA）并在对应结果中给出警告；"
        "请结合研究设计与样本量考虑非参数替代（两组用 Mann-Whitney U，三组及以上用 Kruskal-Wallis）。",
    ]
