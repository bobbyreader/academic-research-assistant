"""把可选的 CSV 数据集转换成可用于论文发表的图件。

设计第一原则：
1. 诚实——只画真实观测点及其不确定性，绝不编造数据或隐藏样本量 n；
2. 可复现——同样的输入、同样的随机种子必须生成完全一致的图件；
3. 可读——采用 Okabe-Ito 色盲友好配色、去除多余边框、字号清晰。

图件类型互补、各司其职（避免重复绘制同一张图）：
  * distribution   —— 逐数值列的直方图 + 核密度曲线（始终不分组）；
  * group_comparison —— 分组箱线图 + 个体观测点 + 中位数标记 + 每组 n
                        （仅当存在有效分组列时）；
  * correlation    —— 相关性最高的一对数值列的散点 + 最小二乘拟合 + 相关系数。

本模块始终以无界面（headless）模式运行：在导入 pyplot 之前先设置 Agg 后端，
因此在服务器 / CI 环境中也不会因缺少显示设备而失败。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# 必须在导入 pyplot 之前指定无界面后端，否则会尝试连接显示设备。
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (必须在 use("Agg") 之后导入)
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


class FigureBuildError(ValueError):
    """当无法安全地构建图件（例如输入文件不可读）时抛出。"""


# Okabe-Ito 色盲友好配色（8 色，十六进制）。
OKABE_ITO: tuple[str, ...] = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)

# 配色注册表，便于未来扩展。
_PALETTES: dict[str, tuple[str, ...]] = {"okabe_ito": OKABE_ITO}

# 分组列允许的组数上下限（与 statistics_engine 保持一致）。
_MIN_GROUPS = 2
_MAX_GROUPS = 12

# 判定「分类列」时允许的最大数值化比例：超过此比例视为数值列。
_NUMERIC_RATIO_THRESHOLD = 0.5


@dataclass
class FigureSpec:
    """描述一张已生成图件的元数据。"""

    figure_id: str  # 例如 "Figure 1"
    filename: str  # 例如 "figure_1_distribution.png"
    title: str
    caption: str
    figure_type: str  # "distribution" | "group_comparison" | "correlation"
    path: Path

    def to_dict(self) -> dict:
        """返回可直接序列化为 JSON 的字典（path 转为字符串）。"""
        return {
            "figure_id": self.figure_id,
            "filename": self.filename,
            "title": self.title,
            "caption": self.caption,
            "figure_type": self.figure_type,
            "path": str(self.path),
        }


@dataclass
class FigureBundle:
    """一次批量构建的结果：图件列表与警告信息。"""

    figures: list[FigureSpec] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """返回可直接序列化为 JSON 的字典。"""
        return {
            "figures": [figure.to_dict() for figure in self.figures],
            "warnings": list(self.warnings),
        }


def _resolve_palette(palette: str) -> tuple[str, ...]:
    """解析配色名，未知名称回退到默认的 Okabe-Ito。"""
    return _PALETTES.get(palette, OKABE_ITO)


def _configure_style() -> str:
    """配置全局绘图样式，返回实际生效的字体名。

    图件标题与坐标轴标签包含中文，因此字体首选必须是「能渲染中文字形」的字体，
    否则中文会渲染为方框（tofu）。策略：
      1. 先尝试 Arial 的替代品——但只在它本身能覆盖 CJK 时才用；
      2. 否则使用系统中第一个可用于 CJK 的字体（PingFang / Hiragino Sans GB /
         Heiti / Songti / Noto Sans CJK 等），这些字体同样包含完整拉丁字形；
      3. 全部缺失时回退到 matplotlib 自带的 DejaVu Sans（可能无法显示中文，
         但绝不抛错）。
    整个过程中绝不因缺少字体而抛错。
    """
    import matplotlib.font_manager as fm

    installed = {font.name for font in fm.fontManager.ttflist}

    def _present(name: str) -> bool:
        return name in installed

    # CJK 候选：覆盖 macOS / Windows / Linux 常见中文字体；这些字体同时含拉丁字形。
    cjk_candidates = [
        "Arial Unicode MS",  # 微软/Adobe 的 Arial 变体，含 CJK
        "PingFang SC",
        "PingFang HK",
        "Hiragino Sans GB",
        "Heiti SC",
        "Heiti TC",
        "STHeiti",
        "Microsoft YaHei",
        "SimHei",
        "Songti SC",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "WenQuanYi Zen Hei",
    ]
    cjk_fonts = [name for name in cjk_candidates if _present(name)]

    # 西文候选：Arial 优先，其次 Helvetica。
    latin_fonts = [name for name in ("Arial", "Helvetica") if _present(name)]

    # 关键：把能渲染中文的字体放在最前，确保中文不会变成方框；
    # 若系统缺少任何 CJK 字体，才退而使用 Arial，最后用 DejaVu Sans 兜底。
    if cjk_fonts:
        chain = cjk_fonts + latin_fonts + ["DejaVu Sans"]
    else:
        chain = latin_fonts + ["DejaVu Sans"]

    # 去重并保序。
    seen: list[str] = []
    for name in chain:
        if name not in seen:
            seen.append(name)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": seen,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.autolayout": False,
            "axes.unicode_minus": False,
        }
    )
    return seen[0] if seen else "DejaVu Sans"


def _despine(ax: Axes) -> None:
    """去除顶部与右侧边框，保持图形简洁可读。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _load_frame(path: Path) -> pd.DataFrame:
    """读取 CSV 为 DataFrame，失败时抛出 FigureBuildError。"""
    if not path.exists():
        raise FigureBuildError(f"数据文件不存在: {path}")
    if path.suffix.lower() != ".csv":
        raise FigureBuildError("当前仅支持 CSV 数据文件")
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except Exception as exc:  # noqa: BLE001 - 统一转换为领域异常
        raise FigureBuildError(f"无法读取 CSV 文件: {exc}") from exc
    if frame.shape[1] == 0:
        raise FigureBuildError("CSV 缺少列")
    return frame


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    """返回被识别为数值型的列名（按原列顺序）。"""
    return [name for name in frame.columns if pd.api.types.is_numeric_dtype(frame[name])]


def _categorical_candidates(frame: pd.DataFrame, numeric_columns: list[str]) -> list[str]:
    """返回候选分类列（与 statistics_engine 的判定口径一致）。

    规则：某列数值化比例 <= 0.5（即基本不是数值列），且去重后取值数介于
    2..12 之间。数值列本身不会被当作候选。
    """
    candidates: list[str] = []
    for column in frame.columns:
        if column in numeric_columns:
            continue
        raw = frame[column]
        non_missing = raw.dropna()
        if non_missing.empty:
            continue
        numeric_ratio = float(pd.to_numeric(non_missing, errors="coerce").notna().mean())
        if numeric_ratio > _NUMERIC_RATIO_THRESHOLD:
            continue
        levels = non_missing.astype(str).str.strip().nunique()
        if _MIN_GROUPS <= levels <= _MAX_GROUPS:
            candidates.append(str(column))
    return candidates


def _resolve_group_column(
    frame: pd.DataFrame,
    numeric_columns: list[str],
    explicit: str | None,
    warnings: list[str],
) -> str | None:
    """确定分组列：显式指定优先，否则在候选唯一时自动识别。

    与 statistics_engine 的行为保持一致——显式指定若不存在会抛错；自动识别时
    仅在恰好一个候选分类列时才采用，多个候选则记录警告并放弃（绝不静默跳过）。
    """
    if explicit is not None:
        if explicit not in frame.columns:
            raise FigureBuildError(f"指定的分组列不存在: {explicit}")
        return explicit

    candidates = _categorical_candidates(frame, numeric_columns)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        warnings.append(
            f"检测到多个候选分组列 {candidates}，已跳过分组相关图件；"
            "请通过 group_column 明确指定。"
        )
    return None


def _series_for(frame: pd.DataFrame, column: str) -> pd.Series:
    """返回去掉缺失值后的数值序列。"""
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _save_figure(
    fig: Figure,
    output_dir: Path,
    base_name: str,
    formats: tuple[str, ...],
    dpi: int,
) -> tuple[str, Path]:
    """把图形按各格式保存，返回 (展示文件名, 主路径)。

    文件名形如 ``figure_1_distribution.png``；多格式时主路径取第一种格式。

    为保证「可复现」：写入 PDF 时把 CreationDate/ModDate 置空，避免每次运行因
    嵌入的时间戳不同而产生字节差异（PNG 本身不含时间戳，天然确定）。
    """
    primary_name = f"{base_name}.{formats[0]}"
    primary_path = output_dir / primary_name
    for fmt in formats:
        target = output_dir / f"{base_name}.{fmt}"
        if fmt == "pdf":
            fig.savefig(
                target,
                dpi=dpi,
                bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None},
            )
        else:
            fig.savefig(target, dpi=dpi, bbox_inches="tight")
        if fmt == formats[0]:
            primary_path = target
    return primary_name, primary_path


def _kde_curve(values: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    """用高斯核密度估计在给定网格上平滑，不依赖 scipy。"""
    n = values.size
    if n < 2:
        return None
    std = float(np.std(values, ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return None
    # Silverman 经验法则选择带宽。
    bandwidth = 1.06 * std * n ** (-1.0 / 5.0)
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        return None
    diff = (grid[:, None] - values[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diff**2) / np.sqrt(2.0 * np.pi)
    density = kernel.sum(axis=1) / (n * bandwidth)
    return density


def _ordered_groups(frame: pd.DataFrame, group_column: str) -> list[str]:
    """返回分组列中去重后的取值字符串（保持原始出现顺序）。"""
    seen: list[str] = []
    for value in frame[group_column].astype(str):
        text = value.strip()
        if text and text.lower() != "nan" and text not in seen:
            seen.append(text)
    return seen


def _split_by_group(
    frame: pd.DataFrame, group_column: str, column: str, groups: list[str]
) -> list[np.ndarray]:
    """按给定分组顺序拆分某数值列，返回每组的有效数值数组。"""
    data: list[np.ndarray] = []
    for group in groups:
        subset = frame.loc[frame[group_column].astype(str).str.strip() == group, column]
        values = pd.to_numeric(subset, errors="coerce").dropna().to_numpy(dtype=float)
        data.append(values)
    return data


def _build_distribution_numeric(
    frame: pd.DataFrame, column: str, palette: tuple[str, ...]
) -> tuple[Figure, str, str] | str:
    """为单个数值列绘制直方图 + 叠加密度曲线。

    返回 ``(图, 标题, 列名)``；若无法绘制则返回跳过原因（字符串）。
    分布图始终不分组——分组信息由 group_comparison 承载，避免内容重复。
    """
    values = _series_for(frame, column)
    n = int(values.size)
    if n < 2 or float(values.std(ddof=1)) == 0.0:
        return f"列「{column}」为常量或样本不足，跳过分布图"

    data = values.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(
        data,
        bins="auto",
        density=True,
        color=palette[2],
        edgecolor="white",
        alpha=0.85,
        label=f"直方图 (n={n})",
    )

    grid = np.linspace(float(data.min()), float(data.max()), 256)
    density = _kde_curve(data, grid)
    if density is not None:
        ax.plot(grid, density, color=palette[5], linewidth=2.0, label="核密度估计")

    ax.set_xlabel(column)
    ax.set_ylabel("密度")
    title = f"{column} 的分布"
    ax.set_title(title)
    ax.legend(frameon=False)
    _despine(ax)
    return fig, title, column


def _build_group_comparison(
    frame: pd.DataFrame, columns: list[str], group_column: str, palette: tuple[str, ...]
) -> list[tuple[Figure, str, str]] | str:
    """分组比较图：箱线图 + 个体观测点 + 中位数标记 + 每组 n。

    对每个数值列返回一张图，元素为 ``(图, 标题, 列名)``；若分组不满足条件则返回
    跳过原因（字符串）。
    """
    groups = _ordered_groups(frame, group_column)
    if not (_MIN_GROUPS <= len(groups) <= _MAX_GROUPS):
        return (
            f"分组列「{group_column}」组数不在 {_MIN_GROUPS}..{_MAX_GROUPS} 之间，"
            "跳过分组比较图"
        )

    rng = np.random.default_rng(0)
    results: list[tuple[Figure, str, str]] = []
    for column in columns:
        data_per_group = _split_by_group(frame, group_column, column, groups)
        labels = [f"{group}\n(n={arr.size})" for group, arr in zip(groups, data_per_group)]

        non_empty = [arr for arr in data_per_group if arr.size > 0]
        if len(non_empty) < _MIN_GROUPS:
            continue

        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        positions = np.arange(1, len(groups) + 1)
        box = ax.boxplot(
            data_per_group,
            positions=positions,
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": palette[0], "linewidth": 1.6},
        )
        for i, patch in enumerate(box["boxes"]):
            patch.set_facecolor(palette[(i + 1) % len(palette)])
            patch.set_alpha(0.55)
            patch.set_edgecolor(palette[0])

        # 中位数标记点，突出稳健的中心位置。
        medians = [float(np.median(arr)) if arr.size else np.nan for arr in data_per_group]
        ax.scatter(
            positions,
            medians,
            marker="D",
            s=36,
            color=palette[0],
            zorder=4,
            label="中位数",
        )

        # 叠加个体观测点（轻微抖动），明确展示真实样本。
        for i, arr in enumerate(data_per_group):
            if arr.size == 0:
                continue
            jitter = rng.uniform(-0.12, 0.12, size=arr.size)
            ax.scatter(
                positions[i] + jitter,
                arr,
                s=18,
                color=palette[6],
                edgecolor="white",
                linewidth=0.4,
                alpha=0.85,
                zorder=3,
                label="个体观测" if i == 0 else None,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel(group_column)
        ax.set_ylabel(column)
        title = f"{column} 按 {group_column} 的分组比较"
        ax.set_title(title)
        ax.legend(frameon=False)
        _despine(ax)
        results.append((fig, title, column))

    if not results:
        return f"分组列「{group_column}」下没有可用的数值分组，跳过比较图"
    return results


def _build_correlation(
    frame: pd.DataFrame, columns: list[str], palette: tuple[str, ...]
) -> tuple[Figure, str, tuple[str, str], float] | str:
    """相关性散点图：选取相关性最高的一对数值列，绘散点 + 最小二乘拟合线 + 相关系数。

    返回 ``(图, 标题, (列一, 列二), r)``；无法绘制时返回跳过原因（字符串）。
    """
    usable = [name for name in columns if _series_for(frame, name).size >= 2]
    if len(usable) < 2:
        return "数值列不足两列，跳过相关图"

    corr = frame[usable].corr()
    best_pair: tuple[str, str] | None = None
    best_abs = -1.0
    for i, first in enumerate(usable):
        for second in usable[i + 1 :]:
            value = corr.loc[first, second]
            if pd.isna(value):
                continue
            if abs(float(value)) > best_abs:
                best_abs = abs(float(value))
                best_pair = (first, second)
    if best_pair is None:
        return "未能计算有效的相关系数，跳过相关图"

    first, second = best_pair
    paired = frame[[first, second]].apply(pd.to_numeric, errors="coerce").dropna()
    n = int(len(paired))
    if n < 2 or paired[first].std(ddof=1) == 0.0 or paired[second].std(ddof=1) == 0.0:
        return "相关列方差为零或样本不足，跳过相关图"

    x = paired[first].to_numpy(dtype=float)
    y = paired[second].to_numpy(dtype=float)
    r = float(np.corrcoef(x, y)[0, 1])

    slope, intercept = np.polyfit(x, y, 1)
    line_x = np.linspace(float(x.min()), float(x.max()), 100)
    line_y = slope * line_x + intercept

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.scatter(
        x,
        y,
        s=26,
        color=palette[5],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
        label=f"个体观测 (n={n})",
    )
    ax.plot(line_x, line_y, color=palette[6], linewidth=2.0, label="最小二乘拟合")
    ax.text(
        0.04,
        0.96,
        f"r = {r:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
    )
    ax.set_xlabel(first)
    ax.set_ylabel(second)
    title = f"{first} 与 {second} 的相关性"
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    _despine(ax)
    return fig, title, (first, second), r


def _next_id(index: int) -> str:
    """按序号生成稳定的图件编号。"""
    return f"Figure {index}"


def build_figures(
    path: Path,
    output_dir: Path,
    *,
    group_column: str | None = None,
    dpi: int = 300,
    palette: str = "okabe_ito",
    formats: tuple[str, ...] = ("png",),
) -> FigureBundle:
    """把 CSV 数据集转换为论文可用的图件集合。

    参数：
        path: 输入 CSV 路径。
        output_dir: 输出目录，不存在时自动创建。
        group_column: 可选的分组列名。为 None 时会像 statistics_engine 那样自动
            识别唯一的分类列（取值 2..12）；若存在多个候选则记录警告并跳过。
        dpi: 位图分辨率。
        palette: 配色名，默认 Okabe-Ito 色盲友好配色。
        formats: 需要导出的文件格式，例如 ("png", "pdf")。

    返回：
        FigureBundle，包含全部已生成图件的元数据与跳过原因（warnings）。
        对无法构建的图件（常量列、组数不足等）仅记录警告，不抛异常；绝不会在
        存在可用信息的情况下静默跳过。
    """
    if not formats:
        formats = ("png",)

    frame = _load_frame(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_style()
    colors = _resolve_palette(palette)

    numeric_columns = _numeric_columns(frame)

    bundle = FigureBundle()
    if not numeric_columns:
        bundle.warnings.append("数据集中没有可用的数值列，未生成任何图件")

    # 分组列：显式指定优先，否则自动识别（与 statistics_engine 口径一致）。
    resolved_group = _resolve_group_column(frame, numeric_columns, group_column, bundle.warnings)
    if resolved_group is None and group_column is not None:
        bundle.warnings.append(
            f"分组列「{group_column}」不可用或无法用于分组，跳过分组相关图件。"
        )
    elif resolved_group is None and group_column is None:
        candidates = _categorical_candidates(frame, numeric_columns)
        if not candidates:
            bundle.warnings.append(
                "未检测到可用于分组的分类列（需要 2..12 个取值），跳过分组相关图件。"
            )

    figures: list[FigureSpec] = []

    def _register(
        fig: Figure,
        figure_type: str,
        title: str,
        caption: str,
        slug: str,
    ) -> None:
        """保存图形并登记元数据。"""
        figure_id = _next_id(len(figures) + 1)
        base_name = f"figure_{len(figures) + 1}_{slug}"
        filename, primary_path = _save_figure(fig, output_dir, base_name, formats, dpi)
        plt.close(fig)
        figures.append(
            FigureSpec(
                figure_id=figure_id,
                filename=filename,
                title=title,
                caption=caption,
                figure_type=figure_type,
                path=primary_path,
            )
        )

    # 1) 分布图：逐数值列的直方图 + 核密度曲线，始终不分组。
    for column in numeric_columns:
        result = _build_distribution_numeric(frame, column, colors)
        if isinstance(result, str):
            bundle.warnings.append(result)
            continue
        fig, title, col_name = result
        n = int(_series_for(frame, col_name).size)
        caption = (
            f"图为「{col_name}」的分布，样本量 n={n}；直方图展示观测频数密度，"
            f"曲线为核密度估计（Silverman 带宽的平滑估计，属于平滑选择而非直接测量），"
            f"均为个体观测的汇总。"
        )
        _register(fig, "distribution", title, caption, "distribution")

    # 2) 分组比较图：仅当存在有效分组列时生成。
    if resolved_group is not None:
        comparison = _build_group_comparison(frame, numeric_columns, resolved_group, colors)
        if isinstance(comparison, str):
            bundle.warnings.append(comparison)
        else:
            for fig, title, col_name in comparison:
                n = int(_series_for(frame, col_name).size)
                caption = (
                    f"图为「{col_name}」按「{resolved_group}」的分组比较，样本量 n={n}；"
                    f"箱体展示中位数与四分位距，菱形为各组中位数，散点为个体观测。"
                )
                _register(fig, "group_comparison", title, caption, "group_comparison")

    # 3) 相关图：仅当至少两个数值列时生成。
    correlation = _build_correlation(frame, numeric_columns, colors)
    if isinstance(correlation, str):
        bundle.warnings.append(correlation)
    else:
        fig, title, pair, r = correlation
        first, second = pair
        paired = frame[[first, second]].apply(pd.to_numeric, errors="coerce").dropna()
        n = int(len(paired))
        caption = (
            f"图为「{first}」与「{second}」的相关性，有效样本量 n={n}；"
            f"散点为个体观测，直线为最小二乘拟合，相关系数 r={r:.3f}。"
        )
        _register(fig, "correlation", title, caption, "correlation")

    bundle.figures = figures
    return bundle
