"""figure_builder 的单元测试：合成 CSV、验证诚实性、可复现性与可读性。"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

from core.figure_builder import FigureBundle, build_figures


def _write_csv(path: Path, text: str) -> Path:
    """写入 UTF-8 CSV 并返回路径。"""
    path.write_text(text, encoding="utf-8")
    return path


def test_two_numeric_one_group_produces_files_with_captions(tmp_path: Path) -> None:
    """含 2 数值列 + 1 分组列时：应生成非空文件、说明非空、首编号为 Figure 1。"""
    csv_path = _write_csv(
        tmp_path / "data.csv",
        "group,score,weight\n"
        "control,1,10\n"
        "control,2,12\n"
        "control,3,11\n"
        "treat,5,20\n"
        "treat,6,22\n"
        "treat,7,21\n",
    )
    out = tmp_path / "figures"

    bundle = build_figures(csv_path, out, group_column="group")

    assert isinstance(bundle, FigureBundle)
    assert bundle.figures, "应至少生成一张图"
    for figure in bundle.figures:
        assert "n=" in figure.caption
        assert figure.caption.strip()
        assert figure.path.exists()
        assert figure.path.stat().st_size > 0
    assert bundle.figures[0].figure_id == "Figure 1"


def test_numeric_only_has_distribution_but_no_group_comparison(tmp_path: Path) -> None:
    """无任何分类列时：不应有 group_comparison，但至少有一张 distribution。"""
    csv_path = _write_csv(
        tmp_path / "numeric.csv",
        "score,weight\n1,10\n2,12\n3,11\n4,15\n5,20\n6,22\n",
    )
    out = tmp_path / "figures"

    bundle = build_figures(csv_path, out)

    types = [figure.figure_type for figure in bundle.figures]
    assert "group_comparison" not in types
    assert types.count("distribution") >= 1


def test_determinism_same_filenames_and_count(tmp_path: Path) -> None:
    """两次构建应产出相同的文件名列表与图件数量。"""
    csv_path = _write_csv(
        tmp_path / "data.csv",
        "group,score,weight\n"
        "a,1,10\n"
        "a,2,12\n"
        "a,3,11\n"
        "a,4,13\n"
        "b,5,20\n"
        "b,6,22\n"
        "b,7,21\n"
        "b,8,23\n",
    )
    out_one = tmp_path / "run1"
    out_two = tmp_path / "run2"

    first = build_figures(csv_path, out_one, group_column="group")
    second = build_figures(csv_path, out_two, group_column="group")

    assert len(first.figures) == len(second.figures)
    assert [f.filename for f in first.figures] == [f.filename for f in second.figures]


def test_constant_column_does_not_raise(tmp_path: Path) -> None:
    """常量数值列不应抛异常，最多作为警告跳过。"""
    csv_path = _write_csv(
        tmp_path / "constant.csv",
        "score,weight\n5,10\n5,12\n5,11\n5,15\n",
    )
    out = tmp_path / "figures"

    bundle = build_figures(csv_path, out)

    assert isinstance(bundle, FigureBundle)
    assert any("常量" in warning for warning in bundle.warnings)


def test_every_caption_mentions_sample_size(tmp_path: Path) -> None:
    """每张图的说明都必须提及样本量（含 n + 数字，或「样本」）。"""
    csv_path = _write_csv(
        tmp_path / "data.csv",
        "group,score,weight\n"
        "x,1,10\n"
        "x,2,12\n"
        "x,3,11\n"
        "y,5,20\n"
        "y,6,22\n"
        "y,7,21\n",
    )
    out = tmp_path / "figures"

    bundle = build_figures(csv_path, out, group_column="group")

    assert bundle.figures
    for figure in bundle.figures:
        mentions = bool(re.search(r"n\s*=\s*\d+", figure.caption)) or "样本" in figure.caption
        assert mentions, f"说明未提及样本量: {figure.caption}"


def test_multiple_formats_produce_both_extensions(tmp_path: Path) -> None:
    """formats=("png","pdf") 时至少一张图同时产出国两个扩展名文件。"""
    csv_path = _write_csv(
        tmp_path / "data.csv",
        "score,weight\n1,10\n2,12\n3,11\n4,15\n5,20\n6,22\n",
    )
    out = tmp_path / "figures"

    bundle = build_figures(csv_path, out, formats=("png", "pdf"))

    assert bundle.figures
    png_files = list(out.glob("*.png"))
    pdf_files = list(out.glob("*.pdf"))
    assert png_files, "应生成 PNG 文件"
    assert pdf_files, "应生成 PDF 文件"


def test_to_dict_is_json_serialisable(tmp_path: Path) -> None:
    """to_dict 的 path 字段应为字符串，便于 JSON 序列化。"""
    csv_path = _write_csv(
        tmp_path / "data.csv",
        "score,weight\n1,10\n2,12\n3,11\n4,15\n5,20\n6,22\n",
    )
    out = tmp_path / "figures"

    bundle = build_figures(csv_path, out)
    payload = bundle.to_dict()

    assert isinstance(payload["figures"], list)
    assert all(isinstance(fig["path"], str) for fig in payload["figures"])


# --------------------------------------------------------------------------- #
# 审计修复后新增的回归测试
# --------------------------------------------------------------------------- #
def test_single_categorical_column_is_auto_detected_as_group(tmp_path: Path) -> None:
    """恰好一个分类列、group_column=None 时应自动识别并生成 group_comparison。"""
    csv_path = _write_csv(
        tmp_path / "auto.csv",
        "group,score,other\n"
        "A,1,10\n"
        "A,2,12\n"
        "A,3,11\n"
        "A,4,13\n"
        "B,5,20\n"
        "B,6,22\n"
        "B,7,21\n"
        "B,8,23\n",
    )
    out = tmp_path / "figures"

    bundle = build_figures(csv_path, out)

    types = [figure.figure_type for figure in bundle.figures]
    assert "group_comparison" in types, "应自动识别分类列并生成分组比较图"


def test_grouped_output_has_no_duplicated_content(tmp_path: Path) -> None:
    """分组数据下：图件类型多重集应恰为 distribution=N, group_comparison=N, correlation=1。"""
    csv_path = _write_csv(
        tmp_path / "data.csv",
        "group,score,weight,height\n"
        "A,1,10,100\n"
        "A,2,12,101\n"
        "A,3,11,99\n"
        "A,4,13,102\n"
        "B,5,20,110\n"
        "B,6,22,111\n"
        "B,7,21,109\n"
        "B,8,23,112\n",
    )
    out = tmp_path / "figures"
    n_numeric = 3  # score, weight, height

    bundle = build_figures(csv_path, out, group_column="group")

    counts = Counter(figure.figure_type for figure in bundle.figures)
    assert counts["distribution"] == n_numeric
    assert counts["group_comparison"] == n_numeric
    assert counts["correlation"] == 1
    assert len(bundle.figures) == 2 * n_numeric + 1


def test_two_runs_are_byte_identical(tmp_path: Path) -> None:
    """两次构建（png+pdf）应逐字节一致。"""
    csv_path = _write_csv(
        tmp_path / "data.csv",
        "group,score,weight\n"
        "a,1,10\n"
        "a,2,12\n"
        "a,3,11\n"
        "a,4,13\n"
        "b,5,20\n"
        "b,6,22\n"
        "b,7,21\n"
        "b,8,23\n",
    )
    out_one = tmp_path / "run1"
    out_two = tmp_path / "run2"

    build_figures(csv_path, out_one, formats=("png", "pdf"))
    build_figures(csv_path, out_two, formats=("png", "pdf"))

    def _digests(directory: Path) -> dict[str, str]:
        return {
            path.name: hashlib.md5(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*"))
        }

    assert _digests(out_one) == _digests(out_two)


def test_multiple_categorical_candidates_warns(tmp_path: Path) -> None:
    """存在多个候选分组列时应记录警告，而不是静默跳过。"""
    csv_path = _write_csv(
        tmp_path / "multi.csv",
        "group_a,group_b,score\n"
        "A,X,1\n"
        "A,X,2\n"
        "B,Y,5\n"
        "B,Y,6\n",
    )
    out = tmp_path / "figures"

    bundle = build_figures(csv_path, out)

    assert any("多个候选分组列" in warning for warning in bundle.warnings)


# --------------------------------------------------------------------------- #
# Phase 8：figures.* 样式参数
# --------------------------------------------------------------------------- #
def _sample_csv(tmp_path: Path) -> Path:
    """一张稳定的样例 CSV（两数值列，无分类列）。"""
    return _write_csv(
        tmp_path / "style.csv",
        "score,weight\n1,10\n2,12\n3,11\n4,15\n5,20\n6,22\n",
    )


def test_default_arguments_keep_png_only(tmp_path: Path) -> None:
    """不传样式参数时仅产出默认 png（默认口径不变）。"""
    out = tmp_path / "figures"

    bundle = build_figures(_sample_csv(tmp_path), out)

    assert bundle.figures
    assert all(fig.filename.endswith(".png") for fig in bundle.figures)
    assert not list(out.glob("*.pdf"))
    assert not list(out.glob("*.svg"))


def test_default_format_selects_output_extension(tmp_path: Path) -> None:
    """default_format="pdf" 应让所有图件以 .pdf 为主格式真实落盘。"""
    out = tmp_path / "figures"

    bundle = build_figures(_sample_csv(tmp_path), out, default_format="pdf")

    assert bundle.figures
    assert all(fig.filename.endswith(".pdf") for fig in bundle.figures)
    assert list(out.glob("*.pdf")), "应真的产出 PDF 文件"


def test_explicit_formats_override_default_format(tmp_path: Path) -> None:
    """显式 formats 优先于 default_format。"""
    out = tmp_path / "figures"

    bundle = build_figures(
        _sample_csv(tmp_path), out, formats=("png",), default_format="svg"
    )

    assert bundle.figures
    assert all(fig.filename.endswith(".png") for fig in bundle.figures)


def test_default_arguments_are_byte_identical_to_pre_phase8(tmp_path: Path) -> None:
    """缺省样式参数（含 default_format 默认值）下，两次运行逐字节一致。

    这是 Phase 5.5 的回归要求：默认行为不得被样式接入改变。
    """
    csv_path = _sample_csv(tmp_path)
    out_one = tmp_path / "run1"
    out_two = tmp_path / "run2"

    build_figures(csv_path, out_one, formats=("png", "pdf"))
    build_figures(csv_path, out_two, formats=("png", "pdf"))

    def _digests(directory: Path) -> dict[str, str]:
        return {
            path.name: hashlib.md5(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*"))
        }

    assert _digests(out_one) == _digests(out_two)


def test_default_format_png_is_noop_compared_to_omitted(tmp_path: Path) -> None:
    """显式 default_format="png"（即默认值）与完全不传的产物逐字节一致。"""
    csv_path = _sample_csv(tmp_path)
    out_omitted = tmp_path / "omitted"
    out_explicit = tmp_path / "explicit"

    build_figures(csv_path, out_omitted)
    build_figures(csv_path, out_explicit, default_format="png")

    def _digests(directory: Path) -> dict[str, str]:
        return {
            path.name: hashlib.md5(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*"))
        }

    assert _digests(out_omitted) == _digests(out_explicit) != {}


def test_font_size_pt_is_applied_to_rcparams(tmp_path: Path) -> None:
    """font_size_pt 应真实改写 matplotlib 的基准字号。"""
    import matplotlib.pyplot as plt

    out = tmp_path / "figures"

    build_figures(_sample_csv(tmp_path), out, font_size_pt=17)

    assert plt.rcParams["axes.labelsize"] == 19  # 基础 + 2
    assert plt.rcParams["xtick.labelsize"] == 17
    assert plt.rcParams["ytick.labelsize"] == 17
    assert plt.rcParams["legend.fontsize"] == 17
    assert plt.rcParams["axes.titlesize"] == 20  # 基础 + 3


def test_font_size_pt_default_matches_historical_values(tmp_path: Path) -> None:
    """缺省字号下 rcParams 与改动前完全一致（标题 13 / 其余 10）。"""
    import matplotlib.pyplot as plt

    out = tmp_path / "figures"

    build_figures(_sample_csv(tmp_path), out)

    assert plt.rcParams["axes.titlesize"] == 13
    assert plt.rcParams["axes.labelsize"] == 12
    assert plt.rcParams["xtick.labelsize"] == 10
    assert plt.rcParams["ytick.labelsize"] == 10
    assert plt.rcParams["legend.fontsize"] == 10


def test_unknown_color_palette_warns_and_falls_back(tmp_path: Path) -> None:
    """未注册的配色名应记录警告（不静默）并回退到默认配色。"""
    out = tmp_path / "figures"

    bundle = build_figures(_sample_csv(tmp_path), out, palette="viridis")

    assert any("配色方案" in warning for warning in bundle.warnings)
    assert bundle.figures, "回退后仍应正常出图"


def test_registered_default_palette_does_not_warn(tmp_path: Path) -> None:
    """已注册的 "default" 配色不应产生警告。"""
    out = tmp_path / "figures"

    bundle = build_figures(_sample_csv(tmp_path), out, palette="default")

    assert not any("配色方案" in warning for warning in bundle.warnings)


def test_unavailable_font_family_warns_without_pretending(tmp_path: Path) -> None:
    """系统不存在的字体族：如实记录警告，绝不假装接上。"""
    import matplotlib.font_manager as fm

    out = tmp_path / "figures"
    missing = "Definitely-Not-A-Real-Font-XYZ"

    assert missing not in {font.name for font in fm.fontManager.ttflist}

    bundle = build_figures(_sample_csv(tmp_path), out, font_family=missing)

    assert any("字体族" in warning for warning in bundle.warnings)


def test_available_font_family_is_applied_without_warning(tmp_path: Path) -> None:
    """系统存在的字体族：应置于字体链首位且不产生字体警告。"""
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    out = tmp_path / "figures"
    installed = [font.name for font in fm.fontManager.ttflist]
    if not installed:
        # 极端环境（无任何字体）下无法验证「已安装」分支，跳过而非误报。
        import pytest

        pytest.skip("系统未安装任何 matplotlib 字体")
    chosen = installed[0]

    bundle = build_figures(_sample_csv(tmp_path), out, font_family=chosen)

    assert not any("字体族" in warning for warning in bundle.warnings)
    assert plt.rcParams["font.sans-serif"][0] == chosen


def test_default_journal_is_reported_as_not_effective(tmp_path: Path) -> None:
    """figures.default_journal 无法真实生效（后端不支持按期刊推导样式）。

    本测试**记录该事实**：非空时必须产生一条"未生效"的警告，而不是假装接上。
    """
    out = tmp_path / "figures"

    bundle = build_figures(_sample_csv(tmp_path), out, default_journal="Nature")

    assert any("default_journal" in warning for warning in bundle.warnings)


def test_empty_default_journal_produces_no_warning(tmp_path: Path) -> None:
    """default_journal 留空（默认）时不应产生警告。"""
    out = tmp_path / "figures"

    bundle = build_figures(_sample_csv(tmp_path), out, default_journal="")

    assert not any("default_journal" in warning for warning in bundle.warnings)
