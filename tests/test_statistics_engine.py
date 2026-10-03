from __future__ import annotations

from pathlib import Path

from core.statistics_engine import analyze_statistics


def _write_csv(tmp_path: Path, content: str, name: str = "data.csv") -> Path:
    """把构造好的 CSV 文本写入临时目录并返回路径。"""
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_two_group_t_test(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "group,score\nA,1\nA,2\nA,3\nA,4\nA,5\nB,6\nB,7\nB,8\nB,9\nB,10\n",
    )

    report = analyze_statistics(path)

    assert len(report.tests) == 1
    test = report.tests[0]
    assert test.test_name == "Welch t-test"
    assert test.effect_size_name == "Cohen's d"
    assert test.n == 10
    assert 0.0 <= test.p_value_adjusted <= 1.0


def test_one_way_anova(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "group,score\nA,1\nA,2\nA,3\nA,4\nB,5\nB,6\nB,7\nB,8\nC,9\nC,10\nC,11\nC,12\n",
    )

    report = analyze_statistics(path)

    assert len(report.tests) == 1
    test = report.tests[0]
    assert "ANOVA" in test.test_name
    assert test.effect_size_name == "eta squared"
    assert test.n == 12


def test_correlation_detected(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "x,y\n"
        "1,2\n2,4\n3,6\n4,8\n5,10\n6,12\n7,14\n8,16\n",
    )

    report = analyze_statistics(path)

    assert report.tests
    test = report.tests[0]
    assert test.effect_size_name in {"Pearson r", "Spearman r"}
    assert test.variables == ["x", "y"]
    assert test.groups == []


def test_holm_adjustment_properties(tmp_path: Path) -> None:
    # 一列分组（2 组 -> t 检验）+ 两列数值（相关），产生 >= 2 个检验。
    path = _write_csv(
        tmp_path,
        "group,score,other\n"
        "A,1,2\nA,2,4\nA,3,6\nA,4,8\nA,5,10\n"
        "B,6,9\nB,7,7\nB,8,5\nB,9,3\nB,10,1\n",
    )

    report = analyze_statistics(path, correction_method="holm")

    assert len(report.tests) >= 2
    for test in report.tests:
        assert test.p_value_adjusted >= test.p_value
        assert 0.0 <= test.p_value_adjusted <= 1.0
    assert min(test.p_value_adjusted for test in report.tests) <= 1.0


def test_small_sample_group_guard(tmp_path: Path) -> None:
    # 分组 B 只有 n=2，不应对其运行 t 检验，且应产生警告。
    path = _write_csv(
        tmp_path,
        "group,score\nA,1\nA,2\nA,3\nB,4\nB,5\n",
    )

    report = analyze_statistics(path)

    assert report.tests == []
    assert any("n=2" in warning or "低于最小样本量" in warning for warning in report.warnings)


def test_missing_values_excluded(tmp_path: Path) -> None:
    # score 列有 2 个缺失值；完整观测应为 10 条。
    path = _write_csv(
        tmp_path,
        "group,score\n"
        "A,1\nA,2\nA,3\nA,4\nA,5\n"
        "B,6\nB,7\nB,8\nB,9\nB,10\n"
        "B,\nA,\n",
    )

    report = analyze_statistics(path)

    assert len(report.tests) == 1
    assert report.tests[0].n == 10


def test_determinism(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "group,score\nA,1\nA,2\nA,3\nA,4\nA,5\nB,6\nB,7\nB,8\nB,9\nB,10\n",
    )

    first = analyze_statistics(path).to_dict()
    second = analyze_statistics(path).to_dict()

    assert first == second


def test_markdown_contains_test_names(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "group,score\nA,1\nA,2\nA,3\nA,4\nA,5\nB,6\nB,7\nB,8\nB,9\nB,10\n",
    )

    markdown = analyze_statistics(path).to_markdown()

    assert "Welch t-test" in markdown
    assert "Cohen's d" in markdown


def test_spearman_has_no_ci_but_pearson_does(tmp_path: Path) -> None:
    # 非正态：明显的重尾/偏态数据 -> 走 Spearman，且置信区间被抑制。
    spearman_path = _write_csv(
        tmp_path,
        "x,y\n1,100\n2,50\n3,40\n4,30\n5,25\n6,20\n7,1\n8,900\n",
        name="spearman.csv",
    )
    spearman_report = analyze_statistics(spearman_path)
    assert len(spearman_report.tests) == 1
    spearman_test = spearman_report.tests[0]
    assert spearman_test.effect_size_name == "Spearman r"
    assert spearman_test.ci_low is None
    assert spearman_test.ci_high is None

    # 近似线性、近似正态（r<1）-> 走 Pearson，且必须有置信区间。
    pearson_path = _write_csv(
        tmp_path,
        "x,y\n1,2\n2,4\n3,6\n4,8\n5,10\n6,12\n7,14\n8,17\n",
        name="pearson.csv",
    )
    pearson_report = analyze_statistics(pearson_path)
    assert len(pearson_report.tests) == 1
    pearson_test = pearson_report.tests[0]
    assert pearson_test.effect_size_name == "Pearson r"
    assert pearson_test.ci_low is not None
    assert pearson_test.ci_high is not None


def test_non_normal_group_test_warns_and_flags_alternative(tmp_path: Path) -> None:
    # B 组包含极端离群值，Shapiro-Wilk 应拒绝正态性。
    path = _write_csv(
        tmp_path,
        "group,score\nA,1\nA,2\nA,3\nA,4\nA,5\nB,6\nB,7\nB,8\nB,9\nB,1000\n",
    )

    report = analyze_statistics(path)

    assert len(report.tests) == 1
    test = report.tests[0]
    assert any("非参数" in warning for warning in test.warnings)
    assert any("非参数" in check for check in report.author_checks)


def test_author_checks_declare_no_post_hoc(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path,
        "group,score\nA,1\nA,2\nA,3\nA,4\nB,5\nB,6\nB,7\nB,8\nC,9\nC,10\nC,11\nC,12\n",
    )

    report = analyze_statistics(path)

    assert any("事后" in check or "post-hoc" in check for check in report.author_checks)
