from __future__ import annotations

from pathlib import Path

import pytest

from core.data_analyzer import DataAnalysisError, analyze_csv


def test_analyze_csv_reports_missing_values_and_numeric_summary(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("group,score\na,1\nb,3\nc,\n", encoding="utf-8")

    summary = analyze_csv(path)

    assert summary["rows"] == 3
    assert summary["missing_by_column"]["score"] == 1
    assert summary["numeric_summary"]["score"]["mean"] == 2.0


def test_analyze_csv_rejects_non_csv_files(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(DataAnalysisError, match="仅支持 CSV"):
        analyze_csv(path)
