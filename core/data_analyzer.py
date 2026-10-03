"""Lightweight, transparent CSV analysis for optional experimental data."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any


class DataAnalysisError(ValueError):
    """Raised when an optional data file cannot be analyzed safely."""


def _number(value: str) -> float | None:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except (TypeError, ValueError):
        return None


def analyze_csv(path: Path) -> dict[str, Any]:
    """Return descriptive statistics without inventing inferential results."""
    if not path.exists():
        raise DataAnalysisError(f"数据文件不存在: {path}")
    if path.suffix.lower() != ".csv":
        raise DataAnalysisError("当前仅支持 CSV 数据文件")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise DataAnalysisError("CSV 缺少表头")
        rows = list(reader)

    summary: dict[str, Any] = {
        "file": str(path),
        "rows": len(rows),
        "columns": reader.fieldnames,
        "missing_by_column": {},
        "numeric_summary": {},
        "note": "这是描述性统计；因果结论和显著性检验仍需根据研究设计由作者确认。",
    }
    for column in reader.fieldnames:
        values = [row.get(column, "").strip() for row in rows]
        summary["missing_by_column"][column] = sum(not value for value in values)
        numeric = [value for value in (_number(value) for value in values) if value is not None]
        if numeric:
            summary["numeric_summary"][column] = {
                "n": len(numeric),
                "mean": statistics.fmean(numeric),
                "median": statistics.median(numeric),
                "min": min(numeric),
                "max": max(numeric),
                "stdev": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
            }
    return summary
