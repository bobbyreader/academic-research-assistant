from __future__ import annotations

from pathlib import Path

import pytest

from core.export_service import ExportError, export_pdf, export_pptx

MARKDOWN = "# Title\n- One point\n---\n# Conclusion\n- Verify\n"


def test_export_pptx_creates_file(tmp_path: Path) -> None:
    markdown = tmp_path / "outline.md"
    markdown.write_text(MARKDOWN, encoding="utf-8")

    pptx = export_pptx(
        markdown,
        tmp_path / "result.pptx",
        Path(__file__).parents[1] / "scripts/export_pptx.py",
    )

    assert pptx.stat().st_size > 0


def test_export_pdf_creates_file_when_backend_available(tmp_path: Path) -> None:
    markdown = tmp_path / "outline.md"
    markdown.write_text(MARKDOWN, encoding="utf-8")

    try:
        pdf = export_pdf(markdown, tmp_path / "result.pdf")
    except ExportError as exc:
        pytest.skip(f"当前环境缺少 PDF 后端（pandoc/xelatex 或 weasyprint 系统库）: {exc}")

    assert pdf.stat().st_size > 0
