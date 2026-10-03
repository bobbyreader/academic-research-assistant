"""Export helpers for generated Markdown manuscripts."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


class ExportError(RuntimeError):
    """Raised when an optional document converter is unavailable."""


def export_pdf(markdown_path: Path, output_path: Path) -> Path:
    """Convert Markdown to PDF with Pandoc and XeLaTeX."""
    pandoc = shutil.which("pandoc")
    xelatex = shutil.which("xelatex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if pandoc and xelatex:
        try:
            subprocess.run(
                [pandoc, str(markdown_path), "-o", str(output_path), "--pdf-engine=xelatex", "--toc"],
                check=True,
                capture_output=True,
                text=True,
            )
            return output_path
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "未知错误").strip()
            raise ExportError(f"PDF 导出失败: {detail}") from exc

    try:
        from markdown import markdown
        from weasyprint import HTML

        html = markdown(markdown_path.read_text(encoding="utf-8"), extensions=["tables"])
        document = f"<!doctype html><html><head><meta charset='utf-8'><style>body{{font-family:Arial,sans-serif;line-height:1.5;margin:2cm}} h1,h2,h3{{page-break-after:avoid}}</style></head><body>{html}</body></html>"
        HTML(string=document, base_url=str(markdown_path.parent)).write_pdf(str(output_path))
    except ImportError as exc:
        missing = ", ".join(name for name, value in (("pandoc", pandoc), ("xelatex", xelatex), ("weasyprint", None)) if not value)
        raise ExportError(f"PDF 导出缺少依赖: {missing}") from exc
    except Exception as exc:
        raise ExportError(f"PDF 导出失败: {exc}") from exc
    return output_path


def export_pptx(markdown_path: Path, output_path: Path, script_path: Path) -> Path:
    """Convert a Markdown outline/manuscript to PPTX using the bundled exporter."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="research-pptx-") as temp_dir:
        input_path = Path(temp_dir) / "outline.md"
        input_path.write_text(markdown_path.read_text(encoding="utf-8"), encoding="utf-8")
        try:
            subprocess.run(
                [sys.executable, str(script_path), str(input_path), "--output", str(output_path)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "未知错误").strip()
            raise ExportError(f"PPTX 导出失败: {detail}") from exc
    return output_path
