from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.export_service import ExportError, export_pdf, export_pptx

MARKDOWN = "# Title\n- One point\n---\n# Conclusion\n- Verify\n"

# 一份覆盖所有受支持构造的文档：标题层级、无序/有序列表、表格、围栏代码、
# 粗体/斜体、链接、分隔线。
RICH_MARKDOWN = """# 一级标题

## 二级标题

### 三级标题

#### 四级标题

##### 五级标题

###### 六级标题

这是一段包含 **粗体**、*斜体* 和 [链接](https://example.com) 的正文。

- 无序第一点
- 无序第二点

1. 有序第一点
2. 有序第二点

| 指标 | 数值 |
| --- | --- |
| p 值 | 0.03 |
| 样本量 | 120 |

```python
def hello() -> str:
    return "world"
```

---
"""


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
    """PDF 必须无条件成功：reportlab 是已声明的依赖，不再跳过。"""
    markdown = tmp_path / "outline.md"
    markdown.write_text(MARKDOWN, encoding="utf-8")

    pdf = export_pdf(markdown, tmp_path / "result.pdf")

    assert pdf.stat().st_size > 0


def test_export_pdf_succeeds_without_pandoc_or_xelatex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没有任何系统后端时，纯 Python reportlab 路径必须产出合法 PDF。"""
    monkeypatch.setattr("core.export_service.shutil.which", lambda _name: None)
    markdown = tmp_path / "outline.md"
    markdown.write_text(MARKDOWN, encoding="utf-8")

    pdf = export_pdf(markdown, tmp_path / "nested" / "result.pdf")

    assert pdf.exists()
    assert pdf.stat().st_size > 0
    assert pdf.read_bytes().startswith(b"%PDF")


def test_export_pdf_renders_chinese(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """中文内容不得抛异常，且能产出非空 PDF（CJK 字体已注册）。"""
    monkeypatch.setattr("core.export_service.shutil.which", lambda _name: None)
    markdown = tmp_path / "report.md"
    markdown.write_text("# 研究报告\n\n- 结论：显著。\n", encoding="utf-8")

    pdf = export_pdf(markdown, tmp_path / "report.pdf")

    assert pdf.stat().st_size > 0
    assert pdf.read_bytes().startswith(b"%PDF")


def test_export_pdf_handles_every_construct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """一次性覆盖所有受支持的 Markdown 构造，不得抛异常。"""
    monkeypatch.setattr("core.export_service.shutil.which", lambda _name: None)
    markdown = tmp_path / "rich.md"
    markdown.write_text(RICH_MARKDOWN, encoding="utf-8")

    pdf = export_pdf(markdown, tmp_path / "rich.pdf")

    assert pdf.stat().st_size > 0


def test_export_pdf_degrades_on_malformed_markdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """畸形 Markdown（未闭合围栏 / 孤立竖线 / 空文件）必须降级而非报错。"""
    monkeypatch.setattr("core.export_service.shutil.which", lambda _name: None)

    for name, content in (
        ("unclosed.md", "# 标题\n\n```python\nprint('x')\n"),
        ("pipe.md", "| 只有一根竖线\n"),
        ("empty.md", ""),
    ):
        markdown = tmp_path / name
        markdown.write_text(content, encoding="utf-8")
        pdf = export_pdf(markdown, tmp_path / f"{name}.pdf")
        assert pdf.read_bytes().startswith(b"%PDF")


def test_export_pdf_prefers_pandoc_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pandoc 与 xelatex 同时存在时优先使用 pandoc，且不触发 reportlab。"""
    calls: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        return f"/usr/local/bin/{name}"

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        output = Path(cmd[cmd.index("-o") + 1])
        output.write_bytes(b"%PDF-1.5\npandoc\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("core.export_service.shutil.which", fake_which)
    monkeypatch.setattr("core.export_service.subprocess.run", fake_run)

    called: list[str] = []
    monkeypatch.setattr(
        "core.export_service._render_with_reportlab",
        lambda *args, **kwargs: called.append("reportlab") or args[1],
    )

    markdown = tmp_path / "outline.md"
    markdown.write_text(MARKDOWN, encoding="utf-8")
    pdf = export_pdf(markdown, tmp_path / "result.pdf")

    assert calls and calls[0][0] == "/usr/local/bin/pandoc"
    assert "--pdf-engine=xelatex" in calls[0]
    assert called == []
    assert pdf.exists()


def test_export_pdf_raises_only_when_all_paths_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """只有当所有路径都失败时，才抛出指明已尝试路径的 ExportError。"""
    monkeypatch.setattr("core.export_service.shutil.which", lambda _name: None)

    def boom(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("模拟渲染失败")

    monkeypatch.setattr("core.export_service._render_with_reportlab", boom)

    markdown = tmp_path / "outline.md"
    markdown.write_text(MARKDOWN, encoding="utf-8")

    with pytest.raises(ExportError) as excinfo:
        export_pdf(markdown, tmp_path / "result.pdf")

    # 无 pandoc/xelatex 时只尝试了 reportlab，错误消息必须点明实际尝试过的路径。
    assert "已尝试" in str(excinfo.value)
    assert "reportlab" in str(excinfo.value)
