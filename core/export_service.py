"""Export helpers for generated Markdown manuscripts."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Table,
    TableStyle,
)


class ExportError(RuntimeError):
    """Raised when every PDF backend has been exhausted."""


# --------------------------------------------------------------------------- #
# 纯 Python 渲染器：Markdown -> PDF（reportlab）
# --------------------------------------------------------------------------- #
# 设计原则说明：任何“回退”都不能依赖系统库，否则它不是回退，而是第二种失败
# 模式。WeasyPrint 需要系统的 Pango/Cairo，在未安装这些库的普通机器上必然
# 抛错，因此这里彻底移除该分支，改用纯 Python 的 reportlab —— 它只依赖
# requirements.txt 中已声明的 Python 包，可在任何安装好依赖的机器上出图。

# 全局标记：CJK 字体注册只做一次，避免重复注册带来的无谓开销。
_CJK_FONT_NAME = "STSong-Light"
_CJK_FONT_READY: bool | None = None


def _ensure_cjk_font() -> tuple[str, str | None]:
    """注册 reportlab 内置的 Unicode CID 中文字体。

    Returns:
        ``(字体名, 警告信息)``。当注册失败时返回 Helvetica 及一条清晰的警告，
        而不是让异常向上冒泡 —— 保证渲染仍然可用。
    """
    global _CJK_FONT_READY
    if _CJK_FONT_READY is None:
        try:
            pdfmetrics.registerFont(UnicodeCIDFont(_CJK_FONT_NAME))
            _CJK_FONT_READY = True
        except Exception:  # noqa: BLE001 - 字体缺失不应导致整个导出崩溃
            _CJK_FONT_READY = False

    if _CJK_FONT_READY:
        return _CJK_FONT_NAME, None
    return "Helvetica", (
        f"未能注册中文字体 {_CJK_FONT_NAME}，已回退到 Helvetica，"
        "内容中的中文可能显示为空白方块。"
    )


# 行内标记：**粗体** / *斜体* / [文本](链接)
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*")
_RE_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_RE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _escape(text: str) -> str:
    """转义 reportlab Paragraph 的 XML 特殊字符。"""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline_to_markup(text: str) -> str:
    """把行内 Markdown 转换为 reportlab Paragraph 支持的 HTML 子集。

    先转义再套用标记，避免用户文本里的 ``<>`` 被当作标签解析。
    """
    escaped = _escape(text)
    escaped = _RE_LINK.sub(r'<link href="\2">\1</link>', escaped)
    escaped = _RE_BOLD.sub(r"<b>\1</b>", escaped)
    escaped = _RE_ITALIC.sub(r"<i>\1</i>", escaped)
    return escaped


def _build_styles(font_name: str) -> dict[str, ParagraphStyle]:
    """基于注册好的字体构造正文/标题等段落样式。"""
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=10.5,
            leading=16,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName=font_name, fontSize=20, leading=26, spaceBefore=12, spaceAfter=8),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName=font_name, fontSize=17, leading=22, spaceBefore=10, spaceAfter=6),
        "h3": ParagraphStyle("H3", parent=base["Heading3"], fontName=font_name, fontSize=14, leading=19, spaceBefore=8, spaceAfter=5),
        "h4": ParagraphStyle("H4", parent=base["Heading4"], fontName=font_name, fontSize=12, leading=17, spaceBefore=6, spaceAfter=4),
        "h5": ParagraphStyle("H5", parent=base["Heading5"], fontName=font_name, fontSize=11, leading=16, spaceBefore=5, spaceAfter=4),
        "h6": ParagraphStyle("H6", parent=base["Heading6"], fontName=font_name, fontSize=10.5, leading=15, spaceBefore=5, spaceAfter=4),
        "code": ParagraphStyle("Code", parent=base["Code"], fontName="Courier", fontSize=9, leading=12, backColor=colors.whitesmoke, borderPadding=4, spaceBefore=4, spaceAfter=6),
        "cell": ParagraphStyle("Cell", parent=base["BodyText"], fontName=font_name, fontSize=9.5, leading=13),
    }


def _make_table(rows: list[list[str]], style: ParagraphStyle) -> Table:
    """把 Markdown 表格的行数据转换成 reportlab Table。"""
    cells = [[Paragraph(_inline_to_markup(cell) or "&nbsp;", style) for cell in row] for row in rows]
    table = Table(cells, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _split_table_row(line: str) -> list[str]:
    """拆分一个以 ``|`` 包裹的表格行，容忍首尾多余的竖线。"""
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _is_table_divider(line: str) -> bool:
    """判断一行是否为 Markdown 表格的分隔行（``|---|---|``）。"""
    body = line.strip().strip("|")
    if not body or "|" not in line:
        return False
    return all(set(cell.strip()) <= set("-: ") and cell.strip() for cell in body.split("|"))


def _markdown_to_flowables(text: str, styles: dict[str, ParagraphStyle]) -> list[object]:
    """把项目实际产出的 Markdown 子集转换为 reportlab 流式元素。

    支持的构造：``#``..``######`` 标题、段落、``-``/``*`` 无序列表、``1.``
    有序列表、``---`` 分隔线、``| a | b |`` 表格、```` ``` ```` 围栏代码块、
    ``**粗体**``、``*斜体*``、``[文本](链接)``。任何畸形输入都降级为普通段落，
    绝不抛异常。
    """
    flowables: list[object] = []
    lines = text.splitlines()
    index = 0
    total = len(lines)

    while index < total:
        raw = lines[index]
        stripped = raw.strip()

        # 围栏代码块：```` ``` ````
        if stripped.startswith("```"):
            index += 1
            code_lines: list[str] = []
            while index < total and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1  # 跳过闭合围栏（缺失时也不会越界）
            flowables.append(Preformatted("\n".join(code_lines), styles["code"]))
            continue

        # 分隔线
        if stripped in {"---", "***", "___"}:
            flowables.append(HRFlowable(width="100%", thickness=0.8, color=colors.grey, spaceBefore=6, spaceAfter=6))
            index += 1
            continue

        # 标题
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            flowables.append(Paragraph(_inline_to_markup(heading.group(2)), styles[f"h{level}"]))
            index += 1
            continue

        # 表格：首行含 ``|`` 且下一行为分隔行
        if "|" in raw and index + 1 < total and _is_table_divider(lines[index + 1]):
            rows = [_split_table_row(raw)]
            index += 2
            while index < total and "|" in lines[index] and lines[index].strip():
                rows.append(_split_table_row(lines[index]))
                index += 1
            if rows and any(rows):
                flowables.append(_make_table(rows, styles["cell"]))
            continue

        # 无序列表
        if re.match(r"^[-*]\s+", stripped):
            items: list[ListItem] = []
            while index < total and re.match(r"^[-*]\s+", lines[index].strip()):
                content = re.sub(r"^[-*]\s+", "", lines[index].strip())
                items.append(ListItem(Paragraph(_inline_to_markup(content), styles["body"])))
                index += 1
            flowables.append(ListFlowable(items, bulletType="bullet", leftIndent=18))
            continue

        # 有序列表
        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while index < total and re.match(r"^\d+\.\s+", lines[index].strip()):
                content = re.sub(r"^\d+\.\s+", "", lines[index].strip())
                items.append(ListItem(Paragraph(_inline_to_markup(content), styles["body"])))
                index += 1
            flowables.append(ListFlowable(items, bulletType="1", leftIndent=18))
            continue

        # 空行
        if not stripped:
            index += 1
            continue

        # 普通段落：连续非空、非特殊行合并为一段
        paragraph_lines = [raw.strip()]
        index += 1
        while index < total:
            nxt = lines[index]
            nxt_stripped = nxt.strip()
            if (
                not nxt_stripped
                or nxt_stripped.startswith("```")
                or nxt_stripped in {"---", "***", "___"}
                or re.match(r"^#{1,6}\s+", nxt_stripped)
                or re.match(r"^[-*]\s+", nxt_stripped)
                or re.match(r"^\d+\.\s+", nxt_stripped)
                or ("|" in nxt and index + 1 < total and _is_table_divider(lines[index + 1]))
            ):
                break
            paragraph_lines.append(nxt_stripped)
            index += 1
        flowables.append(Paragraph(_inline_to_markup(" ".join(paragraph_lines)), styles["body"]))

    return flowables


def _render_with_reportlab(markdown_path: Path, output_path: Path) -> Path:
    """使用 reportlab 将 Markdown 渲染成 PDF（保证可用的回退路径）。"""
    font_name, warning = _ensure_cjk_font()
    if warning:
        print(f"[WARN] PDF 渲染: {warning}", file=sys.stderr)

    styles = _build_styles(font_name)
    try:
        content = markdown_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        content = ""

    flowables = _markdown_to_flowables(content, styles)
    if not flowables:
        # 空文档（或完全无法解析）也要产出合法 PDF，而不是抛异常。
        flowables.append(Paragraph("", styles["body"]))

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=markdown_path.stem,
    )
    document.build(flowables)
    return output_path


def export_pdf(markdown_path: Path, output_path: Path) -> Path:
    """把 Markdown 转换为 PDF，依次尝试两条路径。

    1. **Pandoc + XeLaTeX**：当两者都在 PATH 中时使用，排版质量最高。
    2. **纯 Python reportlab 渲染器**：其余情况下的保底路径，只依赖已声明的
       Python 包，不依赖任何系统库。

    全部路径均失败时抛出 :class:`ExportError`，并在消息中列出已尝试过的路径。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempted: list[str] = []

    pandoc = shutil.which("pandoc")
    xelatex = shutil.which("xelatex")
    if pandoc and xelatex:
        attempted.append("pandoc+xelatex")
        try:
            subprocess.run(
                [pandoc, str(markdown_path), "-o", str(output_path), "--pdf-engine=xelatex", "--toc"],
                check=True,
                capture_output=True,
                text=True,
            )
            return output_path
        except (subprocess.CalledProcessError, OSError) as exc:
            detail = getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc) or "未知错误"
            print(f"[WARN] PDF 导出: pandoc+xelatex 失败，改用 reportlab 回退: {detail.strip()}", file=sys.stderr)

    attempted.append("reportlab")
    try:
        return _render_with_reportlab(markdown_path, output_path)
    except Exception as exc:
        raise ExportError(f"PDF 导出失败（已尝试: {', '.join(attempted)}）: {exc}") from exc


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
