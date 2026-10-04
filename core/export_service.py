"""Export helpers for generated Markdown manuscripts."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
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


#: 支持的 PDF 引擎取值。``auto`` = 现状（优先 pandoc+xelatex，回退 reportlab）。
KNOWN_PDF_ENGINES: frozenset[str] = frozenset({"auto", "pandoc", "reportlab"})

#: 支持的导出格式取值。
KNOWN_EXPORT_FORMATS: frozenset[str] = frozenset({"md", "pdf", "pptx"})

#: ``export.default_format`` 缺省时使用的格式（历史行为）。
DEFAULT_EXPORT_FORMAT = "md"


def resolve_export_format(
    requested: str | None,
    configured_default: str = "",
    allowed_formats: Sequence[str] | None = None,
) -> str:
    """决定实际导出格式：显式请求优先，其次配置默认，最后回退 ``md``。

    这是 ``export.default_format`` 与 ``citation.export_formats`` 的真实生效点：

    1. **解析顺序**：显式请求（``requested``）> 配置默认（``configured_default``）
       > 历史行为 ``md``。
    2. **全局格式合法性**：解析结果必须是 :data:`KNOWN_EXPORT_FORMATS` 之一，
       否则抛 :class:`ExportError`，绝不在"配置写错"时悄悄按另一种格式导出。
    3. **导出白名单**（``citation.export_formats``）：``allowed_formats`` 非空时，
       解析结果还必须落在其中；否则抛 :class:`ExportError`，消息**点名叫出**
       ``citation.export_formats`` 与具体被拒的格式——绝不静默换格式。
       ``allowed_formats=None``（缺省）表示不施加白名单，行为与改动前一致。

    Args:
        requested: 调用方显式请求的格式；``None`` / 空串表示未指定。
        configured_default: 来自 ``export.default_format`` 的默认格式。
        allowed_formats: 来自 ``citation.export_formats`` 的白名单；``None`` 不限。

    Raises:
        ExportError: 解析结果不是已知格式，或不在白名单内。
    """
    choice = (requested or "").strip() or (configured_default or "").strip()
    if not choice:
        choice = DEFAULT_EXPORT_FORMAT
    if choice not in KNOWN_EXPORT_FORMATS:
        raise ExportError(
            f"不支持的导出格式: {choice}（必须是 {sorted(KNOWN_EXPORT_FORMATS)} 之一）"
        )
    if allowed_formats is not None:
        allowed = [item for item in allowed_formats]
        if allowed and choice not in allowed:
            raise ExportError(
                f"导出格式 {choice!r} 不在 citation.export_formats 允许的列表内"
                f"（当前允许: {allowed}）；请在 config/settings.yaml 的 "
                "citation.export_formats 中加入该格式，或改用允许的格式。"
            )
    return choice


def export_pdf(
    markdown_path: Path, output_path: Path, engine: str = "auto"
) -> Path:
    """把 Markdown 转换为 PDF。

    Args:
        markdown_path: 源 Markdown。
        output_path: 目标 PDF。
        engine: PDF 引擎，取值 ``auto`` / ``pandoc`` / ``reportlab``：

            * ``auto``（默认，等价于历史行为）——两者都在 PATH 时用
              **Pandoc + XeLaTeX**（排版质量最高），否则回退到**纯 Python
              reportlab**（只依赖已声明的 Python 包，不依赖任何系统库）；
            * ``pandoc``——**强制**使用 Pandoc + XeLaTeX，二者不可用即抛
              :class:`ExportError`（不回退），以便让"用户显式选择了 pandoc"这一
              意图失败得可见，而不是悄悄换成 reportlab；
            * ``reportlab``——**强制**使用纯 Python 渲染器，跳过 pandoc。

    全部允许的路径均失败时抛出 :class:`ExportError`，并在消息中列出已尝试过的路径。
    """
    if engine not in KNOWN_PDF_ENGINES:
        raise ExportError(
            f"不支持的 PDF 引擎: {engine}（必须是 {sorted(KNOWN_PDF_ENGINES)} 之一）"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attempted: list[str] = []

    if engine == "reportlab":
        attempted.append("reportlab")
        try:
            return _render_with_reportlab(markdown_path, output_path)
        except Exception as exc:
            raise ExportError(
                f"PDF 导出失败（已尝试: {', '.join(attempted)}）: {exc}"
            ) from exc

    pandoc = shutil.which("pandoc")
    xelatex = shutil.which("xelatex")
    if engine == "pandoc":
        # 显式选择 pandoc：可用则用之，不可用即失败，绝不静默回退。
        if not (pandoc and xelatex):
            raise ExportError(
                "PDF 引擎被配置为 pandoc，但未在 PATH 中找到 pandoc 与 xelatex；"
                "请安装二者，或把 export.pdf_engine 改为 auto/reportlab。"
            )
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
            raise ExportError(
                f"PDF 导出失败（已尝试: pandoc+xelatex）: {detail.strip()}"
            ) from exc

    # engine == "auto"：历史行为，pandoc 失败也可回退 reportlab。
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
    else:
        # engine == "auto" 且系统后端缺失：这是**静默降级**。产物看起来正常（合法
        # PDF），但排版来自 reportlab 而非 Pandoc+XeLaTeX，用户必须被告知，否则会
        # 误以为拿到了「高质量排版」的产物。降级本身是设计好的兜底，故只警告、不失败。
        missing = [
            name for name, path in (("pandoc", pandoc), ("xelatex", xelatex)) if not path
        ]
        print(
            "[WARN] PDF 导出: 未在 PATH 中找到 "
            f"{' 与 '.join(missing)}，已改用 reportlab 渲染（排版为回退样式，非 Pandoc+XeLaTeX）。",
            file=sys.stderr,
        )

    attempted.append("reportlab")
    try:
        return _render_with_reportlab(markdown_path, output_path)
    except Exception as exc:
        raise ExportError(f"PDF 导出失败（已尝试: {', '.join(attempted)}）: {exc}") from exc


def _with_speaker_notes(markdown: str) -> str:
    """为大纲的每个 ``# 标题`` 段落补一行 ``Notes:`` 演讲者备注。

    备注内容由该幻灯片的标题与要点**确定性**生成（不含任何随机/时间信息），
    因此相同输入 ⇒ 相同产物。仅在 `include_speaker_notes=True` 时调用。
    """
    lines = markdown.splitlines()
    out: list[str] = []
    current_title = ""
    current_bullets: list[str] = []
    has_notes = False

    def flush() -> None:
        if current_title and not has_notes:
            spoken = "；".join(current_bullets) if current_bullets else "（无要点）"
            out.append(f"Notes: 要点回顾：{spoken}")

    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            flush()
            out.append(line)
            current_title, current_bullets, has_notes = "", [], False
            continue
        if stripped.startswith("# ") and not stripped.startswith("## "):
            flush()
            current_title = stripped[2:].strip()
            current_bullets, has_notes = [], False
            out.append(line)
            continue
        if current_title and stripped.startswith(("- ", "* ")):
            current_bullets.append(stripped[2:].strip())
        elif current_title and stripped.lower().startswith("notes:"):
            has_notes = True
        out.append(line)
    flush()
    return "\n".join(out) + ("\n" if markdown.endswith("\n") else "")


def export_pptx(
    markdown_path: Path,
    output_path: Path,
    script_path: Path,
    template_path: Path | None = None,
    include_speaker_notes: bool = False,
) -> Path:
    """Convert a Markdown outline/manuscript to PPTX using the bundled exporter.

    ``template_path`` 非空时传给导出脚本的 ``--template``（PPTX 模板）；
    ``include_speaker_notes=True`` 时为每张幻灯片补一行 ``Notes:`` 演讲者备注。
    两者缺省（None / False）时与改动前**逐字节一致**。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = markdown_path.read_text(encoding="utf-8")
    if include_speaker_notes:
        markdown = _with_speaker_notes(markdown)
    with tempfile.TemporaryDirectory(prefix="research-pptx-") as temp_dir:
        input_path = Path(temp_dir) / "outline.md"
        input_path.write_text(markdown, encoding="utf-8")
        command = [
            sys.executable,
            str(script_path),
            str(input_path),
            "--output",
            str(output_path),
        ]
        if template_path is not None and str(template_path).strip():
            command.extend(["--template", str(template_path)])
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "未知错误").strip()
            raise ExportError(f"PPTX 导出失败: {detail}") from exc
    return output_path
