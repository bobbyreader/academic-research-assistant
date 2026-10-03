"""引用语义追溯：把手稿**正文**的「句子—被引文献」配对送进证据核验。

Phase 1 的 :func:`core.citation_verifier.verify_citations` 只回答「``[Pn]`` 是否
存在」；Phase 5.1 的 :func:`core.claim_verifier.verify_claims` 回答「引用摘要是否
支撑该论断」，但当时只作用于手稿**之外**的 ``analysis.key_findings``。于是最终
交付物（手稿正文）本身没有任何语义核验：正文里写 "X 导致 Y [P3]" 而 ``P3`` 讲的
是别的东西时，两道确定性关口会全部通过。

本模块补上这一环——把 Phase 5.1 的核验延伸到最终交付物。它**不重新实现**核验
逻辑，而是把正文切分成「带引用标识的句子」，再整批交给
:func:`core.claim_verifier.verify_claims`。因此共享接地规则
（:func:`core.quote_grounding.is_quote_grounded`）、索引漂移防护、覆盖保证、
反编造——全部自动继承，本模块只负责**抽取**与**装配**。

抽取的两个关键约定：

* **入参必须是正文，不是装配后的手稿。** 系统附加的 ``## References`` 章节里
  每一行都是 ``[P1] 作者. 标题. ...`` 形态；若把整篇手稿（含 References）传进来，
  这些行走会被误当成「引用句子」，既污染核验又浪费一次模型调用。请只传模型输出
  的 Markdown 正文；本模块还会对明显的参考文献行做防御（详见
  :func:`_is_reference_entry`）。
* **未知标识不得进入核验。** ``[P99]`` 这类不存在的标识由
  :func:`core.citation_verifier.verify_citations` **硬阻断**负责；本模块只保留
  「论文列表中真实存在」的标识，绝不把未知标识传给
  :func:`core.claim_verifier.verify_claims`（否则会污染其覆盖统计）。

该核验沿用 Phase 5.1 的定位——模型判断而非确定性证据，因此是**建议性**的：
失败降级为警告，绝不阻断流水线，也绝不向调用方抛异常。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from core.claim_verifier import ClaimVerdict, ClaimVerificationReport, verify_claims
from core.research_models import PaperRecord

#: 单句切分符：中英文句末标点。切分后标点被丢弃，因此这里用字符类逐个匹配。
_SENTENCE_BOUNDARY = re.compile(r"[。！？.!?]")

#: 引用标识：只认 ``P<数字>`` 形态（与 :data:`core.citation_verifier.MARKER_PATTERN`
#: 的严格程度一致），便于过滤掉 ``[P]``、``[P3a]`` 等非标识文本。
_CITATION_MARKER = re.compile(r"\[P(\d+)\]")

#: Markdown 标题行（``#`` 或 ``##`` ... 开头，允许最多 3 个前导空格）。
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")

#: ``## References`` 章节标题的识别（大小写不敏感，容忍尾随符号）。
_REFERENCES_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*references\b", re.IGNORECASE)

#: 参考文献行前缀：以 ``[Pn]`` 开头（允许列表符号 ``-`` / ``*`` / ``+``）。
_REFERENCE_ENTRY_PREFIX = re.compile(r"^\s*(?:[-*+]\s*)?\[P\d+\]")

#: 结语/无正文标记：命中即认为正文已结束，之后（含 References）不再抽取。
_END_OF_BODY_MARKER = re.compile(
    r"```|^\s*[-*_]{3,}\s*$|^\s*(?:#{1,6}\s*)?references\b",
    re.IGNORECASE,
)

#: 参考文献行防御用的「作者样式」启发式：年份 / 期刊斜体 / DOI / URL 等。
#: 命中任一特征即投出「像参考文献而不像正文句子」的一票。
_REFERENCE_LOOKS = (
    re.compile(r"\(\d{4}[a-z]?\)"),  # (2021) / (2021a)
    re.compile(r"https?://|doi\.org|10\.\d{4,}"),
    re.compile(r"\*[^*]+\*"),  # *Journal of ...*
    re.compile(r"\b(?:et al\.|ibid\.|retrieved from)\b", re.IGNORECASE),
)


def extract_citing_claims(body: str) -> list[dict[str, Any]]:
    """从手稿正文中抽取「带引用标识的句子」。

    Args:
        body: **模型输出的正文**（Markdown），不是装配后的手稿。不要传入带
            ``## References`` 章节的完整手稿：那一章节每行都是 ``[P1] 作者...``
            形态，会被误当作引用句子。本函数虽对参考文献行做了防御，但正确的
            用法仍是只传正文。

    Returns:
        ``[{"claim": <句子原文>, "citation_ids": ["P1", ...]}, ...]``；
        ``citation_ids`` 按出现顺序去重。没有任何带引用标识的句子时返回 ``[]``。

    切分规则：先按 ``。！？.!?`` 切分并 strip，保留含 ``[Pn]`` 标识的句子，
    一句可有多个标识。明显的参考文献行（整行以 ``[Pn]`` 开头且带作者/年份/
    期刊/DOI 等特征）以及 ``## References`` 章节之后的全部内容都会被忽略。
    """
    claims: list[dict[str, Any]] = []
    for raw_line in _body_lines(body):
        line = raw_line.strip()
        if not line:
            continue
        # 行级防御优先于切分：参考文献条目必须在**整行**上判定。按句切分会把
        # "[P1] Li, W. (2020). Title." 打散，其中只有首片含标记，而作者/年份等
        # 特征都落在后续片段里——逐句判定必然漏判。
        if _is_reference_entry(line):
            continue
        for segment in _SENTENCE_BOUNDARY.split(line):
            sentence = segment.strip()
            if not sentence:
                continue
            citation_ids = _dedupe(_CITATION_MARKER.findall(sentence))
            if not citation_ids:
                continue
            claims.append(
                {"claim": sentence, "citation_ids": [f"P{num}" for num in citation_ids]}
            )
    return claims


def verify_manuscript_claims(
    llm_client: Any,
    body: str,
    papers: Sequence[PaperRecord],
) -> ClaimVerificationReport:
    """核验正文中每个「句子—被引文献」配对是否有摘要支持。

    **始终返回一个 report**（不返回 None）：

    * 正文没有**可核验**的句子时——没有引用标识、只命中被防御掉的参考文献行、
      或所有标识都指向论文列表中不存在的编号——返回一个 ``claims`` 为空的 report，
      并在其 ``warnings`` 中说明「未执行正文级核验」，且**恰好 0 次模型调用**；
    * 否则**恰好 1 次** ``complete_json`` 调用——语义核验整批委托给
      :func:`core.claim_verifier.verify_claims`，其自身的失败降级与覆盖保证
      全部继承。

    Args:
        llm_client: 具备 ``complete_json(system_prompt, user_prompt) -> dict`` 的客户端。
        body: 手稿**正文**（不含 ``## References``）；推荐先用
            :func:`extract_citing_claims` 自检一遍。
        papers: 检索到的文献，顺序定义 ``P1..Pn``。

    Returns:
        一个 :class:`core.claim_verifier.ClaimVerificationReport`；其 ``passed``
        仅为上报信息，绝不用于阻断流水线。
    """
    extracted = extract_citing_claims(body)
    candidates = _resolve_citable(extracted, papers)
    if not candidates:
        return ClaimVerificationReport(
            author_checks=[
                (
                    "正文中没有可核验的「句子—被引文献」配对，本次**未执行**"
                    "正文级语义核验；这不代表正文没有问题，只是没有可自动核验的对象。"
                )
            ],
            warnings=[
                (
                    "未执行正文级核验：正文中没有可核验的「句子—被引文献」配对"
                    "（没有引用标识，或所有标识均不可解析）。"
                )
            ],
        )

    # 整批交给 Phase 5.1 的核验器：恰好一次模型调用，其余规则全部复用。
    # 该函数只对 [{"claim": str, "citation_ids": [str]}] 取用，忽略额外字段，
    # 因此原样传入抽取结果即可（未来若有 provenance 字段也会被透传）。
    return verify_claims(llm_client, claims=candidates, papers=list(papers))


def render_manuscript_claim_markdown(report: ClaimVerificationReport) -> str:
    """把手稿正文核验报告渲染为可落盘的 Markdown 产物。"""
    lines = [
        "# 手稿正文引用语义核验报告（建议性）",
        "",
        f"- 带引用标识的句子总数: {len(report.claims)}",
        f"- 判定为 unsupported 的句子: {len(report.unsupported_claims)} 条",
        (
            "- 结论: "
            + ("通过" if report.passed else "存在需人工核对的句子")
        ),
        (
            "- 说明: 本报告仅为建议，不会阻断流程；核验范围仅为被引文献的**摘要**，"
            "不涉及全文。"
        ),
        "",
    ]

    for verdict in report.claims:
        lines.append(f"## 句子 {verdict.index + 1}: {verdict.claim}")
        lines.append("")
        lines.append(f"- 总体判定: {verdict.overall}")
        lines.append(
            f"- 引用标识: {', '.join(verdict.citation_ids) or '（无）'}"
        )
        if verdict.note:
            lines.append(f"- 备注: {verdict.note}")
        lines.append("")
        lines.extend(_evidence_table(verdict))

    if report.author_checks:
        lines.extend(["## 作者注意事项", ""])
        lines.extend(f"- {item}" for item in report.author_checks)
        lines.append("")

    if report.warnings:
        lines.extend(["## 系统警告", ""])
        lines.extend(f"- {item}" for item in report.warnings)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# 内部实现
# --------------------------------------------------------------------------- #
def _body_lines(body: str) -> list[str]:
    """返回正文行；在 References 章节/代码块/水平线处截断。"""
    lines: list[str] = []
    for raw_line in body.splitlines():
        if _END_OF_BODY_MARKER.search(raw_line):
            break
        lines.append(raw_line)
    return lines


def _is_reference_entry(line: str) -> bool:
    """判断一行是否其实是参考文献条目（含 ``## References`` 章节的那类行）。

    本模块最容易踩的坑就是 ``## References`` 章节：``[P1] Author. Title.`` 这类
    行同样含 ``[P1]``，会被朴素切分当作「引用句子」。这里做一次防御，规则尽量
    保守（宁可漏判也不误杀正文）：

    1. 该行位于 :func:`_body_lines` 截断之后时，根本到不了这里；
    2. **整行**以 ``[Pn]``（可带列表符号）开头。真正的正文句子几乎不会以引用
       标识作为**开头**——``X 导致 Y [P3]`` 的标识在句末，因此这条规则本身
       就非常干净；
    3. 在该前提下，只要还有作者样式特征（年份、DOI/URL、期刊斜体、``et al.``）
       即判定为参考文献行。保留第 3 条是为了避免误杀「``[P3] 提供了反例。``」
       这类以标识开头的正常正文句——它们虽少见，但确实是合法的引用句。

    必须在**整行**上调用（切分之前）：按句切分会把 ``[P1] Li, W. (2020). ...``
    打散，其中只有首片含标记，而年份等特征都落在后续片段里，逐句判定必漏判。
    """
    if not _REFERENCE_ENTRY_PREFIX.match(line):
        return False
    return any(pattern.search(line) for pattern in _REFERENCE_LOOKS)


def _resolve_citable(
    extracted: Sequence[dict[str, Any]],
    papers: Sequence[PaperRecord],
) -> list[dict[str, Any]]:
    """抽取结果 → 可直接提交核验的候选，剔除未知标识。

    只保留形如 ``P<数字>`` 且位于 ``papers`` 范围内的标识；一句中的标识全部
    不可解析时丢弃该句。未知标识由
    :func:`core.citation_verifier.verify_citations` 硬阻断负责，此处**不**把它们
    传给 :func:`verify_claims`，以免污染其覆盖统计。
    """
    total = len(papers)
    candidates: list[dict[str, Any]] = []
    for item in extracted:
        claim = str(item.get("claim", "")).strip()
        raw_ids = item.get("citation_ids")
        ids = raw_ids if isinstance(raw_ids, list) else []
        resolved: list[str] = []
        for citation_id in ids:
            position = _paper_position(str(citation_id))
            if position is not None and 1 <= position <= total:
                resolved.append(str(citation_id))
        if claim and resolved:
            candidates.append(
                {"claim": claim, "citation_ids": _dedupe(resolved)}
            )
    return candidates


def _paper_position(citation_id: str) -> int | None:
    """把 ``"P3"`` 解析为从 1 开始的位置；格式非法返回 None。"""
    match = re.fullmatch(r"P(\d+)", citation_id)
    if match is None:
        return None
    return int(match.group(1))


def _dedupe(values: Sequence[str]) -> list[str]:
    """按首次出现顺序去重。"""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _evidence_table(verdict: ClaimVerdict) -> list[str]:
    """渲染单条句子的证据表；无证据时给出占位行。"""
    if not verdict.evidence:
        return ["- （无引用文献，暂无证据）", ""]
    lines = [
        "| 引用 | 判定 | 原文引用 | 理由 |",
        "| --- | --- | --- | --- |",
    ]
    for item in verdict.evidence:
        quote = item.quote.replace("|", "\\|").replace("\n", " ")
        rationale = item.rationale.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {item.citation_id} | {item.verdict} | {quote or '（无）'} "
            f"| {rationale} |"
        )
    lines.append("")
    return lines
