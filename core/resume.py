"""断点续跑：判定哪些阶段可以安全跳过，并给出可复核的理由。

第一性原理
----------
1. **唯一可信的"某阶段已完成"证据是产物本身**——物理存在、非空、可解析。
   状态标志位由可能崩溃的同一进程写入，且可能先于产物落盘，因此标志位只能
   作为线索，不能作为依据。零字节文件不是证据，无法解析的 JSON 也不是证据。

2. **续跑最大的风险不是"少做"，而是用旧答案回答新问题。** 若研究主题或数据
   文件已经改变，却复用了上一次的检索结果，就会得到一份"引用标识全部存在、
   统计数字全部对得上"的手稿——两道确定性关口都会通过。这与编造证据属于
   同一类错误，必须在结构上杜绝，而不是靠"发生的概率很低"来容忍。

   因此每个阶段的产物都绑定**运行输入指纹**，且指纹是在该阶段产物**落盘之后**
   才被记录的（由 `ResearchService` 在 `progress(..., "completed")` 回调中写入
   `state.json`）。这样"输入已变化"与"产物尚未写出"两种情况都会导致指纹不一致，
   从而拒绝复用。

   指纹按**阶段作用域**比较：每个阶段只比对真正影响其输出的输入（见
   `_STEP_INPUTS`）。否则仅仅调整图表分辨率或评审人数，也会让检索与全部模型
   调用重做一遍，这违背了续跑的初衷。

3. **一旦某阶段不能复用，其后所有阶段都不能复用**：下游产物依赖上游产物。
   于是可复用集合必然是 `RESUME_STEPS` 的**连续前缀**，不存在"跳过中间某一步"
   这种会产生拼接式手稿的情形。

本模块只做判定与读取，不修改任何产物。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.artifact_store import ArtifactStore

#: 可续跑的阶段，顺序与 `ResearchPipeline.run` 的执行顺序一致。
RESUME_STEPS: tuple[str, ...] = ("search", "analysis", "claims", "writing", "review")

#: `state.json` 的 `metadata` 中记录"阶段完成时所用输入指纹"的键。
COMPLETED_STEPS_KEY = "completed_step_fingerprints"

#: 阶段名到进度回调标记的映射。指纹在标记触发时记录，而标记的触发点恰好位于
#: 该阶段产物落盘之后（见 `core/research_pipeline.py`）。
_STEP_MARKERS: dict[str, str] = {
    "search": "search",
    "analysis": "lit_review",
    "claims": "lit_review",
    "writing": "writing",
    "review": "writing",
}

_STEP_LABELS: dict[str, str] = {
    "search": "检索",
    "analysis": "分析与统计",
    "claims": "论断核验",
    "writing": "撰写",
    "review": "同行评审",
}

#: 每个阶段真正依赖的输入。按阶段作用域比较，而不是全局比较：否则只调整图表
#: 分辨率或评审人数也会让检索与全部模型调用重做一遍，这违背了续跑的初衷。
#: 分组是保守的——宁可多跑一个阶段，也不让某个阶段的产物在输入变化后被复用。
_SEARCH_INPUTS: tuple[str, ...] = ("topic", "sources", "max_results")
_DATA_INPUTS: tuple[str, ...] = (
    "topic",
    "sources",
    "max_results",
    "data_sha256",
    "data_name",
    "data_readable",
    "figure_dpi",
)
_REVIEW_INPUTS: tuple[str, ...] = (*_DATA_INPUTS, "reviewer_count")

_STEP_INPUTS: dict[str, tuple[str, ...]] = {
    "search": _SEARCH_INPUTS,
    "analysis": _DATA_INPUTS,
    "claims": _DATA_INPUTS,
    "writing": _DATA_INPUTS,
    "review": _REVIEW_INPUTS,
}

_FIELD_LABELS: dict[str, str] = {
    "topic": "研究主题",
    "sources": "检索源",
    "max_results": "检索条数上限",
    "data_sha256": "数据文件",
    "data_name": "数据文件",
    "data_readable": "数据文件",
    "figure_dpi": "图表分辨率",
    "reviewer_count": "评审人数",
}


@dataclass(frozen=True)
class RunFingerprint:
    """一次运行的输入指纹：决定旧产物能否被复用的全部输入。"""

    topic: str
    sources: tuple[str, ...]
    max_results: int
    data_sha256: str
    data_name: str
    reviewer_count: int
    figure_dpi: int
    data_readable: bool = True

    @classmethod
    def build(
        cls,
        *,
        topic: str,
        sources: Sequence[str],
        max_results: int,
        data_path: Path | None,
        reviewer_count: int,
        figure_dpi: int,
    ) -> RunFingerprint:
        """从本次运行的参数构造指纹。

        数据文件按**内容哈希**参与指纹：改写 CSV 会改变统计结果，因此必须让
        旧产物失效。文件不可读时记为 `data_readable=False`，此时任何复用都被
        拒绝——宁可重跑，也不能在无法确认输入的情况下复用。
        """
        digest = ""
        name = ""
        readable = True
        if data_path is not None:
            path = Path(data_path)
            name = path.name
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                digest = ""
                readable = False
        return cls(
            topic=topic,
            sources=tuple(sorted({item for item in sources if item})),
            max_results=int(max_results),
            data_sha256=digest,
            data_name=name,
            reviewer_count=int(reviewer_count),
            figure_dpi=int(figure_dpi),
            data_readable=readable,
        )

    def to_dict(self) -> dict[str, Any]:
        """序列化为可写入 `state.json` 的字典。"""
        return {
            "topic": self.topic,
            "sources": list(self.sources),
            "max_results": self.max_results,
            "data_sha256": self.data_sha256,
            "data_name": self.data_name,
            "reviewer_count": self.reviewer_count,
            "figure_dpi": self.figure_dpi,
            "data_readable": self.data_readable,
        }

    @classmethod
    def from_dict(cls, payload: object) -> RunFingerprint | None:
        """从持久化字典还原；结构不合法时返回 None（视为无记录）。"""
        if not isinstance(payload, dict):
            return None
        topic = payload.get("topic")
        sources = payload.get("sources")
        max_results = payload.get("max_results")
        data_sha256 = payload.get("data_sha256", "")
        data_name = payload.get("data_name", "")
        reviewer_count = payload.get("reviewer_count", 1)
        figure_dpi = payload.get("figure_dpi", 300)
        data_readable = payload.get("data_readable", True)
        if not isinstance(topic, str) or not isinstance(sources, list):
            return None
        if not all(isinstance(item, str) for item in sources):
            return None
        if not isinstance(max_results, int) or isinstance(max_results, bool):
            return None
        if not isinstance(reviewer_count, int) or isinstance(reviewer_count, bool):
            return None
        if not isinstance(figure_dpi, int) or isinstance(figure_dpi, bool):
            return None
        if not isinstance(data_sha256, str) or not isinstance(data_name, str):
            return None
        if not isinstance(data_readable, bool):
            return None
        return cls(
            topic=topic,
            sources=tuple(sources),
            max_results=max_results,
            data_sha256=data_sha256,
            data_name=data_name,
            reviewer_count=reviewer_count,
            figure_dpi=figure_dpi,
            data_readable=data_readable,
        )


@dataclass(frozen=True)
class ResumeDecision:
    """续跑判定结果：可跳过哪些阶段，以及为什么不能再多跳。"""

    enabled: bool
    fingerprint: RunFingerprint
    reusable: tuple[str, ...]
    reason: str

    def can_skip(self, step: str) -> bool:
        """该阶段是否可以跳过（复用旧产物）。"""
        return step in self.reusable

    def describe(self) -> str:
        """**事前**口径：本次判定计划跳过哪些阶段。

        这只是预测。运行时可能因为产物重水化失败而少复用若干阶段，因此对外
        呈现"实际复用了什么"必须使用 `describe_reuse(实际复用阶段)`，否则就是在
        报告一个比事实更乐观的假象。
        """
        if not self.reusable:
            return f"判定不复用任何旧产物：{self.reason}"
        names = "、".join(_STEP_LABELS[step] for step in self.reusable)
        return f"判定可复用 {len(self.reusable)} 个阶段（{names}）：{self.reason}"


def describe_reuse(reused: Sequence[str]) -> str:
    """**事后**口径：按实际发生的复用生成说明。

    与 `ResumeDecision.describe()` 的区别是刻意的——判定是预测，实际复用是事实。
    两者不一致时，对外呈现必须采用事实。
    """
    reused_set = set(reused)
    kept = [step for step in RESUME_STEPS if step in reused_set]
    rerun = [step for step in RESUME_STEPS if step not in reused_set]
    if not kept:
        return "本次未复用任何旧产物，全部阶段均已重新执行。"
    names = "、".join(_STEP_LABELS[step] for step in kept)
    if not rerun:
        return f"本次复用了全部 {len(kept)} 个阶段（{names}），没有阶段被重新执行。"
    rerun_names = "、".join(_STEP_LABELS[step] for step in rerun)
    return (
        f"本次复用了 {len(kept)} 个阶段（{names}），"
        f"重新执行了 {len(rerun)} 个阶段（{rerun_names}）。"
    )


def parse_recorded_steps(payload: object) -> dict[str, RunFingerprint]:
    """把 `state.json` 中记录的分阶段指纹还原为映射。

    无法解析的条目被忽略（视为无记录），因为"记录损坏"必须退化为"不复用"，
    而不是退化为"放心复用"。
    """
    if not isinstance(payload, dict):
        return {}
    recorded: dict[str, RunFingerprint] = {}
    for marker, value in payload.items():
        if not isinstance(marker, str):
            continue
        fingerprint = RunFingerprint.from_dict(value)
        if fingerprint is not None:
            recorded[marker] = fingerprint
    return recorded


def plan_resume(
    *,
    artifact_store: ArtifactStore,
    project_name: str,
    fingerprint: RunFingerprint,
    previous_steps: Mapping[str, RunFingerprint],
    enabled: bool = True,
) -> ResumeDecision:
    """判定本次运行可以跳过哪些阶段。

    Args:
        artifact_store: 用于检查产物是否存在且可解析。
        project_name: 项目名称。
        fingerprint: 本次运行的输入指纹。
        previous_steps: 上一次运行记录下来的分阶段输入指纹。
        enabled: 用户是否允许复用（`--no-resume` 时为 False）。

    Returns:
        判定结果；`reusable` 为 `RESUME_STEPS` 的连续前缀。
    """
    if not enabled:
        return ResumeDecision(
            False, fingerprint, (), "已按请求禁用断点续跑，从头执行全部阶段。"
        )
    if not fingerprint.data_readable:
        return ResumeDecision(
            True,
            fingerprint,
            (),
            "数据文件无法读取，无法确认旧产物与当前输入一致，从头执行全部阶段。",
        )
    if not previous_steps:
        return ResumeDecision(
            True, fingerprint, (), "该项目没有可复用的运行记录，从头执行全部阶段。"
        )

    reusable: list[str] = []
    reason = "全部阶段的产物齐备且与当前输入一致。"
    for step in RESUME_STEPS:
        marker = _STEP_MARKERS[step]
        recorded = previous_steps.get(marker)
        if recorded is None:
            reason = (
                f"阶段「{_STEP_LABELS[step]}」没有对应的完成记录，"
                "从头执行该阶段及其后续阶段。"
            )
            break
        if not _inputs_match(step, recorded, fingerprint):
            changed = _describe_change(recorded, fingerprint, step)
            reason = (
                f"运行输入已变化（{changed}），"
                f"阶段「{_STEP_LABELS[step]}」及其后续阶段的旧产物不能用于回答当前问题，"
                "需重新执行。"
            )
            break
        complete, why = _step_complete(artifact_store, project_name, step, fingerprint)
        if not complete:
            reason = why
            break
        reusable.append(step)

    return ResumeDecision(True, fingerprint, tuple(reusable), reason)


def _describe_change(
    recorded: RunFingerprint, current: RunFingerprint, step: str
) -> str:
    """列出该阶段真正关心的、发生变化的输入，用于给出可核对的理由。"""
    changed: list[str] = []
    for field in _STEP_INPUTS[step]:
        if getattr(recorded, field) == getattr(current, field):
            continue
        label = _FIELD_LABELS[field]
        if label not in changed:
            changed.append(label)
    return "、".join(changed) if changed else "未知输入"


def _inputs_match(
    step: str, recorded: RunFingerprint, current: RunFingerprint
) -> bool:
    """该阶段所依赖的输入是否完全一致。"""
    return all(
        getattr(recorded, field) == getattr(current, field)
        for field in _STEP_INPUTS[step]
    )


def _json_artifact(
    store: ArtifactStore, project_name: str, stage: str, filename: str
) -> object | None:
    """读取 JSON 产物；缺失、不可读或无法解析时一律返回 None。"""
    path = store.get_artifact(project_name, stage, filename)
    if path is None:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _nonempty_artifact(
    store: ArtifactStore, project_name: str, stage: str, filename: str
) -> bool:
    """产物存在且非空才算数——零字节文件不是证据。"""
    path = store.get_artifact(project_name, stage, filename)
    if path is None:
        return False
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def _step_complete(
    store: ArtifactStore,
    project_name: str,
    step: str,
    fingerprint: RunFingerprint,
) -> tuple[bool, str]:
    """判断单个阶段的产物是否齐备到可以跳过的程度。"""
    if step == "search":
        return _search_complete(store, project_name)
    if step == "analysis":
        return _analysis_complete(store, project_name, fingerprint)
    if step == "claims":
        return _claims_complete(store, project_name)
    if step == "writing":
        return _writing_complete(store, project_name)
    if step == "review":
        return _review_complete(store, project_name, fingerprint)
    return False, f"未知阶段「{step}」，不予复用。"


def _search_complete(store: ArtifactStore, project_name: str) -> tuple[bool, str]:
    literature = _json_artifact(store, project_name, "search", "literature.json")
    if not isinstance(literature, list) or not literature:
        return False, "检索结果 search/literature.json 缺失或不可解析，需重新检索。"
    if not isinstance(
        _json_artifact(store, project_name, "search", "search_report.json"), dict
    ):
        return False, "缺少 search/search_report.json，需重新检索。"
    # 相关性校验（Phase 7）。该关口**始终**产出产物（含 ran=False 的"未执行"记录），
    # 因此这里无条件要求它存在。缺失即意味着检索阶段没有走完。
    #
    # 代价：Phase 7 之前的项目缺少该产物，检索阶段会重跑一次。这是刻意的方向选择——
    # 宁可多跑一个阶段，也不能让续跑**静默跳过一个关口**。
    if not isinstance(
        _json_artifact(store, project_name, "search", "relevance_check.json"), dict
    ):
        return False, (
            "缺少 search/relevance_check.json，"
            "相关性校验没有留下记录，需重新检索。"
        )
    if not _nonempty_artifact(store, project_name, "search", "relevance_check.md"):
        return False, "缺少 search/relevance_check.md，需重新检索。"
    return True, ""


def _analysis_complete(
    store: ArtifactStore, project_name: str, fingerprint: RunFingerprint
) -> tuple[bool, str]:
    if not isinstance(
        _json_artifact(store, project_name, "analysis", "research_analysis.json"), dict
    ):
        return False, "缺少 analysis/research_analysis.json，需重新分析。"
    if not _nonempty_artifact(
        store, project_name, "analysis", "research_analysis.md"
    ):
        return False, "缺少 analysis/research_analysis.md，需重新分析。"
    if fingerprint.data_sha256:
        if not isinstance(
            _json_artifact(store, project_name, "analysis", "statistics_report.json"),
            dict,
        ):
            return False, "缺少 analysis/statistics_report.json，需重新分析数据。"
        if not _nonempty_artifact(
            store, project_name, "analysis", "statistics_report.md"
        ):
            return False, "缺少 analysis/statistics_report.md，需重新分析数据。"
        figures = _json_artifact(store, project_name, "visualization", "figures.json")
        if not isinstance(figures, dict):
            return False, "缺少 visualization/figures.json，需重新生成图表。"
        entries = figures.get("figures")
        if not isinstance(entries, list) or not entries:
            return False, "visualization/figures.json 中没有登记任何图件，需重新生成图表。"
        for entry in entries:
            filename = entry.get("filename") if isinstance(entry, dict) else None
            # 图件文件本身也必须存在：否则手稿会登记一张实际不存在的图。
            if not isinstance(filename, str) or not _nonempty_artifact(
                store, project_name, "visualization", filename
            ):
                return False, "登记的图件文件缺失，需重新生成图表。"
    return True, ""


def _claims_complete(store: ArtifactStore, project_name: str) -> tuple[bool, str]:
    analysis = _json_artifact(store, project_name, "analysis", "research_analysis.json")
    findings = analysis.get("key_findings") if isinstance(analysis, dict) else None
    if not isinstance(findings, list) or not findings:
        # 没有论断需要核验时，本阶段无事可做，视为已完成。
        return True, ""
    if not isinstance(
        _json_artifact(
            store, project_name, "writing", "claim_evidence_verification.json"
        ),
        dict,
    ):
        return False, "缺少 writing/claim_evidence_verification.json，需重新核验论断。"
    if not _nonempty_artifact(
        store, project_name, "writing", "claim_evidence_verification.md"
    ):
        return False, "缺少 writing/claim_evidence_verification.md，需重新核验论断。"
    return True, ""


def _writing_complete(store: ArtifactStore, project_name: str) -> tuple[bool, str]:
    if not _nonempty_artifact(store, project_name, "writing", "manuscript.md"):
        return False, "缺少 writing/manuscript.md，需重新撰写。"
    if not isinstance(
        _json_artifact(store, project_name, "writing", "citation_verification.json"),
        dict,
    ):
        return False, "缺少 writing/citation_verification.json，需重新撰写并复核引用。"
    if not _nonempty_artifact(
        store, project_name, "writing", "citation_verification.md"
    ):
        return False, "缺少 writing/citation_verification.md，需重新撰写并复核引用。"
    # 正文级论断核验（Phase 6）。该关口**始终**产出产物（正文没有引用标识时
    # 也产出一份 claims 为空的报告），因此这里无条件要求它存在。
    #
    # 代价：Phase 6 之前的项目缺少该产物，writing 阶段会重跑一次。这是刻意的
    # 方向选择——宁可多跑一个阶段，也不能让续跑**静默跳过一个关口**。
    if not isinstance(
        _json_artifact(
            store, project_name, "writing", "manuscript_claim_verification.json"
        ),
        dict,
    ):
        return False, (
            "缺少 writing/manuscript_claim_verification.json，"
            "正文级论断核验没有留下记录，需重新撰写。"
        )
    if not _nonempty_artifact(
        store, project_name, "writing", "manuscript_claim_verification.md"
    ):
        return False, "缺少 writing/manuscript_claim_verification.md，需重新撰写。"
    if not _nonempty_artifact(
        store, project_name, "communication", "presentation_outline.md"
    ):
        return False, "缺少 communication/presentation_outline.md，需重新撰写。"
    statistics = _json_artifact(
        store, project_name, "analysis", "statistics_report.json"
    )
    if (
        isinstance(statistics, dict)
        and statistics.get("tests")
        and not isinstance(
            _json_artifact(
                store, project_name, "writing", "statistics_verification.json"
            ),
            dict,
        )
    ):
        missing_gate = (
            "缺少 writing/statistics_verification.json，"
            "统计可追溯性关口没有留下记录，需重新撰写。"
        )
        return False, missing_gate
    return True, ""


def _review_complete(
    store: ArtifactStore, project_name: str, fingerprint: RunFingerprint
) -> tuple[bool, str]:
    if fingerprint.reviewer_count <= 0:
        # 本阶段被配置禁用，无事可做，视为已完成。
        return True, ""
    if not isinstance(
        _json_artifact(store, project_name, "review", "review_reports.json"), dict
    ):
        return False, "缺少 review/review_reports.json，需重新评审。"
    if not _nonempty_artifact(store, project_name, "review", "review_reports.md"):
        return False, "缺少 review/review_reports.md，需重新评审。"
    return True, ""
