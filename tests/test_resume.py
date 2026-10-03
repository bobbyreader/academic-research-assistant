"""`core/resume.py` 的独立复核测试。

这些用例**独立设计**（不复用团队负责人已用过的 15 个场景），聚焦续跑判定的
第一性原理：

* **产物是唯一证据**：缺失 / 零字节 / JSON 损坏 / 登记的图件文件不存在，都必须
  使该阶段不可复用；
* **前缀性**：一旦某阶段不能复用，其后所有阶段都不能复用——可复用集合必然是
  `RESUME_STEPS` 的连续前缀，杜绝"拼接式手稿"；
* **指纹按阶段作用域比较**：只有真正影响该阶段输出的输入变化才会让它失效；
* **事前 vs 事后**：`ResumeDecision.describe()` 是**判定预测**，
  `describe_reuse(...)` 是**实际事实**；部分复用时两者必须给出不同结论，
  对外呈现必须采用事实。

本模块只验证 `resume.py` 的判定/读取逻辑，不启动整条管线。
"""

from __future__ import annotations

import json
from pathlib import Path

from core.artifact_store import ArtifactStore
from core.resume import (
    RESUME_STEPS,
    RunFingerprint,
    describe_reuse,
    parse_recorded_steps,
    plan_resume,
)

ALL_FINGERPRINT_KEYS = {
    "topic": "climate",
    "sources": ("crossref", "pubmed"),
    "max_results": 10,
    "data_sha256": "abc123",
    "data_name": "data.csv",
    "reviewer_count": 3,
    "figure_dpi": 300,
    "data_readable": True,
}


def _fingerprint(**overrides: object) -> RunFingerprint:
    payload = dict(ALL_FINGERPRINT_KEYS)
    payload.update(overrides)
    return RunFingerprint(
        topic=str(payload["topic"]),
        sources=tuple(payload["sources"]),  # type: ignore[arg-type]
        max_results=int(payload["max_results"]),  # type: ignore[arg-type]
        data_sha256=str(payload["data_sha256"]),
        data_name=str(payload["data_name"]),
        reviewer_count=int(payload["reviewer_count"]),  # type: ignore[arg-type]
        figure_dpi=int(payload["figure_dpi"]),  # type: ignore[arg-type]
        data_readable=bool(payload["data_readable"]),
    )


def _previous_steps_everything() -> dict[str, RunFingerprint]:
    """为每个阶段写入与其作用域相容的完成指纹。

    阶段 → 进度回调标记的映射见 `core/resume.py::_STEP_MARKERS`；
    `analysis` 与 `claims` 同为 `lit_review` 标记。
    """
    full = _fingerprint()
    review_only = _fingerprint()  # reviewer_count 也一致
    return {
        "search": full,
        "lit_review": full,
        "writing": full,
        "review": review_only,
    }


def _write_complete_artifacts(store: ArtifactStore, project: str = "p") -> None:
    """写入一套"五阶段全部完成"的产物，供复用判定通过。"""
    store.save_artifact(
        project, "search", "literature.json", json.dumps([{"title": "t", "doi": "10.1/x"}])
    )
    store.save_artifact(
        project, "search", "search_report.json", json.dumps({"result_count": 1})
    )
    # Phase 7 的相关性校验产物。`_search_complete` 无条件要求它存在：该关口始终产出
    # 报告（含 ran=False 的"未执行"记录），因此"缺失"就等于"检索阶段没走完"。
    store.save_artifact(
        project,
        "search",
        "relevance_check.json",
        json.dumps({"ran": True, "verdicts": []}),
    )
    store.save_artifact(
        project, "search", "relevance_check.md", "relevance body"
    )
    # 给出非空 key_findings，使 claims 阶段确有产物需要核验（否则该阶段视为无事可做）。
    store.save_artifact(
        project,
        "analysis",
        "research_analysis.json",
        json.dumps({"key_findings": [{"claim": "c", "citation_ids": ["P1"]}]}),
    )
    store.save_artifact(project, "analysis", "research_analysis.md", "analysis body")
    store.save_artifact(
        project, "analysis", "statistics_report.json", json.dumps({"tests": []})
    )
    store.save_artifact(project, "analysis", "statistics_report.md", "stats body")
    # 数据存在时，_analysis_complete 还要求 figures.json 登记了至少一张**真实存在**
    # 的图件，因此这里同时写出图件文件本身。
    store.save_artifact(project, "visualization", "figure_1.png", b"\x89PNG-fake")
    store.save_artifact(
        project,
        "visualization",
        "figures.json",
        json.dumps({"figures": [{"filename": "figure_1.png"}]}),
    )
    store.save_artifact(project, "writing", "manuscript.md", "manuscript body")
    store.save_artifact(
        project, "writing", "citation_verification.json", json.dumps({"passed": True})
    )
    store.save_artifact(project, "writing", "citation_verification.md", "citation body")
    store.save_artifact(
        project,
        "writing",
        "claim_evidence_verification.json",
        json.dumps({"passed": True, "claims": []}),
    )
    store.save_artifact(
        project, "writing", "claim_evidence_verification.md", "claim body"
    )
    # Phase 6 新增的正文级论断核验产物。`_writing_complete` 无条件要求它存在：
    # 该关口始终产出报告（正文没有引用标识时也产出一份 claims 为空的报告），
    # 因此"缺失"就等于"writing 阶段没完成"。
    store.save_artifact(
        project,
        "writing",
        "manuscript_claim_verification.json",
        json.dumps({"passed": True, "claims": []}),
    )
    store.save_artifact(
        project,
        "writing",
        "manuscript_claim_verification.md",
        "manuscript claim body",
    )
    store.save_artifact(
        project, "communication", "presentation_outline.md", "outline body"
    )
    store.save_artifact(
        project, "review", "review_reports.json", json.dumps({"decision": "accept"})
    )
    store.save_artifact(project, "review", "review_reports.md", "review body")


# --------------------------------------------------------------------------- #
# 1. 全部齐备 → 连续前缀 = 全部阶段
# --------------------------------------------------------------------------- #
def test_all_artifacts_present_reuses_every_stage(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(),
        previous_steps=_previous_steps_everything(),
    )

    assert decision.enabled is True
    assert decision.reusable == RESUME_STEPS
    for step in RESUME_STEPS:
        assert decision.can_skip(step)


# --------------------------------------------------------------------------- #
# 2. 产物缺失 → 前缀在缺失点截断
# --------------------------------------------------------------------------- #
def test_missing_writing_artifact_truncates_prefix_at_writing(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)
    # 删除 writing 阶段的正文（前缀必须在 writing 处截断）。
    stage_dir = tmp_path / "p" / "artifacts" / "writing"
    for path in stage_dir.glob("manuscript.md.v*"):
        path.unlink()

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(),
        previous_steps=_previous_steps_everything(),
    )

    assert decision.reusable == ("search", "analysis", "claims")
    assert not decision.can_skip("writing")
    assert not decision.can_skip("review")


def test_zero_byte_artifact_is_not_evidence(tmp_path: Path) -> None:
    """零字节文件不是"已完成"的证据。"""
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)
    # 把 manuscript.md 覆盖成一个零字节版本（更"新"，但没有内容）。
    store.save_artifact("p", "writing", "manuscript.md", "")

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(),
        previous_steps=_previous_steps_everything(),
    )

    # 最新版本是零字节 → writing 不可复用 → 前缀在 claims 结束。
    assert decision.reusable == ("search", "analysis", "claims")
    assert not decision.can_skip("writing")


def test_corrupted_json_artifact_blocks_that_stage(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)
    store.save_artifact("p", "search", "literature.json", "{not valid json")

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(),
        previous_steps=_previous_steps_everything(),
    )

    assert decision.reusable == ()
    assert not decision.can_skip("search")


def test_registered_figure_file_missing_blocks_analysis(tmp_path: Path) -> None:
    """figures.json 登记了一张实际不存在的图 → analysis 不可复用。"""
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)
    # 登记一张**未**落盘的图件（例如文件被清理）——不得复用 analysis。
    store.save_artifact(
        "p",
        "visualization",
        "figures.json",
        json.dumps({"figures": [{"filename": "figure_9_missing.png"}]}),
    )

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(),
        previous_steps=_previous_steps_everything(),
    )

    assert decision.reusable == ("search",)
    assert not decision.can_skip("analysis")


# --------------------------------------------------------------------------- #
# 3. 输入变化（按阶段作用域）→ 前缀在受影响阶段截断
# --------------------------------------------------------------------------- #
def test_topic_change_invalidates_every_stage(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(topic="a different topic"),
        previous_steps=_previous_steps_everything(),
    )

    assert decision.reusable == ()
    assert "研究主题" in decision.reason


def test_data_content_change_invalidates_from_analysis(tmp_path: Path) -> None:
    """数据文件内容哈希变化 → analysis 及其后阶段失效；search 仍可复用。"""
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(data_sha256="different-digest"),
        previous_steps=_previous_steps_everything(),
    )

    assert decision.reusable == ("search",)
    assert not decision.can_skip("analysis")
    assert "数据文件" in decision.reason


def test_reviewer_count_change_invalidates_only_review(tmp_path: Path) -> None:
    """评审人数变化只应让 review 失效——否则会让全部模型调用重做。"""
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(reviewer_count=5),
        previous_steps=_previous_steps_everything(),
    )

    assert decision.reusable == ("search", "analysis", "claims", "writing")
    assert not decision.can_skip("review")
    assert "评审人数" in decision.reason


def test_figure_dpi_change_invalidates_from_analysis(tmp_path: Path) -> None:
    """分辨率变化影响图件 → analysis 起失效，但 search 不受影响。"""
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(figure_dpi=600),
        previous_steps=_previous_steps_everything(),
    )

    assert decision.reusable == ("search",)
    assert "图表分辨率" in decision.reason


# --------------------------------------------------------------------------- #
# 4. 前缀性（结构性保证：不存在"跳过中间某一步"）
# --------------------------------------------------------------------------- #
def test_reusable_set_is_always_a_contiguous_prefix(tmp_path: Path) -> None:
    """无论缺失哪一阶段产物，reusable 都是 RESUME_STEPS 的连续前缀。"""
    for broken_stage in RESUME_STEPS[1:]:
        store = ArtifactStore(tmp_path / broken_stage)
        _write_complete_artifacts(store)
        # 破坏该阶段的某个关键产物（writing/review 的正文；analysis 的 JSON）。
        target = {
            "analysis": ("analysis", "research_analysis.json"),
            "claims": ("writing", "claim_evidence_verification.json"),
            "writing": ("writing", "manuscript.md"),
            "review": ("review", "review_reports.md"),
        }[broken_stage]
        path = store.get_artifact("p", target[0], target[1])
        assert path is not None
        path.unlink()

        decision = plan_resume(
            artifact_store=store,
            project_name="p",
            fingerprint=_fingerprint(),
            previous_steps=_previous_steps_everything(),
        )

        # 必须是前缀，且截止到该阶段之前。
        assert decision.reusable == RESUME_STEPS[: RESUME_STEPS.index(broken_stage)]
        assert all(
            decision.can_skip(step)
            for step in RESUME_STEPS[: RESUME_STEPS.index(broken_stage)]
        )
        assert not any(
            decision.can_skip(step)
            for step in RESUME_STEPS[RESUME_STEPS.index(broken_stage) :]
        )


# --------------------------------------------------------------------------- #
# 5. enabled=False → 不复用
# --------------------------------------------------------------------------- #
def test_disabled_resume_reuses_nothing(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(),
        previous_steps=_previous_steps_everything(),
        enabled=False,
    )

    assert decision.enabled is False
    assert decision.reusable == ()
    assert all(not decision.can_skip(step) for step in RESUME_STEPS)


# --------------------------------------------------------------------------- #
# 6. 记录缺失 / 损坏 → 退化为"不复用"
# --------------------------------------------------------------------------- #
def test_no_previous_records_reuses_nothing(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(),
        previous_steps={},
    )

    assert decision.reusable == ()


def test_unreadable_data_file_reuses_nothing(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(data_readable=False),
        previous_steps=_previous_steps_everything(),
    )

    assert decision.reusable == ()
    assert "数据文件" in decision.reason


def test_parse_recorded_steps_ignores_corrupted_entries() -> None:
    """记录损坏必须退化为"不复用"，而不是"放心复用"。"""
    valid = _fingerprint().to_dict()
    payload = {
        "search": valid,
        "lit_review": {"topic": 5},  # 结构非法
        "writing": "not-a-dict",
        "review": valid,
    }

    recorded = parse_recorded_steps(payload)

    assert set(recorded) == {"search", "review"}
    assert recorded["search"] == _fingerprint()


def test_parse_recorded_steps_of_non_mapping_is_empty() -> None:
    assert parse_recorded_steps(None) == {}
    assert parse_recorded_steps([1, 2]) == {}


# --------------------------------------------------------------------------- #
# 7. 事前（describe）vs 事后（describe_reuse）——本次修复的核心
# --------------------------------------------------------------------------- #
def test_describe_predicts_and_describe_reuse_reports_fact() -> None:
    """判定可复用三段，但实际只复用了两段 → 两者结论必须不同。

    这正是修复的核心：判定是**预测**，实际复用是**事实**；对外呈现必须采用事实，
    否则会报告一个比现实更乐观的假象。
    """
    store = None  # 本用例只比较两个纯函数，不涉及产物读取。

    decision_reason = "运行输入一致。"
    from core.resume import ResumeDecision

    decision = ResumeDecision(
        enabled=True,
        fingerprint=_fingerprint(),
        reusable=("search", "analysis", "claims"),
        reason=decision_reason,
    )

    predicted = decision.describe()
    actual = describe_reuse(("search", "analysis"))

    # 事前：宣称可复用 3 个阶段。
    assert "3" in predicted
    assert "撰写" not in predicted  # 预测未包含 writing
    # 事后：实际复用 2 个阶段，并且明确指出"撰写"被重新执行。
    assert "2" in actual
    assert "撰写" in actual
    # 两者必须不同——否则就没有区分"预测"与"事实"的意义。
    assert predicted != actual
    assert store is None


def test_describe_reuse_all_reused_differs_from_partial() -> None:
    """全部复用与部分复用的表述必须不同。"""
    all_reused = describe_reuse(RESUME_STEPS)
    partial = describe_reuse(("search", "analysis"))

    assert "全部" in all_reused
    assert all_reused != partial


def test_describe_reuse_none_reused() -> None:
    assert "未复用任何旧产物" in describe_reuse(())


def test_describe_reuse_is_prefix_order_independent() -> None:
    """传入顺序不同，只要集合是同一前缀，表述应一致（按 RESUME_STEPS 规范化）。"""
    shuffled = ("writing", "search", "claims", "analysis")

    assert describe_reuse(shuffled) == describe_reuse(
        ("search", "analysis", "claims", "writing")
    )
