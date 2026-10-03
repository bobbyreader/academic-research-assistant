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
    # --- Phase 8：影响各阶段输出的配置。这里取"已配置的默认值"（即用户没有改动
    # settings.yaml 时的实际取值，与 config/settings.yaml 一致），使既有用例在
    # 行为上完全不变，并代表"配置未改动的一次真实运行"。
    "search_year_range": (0, 0),
    "figures_default_journal": "",
    "figures_default_format": "png",
    "figures_color_palette": "default",
    "figures_font_family": "",
    "figures_font_size_pt": 10,
    "writing_paper_type": "research_article",
    "writing_language": "zh",
    "writing_bilingual_abstract": False,
    "writing_style_guide": "",
    "citation_style": "numeric",
    "review_include_devil_advocate": False,
    "review_consensus_threshold": 0.6,
    "review_score_scale": "0-100",
}


def _fingerprint(**overrides: object) -> RunFingerprint:
    payload = dict(ALL_FINGERPRINT_KEYS)
    payload.update(overrides)
    return RunFingerprint(
        topic=str(payload["topic"]),
        sources=tuple(payload["sources"]),  # type: ignore[arg-type]
        max_results=int(payload["max_results"]),  # type: ignore[call-overload]
        data_sha256=str(payload["data_sha256"]),
        data_name=str(payload["data_name"]),
        reviewer_count=int(payload["reviewer_count"]),  # type: ignore[call-overload]
        figure_dpi=int(payload["figure_dpi"]),  # type: ignore[call-overload]
        data_readable=bool(payload["data_readable"]),
        search_year_range=tuple(payload["search_year_range"]),  # type: ignore[arg-type]
        figures_default_journal=str(payload["figures_default_journal"]),
        figures_default_format=str(payload["figures_default_format"]),
        figures_color_palette=str(payload["figures_color_palette"]),
        figures_font_family=str(payload["figures_font_family"]),
        figures_font_size_pt=int(payload["figures_font_size_pt"]),  # type: ignore[call-overload]
        writing_paper_type=str(payload["writing_paper_type"]),
        writing_language=str(payload["writing_language"]),
        writing_bilingual_abstract=bool(payload["writing_bilingual_abstract"]),
        writing_style_guide=str(payload["writing_style_guide"]),
        citation_style=str(payload["citation_style"]),
        review_include_devil_advocate=bool(payload["review_include_devil_advocate"]),
        review_consensus_threshold=float(payload["review_consensus_threshold"]),  # type: ignore[arg-type]
        review_score_scale=str(payload["review_score_scale"]),
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


# --------------------------------------------------------------------------- #
# 8. Phase 8：配置项进入指纹——"改了设置就必须重跑对应阶段"
#
# 这是本阶段最关键的一条正确性属性。漏掉某配置 → 用户改了设置却复用旧设置下
# 产出的产物，而系统会报告"本次复用了全部阶段"——**这是一句谎**，且用户看不出。
# 反方向（配置进错阶段）只会白花钱，属于安全方向。
# --------------------------------------------------------------------------- #
def test_writing_language_change_invalidates_only_from_writing(tmp_path: Path) -> None:
    """撰写语言变化 → writing 起失效；search/analysis/claims 不受影响。"""
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(writing_language="en"),
        previous_steps=_previous_steps_everything(),
    )

    assert decision.reusable == ("search", "analysis", "claims")
    assert not decision.can_skip("writing")
    assert not decision.can_skip("review")
    assert "撰写语言" in decision.reason


def test_writing_paper_type_and_style_guide_change_invalidate_writing(
    tmp_path: Path,
) -> None:
    """文体与风格约束同样影响手稿正文。"""
    for override, label in (
        ({"writing_paper_type": "review"}, "文体"),
        ({"writing_style_guide": "用被动语态"}, "风格约束"),
        ({"writing_bilingual_abstract": True}, "双语摘要"),
    ):
        store = ArtifactStore(tmp_path / label)
        _write_complete_artifacts(store)

        decision = plan_resume(
            artifact_store=store,
            project_name="p",
            fingerprint=_fingerprint(**override),
            previous_steps=_previous_steps_everything(),
        )

        assert decision.reusable == ("search", "analysis", "claims"), label
        assert label in decision.reason, label


def test_citation_style_change_invalidates_writing(tmp_path: Path) -> None:
    """引用样式改变参考文献渲染结果，因此属于 writing 的输入。"""
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(citation_style="author_year"),
        previous_steps=_previous_steps_everything(),
    )

    assert decision.reusable == ("search", "analysis", "claims")
    assert "引用样式" in decision.reason


def test_search_year_range_change_invalidates_from_search(tmp_path: Path) -> None:
    """年份范围改变检索结果集 → 全部阶段失效（含 search 本身）。"""
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(search_year_range=(2019, 2024)),
        previous_steps=_previous_steps_everything(),
    )

    assert decision.reusable == ()
    assert "检索年份范围" in decision.reason


def test_figures_style_change_invalidates_from_analysis(tmp_path: Path) -> None:
    """图表样式在分析阶段生成 → analysis 起失效，但 search 不受影响。"""
    for override, label in (
        ({"figures_color_palette": "colorblind"}, "图表配色"),
        ({"figures_font_family": "Times New Roman"}, "图表字体"),
        ({"figures_font_size_pt": 12}, "图表字号"),
        ({"figures_default_format": "pdf"}, "图表格式"),
        ({"figures_default_journal": "Nature"}, "图表目标期刊"),
    ):
        store = ArtifactStore(tmp_path / label)
        _write_complete_artifacts(store)

        decision = plan_resume(
            artifact_store=store,
            project_name="p",
            fingerprint=_fingerprint(**override),
            previous_steps=_previous_steps_everything(),
        )

        assert decision.reusable == ("search",), label
        assert label in decision.reason, label


def test_review_config_change_invalidates_only_review(tmp_path: Path) -> None:
    """评审配置只应让 review 失效——否则会让全部模型调用重做。"""
    for override, label in (
        ({"review_include_devil_advocate": True}, "反对意见要求"),
        ({"review_consensus_threshold": 0.9}, "评审一致度阈值"),
        ({"review_score_scale": "0-10"}, "评分刻度"),
    ):
        store = ArtifactStore(tmp_path / label)
        _write_complete_artifacts(store)

        decision = plan_resume(
            artifact_store=store,
            project_name="p",
            fingerprint=_fingerprint(**override),
            previous_steps=_previous_steps_everything(),
        )

        assert decision.reusable == ("search", "analysis", "claims", "writing"), label
        assert not decision.can_skip("review"), label
        assert label in decision.reason, label


def test_writing_config_does_not_invalidate_earlier_stages(tmp_path: Path) -> None:
    """反向约束：撰写配置**不得**进入 search/analysis 的作用域。

    进错阶段只会白花钱（安全方向），但会让"改个语言就重新付费检索"这种浪费复活，
    因此同样要钉死。
    """
    from core.resume import _STEP_INPUTS

    for step in ("search", "analysis", "claims"):
        assert "writing_language" not in _STEP_INPUTS[step], step
        assert "citation_style" not in _STEP_INPUTS[step], step
    for step in ("search", "analysis", "claims", "writing"):
        assert "review_consensus_threshold" not in _STEP_INPUTS[step], step
    assert "search_year_range" not in _STEP_INPUTS["analysis"]


# --------------------------------------------------------------------------- #
# 9. Phase 8：旧记录的保守语义 + 读取健壮性
# --------------------------------------------------------------------------- #
LEGACY_RECORD = {
    "topic": "climate",
    "sources": ["crossref", "pubmed"],
    "max_results": 10,
    "data_sha256": "abc123",
    "data_name": "data.csv",
    "reviewer_count": 3,
    "figure_dpi": 300,
    "data_readable": True,
}


def test_legacy_record_is_conservative_about_phase8_keys() -> None:
    """Phase 8 之前的记录缺失新键 → 取保守哨兵，**不**冒充成"已配置的取值"。

    若默认值取 `config_loader.DEFAULTS`（`"zh"` / `10` / `0.6`），就等于断言旧运行
    当时确实用了这些设置——而我们无法确认。默认值因此刻意与已配置取值不同，
    使对应阶段重跑一次。宁可重跑，也不能把旧设置下的产物冒充成当前设置的结果。
    """
    restored = RunFingerprint.from_dict(LEGACY_RECORD)

    assert restored is not None
    configured = _fingerprint()
    # 逐项：哨兵值必须与"已配置取值"不同，因此旧记录不会与当前运行匹配。
    assert restored.writing_language != configured.writing_language
    assert restored.citation_style != configured.citation_style
    assert restored.figures_font_size_pt != configured.figures_font_size_pt
    assert restored.review_consensus_threshold != configured.review_consensus_threshold
    assert restored != configured
    # 唯一例外：年份范围的哨兵与默认同为"不过滤"，语义确实一致。
    assert restored.search_year_range == (0, 0)
    assert restored.search_year_range == configured.search_year_range


def test_legacy_record_keeps_search_reusable_but_not_later_stages(
    tmp_path: Path,
) -> None:
    """旧记录的 search 仍可复用（年份范围语义一致），analysis 起必须重跑。"""
    store = ArtifactStore(tmp_path)
    _write_complete_artifacts(store)
    restored = RunFingerprint.from_dict(LEGACY_RECORD)
    assert restored is not None

    decision = plan_resume(
        artifact_store=store,
        project_name="p",
        fingerprint=_fingerprint(),
        previous_steps={
            "search": restored,
            "lit_review": restored,
            "writing": restored,
            "review": restored,
        },
    )

    assert decision.reusable == ("search",)


def test_phase8_key_with_wrong_type_makes_record_unusable() -> None:
    """键存在但类型非法 → 记录损坏 → 不复用（而不是"放心复用"）。"""
    for bad_key, bad_value in (
        ("writing_language", 123),
        ("writing_bilingual_abstract", "yes"),
        ("figures_font_size_pt", "10"),
        ("review_consensus_threshold", "0.6"),
        ("search_year_range", [1, 2, 3]),
        ("search_year_range", "2019-2024"),
        ("citation_style", None),
    ):
        payload = dict(LEGACY_RECORD)
        payload[bad_key] = bad_value
        assert RunFingerprint.from_dict(payload) is None, (bad_key, bad_value)


def test_roundtrip_preserves_phase8_config_fields() -> None:
    """序列化往返必须逐字段保真，否则续跑判定会基于失真的输入。"""
    original = _fingerprint(
        search_year_range=(2019, 2024),
        figures_default_journal="Nature",
        figures_default_format="pdf",
        figures_color_palette="colorblind",
        figures_font_family="Times New Roman",
        figures_font_size_pt=12,
        writing_paper_type="review",
        writing_language="en",
        writing_bilingual_abstract=True,
        writing_style_guide="被动语态",
        citation_style="author_year",
        review_include_devil_advocate=True,
        review_consensus_threshold=0.9,
        review_score_scale="0-10",
    )

    assert RunFingerprint.from_dict(original.to_dict()) == original
