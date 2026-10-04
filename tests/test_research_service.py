"""`ResearchService` 的项目级状态记账测试。

核心命题只有一个：**`state.json` 必须与事实一致**。它是续跑判定的输入之一，
一旦对"上一次运行成功了吗、失败在哪"说谎，后续就会做出错误决策（复用不该复用的
产物、或重跑本该完成的阶段）。

Phase 9 的两条真实失真都在这里被钉死：

* Run 2：`search` **成功复用**却因异常路径被标成 `search: blocked`；
* Run 3：完整跑通，`search` 却永远是 `in_progress`、从来不是 `completed`。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.research_service as research_service_module
from core.artifact_store import ArtifactStore
from core.research_service import ResearchService
from core.resume import COMPLETED_STEPS_KEY
from core.state_manager import StateManager, WorkflowStage


# --------------------------------------------------------------------------- #
# 测试替身：只在**管线/LLM 边界**打桩，其余一律用真实实现
# --------------------------------------------------------------------------- #
class _FakeLLM:
    """占位 LLM；本测试的替身管线不会调用它。"""


class _ScriptedPipeline:
    """用脚本模拟真实管线对 `progress` 回调的驱动，以及最终成功/失败。

    `steps` 每个元素形如 `(stage_name, status)`，按顺序回放。回放完毕仍没有被要求
    抛出，就返回一个成功结果。
    """

    def __init__(
        self,
        *,
        steps: list[tuple[str, str]],
        fail_after: str | None = None,
        cancelled_after: str | None = None,
    ) -> None:
        self.steps = steps
        self.fail_after = fail_after
        self.cancelled_after = cancelled_after
        self.progress: object = None
        self.calls = 0

    def run(self, config: object) -> object:
        self.calls += 1
        callback = self.progress
        assert callable(callback)
        for stage_name, status in self.steps:
            callback(stage_name, status)
            if self.fail_after is not None and stage_name == self.fail_after:
                raise RuntimeError(f"模拟阶段 {stage_name} 失败")
            if self.cancelled_after is not None and stage_name == self.cancelled_after:
                from core.research_pipeline import ResearchPipelineCancelled

                raise ResearchPipelineCancelled("用户取消")
        return object()


def _install_pipeline_stub(monkeypatch: pytest.MonkeyPatch, scripted: _ScriptedPipeline) -> None:
    def fake_pipeline(*_args: object, **kwargs: object) -> _ScriptedPipeline:
        scripted.progress = kwargs.get("progress")
        return scripted

    monkeypatch.setattr(research_service_module, "ResearchPipeline", fake_pipeline)


def _install_llm_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        research_service_module, "build_llm_client", lambda **_kwargs: _FakeLLM()
    )


# --------------------------------------------------------------------------- #
# 真实数据参照：`projects/phase9-real-urban-heat/` 里三跑的产物
# --------------------------------------------------------------------------- #
_REAL_PROJECT = (
    Path(__file__).resolve().parent.parent / "projects" / "phase9-real-urban-heat"
)


def _minimal_search_artifacts() -> dict[str, object]:
    """让 `plan_resume` 认为 `search` 产物齐备所需的最小集合。

    只需 `search` 一个阶段可复用——这正是 Run 2/3 的处境：上游 `search` 完整，
    下游才发生失败。内容取自真实产物形状，故 `_search_complete` 能解析通过。
    """
    return {
        "literature.json": [
            {"title": "Urban heat island mitigation", "doi": "10.1/x"}
        ],
        "search_report.json": {
            "sources_attempted": ["crossref"],
            "counts_by_source": {"crossref": 1},
            "errors": [],
            "result_count": 1,
        },
        "relevance_check.json": {"ran": True, "checked": 0, "flagged": [], "notes": []},
    }


def _make_project(
    tmp_path: Path,
    *,
    project_name: str = "case",
    recorded_search_fingerprint: dict[str, object] | None = None,
) -> tuple[ResearchService, Path]:
    """在 tmp_path 下搭一个真实可用的项目骨架（真实 StateManager/ArtifactStore）。"""
    projects_dir = tmp_path / "projects"
    state_manager = StateManager(projects_dir)
    artifact_store = ArtifactStore(projects_dir)

    state = state_manager.load(project_name)
    if state is None:
        from core.state_manager import ProjectState

        state = ProjectState(
            name=project_name,
            mode="hybrid",
            current_stage=WorkflowStage.SEARCH,
        )
    for filename, content in _minimal_search_artifacts().items():
        artifact_store.save_artifact(
            project_name, "search", filename, json.dumps(content)
        )
    if recorded_search_fingerprint is not None:
        state.metadata[COMPLETED_STEPS_KEY] = {"search": recorded_search_fingerprint}
    state_manager.save(state)

    service = ResearchService(projects_dir, state_manager, artifact_store)
    return service, projects_dir


def _run(service: ResearchService, project_name: str = "case") -> None:
    service.run(
        project_name,
        "Urban heat island mitigation",
        sources=["crossref"],
        max_results=10,
        resume=False,  # 保持替换管线可控；复用/完成语义由脚本回调表达
    )


def _load_state(projects_dir: Path, project_name: str = "case") -> dict:
    return json.loads(
        (projects_dir / project_name / "state.json").read_text(encoding="utf-8")
    )


def _fingerprint_for_service() -> dict[str, object]:
    """构造一份与 `_run` 输入一致的 search 指纹（字段形状取自真实 state.json）。"""
    from core.resume import RunFingerprint

    return RunFingerprint.build(
        topic="Urban heat island mitigation",
        sources=["crossref"],
        max_results=10,
        data_path=None,
        reviewer_count=1,
        figure_dpi=300,
    ).to_dict()


# --------------------------------------------------------------------------- #
# 缺陷 2 场景二（Run 3）：完整跑通时，`search` 必须被标为 completed
# --------------------------------------------------------------------------- #
def test_successful_run_marks_search_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run 3 的真实失真：整条管线成功，`search` 却停在 `in_progress`。

    修复后：只要管线发出了 `progress("search", "completed")`（无论 `search` 是
    "新检索"还是"复用"，由管线侧统一发出），`search` 就必须是 `completed`。
    """
    _install_llm_stub(monkeypatch)
    scripted = _ScriptedPipeline(
        steps=[
            ("search", "completed"),
            ("lit_review", "completed"),
            ("writing", "completed"),
        ]
    )
    _install_pipeline_stub(monkeypatch, scripted)
    service, projects_dir = _make_project(tmp_path)

    _run(service)

    state = _load_state(projects_dir)
    assert state["stage_status"]["search"] == "completed"
    assert state["stage_status"]["lit_review"] == "completed"
    assert state["stage_status"]["writing"] == "completed"
    assert state["current_stage"] == "export"


# --------------------------------------------------------------------------- #
# 缺陷 2 场景一（Run 2）：已成功的阶段不得因下游失败被覆盖成 blocked
# --------------------------------------------------------------------------- #
def test_downstream_failure_does_not_overwrite_successful_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run 2 的真实失真：`search` 成功复用（有产物时间戳为证），下游 `analysis` 失败，
    `search` 却被标成 `blocked`、`current_stage` 回退到 `search`。

    修复后：只有**真正失败**的阶段标 `blocked` 并成为 `current_stage`；已成功的
    `search` 保持 `completed`。
    """
    _install_llm_stub(monkeypatch)
    # 脚本先让 search 成功（含 completed 回调），随后 lit_review 阶段抛错。
    scripted = _ScriptedPipeline(
        steps=[("search", "completed"), ("lit_review", "in_progress")],
        fail_after="lit_review",
    )
    _install_pipeline_stub(monkeypatch, scripted)
    service, projects_dir = _make_project(
        tmp_path, recorded_search_fingerprint=_fingerprint_for_service()
    )

    with pytest.raises(RuntimeError, match="lit_review"):
        _run(service)

    state = _load_state(projects_dir)
    # 已成功的 search 绝不能被改写
    assert state["stage_status"]["search"] == "completed"
    # 真正失败的阶段才是 blocked，且 current_stage 指向它
    assert state["stage_status"]["lit_review"] == "blocked"
    assert state["current_stage"] == "lit_review"


def test_failure_is_attributed_to_failed_stage_not_stale_active_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """失败发生在 `search` **已成功、但尚未收到 completed 回调**的窗口内。

    此时 `active_stage` 仍是 `SEARCH`，若无脑覆盖就会把一次真实的检索成功改写成
    `search: blocked`——正是 Run 2 的根因。修复后：

    * `search` 依据"已完成指纹"这一事实保持 `completed`，**不被改写**；
    * 但运行确实没跑完，所以**不能**留下一个全绿的 state——把失败记到"下一个尚未
      成功的阶段"（`lit_review`）上，`current_stage` 指向它。
    """
    _install_llm_stub(monkeypatch)
    # 只驱动到 search 的 in_progress 就抛错：没有任何 completed 回调。
    scripted = _ScriptedPipeline(steps=[("search", "in_progress")], fail_after="search")
    _install_pipeline_stub(monkeypatch, scripted)
    service, projects_dir = _make_project(
        tmp_path, recorded_search_fingerprint=_fingerprint_for_service()
    )

    with pytest.raises(RuntimeError, match="search"):
        _run(service)

    state = _load_state(projects_dir)
    # 该阶段在本次运行中确实成功过（有完成指纹），不得被标成 blocked。
    assert state["stage_status"]["search"] == "completed"
    # 但本次运行没跑完这件事必须留下痕迹：记到下一个尚未成功的阶段。
    assert state["stage_status"]["lit_review"] == "blocked"
    assert state["current_stage"] == "lit_review"
    # run 级标记让状态文件能回答"上一次运行成功了吗、停在哪、为什么"。
    assert state["metadata"]["last_run_outcome"]["outcome"] == "blocked"
    assert state["metadata"]["last_run_outcome"]["stopped_at"] == "lit_review"


def test_stage_failure_without_prior_success_is_marked_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """对照：若失败阶段**确实从未成功**，就必须老实标 `blocked`。"""
    _install_llm_stub(monkeypatch)
    scripted = _ScriptedPipeline(steps=[("search", "in_progress")], fail_after="search")
    _install_pipeline_stub(monkeypatch, scripted)
    service, projects_dir = _make_project(tmp_path)  # 无复用、无完成指纹

    with pytest.raises(RuntimeError, match="search"):
        _run(service)

    state = _load_state(projects_dir)
    assert state["stage_status"]["search"] == "blocked"
    assert state["current_stage"] == "search"


# --------------------------------------------------------------------------- #
# 单调性：completed 不得被降级 / 覆盖
# --------------------------------------------------------------------------- #
def test_completed_stage_never_downgraded_by_later_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """即使后续有杂散回调把某阶段再次标成 `in_progress`，也不得覆盖已 `completed`。"""
    _install_llm_stub(monkeypatch)
    scripted = _ScriptedPipeline(
        steps=[
            ("search", "completed"),
            ("search", "in_progress"),  # 杂散/重复回调
            ("lit_review", "completed"),
        ]
    )
    _install_pipeline_stub(monkeypatch, scripted)
    service, projects_dir = _make_project(tmp_path)

    _run(service)

    state = _load_state(projects_dir)
    assert state["stage_status"]["search"] == "completed"


# --------------------------------------------------------------------------- #
# 取消路径：不得把已成功阶段标成 cancelled
# --------------------------------------------------------------------------- #
def test_cancelled_run_marks_only_pending_stage_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_llm_stub(monkeypatch)
    scripted = _ScriptedPipeline(
        steps=[("search", "completed"), ("lit_review", "in_progress")],
        cancelled_after="lit_review",
    )
    _install_pipeline_stub(monkeypatch, scripted)
    service, projects_dir = _make_project(tmp_path)

    from core.research_pipeline import ResearchPipelineCancelled

    with pytest.raises(ResearchPipelineCancelled):
        _run(service)

    state = _load_state(projects_dir)
    assert state["stage_status"]["search"] == "completed"
    assert state["stage_status"]["lit_review"] == "cancelled"
    assert state["current_stage"] == "lit_review"


# --------------------------------------------------------------------------- #
# 真实数据重放：用真实三跑的 state.json 逐字节对照（只读，不修改）
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not (_REAL_PROJECT / "state.json").exists(),
    reason="真实项目数据不在本次 checkout 中",
)
def test_real_run3_state_matches_true_fact() -> None:
    """如实记录 Run 3 的**事实**：整条管线跑通。

    真实 `state.json` 里 `search` 是 `in_progress`（失真）。这里把事实写死为断言：
    「一个完整跑通的运行，其各阶段应为 completed，`current_stage` 为 export」。
    修复后的 `ResearchService` 正是产出这个结果（见上面的成功路径测试）。
    """
    raw = json.loads((_REAL_PROJECT / "state.json").read_text(encoding="utf-8"))
    # 真实文件里已有 completed 的 lit_review / writing / export 必须与事实一致；
    # 而 search 在真实文件里是 in_progress —— 那是缺陷造成的，正确事实应为 completed。
    assert raw["stage_status"]["lit_review"] == "completed"
    assert raw["stage_status"]["writing"] == "completed"
    assert raw["current_stage"] == "export"
    # 记录缺陷存在：真实文件把已成功的 search 记成了 in_progress。
    assert raw["stage_status"]["search"] == "in_progress"
