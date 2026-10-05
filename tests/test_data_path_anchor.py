"""`--data` 相对路径锚定**运行根**（Phase 9 收尾裁决）。

背景（真实证据）：从**别的 cwd** 调用时，`--data tests/fixtures/xxx.csv` 找不到
文件——相对路径按**调用者进程 cwd** 解析，而消费点 ``analyze_csv(config.data_path)``
/ ``build_figures(config.data_path, ...)``（``core/research_pipeline.py``）隐式依赖
进程 cwd。

裁决：``--data`` 是**用户提供的数据文件**，按 Phase 8 双锚点原则归"工作区数据"，
锚定**运行根**（``Orchestrator(base_dir=...)`` / ``create_app(base_dir)`` / 未指定
时为 ``Path.cwd()``）。相对路径 → 相对运行根；绝对路径 → 原样。

本文件钉死四条硬约束：
1. 从**别的 cwd** 调用时，相对 ``--data`` 仍按**运行根**找到文件（核心证据）；
2. 绝对路径原样使用；
3. 指纹 ``data_name`` / ``data_sha256`` 在相对/绝对两种写法下**一致**
   （同一份文件不因写法不同而改变续跑判定）；
4. 未指定 ``base_dir`` 时行为与"按 cwd 解析"一致（既有路径逐字节不变）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import core.research_service as research_service_module
from core.artifact_store import ArtifactStore, resolve_repo_path
from core.research_service import ResearchService
from core.resume import COMPLETED_STEPS_KEY, RunFingerprint
from core.state_manager import ProjectState, StateManager, WorkflowStage
from orchestrator import Orchestrator

_DATASET = "group,value\nA,1\nA,2\nB,3\nB,4\n"


# --------------------------------------------------------------------------- #
# 替身：只在**管线边界**打桩，用于捕获真正传给管线的 `data_path`
# --------------------------------------------------------------------------- #
class _CapturingPipeline:
    """记录 ``ResearchPipelineConfig`` 后直接返回成功，不触网、不调模型。

    它让我们**可核对**地看到：服务层究竟把哪条路径传给了下游消费点
    （``analyze_csv`` / ``build_figures`` / 重水化都用它）。
    """

    def __init__(self) -> None:
        self.progress: object = None
        self.config: object = None
        self.instances: list[_CapturingPipeline] = []

    def run(self, config: object) -> object:
        self.config = config
        return object()


class _FakeLLM:
    """占位 LLM；替身管线不会调用它。"""


def _install_pipeline_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> list[_CapturingPipeline]:
    captured: list[_CapturingPipeline] = []

    def fake_pipeline(*_args: object, **kwargs: object) -> _CapturingPipeline:
        pipe = _CapturingPipeline()
        pipe.progress = kwargs.get("progress")
        captured.append(pipe)
        return pipe

    monkeypatch.setattr(research_service_module, "ResearchPipeline", fake_pipeline)
    monkeypatch.setattr(
        research_service_module, "build_llm_client", lambda **_kwargs: _FakeLLM()
    )
    return captured


def _make_service(
    root: Path, *, base_dir: Path | None = None
) -> tuple[ResearchService, Path]:
    projects_dir = root / "projects"
    state_manager = StateManager(projects_dir)
    artifact_store = ArtifactStore(projects_dir)
    state_manager.save(
        ProjectState(
            name="case",
            mode="hybrid",
            current_stage=WorkflowStage.BRAINSTORMING,
        )
    )
    service = ResearchService(
        projects_dir, state_manager, artifact_store, base_dir=base_dir
    )
    return service, projects_dir


# --------------------------------------------------------------------------- #
# 硬约束 1：从别的 cwd 调用，相对 --data 仍按**运行根**找到文件（核心证据）
# --------------------------------------------------------------------------- #
def test_relative_data_resolves_against_run_root_not_caller_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """运行根有一份 ``data.csv``；调用者 cwd 换到别处，相对 ``--data`` 仍能找到。

    这是本修复的**核心证据**：解析锚点是**运行根**，不是进程 cwd。
    """
    run_root = tmp_path / "run_root"
    run_root.mkdir()
    (run_root / "data.csv").write_text(_DATASET, encoding="utf-8")

    # 调用者的 cwd 指向一个**不含** data.csv 的目录。
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert not (elsewhere / "data.csv").exists()
    monkeypatch.chdir(elsewhere)

    captured = _install_pipeline_stub(monkeypatch)
    service, _ = _make_service(run_root, base_dir=run_root)

    service.run(
        "case",
        "climate adaptation",
        sources=["crossref"],
        max_results=2,
        data_path=Path("data.csv"),  # 相对路径
        resume=False,
    )

    passed = captured[0].config
    assert passed is not None
    resolved = passed.data_path  # type: ignore[attr-defined]
    # 解析到运行根，而不是调用者 cwd（/elsewhere 下不存在该文件）。
    assert resolved == run_root / "data.csv"
    assert resolved.exists()
    assert not (elsewhere / "data.csv").exists()


# --------------------------------------------------------------------------- #
# 硬约束 2：绝对路径原样使用
# --------------------------------------------------------------------------- #
def test_absolute_data_used_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """绝对路径一律原样使用——不做任何拼接、不受运行根影响。"""
    run_root = tmp_path / "run_root"
    run_root.mkdir()
    data = tmp_path / "somewhere_else" / "data.csv"
    data.parent.mkdir()
    data.write_text(_DATASET, encoding="utf-8")

    monkeypatch.chdir(tmp_path)  # cwd 与数据文件、运行根都不同
    captured = _install_pipeline_stub(monkeypatch)
    service, _ = _make_service(run_root, base_dir=run_root)

    service.run(
        "case",
        "climate adaptation",
        sources=["crossref"],
        max_results=2,
        data_path=data,  # 绝对路径
        resume=False,
    )

    passed = captured[0].config
    assert passed.data_path == data  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# 硬约束 3：指纹在相对/绝对两种写法下一致
# --------------------------------------------------------------------------- #
def test_fingerprint_identical_for_relative_and_absolute_writing(
    tmp_path: Path,
) -> None:
    """同一文件、同一相对路径、cwd 不同 → 指纹必须完全相等。

    若哪天有人把**绝对路径**写进指纹，续跑就会在"换个目录跑"时**无谓地全量重做**
    （用户白花钱），而不会有任何测试发现。这里把 ``data_name`` / ``data_sha256``
    与**整个指纹**都钉死：
    * ``data_name`` 是文件名（``path.name``），不含目录；
    * ``data_sha256`` 是内容哈希，与路径无关。
    """
    run_root = tmp_path / "run_root"
    run_root.mkdir()
    data = run_root / "data.csv"
    data.write_text(_DATASET, encoding="utf-8")

    # 相对写法：模拟服务层——相对 ``--data`` 经**运行根**解析为绝对路径后再进指纹。
    resolved_from_relative = resolve_repo_path(run_root, "data.csv")
    assert resolved_from_relative == data
    fingerprint_relative = RunFingerprint.build(
        topic="climate adaptation",
        sources=["crossref"],
        max_results=2,
        data_path=resolved_from_relative,
        reviewer_count=1,
        figure_dpi=300,
    )
    fingerprint_absolute = RunFingerprint.build(
        topic="climate adaptation",
        sources=["crossref"],
        max_results=2,
        data_path=data,  # 绝对写法
        reviewer_count=1,
        figure_dpi=300,
    )

    assert fingerprint_relative.data_name == "data.csv"
    assert fingerprint_absolute.data_name == "data.csv"
    assert fingerprint_relative.data_sha256 == fingerprint_absolute.data_sha256
    assert fingerprint_relative.data_sha256 != ""
    assert fingerprint_relative == fingerprint_absolute


def test_fingerprint_data_digest_matches_file_content(tmp_path: Path) -> None:
    """``data_sha256`` 必须是**同一份内容**的哈希（绝对化不得改变它）。"""
    import hashlib

    run_root = tmp_path / "run_root"
    run_root.mkdir()
    data = run_root / "data.csv"
    data.write_text(_DATASET, encoding="utf-8")

    fingerprint = RunFingerprint.build(
        topic="t",
        sources=["crossref"],
        max_results=2,
        data_path=data,
        reviewer_count=1,
        figure_dpi=300,
    )
    assert fingerprint.data_sha256 == hashlib.sha256(data.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# 硬约束 3b：真实链路（Orchestrator）下，相对与绝对写法得到同一份指纹
# --------------------------------------------------------------------------- #
def _run_via_orchestrator(
    run_root: Path,
    *,
    data_arg: Path,
    caller_cwd: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    """经真实 ``Orchestrator`` 跑一次（管线边界打桩），返回落盘的 ``search`` 指纹。

    `progress("search","completed")` 会把当次指纹写入 state.json；我们直接读它，
    从而拿到**真实服务层**构造出的指纹（不是测试自己拼的）。
    """
    monkeypatch.chdir(caller_cwd)
    captured = _install_pipeline_stub(monkeypatch)

    orchestrator = Orchestrator(run_root)
    orchestrator.init_project("case", "hybrid")
    orchestrator.run_real_research(
        "case",
        "climate adaptation",
        sources=["crossref"],
        max_results=2,
        data_path=data_arg,
        resume=False,
    )

    # 手动驱动 progress 到 search/completed，让真实指纹落盘（等价于真实运行里
    # 检索阶段成功完成的时刻）。
    pipe = captured[0]
    callback = pipe.progress
    assert callable(callback)
    callback("search", "in_progress")
    callback("search", "completed")

    state = json.loads(
        (run_root / "projects" / "case" / "state.json").read_text(encoding="utf-8")
    )
    return state["metadata"][COMPLETED_STEPS_KEY]["search"]


def test_orchestrator_fingerprint_same_for_relative_and_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实链路上：同一文件、相对与绝对两种写法 → ``search`` 指纹逐字段相等。"""
    run_root = tmp_path / "run_root"
    run_root.mkdir()
    data = run_root / "data.csv"
    data.write_text(_DATASET, encoding="utf-8")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    relative_fp = _run_via_orchestrator(
        run_root,
        data_arg=Path("data.csv"),
        caller_cwd=elsewhere,  # 从别的 cwd 调用
        monkeypatch=monkeypatch,
    )

    # 清理项目，避免复用干扰（用绝对写法再跑一次，落到独立运行根）。
    run_root_2 = tmp_path / "run_root_2"
    run_root_2.mkdir()
    data_2 = run_root_2 / "data.csv"
    data_2.write_text(_DATASET, encoding="utf-8")
    absolute_fp = _run_via_orchestrator(
        run_root_2,
        data_arg=data_2,
        caller_cwd=tmp_path,
        monkeypatch=monkeypatch,
    )

    assert relative_fp["data_name"] == "data.csv"
    assert absolute_fp["data_name"] == "data.csv"
    assert relative_fp["data_sha256"] == absolute_fp["data_sha256"]
    assert relative_fp == absolute_fp


# --------------------------------------------------------------------------- #
# 硬约束 4：未指定 base_dir 时行为与"按 cwd 解析"一致（既有路径不变）
# --------------------------------------------------------------------------- #
def test_service_without_base_dir_falls_back_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未显式传入 ``base_dir`` 时锚点回退 ``Path.cwd()``——这正是改动前的隐式行为。

    这条保证"从仓库根跑"这条既有路径**逐字节不变**。
    """
    workdir = tmp_path / "cwd_here"
    workdir.mkdir()
    (workdir / "data.csv").write_text(_DATASET, encoding="utf-8")
    monkeypatch.chdir(workdir)

    captured = _install_pipeline_stub(monkeypatch)
    service, _ = _make_service(workdir, base_dir=None)  # 不传 base_dir

    service.run(
        "case",
        "climate adaptation",
        sources=["crossref"],
        max_results=2,
        data_path=Path("data.csv"),
        resume=False,
    )

    passed = captured[0].config
    assert passed.data_path == workdir / "data.csv"  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# 元数据诚实性：``metadata["data_path"]`` 保留**用户原始值**（相对就是相对）
# --------------------------------------------------------------------------- #
def test_metadata_records_user_supplied_value_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``state.json`` 的 ``data_path`` 如实记录用户传入的原值，不是解析后的绝对路径。

    否则"从仓库根跑"这条既有路径的 ``state.json`` 会变化，违反逐字节不变。
    """
    run_root = tmp_path / "run_root"
    run_root.mkdir()
    (run_root / "data.csv").write_text(_DATASET, encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    _install_pipeline_stub(monkeypatch)
    service, projects_dir = _make_service(run_root, base_dir=run_root)

    service.run(
        "case",
        "climate adaptation",
        sources=["crossref"],
        max_results=2,
        data_path=Path("data.csv"),
        resume=False,
    )

    state = json.loads(
        (projects_dir / "case" / "state.json").read_text(encoding="utf-8")
    )
    assert state["metadata"]["data_path"] == "data.csv"


def test_metadata_without_data_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未提供 ``--data`` 时元数据为 ``None``，且不触发任何解析。"""
    captured = _install_pipeline_stub(monkeypatch)
    service, projects_dir = _make_service(tmp_path)

    service.run(
        "case",
        "climate adaptation",
        sources=["crossref"],
        max_results=2,
        data_path=None,
        resume=False,
    )

    assert captured[0].config.data_path is None  # type: ignore[attr-defined]
    state = json.loads(
        (projects_dir / "case" / "state.json").read_text(encoding="utf-8")
    )
    assert state["metadata"]["data_path"] is None
