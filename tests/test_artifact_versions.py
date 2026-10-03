"""产物版本排序回归测试。

背景（真实缺陷）：版本排序原先使用**字典序**，当版本数 >= 10 时
``x.v10`` 会排在 ``x.v2`` 之前，于是 ``sorted(...)[-1]`` 把 **v2 当成最新版本**。
断点续跑会因此读到**过期产物**并复用一份已被取代的结果——两道确定性关口都会
通过，这正是本项目竭力避免的"用旧答案回答新问题"。

本文件锁定：`get_artifact` 返回真正的最大数值版本；`save_artifact` 的下一版本号
不受字典序影响；`list_artifacts` / `get_all_artifacts` 按数值序展示；异常文件名
（``x.v``、``x.vabc``）不导致崩溃。
"""

from __future__ import annotations

from pathlib import Path

from core.artifact_store import ArtifactStore


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "projects")


def test_get_artifact_returns_true_latest_after_twelve_versions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for index in range(1, 13):
        store.save_artifact("p", "s", "x.txt", f"content-{index}")

    latest = store.get_artifact("p", "s", "x.txt")

    assert latest is not None
    assert latest.name == "x.txt.v12", (
        f"字典序缺陷：最新版本应为 v12，实际取到 {latest.name}"
    )
    assert latest.read_text(encoding="utf-8") == "content-12"


def test_next_version_after_twelve_is_thirteen(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for index in range(1, 13):
        store.save_artifact("p", "s", "x.txt", f"content-{index}")

    saved = store.save_artifact("p", "s", "x.txt", "content-13")

    assert saved.name == "x.txt.v13", (
        f"下一版本号应为 13，实际为 {saved.name}"
    )
    assert store.get_artifact("p", "s", "x.txt").name == "x.txt.v13"


def test_specific_version_lookup_still_works(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for index in range(1, 13):
        store.save_artifact("p", "s", "x.txt", f"content-{index}")

    assert store.get_artifact("p", "s", "x.txt", 2).name == "x.txt.v2"
    assert store.get_artifact("p", "s", "x.txt", 10).name == "x.txt.v10"
    # 不存在的版本号返回 None，而不是回退到其它版本。
    assert store.get_artifact("p", "s", "x.txt", 99) is None


def test_list_artifacts_orders_versions_numerically(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for index in range(1, 13):
        store.save_artifact("p", "s", "x.txt", f"content-{index}")

    listed = store.list_artifacts("p")

    assert listed[0] == "s/x.txt.v1"
    assert listed[-1] == "s/x.txt.v12"
    # v2 必须排在 v10 之前（字典序会反过来）。
    assert listed.index("s/x.txt.v2") < listed.index("s/x.txt.v10")
    assert [name.split(".v")[-1] for name in listed] == [
        str(index) for index in range(1, 13)
    ]


def test_get_all_artifacts_orders_versions_numerically(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for index in range(1, 13):
        store.save_artifact("p", "s", "x.txt", f"content-{index}")

    grouped = store.get_all_artifacts("p")

    names = [path.name for path in grouped["s"]]
    assert names[0] == "x.txt.v1"
    assert names[-1] == "x.txt.v12"
    assert names.index("x.txt.v2") < names.index("x.txt.v10")


def test_malformed_version_filenames_do_not_crash(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for index in range(1, 4):
        store.save_artifact("p", "s", "x.txt", f"content-{index}")

    stage_dir = tmp_path / "projects" / "p" / "artifacts" / "s"
    (stage_dir / "x.txt.v").write_text("junk", encoding="utf-8")
    (stage_dir / "x.txt.vabc").write_text("junk", encoding="utf-8")
    (stage_dir / "x.txt.v3.meta").write_text("{}", encoding="utf-8")

    # 最大合法版本号仍是 3，不受异常文件名影响。
    assert store.get_artifact("p", "s", "x.txt").name == "x.txt.v3"
    # 更关键：异常文件名不得让"最新版本"退化为它。
    assert store.get_artifact("p", "s", "x.txt") is not None

    # 下一版本号按现有最大合法版本 + 1 = 4，而不是被异常文件名干扰。
    saved = store.save_artifact("p", "s", "x.txt", "content-4")
    assert saved.name == "x.txt.v4"


def test_ordering_is_unaffected_by_creation_order(tmp_path: Path) -> None:
    """即使版本文件以乱序落盘，"最新版本"仍由数值大小决定，而不是文件创建顺序。

    这里直接构造字面量版本文件（``save_artifact`` 永远写入"下一个"版本号，因此
    不会产生这种历史遗留/乱序布局；但排序逻辑必须对其稳健）。
    """
    store = _store(tmp_path)
    stage_dir = tmp_path / "projects" / "p" / "artifacts" / "s"
    stage_dir.mkdir(parents=True, exist_ok=True)
    for index in [10, 2, 1, 11, 3, 12, 5]:
        (stage_dir / f"x.txt.v{index}").write_text(f"content-{index}", encoding="utf-8")

    assert store.get_artifact("p", "s", "x.txt").name == "x.txt.v12"
    # 下一版本号 = max(12, 7) + 1 = 13。
    assert store.save_artifact("p", "s", "x.txt", "next").name == "x.txt.v13"


def test_rollback_targets_numeric_version(tmp_path: Path) -> None:
    """rollback 复用 get_artifact，因此也必须按数值序定位目标版本。"""
    store = _store(tmp_path)
    for index in range(1, 13):
        store.save_artifact("p", "s", "x.txt", f"content-{index}")

    rolled = store.rollback("p", "s", "x.txt", target_version=2)

    assert rolled is not None
    assert rolled.name == "x.txt.v13"
    assert rolled.read_text(encoding="utf-8") == "content-2"
