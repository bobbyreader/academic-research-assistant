"""产出物存储和版本管理模块。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ArtifactVersion:
    """产出物版本信息。"""

    version: int
    path: Path
    created_at: str
    metadata: dict[str, Any]


def _version_number(path: Path, filename: str) -> int | None:
    """从 ``<filename>.v<N>`` 文件名中解析数值版本号。

    必须按**数值**而非字符串比较：字典序会把 ``x.v10`` 排在 ``x.v2`` 之前，
    从而让 ``[-1]`` 取到 ``v2`` 这样的过期产物。无法解析的文件名（``x.v``、
    ``x.vabc``）返回 ``None``，绝不抛异常。
    """
    prefix = f"{filename}.v"
    name = path.name
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _latest_path(paths: list[Path], filename: str) -> Path | None:
    """在候选路径中返回数值版本号最大的一个（忽略无法解析的文件名）。

    该函数同时用于 ``get_artifact`` 的"最新版本"查询与 ``save_artifact`` 的
    "下一个版本号"计算，确保两处口径一致。
    """
    best_path: Path | None = None
    best_version = -1
    for path in paths:
        version = _version_number(path, filename)
        if version is None or version <= best_version:
            continue
        best_path = path
        best_version = version
    return best_path


def _next_version(paths: list[Path], filename: str) -> int:
    """返回下一个可用版本号：``max(现有最大数值版本号, 现有合法文件数) + 1``。

    取二者较大值是为了同时满足两点：
    * 数值版本号 + 1（修复字典序缺陷：``x.v10`` 之后的下一版本必须是 11）；
    * 绝不复用已存在的版本号——即使版本号并非从 1 连续（例如历史遗留或手工构造
      的文件），也不会覆盖已有产物（此时以合法文件数兜底）。

    这样无论版本是否连续、写入顺序如何，``save_artifact`` 都不会覆盖既有产物。
    """
    best_version = 0
    count = 0
    for path in paths:
        version = _version_number(path, filename)
        if version is None:
            continue
        count += 1
        best_version = max(best_version, version)
    return max(best_version, count) + 1


def _artifact_shown_name(path: Path) -> tuple[str, int, str]:
    """文件路径的排序键：按 (文件名字符串, 数值版本号, 原始名) 排序。

    ``x.v2`` 与 ``x.v10`` 的文件名字符串不同，因此先比字符串会把两个版本排到
    一起的更"靠字典序"的位置；这里以数值版本号作为主键的一部分，保证
    ``x.v2`` 一定排在 ``x.v10`` 之前。异常文件名（无法解析版本）以 ``-1`` 兜底。
    """
    name = path.name
    marker = name.rfind(".v")
    if marker < 0:
        return (name, -1, name)
    filename = name[:marker]
    version = _version_number(path, filename)
    return (filename, version if version is not None else -1, name)


def _artifact_sort_key(reference: str) -> tuple[str, str, int, str]:
    """``list_artifacts`` 的排序键，输入形如 ``<stage>/<filename>.v<N>``。

    返回 ``(阶段名, 文件名字符串, 数值版本号, 原始名)``。
    """
    stage, _, name = reference.partition("/")
    return (stage, *(_artifact_shown_name(Path(name))))


class ArtifactStore:
    """产出物存储管理器。"""

    def __init__(self, projects_dir: Path) -> None:
        """初始化存储管理器。

        Args:
            projects_dir: 项目存储目录。
        """
        self.projects_dir = projects_dir

    def save_artifact(
        self,
        project_name: str,
        stage: str,
        filename: str,
        content: str | bytes,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """保存产出物（自动版本管理）。

        Args:
            project_name: 项目名称。
            stage: 工作流阶段。
            filename: 文件名。
            content: 文件内容。
            metadata: 元数据。

        Returns:
            保存的文件路径。
        """
        stage_dir = self.projects_dir / project_name / "artifacts" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        # 检查现有版本：按数值版本号取最大值 + 1，而不是按文件个数（字典序会让
        # `x.v10` 排在 `x.v2` 之前，个数统计也会被异常文件名干扰）。
        candidates = [
            path for path in stage_dir.glob(f"{filename}.v*")
            if not path.name.endswith(".meta")
        ]
        version = _next_version(candidates, filename)

        # 保存新版本
        versioned_filename = f"{filename}.v{version}"
        file_path = stage_dir / versioned_filename

        if isinstance(content, str):
            file_path.write_text(content, encoding="utf-8")
        else:
            file_path.write_bytes(content)

        # 保存元数据
        meta_file = stage_dir / f"{filename}.v{version}.meta"
        meta_content = {
            "version": version,
            "created_at": datetime.now(UTC).isoformat(),
            "original_filename": filename,
            **(metadata or {}),
        }
        meta_file.write_text(
            __import__("json").dumps(meta_content, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return file_path

    def get_artifact(
        self,
        project_name: str,
        stage: str,
        filename: str,
        version: int | None = None,
    ) -> Path | None:
        """获取产出物。

        Args:
            project_name: 项目名称。
            stage: 工作流阶段。
            filename: 文件名。
            version: 版本号，None 表示最新版本。

        Returns:
            文件路径，不存在时返回 None。
        """
        stage_dir = self.projects_dir / project_name / "artifacts" / stage

        if version is not None:
            file_path = stage_dir / f"{filename}.v{version}"
            return file_path if file_path.exists() else None

        # 获取最新版本：按数值版本号取最大者（字典序会在版本数 >= 10 时返回过期产物）
        existing = [
            path for path in stage_dir.glob(f"{filename}.v*")
            if not path.name.endswith(".meta")
        ]
        return _latest_path(existing, filename)

    def list_artifacts(self, project_name: str) -> list[str]:
        """列出项目的所有产出物。

        Args:
            project_name: 项目名称。

        Returns:
            产出物描述列表。
        """
        artifacts_dir = self.projects_dir / project_name / "artifacts"
        if not artifacts_dir.exists():
            return []

        artifacts = []
        for stage_dir in artifacts_dir.iterdir():
            if stage_dir.is_dir():
                for file_path in stage_dir.glob("*.v*"):
                    if not file_path.name.endswith(".meta"):
                        artifacts.append(f"{stage_dir.name}/{file_path.name}")

        return sorted(artifacts, key=_artifact_sort_key)

    def get_all_artifacts(self, project_name: str) -> dict[str, list[Path]]:
        """获取项目的所有产出物（按阶段组织）。

        Args:
            project_name: 项目名称。

        Returns:
            阶段名到文件路径列表的映射。
        """
        artifacts_dir = self.projects_dir / project_name / "artifacts"
        if not artifacts_dir.exists():
            return {}

        result: dict[str, list[Path]] = {}
        for stage_dir in artifacts_dir.iterdir():
            if stage_dir.is_dir():
                files = [
                    f for f in stage_dir.glob("*.v*")
                    if not f.name.endswith(".meta")
                ]
                if files:
                    result[stage_dir.name] = sorted(files, key=_artifact_shown_name)

        return result

    def rollback(
        self,
        project_name: str,
        stage: str,
        filename: str,
        target_version: int,
    ) -> Path | None:
        """回滚到指定版本（创建新版本内容为旧版本）。

        Args:
            project_name: 项目名称。
            stage: 工作流阶段。
            filename: 文件名。
            target_version: 目标版本号。

        Returns:
            新创建的文件路径，失败时返回 None。
        """
        old_path = self.get_artifact(project_name, stage, filename, target_version)
        if old_path is None:
            return None

        content = old_path.read_text(encoding="utf-8")
        return self.save_artifact(
            project_name,
            stage,
            filename,
            content,
            metadata={"rollback_from": target_version},
        )
