"""产出物存储和版本管理模块。"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ArtifactVersion:
    """产出物版本信息。"""

    version: int
    path: Path
    created_at: str
    metadata: dict[str, Any]


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

        # 检查现有版本
        existing = sorted(stage_dir.glob(f"{filename}.v*"))
        version = len(existing) + 1

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
            "created_at": datetime.now().isoformat(),
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

        # 获取最新版本
        existing = sorted(stage_dir.glob(f"{filename}.v*"))
        if not existing:
            return None
        return existing[-1]

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

        return sorted(artifacts)

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
                    result[stage_dir.name] = sorted(files)

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
