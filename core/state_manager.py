"""项目状态管理模块。

跟踪工作流执行状态，支持 checkpoint 保存和恢复。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class WorkflowStage(Enum):
    """工作流阶段。

    仅保留真实主干 `core/research_pipeline.py` 实际驱动的阶段。
    """

    BRAINSTORMING = "brainstorming"
    SEARCH = "search"
    LIT_REVIEW = "lit_review"
    WRITING = "writing"
    EXPORT = "export"


@dataclass
class ProjectState:
    """项目状态。"""

    name: str
    mode: str
    current_stage: WorkflowStage
    stage_status: dict[WorkflowStage, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """初始化默认阶段状态。"""
        if not self.stage_status:
            for stage in WorkflowStage:
                self.stage_status[stage] = "pending"
            self.stage_status[self.current_stage] = "in_progress"


def _restore_workflow_stage(raw: object) -> WorkflowStage | None:
    """把持久化的阶段名还原为枚举；未知阶段返回 None。

    历史 `state.json` 可能包含已删除的阶段名（如 `polishing`）。删除枚举成员
    属于数据迁移：直接 `WorkflowStage(value)` 会抛 `ValueError`，让旧项目整体
    无法加载。这里对未知阶段返回 None，由调用方跳过，从而保住仍有效的阶段。
    """
    if isinstance(raw, WorkflowStage):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        return WorkflowStage(raw)
    except ValueError:
        return None


def _restore_stage_status(raw: object) -> dict[WorkflowStage, str]:
    """还原阶段状态映射，静默丢弃已删除的历史阶段名。

    保留仍有效阶段的原始状态，并用 `"pending"` 补齐缺失的有效阶段：旧
    `state.json` 里被删除的阶段无法再被任何代码驱动，跳过它们不会丢失有效
    进度，但下游可能按名字索引阶段，因此补齐使恢复后的 map 与新建项目结构一致。
    """
    source = raw if isinstance(raw, dict) else {}
    restored: dict[WorkflowStage, str] = {}
    for key, value in source.items():
        stage = _restore_workflow_stage(key)
        if stage is not None:
            restored[stage] = value
    for stage in WorkflowStage:
        restored.setdefault(stage, "pending")
    return restored


def _resolve_current_stage(
    raw: object, stage_status: dict[WorkflowStage, str]
) -> WorkflowStage:
    """把持久化的当前阶段还原为枚举，且保证结果永不为 None。

    删除枚举成员是数据迁移：历史项目可能停在已删除的阶段（如 `polishing`）。
    直接 `WorkflowStage(value)` 会抛 `ValueError`，而返回 None 会让下游
    `state.current_stage.value`（CLI 与 Web）在加载“成功”后崩溃，等于把一个
    已暂停的项目变成硬断点。因此未知时按固定规则收敛：
    1. 第一个状态为 `"in_progress"` 的有效阶段，否则
    2. 第一个状态不为 `"completed"` 的有效阶段，否则
    3. `WorkflowStage.EXPORT`。
    """
    stage = _restore_workflow_stage(raw)
    if stage is not None:
        return stage
    for candidate in WorkflowStage:
        if stage_status.get(candidate) == "in_progress":
            return candidate
    for candidate in WorkflowStage:
        if stage_status.get(candidate) != "completed":
            return candidate
    return WorkflowStage.EXPORT


class StateManager:
    """项目状态管理器。"""

    def __init__(self, projects_dir: Path) -> None:
        """初始化状态管理器。

        Args:
            projects_dir: 项目存储目录。
        """
        self.projects_dir = projects_dir
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: ProjectState) -> None:
        """保存项目状态。

        Args:
            state: 项目状态对象。
        """
        state.updated_at = datetime.now(UTC).isoformat()
        state_file = self.projects_dir / state.name / "state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)

        # 转换为可序列化格式
        data = asdict(state)
        data["current_stage"] = state.current_stage.value
        data["stage_status"] = {k.value: v for k, v in state.stage_status.items()}

        state_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, project_name: str) -> ProjectState | None:
        """加载项目状态。

        Args:
            project_name: 项目名称。

        Returns:
            项目状态对象，不存在时返回 None。
        """
        state_file = self.projects_dir / project_name / "state.json"
        if not state_file.exists():
            return None

        data = json.loads(state_file.read_text(encoding="utf-8"))

        # 恢复枚举类型；跳过已删除的历史阶段名，避免旧项目无法加载。
        # stage_status 先还原，current_stage 的收敛规则依赖它。
        data["stage_status"] = _restore_stage_status(data["stage_status"])
        data["current_stage"] = _resolve_current_stage(
            data["current_stage"], data["stage_status"]
        )

        return ProjectState(**data)

    def save_checkpoint(self, project_name: str, state: ProjectState) -> Path:
        """保存 checkpoint。

        Args:
            project_name: 项目名称。
            state: 当前状态。

        Returns:
            checkpoint 文件路径。
        """
        checkpoint_dir = self.projects_dir / project_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        checkpoint_file = checkpoint_dir / f"checkpoint_{timestamp}.json"

        data = asdict(state)
        data["current_stage"] = state.current_stage.value
        data["stage_status"] = {k.value: v for k, v in state.stage_status.items()}

        checkpoint_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return checkpoint_file

    def load_latest_checkpoint(self, project_name: str) -> ProjectState | None:
        """加载最新的 checkpoint。

        Args:
            project_name: 项目名称。

        Returns:
            最新的项目状态，不存在时返回 None。
        """
        checkpoint_dir = self.projects_dir / project_name / "checkpoints"
        if not checkpoint_dir.exists():
            return None

        checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.json"), reverse=True)
        if not checkpoints:
            return None

        data = json.loads(checkpoints[0].read_text(encoding="utf-8"))
        data["stage_status"] = _restore_stage_status(data["stage_status"])
        data["current_stage"] = _resolve_current_stage(
            data["current_stage"], data["stage_status"]
        )

        return ProjectState(**data)

    def list_checkpoints(self, project_name: str) -> list[Path]:
        """列出项目的所有 checkpoint。

        Args:
            project_name: 项目名称。

        Returns:
            checkpoint 文件路径列表。
        """
        checkpoint_dir = self.projects_dir / project_name / "checkpoints"
        if not checkpoint_dir.exists():
            return []

        return sorted(checkpoint_dir.glob("checkpoint_*.json"), reverse=True)
