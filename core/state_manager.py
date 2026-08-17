"""项目状态管理模块。

跟踪工作流执行状态，支持 checkpoint 保存和恢复。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class WorkflowStage(Enum):
    """工作流阶段。"""

    BRAINSTORMING = "brainstorming"
    SEARCH = "search"
    LIT_REVIEW = "lit_review"
    STATISTICS = "statistics"
    VISUALIZATION = "visualization"
    WRITING = "writing"
    POLISHING = "polishing"
    REVIEW = "review"
    ARS_INTEGRATION = "ars_integration"
    PAPER2PPT = "paper2ppt"
    EXPORT = "export"


@dataclass
class StageRecord:
    """单阶段执行记录：用于跨阶段传递数据。"""

    stage: WorkflowStage
    status: str = "pending"            # pending / in_progress / completed / blocked
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ProjectState:
    """项目状态。"""

    name: str
    mode: str
    current_stage: WorkflowStage
    stage_status: dict[WorkflowStage, str] = field(default_factory=dict)
    stage_records: dict[WorkflowStage, StageRecord] = field(default_factory=dict)
    research_topic: str = ""           # 横跨所有阶段的研究主题
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """初始化默认阶段状态。"""
        if not self.stage_status:
            for stage in WorkflowStage:
                self.stage_status[stage] = "pending"
            self.stage_status[self.current_stage] = "in_progress"
        if not self.stage_records:
            for stage in WorkflowStage:
                self.stage_records[stage] = StageRecord(stage=stage)

    def update_stage(
        self,
        stage: WorkflowStage,
        status: str | None = None,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """更新阶段记录（仅覆盖显式传入的字段）。"""
        rec = self.stage_records.get(stage) or StageRecord(stage=stage)
        if status is not None:
            rec.status = status
            self.stage_status[stage] = status
        if inputs is not None:
            rec.inputs = {**rec.inputs, **inputs}
        if outputs is not None:
            rec.outputs = {**rec.outputs, **outputs}
        if error is not None:
            rec.error = error
        rec.updated_at = datetime.now().isoformat()
        self.stage_records[stage] = rec


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
        state.updated_at = datetime.now().isoformat()
        state_file = self.projects_dir / state.name / "state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)

        # 转换为可序列化格式
        data = asdict(state)
        data["current_stage"] = state.current_stage.value
        data["stage_status"] = {k.value: v for k, v in state.stage_status.items()}
        data["stage_records"] = {
            k.value: {**asdict(v), "stage": v.stage.value}
            for k, v in state.stage_records.items()
        }

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

        # 恢复枚举类型
        data["current_stage"] = WorkflowStage(data["current_stage"])
        data["stage_status"] = {WorkflowStage(k): v for k, v in data["stage_status"].items()}

        # 恢复 stage_records（可能不存在——旧版本兼容）
        raw_records = data.pop("stage_records", {}) or {}
        records: dict[WorkflowStage, StageRecord] = {}
        for k, v in raw_records.items():
            try:
                rec = StageRecord(
                    stage=WorkflowStage(v.get("stage", k)),
                    status=v.get("status", "pending"),
                    inputs=v.get("inputs", {}) or {},
                    outputs=v.get("outputs", {}) or {},
                    error=v.get("error"),
                    updated_at=v.get("updated_at", datetime.now().isoformat()),
                )
                records[rec.stage] = rec
            except (ValueError, KeyError):
                continue
        data["stage_records"] = records

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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = checkpoint_dir / f"checkpoint_{timestamp}.json"

        data = asdict(state)
        data["current_stage"] = state.current_stage.value
        data["stage_status"] = {k.value: v for k, v in state.stage_status.items()}
        data["stage_records"] = {
            k.value: {**asdict(v), "stage": v.stage.value}
            for k, v in state.stage_records.items()
        }

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
        data["current_stage"] = WorkflowStage(data["current_stage"])
        data["stage_status"] = {WorkflowStage(k): v for k, v in data["stage_status"].items()}
        raw_records = data.pop("stage_records", {}) or {}
        records: dict[WorkflowStage, StageRecord] = {}
        for k, v in raw_records.items():
            try:
                rec = StageRecord(
                    stage=WorkflowStage(v.get("stage", k)),
                    status=v.get("status", "pending"),
                    inputs=v.get("inputs", {}) or {},
                    outputs=v.get("outputs", {}) or {},
                    error=v.get("error"),
                    updated_at=v.get("updated_at", datetime.now().isoformat()),
                )
                records[rec.stage] = rec
            except (ValueError, KeyError):
                continue
        data["stage_records"] = records

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
