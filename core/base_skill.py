"""Skill 基类模块。

定义所有 Skill 模块的统一基类，提供标准化的输入/输出接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

from core.artifact_store import ArtifactStore
from core.skill_bridge import SkillBridge
from core.state_manager import ProjectState, StateManager, WorkflowStage

# 泛型类型变量
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class SkillContext:
    """Skill 执行上下文。

    包含 Skill 运行所需的所有依赖和配置信息。
    """

    project_name: str
    project_dir: Path
    state_manager: StateManager
    artifact_store: ArtifactStore
    skill_bridge: SkillBridge
    config: dict[str, Any] = field(default_factory=dict)
    state: ProjectState | None = None


@dataclass
class SkillResult:
    """Skill 执行结果。

    封装 Skill 执行后的输出数据和元信息。
    """

    success: bool
    data: Any = None
    error_message: str | None = None
    artifacts: list[Path] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSkill(ABC, Generic[InputT, OutputT]):
    """Skill 统一基类。

    所有具体的 Skill 模块（如 brainstorming, search 等）都应继承此类，
    并实现 `execute` 方法。

    类型参数:
        InputT: 输入数据类型
        OutputT: 输出数据类型

    示例:
        >>> class BrainstormSkill(BaseSkill[BrainstormInput, BrainstormOutput]):
        ...     @property
        ...     def name(self) -> str:
        ...         return "brainstorming"
        ...
        ...     @property
        ...     def stage(self) -> WorkflowStage:
        ...         return WorkflowStage.BRAINSTORMING
        ...
        ...     def execute(self, input_data: BrainstormInput) -> SkillResult:
        ...         # 实现具体逻辑
        ...         pass
    """

    def __init__(self, context: SkillContext) -> None:
        """初始化 Skill。

        Args:
            context: Skill 执行上下文。
        """
        self.context = context
        self._validate_context()

    def _validate_context(self) -> None:
        """验证上下文完整性。"""
        if not self.context.project_name:
            raise ValueError("project_name 不能为空")
        if not self.context.project_dir.exists():
            raise FileNotFoundError(
                f"项目目录不存在: {self.context.project_dir}"
            )

    @property
    @abstractmethod
    def name(self) -> str:
        """Skill 名称（唯一标识）。"""
        ...

    @property
    @abstractmethod
    def stage(self) -> WorkflowStage:
        """对应的工作流阶段。"""
        ...

    @property
    def description(self) -> str:
        """Skill 描述信息。"""
        return self.__doc__ or ""

    @abstractmethod
    def execute(self, input_data: InputT) -> SkillResult:
        """执行 Skill 核心逻辑。

        Args:
            input_data: 输入数据。

        Returns:
            执行结果封装。

        Raises:
            NotImplementedError: 子类必须实现此方法。
        """
        ...

    def pre_execute(self, input_data: InputT) -> None:
        """执行前置钩子（可选重写）。

        在 `execute` 之前调用，可用于参数验证、状态更新等。

        Args:
            input_data: 输入数据。
        """
        # 更新阶段状态为进行中
        if self.context.state:
            self.context.state.update_stage(
                self.stage,
                status="in_progress",
                inputs={"skill": self.name},
            )
            self.context.state_manager.save(self.context.state)

    def post_execute(self, result: SkillResult) -> None:
        """执行后置钩子（可选重写）。

        在 `execute` 之后调用，可用于结果验证、状态更新等。

        Args:
            result: 执行结果。
        """
        if self.context.state:
            status = "completed" if result.success else "blocked"
            outputs = {"skill": self.name, "success": result.success}
            if result.error_message:
                outputs["error"] = result.error_message

            self.context.state.update_stage(
                self.stage,
                status=status,
                outputs=outputs,
            )
            self.context.state_manager.save(self.context.state)

    def run(self, input_data: InputT) -> SkillResult:
        """运行 Skill（模板方法）。

        按顺序执行: pre_execute -> execute -> post_execute

        Args:
            input_data: 输入数据。

        Returns:
            执行结果。
        """
        self.pre_execute(input_data)
        try:
            result = self.execute(input_data)
        except Exception as e:
            result = SkillResult(
                success=False,
                error_message=str(e),
            )
        self.post_execute(result)
        return result

    def save_artifact(
        self,
        name: str,
        content: str | bytes,
        ext: str = ".md",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """保存产出物到 ArtifactStore。

        Args:
            name: 产出物名称。
            content: 文件内容。
            ext: 文件扩展名。
            metadata: 附加元数据。

        Returns:
            保存后的文件路径。
        """
        info = self.context.artifact_store.save(
            project_name=self.context.project_name,
            stage=self.stage.value,
            name=name,
            content=content,
            ext=ext,
            metadata=metadata,
        )
        return info.path

    def load_artifact(
        self,
        name: str,
        stage: str | None = None,
        version: int | None = None,
        ext: str = ".md",
    ) -> str | bytes | None:
        """从 ArtifactStore 加载产出物。

        Args:
            name: 产出物名称。
            stage: 阶段名称，默认为当前 Skill 对应阶段。
            version: 版本号，None 表示最新版本。
            ext: 文件扩展名。

        Returns:
            文件内容。
        """
        target_stage = stage or self.stage.value
        return self.context.artifact_store.load_content(
            project_name=self.context.project_name,
            stage=target_stage,
            name=name,
            version=version,
            ext=ext,
        )

    def get_previous_output(
        self,
        stage: WorkflowStage,
        key: str,
    ) -> Any | None:
        """获取前一阶段的输出数据。

        Args:
            stage: 目标阶段。
            key: 输出数据的键名。

        Returns:
            输出数据，不存在时返回 None。
        """
        if not self.context.state:
            return None

        record = self.context.state.stage_records.get(stage)
        if record:
            return record.outputs.get(key)
        return None

    def __repr__(self) -> str:
        """字符串表示。"""
        return f"<{self.__class__.__name__}(name={self.name}, stage={self.stage.value})>"
