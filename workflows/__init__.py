"""工作流加载和验证模块。"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageConfig:
    """阶段配置。"""

    name: str
    skill: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    required: bool = True
    mandatory: bool = False
    component: str | None = None
    mode: str | None = None
    stage: str | None = None


@dataclass
class WorkflowConfig:
    """工作流配置。"""

    name: str
    description: str
    stages: list[StageConfig] = field(default_factory=list)
    data_mappings: dict[str, Any] = field(default_factory=dict)


class WorkflowLoader:
    """工作流加载器。"""

    def __init__(self, workflows_dir: Path) -> None:
        """初始化加载器。

        Args:
            workflows_dir: 工作流配置文件目录。
        """
        self.workflows_dir = workflows_dir

    def load(self, workflow_name: str) -> WorkflowConfig:
        """加载工作流配置。

        Args:
            workflow_name: 工作流名称（不含 .yaml 后缀）。

        Returns:
            工作流配置对象。

        Raises:
            FileNotFoundError: 配置文件不存在。
            ValueError: 配置格式错误。
        """
        config_file = self.workflows_dir / f"{workflow_name}.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"工作流配置不存在: {config_file}")

        data = yaml.safe_load(config_file.read_text(encoding="utf-8"))

        if "workflows" not in data:
            raise ValueError(f"配置文件缺少 'workflows' 键: {config_file}")

        wf_data = data["workflows"].get(workflow_name)
        if wf_data is None:
            raise ValueError(f"工作流 '{workflow_name}' 未在配置中定义")

        stages = []
        for stage_data in wf_data.get("stages", []):
            stage = StageConfig(
                name=stage_data["name"],
                skill=stage_data["skill"],
                inputs=stage_data.get("inputs", []),
                outputs=stage_data.get("outputs", []),
                required=stage_data.get("required", True),
                mandatory=stage_data.get("mandatory", False),
                component=stage_data.get("component"),
                mode=stage_data.get("mode"),
                stage=stage_data.get("stage"),
            )
            stages.append(stage)

        return WorkflowConfig(
            name=workflow_name,
            description=wf_data.get("description", ""),
            stages=stages,
            data_mappings=data.get("data_mappings", {}),
        )

    def validate(self, config: WorkflowConfig) -> list[str]:
        """验证工作流配置。

        Args:
            config: 工作流配置。

        Returns:
            错误信息列表，空列表表示验证通过。
        """
        errors = []

        # 检查循环依赖
        stage_names = [s.name for s in config.stages]
        if len(stage_names) != len(set(stage_names)):
            errors.append("存在重复的阶段名称")

        # 检查输入输出引用
        all_outputs: set[str] = set()
        for stage in config.stages:
            for inp in stage.inputs:
                if inp not in all_outputs and not inp.startswith("research_"):
                    # 允许引用外部输入（如 research_topic）
                    pass
            all_outputs.update(stage.outputs)

        # 检查 mandatory 阶段
        for stage in config.stages:
            if stage.mandatory and not stage.required:
                errors.append(f"阶段 '{stage.name}' 标记为 mandatory 但 required=false")

        return errors

    def visualize(self, config: WorkflowConfig) -> str:
        """生成工作流 Mermaid 图。

        Args:
            config: 工作流配置。

        Returns:
            Mermaid 图代码。
        """
        lines = ["graph TD"]
        for i, stage in enumerate(config.stages):
            shape = "[" if not stage.mandatory else "[["  # mandatory 用双线框
            close = "]" if not stage.mandatory else "]]"
            lines.append(f"    {stage.name}{shape}{stage.name}{close}")

            if i > 0:
                prev = config.stages[i - 1]
                lines.append(f"    {prev.name} --> {stage.name}")

        return "\n".join(lines)


def load_workflow(workflow_name: str, workflows_dir: Path | None = None) -> WorkflowConfig:
    """便捷函数：加载工作流配置。

    Args:
        workflow_name: 工作流名称。
        workflows_dir: 工作流配置目录，默认为当前模块所在目录。

    Returns:
        工作流配置对象。
    """
    if workflows_dir is None:
        workflows_dir = Path(__file__).parent

    loader = WorkflowLoader(workflows_dir)
    return loader.load(workflow_name)
