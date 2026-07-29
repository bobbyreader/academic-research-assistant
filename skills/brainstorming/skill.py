"""科学头脑风暴技能模块。

提供 5 阶段研究构思工作流：理解背景 → 发散探索 → 建立联系 → 批判性评估 → 综合与后续步骤。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.skill_bridge import BaseSkill, SkillOutput, SkillStatus


@dataclass
class BrainstormingStage:
    """头脑风暴阶段定义。"""

    name: str
    description: str
    prompts: list[str] = field(default_factory=list)


class BrainstormingSkill(BaseSkill):
    """科学头脑风暴技能。

    作为研究构思伙伴，帮助生成假设、探索跨学科联系、
    挑战假设、开发方法论、识别研究空白。
    """

    name: str = "brainstorming"
    version: str = "1.0.0"
    description: str = "研究构思伙伴：生成假设、探索跨学科联系、识别研究空白"

    STAGES: list[BrainstormingStage] = [
        BrainstormingStage(
            name="understand_context",
            description="理解研究背景、兴趣、挑战和约束",
            prompts=[
                "你现在对研究的哪个方面最感兴趣？",
                "什么问题让你夜不能寐？",
                "你正在做哪些可能值得质疑的假设？",
                "有没有不符合当前模型的意外发现？",
            ],
        ),
        BrainstormingStage(
            name="divergent_exploration",
            description="不加评判地产生广泛想法",
            prompts=[
                "让我们从其他领域借鉴一些概念...",
                "如果相反的情况是真的呢？",
                "在不同尺度上这个问题会呈现什么面貌？",
                "如果能测量任何东西，你会测量什么？",
                "如果必须用19世纪的技术解决呢？",
            ],
        ),
        BrainstormingStage(
            name="connect_ideas",
            description="识别模式、主题和意外联系",
            prompts=[
                "我注意到几个想法涉及某个共同主题——如果将它们结合起来会怎样？",
                "这三种方法有什么共同点？是否有更深层的东西？",
                "你看到的最意外的联系是什么？",
            ],
        ),
        BrainstormingStage(
            name="critical_evaluation",
            description="对有前景的想法进行建设性评估",
            prompts=[
                "实际测试这个需要什么？",
                "第一个小实验是什么？",
                "可以利用哪些现有数据或工具？",
                "还需要谁参与？",
                "最大的障碍是什么，如何克服？",
            ],
        ),
        BrainstormingStage(
            name="synthesize_next_steps",
            description="凝聚洞察并创建具体前进路径",
            prompts=[
                "总结最有前景的方向",
                "建议即时后续步骤（文献搜索、试点实验、合作）",
                "捕获未来探索的关键问题",
            ],
        ),
    ]

    TECHNIQUES: dict[str, str] = {
        "cross_domain_analogy": "跨领域类比：从其他科学领域借鉴概念",
        "assumption_reversal": "假设反转：识别核心假设并翻转它们",
        "scale_shifting": "尺度转换：在不同空间/时间尺度上探索问题",
        "constraint_removal": "约束移除：想象无限制条件下的解决方案",
        "constraint_addition": "约束添加：用历史技术限制激发创新",
        "interdisciplinary_fusion": "跨学科融合：结合不同领域方法论",
        "technology_speculation": "技术推测：想象新兴技术的应用",
    }

    def execute(self, input_data: dict[str, Any]) -> SkillOutput:
        """执行头脑风暴工作流。

        Args:
            input_data: 包含以下键的字典：
                - research_topic: 研究主题
                - current_stage: 当前阶段（可选，默认为 understand_context）
                - conversation_history: 对话历史（可选）
                - user_responses: 用户对各阶段提示的回应（可选）

        Returns:
            SkillOutput 包含头脑风暴结果和下一步建议。
        """
        output = self._create_output()

        research_topic = input_data.get("research_topic", "")
        if not research_topic:
            output.add_error("缺少研究主题 (research_topic)")
            return output

        current_stage_name = input_data.get("current_stage", "understand_context")
        user_responses = input_data.get("user_responses", {})

        # 找到当前阶段
        current_stage = None
        for stage in self.STAGES:
            if stage.name == current_stage_name:
                current_stage = stage
                break

        if current_stage is None:
            output.add_error(f"未知阶段: {current_stage_name}")
            return output

        # 构建结果
        result = {
            "research_topic": research_topic,
            "current_stage": current_stage.name,
            "stage_description": current_stage.description,
            "prompts": current_stage.prompts,
            "techniques_available": list(self.TECHNIQUES.keys()),
            "next_stage": self._get_next_stage(current_stage.name),
            "all_stages": [s.name for s in self.STAGES],
        }

        # 如果有用户回应，生成阶段总结
        if user_responses:
            result["stage_summary"] = self._summarize_responses(
                current_stage, user_responses
            )

        output.data = result
        output.add_author_check("请确认当前阶段是否已充分探索，或需要继续深入")

        return output

    def _get_next_stage(self, current_stage: str) -> str | None:
        """获取下一阶段名称。"""
        stage_order = [s.name for s in self.STAGES]
        try:
            idx = stage_order.index(current_stage)
            return stage_order[idx + 1] if idx + 1 < len(stage_order) else None
        except ValueError:
            return None

    def _summarize_responses(
        self,
        stage: BrainstormingStage,
        responses: dict[str, str],
    ) -> dict[str, Any]:
        """总结用户回应（占位实现，实际应调用 LLM）。"""
        return {
            "stage": stage.name,
            "response_count": len(responses),
            "key_insights": ["[待 LLM 生成]"],
            "emerging_themes": ["[待 LLM 生成]"],
            "recommended_next_steps": ["[待 LLM 生成]"],
        }

    def get_stage_prompts(self, stage_name: str) -> list[str]:
        """获取指定阶段的提示问题列表。"""
        for stage in self.STAGES:
            if stage.name == stage_name:
                return stage.prompts
        return []

    def get_technique_description(self, technique: str) -> str:
        """获取指定头脑风暴技术的描述。"""
        return self.TECHNIQUES.get(technique, "未知技术")
