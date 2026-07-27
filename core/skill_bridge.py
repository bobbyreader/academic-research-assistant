"""Skill 基类和数据桥接模块。

定义 BaseSkill 抽象基类、SkillOutput 标准输出格式，
以及 Skill 间数据传递的统一接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


# ============ 标准数据格式定义 ============


class EvidenceType(Enum):
    """证据类型枚举。"""

    EMPIRICAL = "empirical"          # 实证研究
    THEORETICAL = "theoretical"      # 理论推导
    META_ANALYSIS = "meta_analysis"  # 元分析
    REVIEW = "review"                # 综述
    CASE_STUDY = "case_study"        # 案例研究


class FigureType(Enum):
    """图表类型枚举。"""

    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    NETWORK = "network"
    FLOWCHART = "flowchart"
    TABLE = "table"
    IMAGE = "image"


@dataclass
class Citation:
    """文献引用数据格式。

    符合多种引用格式（APA/Nature/Vancouver）的通用表示。
    """

    id: str                          # 唯一标识（如 DOI 或内部 ID）
    title: str
    authors: list[str]
    year: int
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: str | None = None
    url: str | None = None
    abstract: str | None = None
    keywords: list[str] = field(default_factory=list)
    citation_count: int | None = None
    impact_factor: float | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)  # 原始 API 返回

    def to_apa(self) -> str:
        """转换为 APA 格式字符串。"""
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        parts = [f"{authors_str} ({self.year}). {self.title}."]
        if self.journal:
            parts.append(f"*{self.journal}*")
            if self.volume:
                parts.append(f", {self.volume}")
            if self.pages:
                parts.append(f", {self.pages}")
        if self.doi:
            parts.append(f". https://doi.org/{self.doi}")
        return "".join(parts)

    def to_nature(self) -> str:
        """转换为 Nature 格式字符串。"""
        authors_str = ", ".join(self.authors[:5])
        if len(self.authors) > 5:
            authors_str += " et al."
        parts = [f"{authors_str} {self.title}."]
        if self.journal:
            parts.append(f" *{self.journal}*")
            if self.volume:
                parts.append(f" **{self.volume}**")
            if self.pages:
                parts.append(f", {self.pages}")
        parts.append(f" ({self.year}).")
        return "".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Citation:
        """从字典反序列化。"""
        return cls(**data)


@dataclass
class Claim:
    """研究论断数据格式。

    表示论文中的一个核心主张或假设。
    """

    id: str
    text: str                        # 论断内容
    section: str                     # 所属章节（如 "Introduction", "Results"）
    confidence: Literal["high", "medium", "low"] = "medium"
    supporting_evidence: list[str] = field(default_factory=list)   # Evidence ID 列表
    contradicting_evidence: list[str] = field(default_factory=list)  # Evidence ID 列表
    related_citations: list[str] = field(default_factory=list)       # Citation ID 列表
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Claim:
        """从字典反序列化。"""
        return cls(**data)


@dataclass
class Evidence:
    """研究证据数据格式。

    表示支持或反驳某个 Claim 的具体证据。
    """

    id: str
    claim_id: str                    # 关联的 Claim ID
    type: EvidenceType
    source_citation_id: str          # 来源文献 Citation ID
    summary: str                     # 证据摘要
    strength: Literal["strong", "moderate", "weak"] = "moderate"
    page_reference: str | None = None
    quote: str | None = None         # 原文引用
    extracted_data: dict[str, Any] = field(default_factory=dict)  # 提取的数据点

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        data = asdict(self)
        data["type"] = self.type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        """从字典反序列化。"""
        data = data.copy()
        data["type"] = EvidenceType(data["type"])
        return cls(**data)


@dataclass
class Figure:
    """图表数据格式。

    表示论文中的图表及其元数据。
    """

    id: str
    title: str
    type: FigureType
    file_path: str | None = None     # 图片文件路径（如有）
    data_source: str | None = None   # 数据来源描述
    caption: str = ""                # 图注
    section: str = "Results"         # 所属章节
    width: float | None = None       # 宽度（cm）
    height: float | None = None      # 高度（cm）
    dpi: int = 300
    raw_data: dict[str, Any] = field(default_factory=dict)  # 原始绘图数据

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        data = asdict(self)
        data["type"] = self.type.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Figure:
        """从字典反序列化。"""
        data = data.copy()
        data["type"] = FigureType(data["type"])
        return cls(**data)


@dataclass
class ManuscriptSection:
    """论文章节数据格式。

    表示手稿中的一个逻辑章节。
    """

    id: str
    title: str
    order: int                       # 章节顺序
    content: str = ""                # 正文内容（Markdown）
    target_word_count: int | None = None
    actual_word_count: int = 0
    claims: list[str] = field(default_factory=list)      # 包含的 Claim ID 列表
    figures: list[str] = field(default_factory=list)     # 包含的 Figure ID 列表
    citations: list[str] = field(default_factory=list)   # 引用的 Citation ID 列表
    status: Literal["draft", "review", "final"] = "draft"
    parent_section: str | None = None  # 父章节 ID（用于层级结构）

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManuscriptSection:
        """从字典反序列化。"""
        return cls(**data)


# ============ Skill 基类定义 ============


class SkillStatus(Enum):
    """Status of a skill execution."""

    READY = "ready"
    READY_WITH_AUTHOR_CHECKS = "ready_with_author_checks"
    BLOCKED = "blocked"


@dataclass
class SkillOutput:
    """Standardized output container for all skill executions.

    Attributes:
        status: Execution status indicating readiness level.
        data: Main payload containing skill-specific results.
        errors: List of error messages encountered during execution.
        warnings: List of non-blocking issues to review.
        metadata: Additional context such as timestamps, versions, etc.
        author_checks: Items requiring manual author verification.
    """

    status: SkillStatus
    data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    author_checks: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        """Check if the output is ready for use."""
        return self.status in (
            SkillStatus.READY,
            SkillStatus.READY_WITH_AUTHOR_CHECKS,
        )

    def add_error(self, message: str) -> None:
        """Record an error and potentially block the output."""
        self.errors.append(message)
        if self.status != SkillStatus.BLOCKED:
            self.status = SkillStatus.BLOCKED

    def add_warning(self, message: str) -> None:
        """Record a non-blocking warning."""
        self.warnings.append(message)
        if self.status == SkillStatus.READY:
            self.status = SkillStatus.READY_WITH_AUTHOR_CHECKS

    def add_author_check(self, item: str) -> None:
        """Flag an item for manual author verification."""
        self.author_checks.append(item)
        if self.status == SkillStatus.READY:
            self.status = SkillStatus.READY_WITH_AUTHOR_CHECKS


class BaseSkill(ABC):
    """Abstract base class for all Nature Skills workflow modules.

    Each skill encapsulates a specific research workflow capability
    (e.g., brainstorming, literature review, statistics audit).
    Subclasses must implement the `execute` method with proper
    error handling and status reporting.
    """

    #: Human-readable name of the skill.
    name: str = "base_skill"

    #: Semantic version of the skill implementation.
    version: str = "0.1.0"

    #: Brief description of what the skill does.
    description: str = "Abstract base skill"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the skill with optional configuration.

        Args:
            config: Skill-specific configuration dictionary.
        """
        self.config = config or {}
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Override in subclasses to enforce required config keys.

        Raises:
            ValueError: If required configuration is missing or invalid.
        """
        pass

    @abstractmethod
    def execute(self, input_data: dict[str, Any]) -> SkillOutput:
        """Execute the skill's main workflow.

        Args:
            input_data: Input parameters and data for the workflow.

        Returns:
            SkillOutput containing results, status, and any issues.

        Raises:
            NotImplementedError: If subclass does not implement this method.
        """
        raise NotImplementedError

    def _create_output(
        self,
        status: SkillStatus = SkillStatus.READY,
        data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> SkillOutput:
        """Factory method for creating standardized SkillOutput.

        Args:
            status: Initial execution status.
            data: Main result payload.
            **kwargs: Additional fields for SkillOutput.

        Returns:
            Configured SkillOutput instance.
        """
        return SkillOutput(
            status=status,
            data=data or {},
            metadata={
                "skill_name": self.name,
                "skill_version": self.version,
                **kwargs.pop("metadata", {}),
            },
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} version={self.version!r}>"


# ============ Skill 桥接器 ============


@dataclass
class SkillMessage:
    """Skill 间传递的消息封装。"""

    source_skill: str
    target_skill: str
    data_type: str                   # 数据类型名（如 "Citation", "Claim"）
    payload: dict[str, Any]          # 序列化后的数据
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: str | None = None  # 关联 ID（用于追踪请求-响应）


class SkillBridge:
    """Skill 间数据桥接器。

    提供统一的数据序列化、验证和传递接口，
    确保不同 Skill 之间的数据格式一致性。
    """

    # 注册的数据类型映射
    _registry: dict[str, type] = {
        "Citation": Citation,
        "Claim": Claim,
        "Evidence": Evidence,
        "Figure": Figure,
        "ManuscriptSection": ManuscriptSection,
    }

    def __init__(self) -> None:
        """初始化桥接器。"""
        self._message_queue: list[SkillMessage] = []

    def register_type(self, name: str, data_class: type) -> None:
        """注册自定义数据类型。

        Args:
            name: 类型名称。
            data_class: 数据类（需实现 to_dict/from_dict）。
        """
        self._registry[name] = data_class

    def serialize(self, obj: Any) -> dict[str, Any]:
        """将数据对象序列化为字典。

        Args:
            obj: 数据对象（需为注册类型或 dataclass）。

        Returns:
            序列化后的字典。

        Raises:
            ValueError: 未注册的类型。
        """
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        elif hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        else:
            raise ValueError(f"无法序列化类型: {type(obj).__name__}")

    def deserialize(self, type_name: str, data: dict[str, Any]) -> Any:
        """将字典反序列化为数据对象。

        Args:
            type_name: 注册的类型名称。
            data: 序列化数据。

        Returns:
            反序列化后的对象。

        Raises:
            ValueError: 未注册的类型。
        """
        if type_name not in self._registry:
            raise ValueError(f"未注册的数据类型: {type_name}")

        cls = self._registry[type_name]
        if hasattr(cls, "from_dict"):
            return cls.from_dict(data)
        else:
            return cls(**data)

    def create_message(
        self,
        source_skill: str,
        target_skill: str,
        data: Any,
        correlation_id: str | None = None,
    ) -> SkillMessage:
        """创建 Skill 间传递的消息。

        Args:
            source_skill: 来源 Skill 名称。
            target_skill: 目标 Skill 名称。
            data: 数据对象。
            correlation_id: 关联 ID。

        Returns:
            封装后的消息对象。
        """
        type_name = type(data).__name__
        payload = self.serialize(data)

        return SkillMessage(
            source_skill=source_skill,
            target_skill=target_skill,
            data_type=type_name,
            payload=payload,
            correlation_id=correlation_id,
        )

    def send(self, message: SkillMessage) -> None:
        """发送消息到队列。

        Args:
            message: 消息对象。
        """
        self._message_queue.append(message)

    def receive(
        self,
        target_skill: str,
        data_type: str | None = None,
    ) -> SkillMessage | None:
        """接收指定 Skill 的消息。

        Args:
            target_skill: 目标 Skill 名称。
            data_type: 可选的数据类型过滤。

        Returns:
            匹配的消息，无匹配时返回 None。
        """
        for i, msg in enumerate(self._message_queue):
            if msg.target_skill != target_skill:
                continue
            if data_type and msg.data_type != data_type:
                continue
            return self._message_queue.pop(i)
        return None

    def receive_all(
        self,
        target_skill: str,
    ) -> list[SkillMessage]:
        """接收指定 Skill 的所有消息。

        Args:
            target_skill: 目标 Skill 名称。

        Returns:
            消息列表。
        """
        messages = [
            msg for msg in self._message_queue
            if msg.target_skill == target_skill
        ]
        self._message_queue = [
            msg for msg in self._message_queue
            if msg.target_skill != target_skill
        ]
        return messages

    def extract_payload(self, message: SkillMessage) -> Any:
        """从消息中提取并反序列化数据。

        Args:
            message: 消息对象。

        Returns:
            反序列化后的数据对象。
        """
        return self.deserialize(message.data_type, message.payload)

    def clear_queue(self) -> None:
        """清空消息队列。"""
        self._message_queue.clear()

    @property
    def queue_size(self) -> int:
        """当前队列中的消息数量。"""
        return len(self._message_queue)
