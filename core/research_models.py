"""Shared data contracts for the real research pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


class ArtifactDecodeError(ValueError):
    """产物无法还原为内存对象。

    刻意继承 :class:`ValueError` 而非 :class:`TypeError`：出错的输入来自**磁盘上的
    持久化产物**（不可信数据），而不是调用方传错的参数，因此这是数据损坏问题而非
    编程错误。调用方只需捕获这一个类型（它同时是 ``ValueError``，向上兼容既有契约），
    即可统一退化为"重新执行该阶段"。
    """


def _require_object(value: object, owner: str) -> dict[str, Any]:
    """要求是映射（dict）；否则抛 ``ValueError``。"""
    if not isinstance(value, dict):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 期望 payload 为 dict，实际为 {type(value).__name__}"
        )
    return value


def _require_str(data: dict[str, Any], field: str, owner: str) -> str:
    value = data.get(field)
    if not isinstance(value, str):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field}' 期望 str，"
            f"实际为 {type(value).__name__}"
        )
    return value


def _require_optional_int(data: dict[str, Any], field: str, owner: str) -> int | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field}' 期望 int 或 null，"
            f"实际为 {type(value).__name__}"
        )
    return value


def _require_str_list(data: dict[str, Any], field: str, owner: str) -> list[str]:
    value = data.get(field)
    if not isinstance(value, list):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{field}' 期望 list[str]，"
            f"实际为 {type(value).__name__}"
        )
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ArtifactDecodeError(
                f"{owner}.from_dict 字段 '{field}' 的第 {index} 个元素期望 str，"
                f"实际为 {type(item).__name__}"
            )
    return list(value)


@dataclass
class PaperRecord:
    """Normalized metadata returned by a literature source."""

    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str = ""
    doi: str = ""
    abstract: str = ""
    url: str = ""
    source: str = ""
    citation_count: int | None = None
    external_id: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: object) -> PaperRecord:
        """从 :meth:`to_dict` 的输出还原；结构不合法时抛 ``ValueError``。

        ``raw_data`` 是自由格式的来源原始数据，仅要求为 dict；其内部字段不做
        约束（不同检索源结构不同）。
        """
        data = _require_object(payload, cls.__name__)
        raw_data = data.get("raw_data")
        if not isinstance(raw_data, dict):
            raise ArtifactDecodeError(
                f"{cls.__name__}.from_dict 字段 'raw_data' 期望 dict，"
                f"实际为 {type(raw_data).__name__}"
            )
        return cls(
            title=_require_str(data, "title", cls.__name__),
            authors=_require_str_list(data, "authors", cls.__name__),
            year=_require_optional_int(data, "year", cls.__name__),
            journal=_require_str(data, "journal", cls.__name__),
            doi=_require_str(data, "doi", cls.__name__),
            abstract=_require_str(data, "abstract", cls.__name__),
            url=_require_str(data, "url", cls.__name__),
            source=_require_str(data, "source", cls.__name__),
            citation_count=_require_optional_int(data, "citation_count", cls.__name__),
            external_id=_require_str(data, "external_id", cls.__name__),
            raw_data=dict(raw_data),
        )


@dataclass
class SearchReport:
    """Search output, including non-fatal upstream failures.

    ``papers`` 的长度是**最终收录条数**，受 ``max_results``（总量上限）约束；
    被上限裁掉的条数与排序依据记录在下列新增字段中，保证"绝不静默丢弃"。
    所有新增字段均带默认值，向后兼容既有的构造与序列化调用。
    """

    papers: list[PaperRecord]
    errors: list[str] = field(default_factory=list)
    sources_attempted: list[str] = field(default_factory=list)
    counts_by_source: dict[str, int] = field(default_factory=dict)
    #: 去重前从各来源收到的原始条数。
    total_found: int = 0
    #: 去重后的候选条数（= 排序输入规模）。
    deduplicated_count: int = 0
    #: 因总量上限（``max_results``）被裁掉的条数。
    dropped_by_limit: int = 0
    #: 人类可读的排序依据，用于如实报告"找到了多少、留下了多少、为什么"。
    ranking_reasons: list[str] = field(default_factory=list)
