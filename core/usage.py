"""模型用量统计：只记录**提供商真实上报**的数字，绝不估算。

设计底线（不可让步）：

* 提供商没有上报用量时，记 ``reported=False`` 且 token 字段为 ``None``；
  **不得**用字符数、词数或任何启发式填充——那是把猜测伪装成成本。
* 部分调用有上报、部分没有时，累加值**只覆盖已上报的调用**，且 :meth:`UsageReport.note`
  必须让读者明白这一点，而不是把局部数字当成全部。

本模块刻意不依赖 :mod:`core.llm_client`（后者反过来从本模块导入 :class:`TokenCounts`），
因此对 ``LLMClient`` 协议的引用只出现在 ``TYPE_CHECKING`` 下，避免循环导入。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from core.research_models import ArtifactDecodeError

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期存在，避免与 llm_client 循环导入
    from core.llm_client import LLMClient


@dataclass(frozen=True)
class TokenCounts:
    """提供商上报的用量。

    ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` 是各家都有的核心口径；
    ``cached_input_tokens`` / ``reasoning_output_tokens`` 是可选的补充口径，仅在提供商
    确实给出了对应数字时才填写，缺失即为 ``None``（表示"提供商没给"，不代表 0）。
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None


@dataclass(frozen=True)
class UsageRecord:
    """一次模型调用的用量。

    ``reported`` 为 ``False`` 时，所有 token 字段必须为 ``None``——这是"该次调用
    没有拿到用量"的唯一诚实表示，而不是填 0 或估算值。
    """

    provider: str
    model: str
    operation: str
    reported: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None


@dataclass
class UsageReport:
    """一个会话内所有模型调用的用量汇总。"""

    records: list[UsageRecord] = field(default_factory=list)
    provider: str = ""
    model: str = ""

    @property
    def calls(self) -> int:
        """模型调用总次数（含未上报的）。"""
        return len(self.records)

    @property
    def reported_calls(self) -> int:
        """提供商确实上报了用量的调用次数。"""
        return sum(1 for record in self.records if record.reported)

    @property
    def unreported_calls(self) -> int:
        """未获得用量的调用次数（提供商未提供，而非为 0）。"""
        return self.calls - self.reported_calls

    @property
    def prompt_tokens(self) -> int:
        """已上报调用的输入 token 合计；不覆盖未上报的调用。"""
        return sum(
            record.prompt_tokens or 0 for record in self.records if record.reported
        )

    @property
    def completion_tokens(self) -> int:
        """已上报调用的输出 token 合计；不覆盖未上报的调用。"""
        return sum(
            record.completion_tokens or 0 for record in self.records if record.reported
        )

    @property
    def total_tokens(self) -> int:
        """已上报调用的总 token 合计；不覆盖未上报的调用。"""
        return sum(
            record.total_tokens or 0 for record in self.records if record.reported
        )

    @property
    def cached_input_tokens(self) -> int:
        """已上报调用中，提供了缓存输入口径的那些记录的合计。"""
        return sum(
            record.cached_input_tokens or 0
            for record in self.records
            if record.reported and record.cached_input_tokens is not None
        )

    @property
    def reasoning_output_tokens(self) -> int:
        """已上报调用中，提供了推理输出口径的那些记录的合计。"""
        return sum(
            record.reasoning_output_tokens or 0
            for record in self.records
            if record.reported and record.reasoning_output_tokens is not None
        )

    def note(self) -> str:
        """中文一句话摘要，**如实区分**已上报与未上报。

        典型输出：

        * 全部上报：``本次共 4 次模型调用，其中 4 次由提供商上报用量：
          输入 1234 / 输出 567 tokens。``
        * 部分上报：``本次共 4 次模型调用，其中 3 次由提供商上报用量：
          输入 900 / 输出 400 tokens（这些数字只覆盖已上报的 3 次调用）；
          另有 1 次调用未获得用量（该 provider 不提供）。``
        * 全部未上报：``本次共 2 次模型调用，均未获得用量
          （该 provider 不提供；不估算）。``
        """
        total = self.calls
        reported = self.reported_calls
        unreported = self.unreported_calls

        if total == 0:
            return "暂无模型调用记录。"

        if reported == 0:
            return (
                f"本次共 {total} 次模型调用，均未获得用量"
                "（该 provider 不提供；不估算）。"
            )

        detail = (
            f"输入 {self.prompt_tokens} / 输出 {self.completion_tokens} tokens"
        )
        if unreported == 0:
            return (
                f"本次共 {total} 次模型调用，其中 {reported} 次由提供商上报用量："
                f"{detail}。"
            )
        return (
            f"本次共 {total} 次模型调用，其中 {reported} 次由提供商上报用量："
            f"{detail}（这些数字只覆盖已上报的 {reported} 次调用）；"
            f"另有 {unreported} 次调用未获得用量（该 provider 不提供）。"
        )

    def warnings(self) -> list[str]:
        """需要提请用户注意的事项，例如存在无法上报用量的调用。"""
        messages: list[str] = []
        unreported = self.unreported_calls
        if unreported:
            messages.append(
                f"有 {unreported} 次模型调用未获得用量：提供商没有返回用量数据，"
                "因此上述 token 合计不包含这些调用；此处不做任何估算。"
            )
        return messages

    def to_dict(self) -> dict[str, Any]:
        """序列化为可落盘的普通结构。"""
        return {
            "provider": self.provider,
            "model": self.model,
            "calls": self.calls,
            "reported_calls": self.reported_calls,
            "unreported_calls": self.unreported_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_output_tokens": self.reasoning_output_tokens,
            "note": self.note(),
            "warnings": self.warnings(),
            "records": [asdict(record) for record in self.records],
        }

    @classmethod
    def from_dict(cls, payload: object) -> UsageReport:
        """从 :meth:`to_dict` 的输出还原；结构非法时抛 :class:`ArtifactDecodeError`。"""
        data = _require_object(payload, cls.__name__)
        provider = _require_str(data, "provider", cls.__name__)
        model = _require_str(data, "model", cls.__name__)
        raw_records = data.get("records")
        if not isinstance(raw_records, list):
            raise ArtifactDecodeError(
                f"{cls.__name__}.from_dict 字段 'records' 期望 list，"
                f"实际为 {type(raw_records).__name__}"
            )
        records = [_decode_record(item, index, cls.__name__) for index, item in enumerate(raw_records)]
        return cls(records=records, provider=provider, model=model)


def _require_object(value: object, owner: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 期望 payload 为 dict，实际为 {type(value).__name__}"
        )
    return value


def _require_str(data: dict[str, Any], key: str, owner: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{key}' 期望 str，实际为 {type(value).__name__}"
        )
    return value


def _require_bool(data: dict[str, Any], key: str, owner: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{key}' 期望 bool，实际为 {type(value).__name__}"
        )
    return value


def _require_optional_int(data: dict[str, Any], key: str, owner: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ArtifactDecodeError(
            f"{owner}.from_dict 字段 '{key}' 期望 int 或 null，"
            f"实际为 {type(value).__name__}"
        )
    return value


def _decode_record(item: object, index: int, owner: str) -> UsageRecord:
    where = f"{owner}.from_dict 的 records[{index}]"
    data = _require_object(item, where)
    reported = _require_bool(data, "reported", where)
    return UsageRecord(
        provider=_require_str(data, "provider", where),
        model=_require_str(data, "model", where),
        operation=_require_str(data, "operation", where),
        reported=reported,
        prompt_tokens=_require_optional_int(data, "prompt_tokens", where),
        completion_tokens=_require_optional_int(data, "completion_tokens", where),
        total_tokens=_require_optional_int(data, "total_tokens", where),
        cached_input_tokens=_require_optional_int(data, "cached_input_tokens", where),
        reasoning_output_tokens=_require_optional_int(
            data, "reasoning_output_tokens", where
        ),
    )


class UsageTrackingClient:
    """包装任意 ``LLMClient``，逐次记录调用与用量。实现 ``LLMClient`` 协议。

    这是一个**纯观测**的装饰器：

    * 原样透传 ``complete`` / ``complete_json`` 的返回值与异常——不吞、不改写；
    * 每次调用后，若内层客户端提供 ``last_usage() -> TokenCounts | None``，就记录
      其返回的**真实**用量；不存在或返回 ``None`` 时记 ``reported=False``。

    包装器自己填 ``provider`` / ``model`` / ``operation``，因为内层客户端未必把这些
    信息回吐给用量数据。
    """

    def __init__(self, inner: LLMClient, *, provider: str, model: str) -> None:
        self._inner = inner
        self._report = UsageReport(
            provider=str(provider or ""),
            model=str(model or ""),
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        result = _invoke_complete(self._inner, system_prompt, user_prompt)
        self._record("complete")
        return result

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        result = _invoke_complete_json(self._inner, system_prompt, user_prompt)
        self._record("complete_json")
        return result

    @property
    def report(self) -> UsageReport:
        return self._report

    def _record(self, operation: str) -> None:
        usage = _read_last_usage(self._inner)
        if usage is None:
            self._report.records.append(
                UsageRecord(
                    provider=self._report.provider,
                    model=self._report.model,
                    operation=operation,
                    reported=False,
                )
            )
            return
        total = usage.total_tokens
        if total is None:  # 防御：即便内层给了 TokenCounts，也不臆造 total
            total = usage.prompt_tokens + usage.completion_tokens
        self._report.records.append(
            UsageRecord(
                provider=self._report.provider,
                model=self._report.model,
                operation=operation,
                reported=True,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=total,
                cached_input_tokens=usage.cached_input_tokens,
                reasoning_output_tokens=usage.reasoning_output_tokens,
            )
        )


def _invoke_complete(
    inner: LLMClient, system_prompt: str, user_prompt: str
) -> str:
    result: str = inner.complete(system_prompt, user_prompt)
    return result


def _invoke_complete_json(
    inner: LLMClient, system_prompt: str, user_prompt: str
) -> dict[str, Any]:
    result: dict[str, Any] = inner.complete_json(system_prompt, user_prompt)
    return result


def _read_last_usage(inner: object) -> TokenCounts | None:
    """读取内层客户端的 ``last_usage()``；不存在或返回非法值时为 ``None``。

    返回 ``None`` 一律表示"未获得用量"，由调用方记 ``reported=False``——绝不估算。
    """
    reader = getattr(inner, "last_usage", None)
    if not callable(reader):
        return None
    usage = reader()
    if isinstance(usage, TokenCounts):
        return usage
    return None


__all__ = [
    "TokenCounts",
    "UsageRecord",
    "UsageReport",
    "UsageTrackingClient",
]
