"""模型用量统计的测试：核心是**绝不估算**与**如实区分已上报/未上报**。"""

from __future__ import annotations

import pytest

from core.research_models import ArtifactDecodeError
from core.usage import (
    TokenCounts,
    UsageRecord,
    UsageReport,
    UsageTrackingClient,
)


class FakeClient:
    """返回固定正文的假客户端；``usage`` 为 ``None`` 时模拟"拿不到用量"。"""

    def __init__(self, usage: TokenCounts | None = None) -> None:
        self._usage = usage
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return "draft"

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls += 1
        return {"ok": True}

    def last_usage(self) -> TokenCounts | None:
        return self._usage


class BareClient:
    """完全不提供 ``last_usage`` 的假客户端。"""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return "bare draft"

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls += 1
        return {"bare": True}


class ExplodingClient:
    """调用即抛异常，用于验证包装器原样透传。"""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("boom")

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        raise ValueError("bad json")


def _usage(
    prompt: int = 100,
    completion: int = 20,
    *,
    cached: int | None = None,
    reasoning: int | None = None,
) -> TokenCounts:
    return TokenCounts(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cached_input_tokens=cached,
        reasoning_output_tokens=reasoning,
    )


# --------------------------------------------------------------------------- #
# TokenCounts / UsageRecord 基础
# --------------------------------------------------------------------------- #
def test_token_counts_optional_fields_default_to_none() -> None:
    counts = TokenCounts(1, 2, 3)

    assert counts.cached_input_tokens is None
    assert counts.reasoning_output_tokens is None


# --------------------------------------------------------------------------- #
# UsageReport 累加语义
# --------------------------------------------------------------------------- #
def test_report_totals_only_sum_reported_records() -> None:
    report = UsageReport(provider="p", model="m")
    report.records.append(
        UsageRecord("p", "m", "complete", True, 100, 20, 120)
    )
    report.records.append(UsageRecord("p", "m", "complete_json", False))

    assert report.calls == 2
    assert report.reported_calls == 1
    assert report.unreported_calls == 1
    assert report.prompt_tokens == 100
    assert report.completion_tokens == 20
    assert report.total_tokens == 120


def test_report_skips_none_optional_tokens_when_summing() -> None:
    report = UsageReport()
    report.records.append(
        UsageRecord("p", "m", "complete", True, 10, 1, 11, None, None)
    )
    report.records.append(
        UsageRecord("p", "m", "complete", True, 20, 2, 22, 5, 3)
    )

    assert report.cached_input_tokens == 5
    assert report.reasoning_output_tokens == 3


# --------------------------------------------------------------------------- #
# note()：如实区分
# --------------------------------------------------------------------------- #
def test_note_reports_all_calls_reported() -> None:
    report = UsageReport(provider="openai_compatible", model="gpt")
    for _ in range(4):
        report.records.append(UsageRecord("openai_compatible", "gpt", "complete", True, 300, 100, 400))

    note = report.note()

    assert "4 次模型调用" in note
    assert "4 次由提供商上报" in note
    assert "输入 1200 / 输出 400 tokens" in note
    assert "未获得用量" not in note


def test_note_distinguishes_partial_coverage() -> None:
    report = UsageReport(provider="openai_compatible", model="gpt")
    report.records.append(UsageRecord("p", "m", "complete", True, 900, 400, 1300))
    report.records.append(UsageRecord("p", "m", "complete", False))

    note = report.note()

    assert "2 次模型调用" in note
    assert "1 次由提供商上报" in note
    assert "输入 900 / 输出 400 tokens" in note
    assert "只覆盖已上报的 1 次调用" in note
    assert "另有 1 次调用未获得用量" in note


def test_note_states_all_unreported_without_estimation() -> None:
    report = UsageReport(provider="codex_cli", model="")
    report.records.append(UsageRecord("codex_cli", "", "complete", False))
    report.records.append(UsageRecord("codex_cli", "", "complete_json", False))

    note = report.note()

    assert "2 次模型调用" in note
    assert "均未获得用量" in note
    assert "该 provider 不提供" in note
    # 绝不能出现任何"估算"字样与伪造数字
    assert "估算" not in note.replace("不估算", "")
    assert "0 token" not in note


def test_note_handles_empty_report() -> None:
    assert UsageReport().note() == "暂无模型调用记录。"


def test_warnings_flags_unreported_calls() -> None:
    report = UsageReport()
    report.records.append(UsageRecord("p", "m", "complete", True, 1, 1, 2))
    report.records.append(UsageRecord("p", "m", "complete", False))

    warnings = report.warnings()

    assert len(warnings) == 1
    assert "1 次模型调用未获得用量" in warnings[0]
    assert "不做任何估算" in warnings[0]


def test_warnings_empty_when_all_reported() -> None:
    report = UsageReport()
    report.records.append(UsageRecord("p", "m", "complete", True, 1, 1, 2))

    assert report.warnings() == []


# --------------------------------------------------------------------------- #
# to_dict / from_dict 往返 + 非法负载
# --------------------------------------------------------------------------- #
def test_to_dict_round_trip() -> None:
    report = UsageReport(provider="openai_compatible", model="gpt-4o-mini")
    report.records.append(
        UsageRecord("openai_compatible", "gpt-4o-mini", "complete", True, 10, 2, 12, 1, 0)
    )
    report.records.append(UsageRecord("openai_compatible", "gpt-4o-mini", "complete_json", False))

    payload = report.to_dict()

    assert payload["calls"] == 2
    assert payload["reported_calls"] == 1
    assert payload["unreported_calls"] == 1
    assert isinstance(payload["records"], list)
    assert payload["records"][0]["reported"] is True

    restored = UsageReport.from_dict(payload)

    assert restored == report
    assert restored.to_dict() == payload


def test_from_dict_rejects_non_object() -> None:
    with pytest.raises(ArtifactDecodeError):
        UsageReport.from_dict(["not", "a", "dict"])


def test_from_dict_rejects_non_list_records() -> None:
    with pytest.raises(ArtifactDecodeError):
        UsageReport.from_dict({"provider": "p", "model": "m", "records": "nope"})


def test_from_dict_rejects_non_str_provider() -> None:
    with pytest.raises(ArtifactDecodeError):
        UsageReport.from_dict({"provider": 1, "model": "m", "records": []})


def test_from_dict_rejects_non_bool_reported() -> None:
    with pytest.raises(ArtifactDecodeError):
        UsageReport.from_dict(
            {
                "provider": "p",
                "model": "m",
                "records": [
                    {"provider": "p", "model": "m", "operation": "complete", "reported": "yes"}
                ],
            }
        )


def test_from_dict_rejects_non_int_token() -> None:
    with pytest.raises(ArtifactDecodeError):
        UsageReport.from_dict(
            {
                "provider": "p",
                "model": "m",
                "records": [
                    {
                        "provider": "p",
                        "model": "m",
                        "operation": "complete",
                        "reported": True,
                        "prompt_tokens": "many",
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    }
                ],
            }
        )


def test_artifact_decode_error_is_value_error() -> None:
    assert issubclass(ArtifactDecodeError, ValueError)


def test_to_dict_key_set_is_pinned() -> None:
    """键集是跨模块契约：pipeline 按 ``"records"`` 读逐次明细。

    这次不一致曾经是**静默的**——读错键只是跳过明细、不报错，产物看起来正常但少
    了一大块。把键集钉死，任何改名都会立刻让测试失败。
    """
    report = UsageReport(provider="p", model="m")
    expected = {
        "provider",
        "model",
        "calls",
        "reported_calls",
        "unreported_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_input_tokens",
        "reasoning_output_tokens",
        "note",
        "warnings",
        "records",
    }
    assert set(report.to_dict()) == expected
    # records 内每个元素的键集同样稳定
    record = UsageRecord(
        provider="p",
        model="m",
        operation="complete",
        reported=True,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
    )
    assert set(UsageReport(records=[record]).to_dict()["records"][0]) == {
        "provider",
        "model",
        "operation",
        "reported",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_input_tokens",
        "reasoning_output_tokens",
    }


# --------------------------------------------------------------------------- #
# UsageTrackingClient：透传与记录
# --------------------------------------------------------------------------- #
def test_wrapper_records_reported_usage() -> None:
    inner = FakeClient(_usage(1234, 567))

    tracked = UsageTrackingClient(inner, provider="openai_compatible", model="gpt-4o-mini")
    assert tracked.complete("s", "u") == "draft"

    report = tracked.report
    assert report.calls == 1
    assert report.reported_calls == 1
    assert report.prompt_tokens == 1234
    assert report.completion_tokens == 567
    assert report.total_tokens == 1801
    assert report.provider == "openai_compatible"
    assert report.model == "gpt-4o-mini"
    assert report.records[0].operation == "complete"


def test_wrapper_marks_complete_json_operation() -> None:
    inner = FakeClient(_usage(5, 5))

    tracked = UsageTrackingClient(inner, provider="gemini", model="gemini-2.5-flash")
    assert tracked.complete_json("s", "u") == {"ok": True}

    assert tracked.report.records[0].operation == "complete_json"
    assert tracked.report.prompt_tokens == 5


def test_wrapper_never_estimates_when_usage_absent() -> None:
    inner = BareClient()

    tracked = UsageTrackingClient(inner, provider="codex_cli", model="")
    tracked.complete("s", "u")
    tracked.complete("s", "u")

    report = tracked.report
    assert report.calls == 2
    assert report.reported_calls == 0
    assert report.unreported_calls == 2
    # 绝不允许用任何启发式补出数字
    assert report.prompt_tokens == 0
    assert report.completion_tokens == 0
    assert report.total_tokens == 0
    assert all(record.reported is False for record in report.records)
    assert all(record.prompt_tokens is None for record in report.records)


def test_wrapper_never_estimates_when_last_usage_returns_none() -> None:
    inner = FakeClient(None)

    tracked = UsageTrackingClient(inner, provider="codex_cli", model="")
    tracked.complete("s", "u")

    report = tracked.report
    assert report.reported_calls == 0
    assert report.prompt_tokens == 0
    assert "估算" not in report.note().replace("不估算", "")


def test_wrapper_partial_coverage_accumulates_only_reported() -> None:
    class FlakyClient:
        def __init__(self) -> None:
            self.step = 0

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            self.step += 1
            return "draft"

        def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
            raise AssertionError("unused")

        def last_usage(self) -> TokenCounts | None:
            # 第 1 次上报，第 2 次拿不到
            return _usage(500, 100) if self.step == 1 else None

    tracked = UsageTrackingClient(FlakyClient(), provider="openai_compatible", model="gpt")
    tracked.complete("s", "u")
    tracked.complete("s", "u")

    report = tracked.report
    assert report.calls == 2
    assert report.reported_calls == 1
    assert report.prompt_tokens == 500
    assert report.completion_tokens == 100
    assert "只覆盖已上报的 1 次调用" in report.note()


def test_wrapper_propagates_return_values_untouched() -> None:
    payload = {"key": "value", "nested": [1, 2, 3]}

    class JsonClient:
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            return "text"

        def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
            return payload

    tracked = UsageTrackingClient(JsonClient(), provider="p", model="m")

    assert tracked.complete("s", "u") == "text"
    assert tracked.complete_json("s", "u") is payload


def test_wrapper_propagates_exceptions_unchanged() -> None:
    tracked = UsageTrackingClient(ExplodingClient(), provider="p", model="m")

    with pytest.raises(RuntimeError, match="boom"):
        tracked.complete("s", "u")
    with pytest.raises(ValueError, match="bad json"):
        tracked.complete_json("s", "u")


def test_wrapper_records_failed_call_as_unreported_not_zero() -> None:
    """失败路径：调用**发起过**，就必须留下一条记录，且不得计为 0 开销。

    这是本模块最核心的承诺——"不知道花了多少" ≠ "没花钱"。真实事故：一次超时的
    codex 调用被完全漏记，`usage_report.json` 写成 `calls: 0`，用户会以为这次没花钱。
    因此这里钉死：失败调用必须记为 1 次「未上报」，而不是 0 次、也不是 0 开销。
    """
    tracked = UsageTrackingClient(ExplodingClient(), provider="p", model="m")

    with pytest.raises(RuntimeError):
        tracked.complete("s", "u")

    report = tracked.report
    assert report.calls == 1
    assert report.reported_calls == 0
    assert report.unreported_calls == 1
    # 未知不能被记成 0：token 字段必须是 None，合计才允许是 0（因为无任何已上报调用）。
    assert all(record.reported is False for record in report.records)
    assert all(record.prompt_tokens is None for record in report.records)
    assert report.note() != "暂无模型调用记录。"
    assert "1 次模型调用" in report.note()
    assert report.warnings() != []


def test_wrapper_records_failed_complete_json_operation() -> None:
    """`complete_json` 的失败路径同样必须留下记录，且标明是哪种操作。"""
    tracked = UsageTrackingClient(ExplodingClient(), provider="codex_cli", model="")

    with pytest.raises(ValueError, match="bad json"):
        tracked.complete_json("s", "u")

    report = tracked.report
    assert report.calls == 1
    assert report.reported_calls == 0
    assert report.records[0].operation == "complete_json"
    assert report.records[0].reported is False
    assert report.records[0].total_tokens is None


def test_wrapper_records_failure_after_a_successful_reported_call() -> None:
    """先成功一次（有用量）再失败一次：两条记录都在，且部分覆盖措辞如实。"""

    class SucceedThenExplode:
        def __init__(self) -> None:
            self.step = 0

        def complete(self, system_prompt: str, user_prompt: str) -> str:
            self.step += 1
            if self.step == 1:
                return "ok"
            raise RuntimeError("timeout")

        def last_usage(self) -> TokenCounts | None:
            return _usage(500, 100)

    tracked = UsageTrackingClient(SucceedThenExplode(), provider="codex_cli", model="")
    assert tracked.complete("s", "u") == "ok"
    with pytest.raises(RuntimeError, match="timeout"):
        tracked.complete("s", "u")

    report = tracked.report
    assert report.calls == 2
    assert report.reported_calls == 1
    assert report.unreported_calls == 1
    # 合计只覆盖已上报那次；失败那次不得被算成 0 也不能被估出来。
    assert report.prompt_tokens == 500
    assert report.completion_tokens == 100
    assert report.total_tokens == 600
    assert "只覆盖已上报的 1 次调用" in report.note()
    assert "另有 1 次调用未获得用量" in report.note()


def test_wrapper_normalizes_none_provider_and_model() -> None:
    tracked = UsageTrackingClient(
        BareClient(), provider=None, model=None  # type: ignore[arg-type]
    )
    tracked.complete("s", "u")

    assert tracked.report.provider == ""
    assert tracked.report.model == ""
