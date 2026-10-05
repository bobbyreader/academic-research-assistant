from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from core.llm_client import (
    CodexCLIClient,
    GeminiClient,
    LLMClientError,
    LLMSettings,
    OpenAICompatibleClient,
    build_llm_client,
    parse_json_response,
)
from core.usage import TokenCounts, UsageTrackingClient


class FakeLLMTransport:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def post_json(self, url: str, payload: dict, *, headers: dict | None = None) -> dict:
        self.calls.append((url, payload))
        return self.response


class FailingLLMTransport:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def post_json(self, url: str, payload: dict, *, headers: dict | None = None) -> dict:
        raise self.error


def test_parse_json_response_accepts_markdown_fenced_json() -> None:
    result = parse_json_response('```json\n{"answer": "ok"}\n```')

    assert result == {"answer": "ok"}


def test_parse_json_response_rejects_non_object_payload() -> None:
    with pytest.raises(LLMClientError, match="JSON 对象"):
        parse_json_response('["not", "an", "object"]')


def test_parse_json_response_reports_invalid_json() -> None:
    with pytest.raises(LLMClientError, match="JSON"):
        parse_json_response("not json")


def test_openai_compatible_client_extracts_content() -> None:
    transport = FakeLLMTransport(
        {"choices": [{"message": {"content": "draft content"}}]}
    )
    client = OpenAICompatibleClient(
        LLMSettings("openai_compatible", "key", "model", "https://example.test/v1"),
        transport,
    )

    assert client.complete("system", "user") == "draft content"
    assert transport.calls[0][0].endswith("/chat/completions")


def test_gemini_client_extracts_content() -> None:
    transport = FakeLLMTransport(
        {"candidates": [{"content": {"parts": [{"text": "gemini draft"}]}}]}
    )
    client = GeminiClient(
        LLMSettings("gemini", "key", "gemini-model", "https://example.test/v1beta"),
        transport,
    )

    assert client.complete("system", "user") == "gemini draft"


def test_gemini_client_explains_unsupported_location() -> None:
    from core.http_client import HttpClientError

    client = GeminiClient(
        LLMSettings("gemini", "key", "gemini-2.5-flash"),
        FailingLLMTransport(
            HttpClientError(
                'HTTP 400: {"message": "User location is not supported"}',
                status_code=400,
            )
        ),
    )

    with pytest.raises(LLMClientError, match="不支持当前网络所在地区"):
        client.complete("system", "user")


def test_gemini_client_explains_missing_model() -> None:
    from core.http_client import HttpClientError

    client = GeminiClient(
        LLMSettings("gemini", "key", "unavailable-model"),
        FailingLLMTransport(HttpClientError("HTTP 404", status_code=404)),
    )

    with pytest.raises(LLMClientError, match="找不到或无法使用模型 'unavailable-model'"):
        client.complete("system", "user")


def test_codex_cli_client_uses_read_only_ephemeral_project_workspace(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_runner(command: list[str], prompt: str, output_path: Path, timeout: int) -> None:
        captured.update(
            {"command": command, "prompt": prompt, "output_path": output_path, "timeout": timeout}
        )
        output_path.write_text("codex draft", encoding="utf-8")

    client = CodexCLIClient(
        LLMSettings("codex_cli", "", "", timeout=45),
        workspace_dir=tmp_path,
        executable="/usr/local/bin/codex",
        runner=fake_runner,
    )

    assert client.complete("system rules", "research evidence") == "codex draft"
    command = captured["command"]
    assert isinstance(command, list)
    assert "--sandbox" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--ephemeral" in command
    assert "--ignore-rules" in command
    assert str(tmp_path) in command
    assert "system rules" in captured["prompt"]
    assert "research evidence" in captured["prompt"]


def test_codex_cli_client_reports_missing_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("core.llm_client.shutil.which", lambda _: None)
    client = CodexCLIClient(
        LLMSettings("codex_cli", "", ""),
        workspace_dir=tmp_path,
    )

    with pytest.raises(LLMClientError, match="未找到 Codex CLI"):
        client.complete("system", "user")


def test_build_codex_cli_client_uses_explicit_executable_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_CLI_PATH", "/opt/local/bin/codex")

    client = build_llm_client(provider="codex_cli", workspace_dir=tmp_path)

    assert isinstance(client, CodexCLIClient)
    assert client.executable == "/opt/local/bin/codex"


# --------------------------------------------------------------------------- #
# 用量解析：openai_compatible
# --------------------------------------------------------------------------- #
def test_openai_client_parses_reported_usage() -> None:
    transport = FakeLLMTransport(
        {
            "choices": [{"message": {"content": "draft"}}],
            "usage": {
                "prompt_tokens": 1234,
                "completion_tokens": 567,
                "total_tokens": 1801,
                "prompt_tokens_details": {"cached_tokens": 256},
                "completion_tokens_details": {"reasoning_tokens": 128},
            },
        }
    )
    client = OpenAICompatibleClient(
        LLMSettings("openai_compatible", "key", "model", "https://example.test/v1"),
        transport,
    )

    assert client.complete("system", "user") == "draft"

    usage = client.last_usage()
    assert usage == TokenCounts(1234, 567, 1801, 256, 128)


def test_openai_client_usage_missing_is_none() -> None:
    transport = FakeLLMTransport({"choices": [{"message": {"content": "draft"}}]})
    client = OpenAICompatibleClient(
        LLMSettings("openai_compatible", "key", "model", "https://example.test/v1"),
        transport,
    )

    assert client.complete("system", "user") == "draft"
    assert client.last_usage() is None


def test_openai_client_usage_wrong_types_is_none() -> None:
    transport = FakeLLMTransport(
        {
            "choices": [{"message": {"content": "draft"}}],
            "usage": {"prompt_tokens": "many", "completion_tokens": 1, "total_tokens": 2},
        }
    )
    client = OpenAICompatibleClient(
        LLMSettings("openai_compatible", "key", "model", "https://example.test/v1"),
        transport,
    )

    assert client.complete("system", "user") == "draft"
    assert client.last_usage() is None


def test_openai_client_usage_total_derived_from_reported_parts() -> None:
    transport = FakeLLMTransport(
        {
            "choices": [{"message": {"content": "draft"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }
    )
    client = OpenAICompatibleClient(
        LLMSettings("openai_compatible", "key", "model", "https://example.test/v1"),
        transport,
    )

    assert client.complete("system", "user") == "draft"
    usage = client.last_usage()
    assert usage is not None
    assert usage.total_tokens == 120


def test_openai_client_usage_does_not_leak_between_calls() -> None:
    class SwitchingTransport:
        def __init__(self) -> None:
            self.calls = 0

        def post_json(
            self, url: str, payload: dict, *, headers: dict | None = None
        ) -> dict:
            self.calls += 1
            response: dict = {"choices": [{"message": {"content": "draft"}}]}
            if self.calls == 1:
                response["usage"] = {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                }
            return response

    client = OpenAICompatibleClient(
        LLMSettings("openai_compatible", "key", "model", "https://example.test/v1"),
        SwitchingTransport(),
    )

    client.complete("system", "user")
    assert client.last_usage() == TokenCounts(10, 1, 11)
    client.complete("system", "user")
    # 第二次没有 usage：必须清空，不能复用上一次的数字
    assert client.last_usage() is None


# --------------------------------------------------------------------------- #
# 用量解析：gemini
# --------------------------------------------------------------------------- #
def test_gemini_client_parses_reported_usage() -> None:
    transport = FakeLLMTransport(
        {
            "candidates": [{"content": {"parts": [{"text": "gemini draft"}]}}],
            "usageMetadata": {
                "promptTokenCount": 800,
                "candidatesTokenCount": 200,
                "totalTokenCount": 1000,
                "cachedContentTokenCount": 320,
                "thoughtsTokenCount": 64,
            },
        }
    )
    client = GeminiClient(
        LLMSettings("gemini", "key", "gemini-model", "https://example.test/v1beta"),
        transport,
    )

    assert client.complete("system", "user") == "gemini draft"
    assert client.last_usage() == TokenCounts(800, 200, 1000, 320, 64)


def test_gemini_client_usage_missing_is_none() -> None:
    transport = FakeLLMTransport(
        {"candidates": [{"content": {"parts": [{"text": "gemini draft"}]}}]}
    )
    client = GeminiClient(
        LLMSettings("gemini", "key", "gemini-model", "https://example.test/v1beta"),
        transport,
    )

    assert client.complete("system", "user") == "gemini draft"
    assert client.last_usage() is None


def test_gemini_client_usage_wrong_types_is_none() -> None:
    transport = FakeLLMTransport(
        {
            "candidates": [{"content": {"parts": [{"text": "gemini draft"}]}}],
            "usageMetadata": {
                "promptTokenCount": 800,
                "candidatesTokenCount": None,
            },
        }
    )
    client = GeminiClient(
        LLMSettings("gemini", "key", "gemini-model", "https://example.test/v1beta"),
        transport,
    )

    assert client.complete("system", "user") == "gemini draft"
    assert client.last_usage() is None


# --------------------------------------------------------------------------- #
# 用量解析：codex_cli（实证：--json 的 turn.completed 事件带 usage）
# --------------------------------------------------------------------------- #
def test_codex_cli_command_enables_json_event_stream(tmp_path: Path) -> None:
    client = CodexCLIClient(
        LLMSettings("codex_cli", "", ""),
        workspace_dir=tmp_path,
        executable="/usr/local/bin/codex",
    )

    command = client._command(tmp_path / "out.txt")

    assert "--json" in command


def test_codex_cli_parses_usage_from_turn_completed(tmp_path: Path) -> None:
    usage_event = json.dumps(
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 23365,
                "cached_input_tokens": 7168,
                "cache_write_input_tokens": 0,
                "output_tokens": 6,
                "reasoning_output_tokens": 0,
            },
        }
    )
    stdout = (
        '{"type":"thread.started","thread_id":"abc"}\n'
        '{"type":"turn.started"}\n'
        '{"type":"item.completed","item":{"type":"agent_message","text":"PONG"}}\n'
        f"{usage_event}"
    )

    def fake_runner(command: list[str], prompt: str, output_path: Path, timeout: int) -> str:
        output_path.write_text("codex draft", encoding="utf-8")
        return stdout

    client = CodexCLIClient(
        LLMSettings("codex_cli", "", ""),
        workspace_dir=tmp_path,
        executable="/usr/local/bin/codex",
        runner=fake_runner,
    )

    assert client.complete("system", "user") == "codex draft"

    usage = client.last_usage()
    assert usage == TokenCounts(23365, 6, 23371, 7168, 0)


def test_codex_cli_usage_none_when_event_missing(tmp_path: Path) -> None:
    def fake_runner(command: list[str], prompt: str, output_path: Path, timeout: int) -> str:
        output_path.write_text("codex draft", encoding="utf-8")
        return '{"type":"thread.started","thread_id":"abc"}'

    client = CodexCLIClient(
        LLMSettings("codex_cli", "", ""),
        workspace_dir=tmp_path,
        executable="/usr/local/bin/codex",
        runner=fake_runner,
    )

    assert client.complete("system", "user") == "codex draft"
    assert client.last_usage() is None


def test_codex_cli_usage_ignores_non_json_noise_lines(tmp_path: Path) -> None:
    usage_event = json.dumps(
        {
            "type": "turn.completed",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    )
    stdout = (
        "Reading additional input from stdin...\n"
        f"{usage_event}\n"
        "not json at all"
    )

    def fake_runner(command: list[str], prompt: str, output_path: Path, timeout: int) -> str:
        output_path.write_text("codex draft", encoding="utf-8")
        return stdout

    client = CodexCLIClient(
        LLMSettings("codex_cli", "", ""),
        workspace_dir=tmp_path,
        executable="/usr/local/bin/codex",
        runner=fake_runner,
    )

    assert client.complete("system", "user") == "codex draft"
    assert client.last_usage() == TokenCounts(10, 2, 12)


def test_codex_cli_usage_none_when_usage_fields_wrong_type(tmp_path: Path) -> None:
    stdout = '{"type":"turn.completed","usage":{"input_tokens":"many","output_tokens":2}}'

    def fake_runner(command: list[str], prompt: str, output_path: Path, timeout: int) -> str:
        output_path.write_text("codex draft", encoding="utf-8")
        return stdout

    client = CodexCLIClient(
        LLMSettings("codex_cli", "", ""),
        workspace_dir=tmp_path,
        executable="/usr/local/bin/codex",
        runner=fake_runner,
    )

    assert client.complete("system", "user") == "codex draft"
    assert client.last_usage() is None


def test_codex_cli_usage_parse_failure_never_breaks_complete(tmp_path: Path) -> None:
    def fake_runner(command: list[str], prompt: str, output_path: Path, timeout: int) -> str:
        output_path.write_text("codex draft", encoding="utf-8")
        return "{ this is not valid json"

    client = CodexCLIClient(
        LLMSettings("codex_cli", "", ""),
        workspace_dir=tmp_path,
        executable="/usr/local/bin/codex",
        runner=fake_runner,
    )

    assert client.complete("system", "user") == "codex draft"
    assert client.last_usage() is None


def test_codex_cli_runner_returning_none_yields_no_usage(tmp_path: Path) -> None:
    def fake_runner(command: list[str], prompt: str, output_path: Path, timeout: int) -> None:
        output_path.write_text("codex draft", encoding="utf-8")

    client = CodexCLIClient(
        LLMSettings("codex_cli", "", ""),
        workspace_dir=tmp_path,
        executable="/usr/local/bin/codex",
        runner=fake_runner,
    )

    assert client.complete("system", "user") == "codex draft"
    assert client.last_usage() is None


# --------------------------------------------------------------------------- #
# 包装器 + 真实客户端联动：codex_cli 未上报 → 绝不估算
# --------------------------------------------------------------------------- #
def test_wrapper_over_codex_cli_records_unreported_without_estimation(
    tmp_path: Path,
) -> None:
    def fake_runner(command: list[str], prompt: str, output_path: Path, timeout: int) -> None:
        output_path.write_text("codex draft", encoding="utf-8")

    inner = CodexCLIClient(
        LLMSettings("codex_cli", "", ""),
        workspace_dir=tmp_path,
        executable="/usr/local/bin/codex",
        runner=fake_runner,
    )
    tracked = UsageTrackingClient(inner, provider="codex_cli", model="")

    assert tracked.complete("system", "user") == "codex draft"

    report = tracked.report
    assert report.reported_calls == 0
    assert report.prompt_tokens == 0
    assert "未获得用量" in report.note()
    assert "估算" not in report.note().replace("不估算", "")


def test_wrapper_over_openai_reports_real_usage() -> None:
    transport = FakeLLMTransport(
        {
            "choices": [{"message": {"content": "draft"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
    )
    inner = OpenAICompatibleClient(
        LLMSettings("openai_compatible", "key", "model", "https://example.test/v1"),
        transport,
    )
    tracked = UsageTrackingClient(inner, provider="openai_compatible", model="model")

    tracked.complete("system", "user")

    report = tracked.report
    assert report.reported_calls == 1
    assert report.prompt_tokens == 10
    assert report.completion_tokens == 2
    assert report.total_tokens == 12


# --------------------------------------------------------------------------- #
# 跨来源一致性：LLMSettings.timeout 必须等于 config/settings.yaml 的
# llm.timeout_seconds。这两处是**同一个旋钮的两个来源**：一旦不一致，用户
# "删掉配置键"就会静默改变行为——这正是 Phase 8 发现的那类缺陷（DEFAULTS 声称
# llm.provider 默认 codex_cli，而代码真实回退是 gemini）。本测试钉死这种漂移。
# --------------------------------------------------------------------------- #
def _yaml_timeout_seconds() -> int:
    """从 config/settings.yaml 读取 llm.timeout_seconds 的原始值。"""
    settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    data = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    return data["llm"]["timeout_seconds"]


def test_llm_settings_timeout_matches_settings_yaml() -> None:
    """LLMSettings.timeout 必须等于 config/settings.yaml 的 llm.timeout_seconds。

    二者是同一个超时旋钮的两个来源（settings.yaml 由 ResearchService 原始 dict 读取、
    LLMSettings 是代码内回退默认）。**它们不一致时，"删掉配置键"会静默改变行为**——
    这类漂移必须让测试变红，而不能悄悄发生。
    """
    assert LLMSettings.timeout == _yaml_timeout_seconds()


def test_build_llm_client_default_timeout_matches_settings_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未显式传 timeout 时，构建出的客户端必须取自与 settings.yaml 一致的默认值。

    钉死三个 provider 分支共享的 ``timeout if timeout is not None else
    LLMSettings.timeout`` 回退路径——任一分支若硬编码别的值，测试即失败。
    """
    for env_var in ("ARS_LLM_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(env_var, raising=False)
    expected = _yaml_timeout_seconds()

    for provider in ("openai_compatible", "gemini", "codex_cli"):
        client = build_llm_client(
            provider=provider, api_key="test-key", workspace_dir=tmp_path
        )
        assert client.settings.timeout == expected, provider


# --------------------------------------------------------------------------- #
# 超时错误消息：必须给出**配置的上限**与**实测耗时**（不得编造）
# --------------------------------------------------------------------------- #
def test_codex_cli_timeout_message_includes_configured_limit_and_elapsed(
    tmp_path: Path,
) -> None:
    """超时消息必须含配置上限（取自 settings.timeout）与实测耗时。

    用一个很小的假超时——**绝不让测试真的等 600 秒**。上限用远小于 600 的值，
    以证明该数字确实来自 ``settings.timeout``（配置）而非写死的字面量。
    """
    def timeout_runner(
        command: list[str], prompt: str, output_path: Path, timeout: int
    ) -> str:
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    client = CodexCLIClient(
        LLMSettings("codex_cli", "", "", timeout=1),
        workspace_dir=tmp_path,
        executable="/usr/local/bin/codex",
        runner=timeout_runner,
    )

    with pytest.raises(LLMClientError) as excinfo:
        client.complete("system", "user")

    message = str(excinfo.value)
    # 上限取自配置（此处 1），不是写死 600。
    assert "llm.timeout_seconds=1" in message
    # 实测耗时出现在消息里（真实 time.monotonic 差值，格式如 "0.0 秒"）。
    assert "秒" in message
    assert "已等待" in message
    # 不可只写"请稍后重试"——用户必须能据此判断该改什么。
    assert "config/settings.yaml" in message


def test_codex_cli_timeout_message_uses_actual_configured_value_not_literal(
    tmp_path: Path,
) -> None:
    """换一个配置上限，消息里的上限随之改变——数字来自配置，不是硬编码。"""
    def timeout_runner(
        command: list[str], prompt: str, output_path: Path, timeout: int
    ) -> str:
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    client = CodexCLIClient(
        LLMSettings("codex_cli", "", "", timeout=123),
        workspace_dir=tmp_path,
        executable="/usr/local/bin/codex",
        runner=timeout_runner,
    )

    with pytest.raises(LLMClientError, match="llm.timeout_seconds=123"):
        client.complete("system", "user")

    # 超时链保持：from exc，且异常类型仍为可读的 LLMClientError（非崩溃）。
    assert issubclass(LLMClientError, RuntimeError)
