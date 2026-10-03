from __future__ import annotations

import json
from pathlib import Path

import pytest

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
