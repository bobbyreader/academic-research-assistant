from __future__ import annotations

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
