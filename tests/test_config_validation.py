"""Tests for core.config_validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from core.config_validation import (
    ConfigValidationError,
    SettingsValidation,
    load_validated_settings,
    validate_settings,
)

# A settings mapping shaped like the repository's config/settings.yaml, i.e. it
# only contains keys that some code path actually reads.
VALID_SETTINGS: dict = {
    "api_keys": {"pubmed_email": "", "crossref_email": ""},
    "llm": {
        "provider": "codex_cli",
        "model": "",
        "base_url": "",
        "timeout_seconds": 120,
    },
    "figures": {"default_dpi": 300},
    "review": {"reviewer_count": 1},
}


def _messages_contain(messages: list[str], needle: str) -> bool:
    return any(needle in message for message in messages)


def test_config_validation_error_is_value_error() -> None:
    assert issubclass(ConfigValidationError, ValueError)


def test_valid_settings_have_no_errors() -> None:
    result = validate_settings(VALID_SETTINGS)

    assert result.valid is True
    assert result.errors == []


def test_provider_aliases_are_accepted() -> None:
    for provider in ("openai", "compatible", "codex"):
        settings = {**VALID_SETTINGS, "llm": {"provider": provider}}
        assert validate_settings(settings).valid is True, provider


def test_non_int_reviewer_count_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "review": {"reviewer_count": "three"}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "review.reviewer_count")


def test_bool_reviewer_count_is_rejected() -> None:
    settings = {**VALID_SETTINGS, "review": {"reviewer_count": True}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "review.reviewer_count")


def test_zero_reviewer_count_is_valid() -> None:
    settings = {**VALID_SETTINGS, "review": {"reviewer_count": 0}}

    assert validate_settings(settings).valid is True


def test_unknown_provider_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "llm": {"provider": "gpt4"}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "llm.provider")


def test_unknown_llm_key_is_a_warning_only() -> None:
    settings = {**VALID_SETTINGS, "llm": {**VALID_SETTINGS["llm"], "whatever": 1}}

    result = validate_settings(settings)

    assert result.valid is True
    assert result.errors == []
    assert _messages_contain(result.warnings, "llm.whatever")


def test_unknown_top_level_key_is_a_warning_only() -> None:
    settings = {**VALID_SETTINGS, "brand_new_section": {"enabled": True}}

    result = validate_settings(settings)

    assert result.valid is True
    assert _messages_contain(result.warnings, "brand_new_section")


def test_non_mapping_input_is_an_error() -> None:
    result = validate_settings([])

    assert result.valid is False
    assert _messages_contain(result.errors, "settings 必须是映射对象")


def test_non_mapping_section_names_the_section() -> None:
    settings = {**VALID_SETTINGS, "review": "off"}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "review")


def test_negative_timeout_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "llm": {"timeout_seconds": -1}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "llm.timeout_seconds")


def test_non_positive_dpi_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "figures": {"default_dpi": 0}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "figures.default_dpi")


def test_non_string_api_key_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "api_keys": {"pubmed_email": 123}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "api_keys.pubmed_email")


def test_empty_api_key_is_valid() -> None:
    settings = {**VALID_SETTINGS, "api_keys": {"semantic_scholar_key": ""}}

    assert validate_settings(settings).valid is True


def test_load_validated_settings_reads_real_yaml(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        yaml.safe_dump(VALID_SETTINGS), encoding="utf-8"
    )

    settings, validation = load_validated_settings(tmp_path)

    assert settings["llm"]["provider"] == "codex_cli"
    assert validation.valid is True
    assert validation.errors == []


def test_load_validated_settings_missing_file_does_not_raise(tmp_path: Path) -> None:
    settings, validation = load_validated_settings(tmp_path)

    assert settings == {}
    assert validation.valid is True
    assert validation.warnings


def test_load_validated_settings_reports_bad_yaml_values(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        "review:\n  reviewer_count: \"three\"\n", encoding="utf-8"
    )

    _, validation = load_validated_settings(tmp_path)

    assert validation.valid is False
    assert _messages_contain(validation.errors, "review.reviewer_count")


def test_validation_is_deterministic() -> None:
    settings = {
        "review": {"reviewer_count": "three", "mystery": 1},
        "llm": {"provider": "gpt4", "whatever": 2},
        "figures": {"default_dpi": "big"},
    }

    first = validate_settings(settings).to_dict()
    second = validate_settings(settings).to_dict()

    assert first == second


def test_to_dict_shape() -> None:
    payload = SettingsValidation(errors=["e"], warnings=["w"]).to_dict()

    assert payload == {"valid": False, "errors": ["e"], "warnings": ["w"]}
