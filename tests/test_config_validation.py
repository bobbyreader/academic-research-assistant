"""Tests for core.config_validation and core.config_loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.config_loader import (
    DEFAULTS,
    get_bool,
    get_float,
    get_int,
    get_int_pair,
    get_mapping,
    get_str,
    get_str_list,
    load_settings,
)
from core.config_validation import (
    KNOWN_SOURCES,
    ConfigValidationError,
    SettingsValidation,
    load_validated_settings,
    validate_settings,
)

# A settings mapping shaped like the repository's config/settings.yaml, i.e. it
# only contains keys that some code path actually reads.
VALID_SETTINGS: dict = {
    "api_keys": {"pubmed_email": "", "semantic_scholar_key": "", "crossref_email": ""},
    "llm": {
        "provider": "codex_cli",
        "model": "",
        "base_url": "",
        "timeout_seconds": 120,
    },
    "paths": {
        "projects_dir": "projects",
        "output_dir": "output",
        "scripts_dir": "scripts",
    },
    "search": {
        "default_sources": ["crossref", "pubmed", "semantic_scholar"],
        "max_results": 10,
        "year_range": [0, 0],
    },
    "citation": {
        "default_style": "numeric",
        "export_formats": ["md", "pdf", "pptx"],
    },
    "writing": {
        "default_paper_type": "",
        "default_language": "",
        "bilingual_abstract": False,
        "style_guide": "",
    },
    "export": {
        "default_format": "md",
        "pdf_engine": "auto",
        "pptx_template": "",
        "include_speaker_notes": False,
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
        "file": "",
    },
    "figures": {
        "default_dpi": 300,
        "default_journal": "",
        "default_format": "png",
        "color_palette": "default",
        "font_family": "",
        "font_size_pt": 10,
    },
    "review": {
        "reviewer_count": 1,
        "include_devil_advocate": False,
        "consensus_threshold": 0.0,
        "score_scale": "0-100",
    },
}


def _messages_contain(messages: list[str], needle: str) -> bool:
    return any(needle in message for message in messages)


def _dotted_keys(mapping: dict, prefix: str = "") -> set[str]:
    """Flatten a nested mapping into dotted keys (leaves only)."""
    keys: set[str] = set()
    for key, value in mapping.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            keys |= _dotted_keys(value, dotted)
        else:
            keys.add(dotted)
    return keys


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


# ---------------------------------------------------------------------------
# paths.*
# ---------------------------------------------------------------------------


def test_paths_empty_string_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "paths": {"projects_dir": "  "}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "paths.projects_dir")


def test_paths_non_string_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "paths": {"output_dir": 7}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "paths.output_dir")


def test_paths_unknown_key_is_a_warning() -> None:
    settings = {**VALID_SETTINGS, "paths": {**VALID_SETTINGS["paths"], "tmp_dir": "tmp"}}

    result = validate_settings(settings)

    assert result.valid is True
    assert _messages_contain(result.warnings, "paths.tmp_dir")


# ---------------------------------------------------------------------------
# search.*
# ---------------------------------------------------------------------------


def test_search_unknown_source_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "search": {**VALID_SETTINGS["search"], "default_sources": ["google"]}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "search.default_sources")


def test_search_empty_sources_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "search": {**VALID_SETTINGS["search"], "default_sources": []}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "search.default_sources")


def test_search_non_string_source_element_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "search": {**VALID_SETTINGS["search"], "default_sources": ["crossref", 1]}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "search.default_sources")


def test_search_known_sources_are_accepted() -> None:
    settings = {**VALID_SETTINGS, "search": {**VALID_SETTINGS["search"], "default_sources": sorted(KNOWN_SOURCES)}}

    assert validate_settings(settings).valid is True


def test_search_non_positive_max_results_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "search": {**VALID_SETTINGS["search"], "max_results": 0}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "search.max_results")


def test_search_year_range_zero_zero_is_valid() -> None:
    settings = {**VALID_SETTINGS, "search": {**VALID_SETTINGS["search"], "year_range": [0, 0]}}

    assert validate_settings(settings).valid is True


def test_search_year_range_bounded_is_valid() -> None:
    settings = {**VALID_SETTINGS, "search": {**VALID_SETTINGS["search"], "year_range": [2019, 2024]}}

    assert validate_settings(settings).valid is True


def test_search_year_range_start_after_end_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "search": {**VALID_SETTINGS["search"], "year_range": [2024, 2019]}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "search.year_range")


def test_search_year_range_wrong_length_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "search": {**VALID_SETTINGS["search"], "year_range": [2020]}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "search.year_range")


def test_search_year_range_negative_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "search": {**VALID_SETTINGS["search"], "year_range": [-1, 2024]}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "search.year_range")


def test_search_unknown_key_is_a_warning() -> None:
    settings = {**VALID_SETTINGS, "search": {**VALID_SETTINGS["search"], "boost": 3}}

    result = validate_settings(settings)

    assert result.valid is True
    assert _messages_contain(result.warnings, "search.boost")


# ---------------------------------------------------------------------------
# citation.*
# ---------------------------------------------------------------------------


def test_citation_default_style_not_supported_is_an_error() -> None:
    settings = {
        **VALID_SETTINGS,
        "citation": {**VALID_SETTINGS["citation"], "default_style": "apa"},
    }

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "citation.default_style")


def test_citation_supported_styles_is_now_an_unknown_key_warning() -> None:
    # supported_styles 已移除（样式是代码能力事实，不是用户偏好）。
    settings = {
        **VALID_SETTINGS,
        "citation": {**VALID_SETTINGS["citation"], "supported_styles": ["numeric"]},
    }

    result = validate_settings(settings)

    assert result.valid is True
    assert _messages_contain(result.warnings, "citation.supported_styles")


def test_citation_export_formats_empty_is_an_error() -> None:
    settings = {
        **VALID_SETTINGS,
        "citation": {**VALID_SETTINGS["citation"], "export_formats": []},
    }

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "citation.export_formats")


def test_citation_export_formats_unknown_format_is_an_error() -> None:
    settings = {
        **VALID_SETTINGS,
        "citation": {**VALID_SETTINGS["citation"], "export_formats": ["md", "docx"]},
    }

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "citation.export_formats")
    assert _messages_contain(result.errors, "docx")


def test_citation_known_styles_are_accepted() -> None:
    from core.citation_styles import known_styles

    for style in known_styles():
        settings = {
            **VALID_SETTINGS,
            "citation": {**VALID_SETTINGS["citation"], "default_style": style},
        }
        assert validate_settings(settings).valid is True, style


# ---------------------------------------------------------------------------
# 跨节一致性：export.default_format ∈ citation.export_formats
# ---------------------------------------------------------------------------


def test_export_default_format_not_in_export_formats_is_an_error() -> None:
    settings = {
        **VALID_SETTINGS,
        "citation": {**VALID_SETTINGS["citation"], "export_formats": ["md", "pdf"]},
        "export": {**VALID_SETTINGS["export"], "default_format": "pptx"},
    }

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "export.default_format")


def test_export_default_format_in_export_formats_is_valid() -> None:
    settings = {
        **VALID_SETTINGS,
        "citation": {**VALID_SETTINGS["citation"], "export_formats": ["md", "pdf"]},
        "export": {**VALID_SETTINGS["export"], "default_format": "pdf"},
    }

    assert validate_settings(settings).valid is True


def test_citation_unknown_key_is_a_warning() -> None:
    settings = {
        **VALID_SETTINGS,
        "citation": {**VALID_SETTINGS["citation"], "csl": "ieee"},
    }

    result = validate_settings(settings)

    assert result.valid is True
    assert _messages_contain(result.warnings, "citation.csl")


# ---------------------------------------------------------------------------
# writing.*
# ---------------------------------------------------------------------------


def test_writing_bad_language_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "writing": {**VALID_SETTINGS["writing"], "default_language": "fr"}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "writing.default_language")


def test_writing_empty_paper_type_is_valid() -> None:
    # 留空 = 不追加文体约束，沿用既有提示词。
    settings = {**VALID_SETTINGS, "writing": {**VALID_SETTINGS["writing"], "default_paper_type": ""}}

    assert validate_settings(settings).valid is True


def test_writing_empty_language_is_valid() -> None:
    settings = {**VALID_SETTINGS, "writing": {**VALID_SETTINGS["writing"], "default_language": ""}}

    assert validate_settings(settings).valid is True


def test_writing_non_string_paper_type_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "writing": {**VALID_SETTINGS["writing"], "default_paper_type": 7}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "writing.default_paper_type")


def test_writing_non_bool_bilingual_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "writing": {**VALID_SETTINGS["writing"], "bilingual_abstract": "yes"}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "writing.bilingual_abstract")


def test_writing_unknown_key_is_a_warning() -> None:
    settings = {**VALID_SETTINGS, "writing": {**VALID_SETTINGS["writing"], "tone": "formal"}}

    result = validate_settings(settings)

    assert result.valid is True
    assert _messages_contain(result.warnings, "writing.tone")


# ---------------------------------------------------------------------------
# export.*
# ---------------------------------------------------------------------------


def test_export_unknown_format_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "export": {**VALID_SETTINGS["export"], "default_format": "docx"}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "export.default_format")


def test_export_unknown_pdf_engine_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "export": {**VALID_SETTINGS["export"], "pdf_engine": "latex"}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "export.pdf_engine")


def test_export_non_bool_speaker_notes_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "export": {**VALID_SETTINGS["export"], "include_speaker_notes": 1}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "export.include_speaker_notes")


def test_export_unknown_key_is_a_warning() -> None:
    settings = {**VALID_SETTINGS, "export": {**VALID_SETTINGS["export"], "dpi": 300}}

    result = validate_settings(settings)

    assert result.valid is True
    assert _messages_contain(result.warnings, "export.dpi")


# ---------------------------------------------------------------------------
# logging.*
# ---------------------------------------------------------------------------


def test_logging_bad_level_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "logging": {**VALID_SETTINGS["logging"], "level": "VERBOSE"}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "logging.level")


def test_logging_level_is_case_insensitive() -> None:
    settings = {**VALID_SETTINGS, "logging": {**VALID_SETTINGS["logging"], "level": "debug"}}

    assert validate_settings(settings).valid is True


def test_logging_empty_format_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "logging": {**VALID_SETTINGS["logging"], "format": ""}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "logging.format")


def test_logging_empty_file_is_valid() -> None:
    settings = {**VALID_SETTINGS, "logging": {**VALID_SETTINGS["logging"], "file": ""}}

    assert validate_settings(settings).valid is True


def test_logging_non_string_file_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "logging": {**VALID_SETTINGS["logging"], "file": 3}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "logging.file")


def test_logging_unknown_key_is_a_warning() -> None:
    settings = {**VALID_SETTINGS, "logging": {**VALID_SETTINGS["logging"], "color": True}}

    result = validate_settings(settings)

    assert result.valid is True
    assert _messages_contain(result.warnings, "logging.color")


# ---------------------------------------------------------------------------
# figures.*
# ---------------------------------------------------------------------------


def test_figures_bad_format_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "figures": {**VALID_SETTINGS["figures"], "default_format": "jpg"}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "figures.default_format")


def test_figures_non_positive_font_size_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "figures": {**VALID_SETTINGS["figures"], "font_size_pt": 0}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "figures.font_size_pt")


def test_figures_non_string_journal_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "figures": {**VALID_SETTINGS["figures"], "default_journal": 5}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "figures.default_journal")


def test_figures_unknown_key_is_a_warning() -> None:
    settings = {**VALID_SETTINGS, "figures": {**VALID_SETTINGS["figures"], "transparent": True}}

    result = validate_settings(settings)

    assert result.valid is True
    assert _messages_contain(result.warnings, "figures.transparent")


# ---------------------------------------------------------------------------
# review.*
# ---------------------------------------------------------------------------


def test_review_threshold_out_of_range_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "review": {**VALID_SETTINGS["review"], "consensus_threshold": 1.5}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "review.consensus_threshold")


def test_review_non_numeric_threshold_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "review": {**VALID_SETTINGS["review"], "consensus_threshold": "high"}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "review.consensus_threshold")


def test_review_bool_threshold_is_an_error() -> None:
    # ``True`` is an int subclass and must not sneak through as 1.
    settings = {**VALID_SETTINGS, "review": {**VALID_SETTINGS["review"], "consensus_threshold": True}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "review.consensus_threshold")


def test_review_int_threshold_in_range_is_valid() -> None:
    settings = {**VALID_SETTINGS, "review": {**VALID_SETTINGS["review"], "consensus_threshold": 1}}

    assert validate_settings(settings).valid is True


def test_review_non_bool_devil_advocate_is_an_error() -> None:
    settings = {**VALID_SETTINGS, "review": {**VALID_SETTINGS["review"], "include_devil_advocate": "yes"}}

    result = validate_settings(settings)

    assert result.valid is False
    assert _messages_contain(result.errors, "review.include_devil_advocate")


def test_review_unknown_key_is_a_warning() -> None:
    settings = {**VALID_SETTINGS, "review": {**VALID_SETTINGS["review"], "rounds": 2}}

    result = validate_settings(settings)

    assert result.valid is True
    assert _messages_contain(result.warnings, "review.rounds")


# ---------------------------------------------------------------------------
# api_keys.* (only the three real keys)
# ---------------------------------------------------------------------------


def test_api_keys_unknown_slot_is_a_warning() -> None:
    settings = {**VALID_SETTINGS, "api_keys": {**VALID_SETTINGS["api_keys"], "scopus_key": "x"}}

    result = validate_settings(settings)

    assert result.valid is True
    assert _messages_contain(result.warnings, "api_keys.scopus_key")


# ---------------------------------------------------------------------------
# DEFAULTS <-> settings.yaml drift pin
# ---------------------------------------------------------------------------

#: settings.yaml 中**直接以原始 dict 读取、不经 get_* 访问器**的键。它们的未配置
#: 行为由**代码**定义，而非 DEFAULTS：
#:   * llm.provider / llm.model / llm.base_url / llm.timeout_seconds ——
#:     core.research_service 用 ``llm_settings.get(...)`` 读取；provider 未配置时
#:     真实回退是 codex CLI 之外的 "gemini"（见 core/llm_client.build_llm_client）。
#:   * api_keys.pubmed_email / semantic_scholar_key / crossref_email ——
#:     core.research_service 用 ``settings.get("api_keys", {})`` 读取。
#: 这条白名单是**显式**的：任何新键若既不在 DEFAULTS 也不在此处，测试即失败——
#: 保证「每个键都有可追溯的读取路径」。
_DIRECT_READ_KEYS: frozenset[str] = frozenset(
    {
        "llm.provider",
        "llm.model",
        "llm.base_url",
        "llm.timeout_seconds",
        "api_keys.pubmed_email",
        "api_keys.semantic_scholar_key",
        "api_keys.crossref_email",
    }
)


def _load_yaml_settings() -> dict:
    settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"
    return yaml.safe_load(settings_path.read_text(encoding="utf-8"))


def test_defaults_is_subset_of_yaml() -> None:
    """DEFAULTS 里的每个键都必须在 settings.yaml 中存在。"""
    yaml_keys = _dotted_keys(_load_yaml_settings())

    assert set(DEFAULTS) <= yaml_keys


def test_every_yaml_key_has_a_traceable_read_path() -> None:
    """settings.yaml 的每个键要么在 DEFAULTS，要么在"直接读取"白名单中。"""
    yaml_keys = _dotted_keys(_load_yaml_settings())
    covered = set(DEFAULTS) | _DIRECT_READ_KEYS

    assert yaml_keys <= covered, f"无读取路径的键: {sorted(yaml_keys - covered)}"


def test_direct_read_allowlist_has_no_stale_entries() -> None:
    """白名单里的键必须真实存在于 settings.yaml（防止白名单腐化）。"""
    yaml_keys = _dotted_keys(_load_yaml_settings())

    assert _DIRECT_READ_KEYS <= yaml_keys, sorted(_DIRECT_READ_KEYS - yaml_keys)


def test_defaults_match_yaml_values() -> None:
    yaml_settings = _load_yaml_settings()

    for dotted, default in DEFAULTS.items():
        section, _, leaf = dotted.partition(".")
        assert yaml_settings[section][leaf] == default, dotted


def test_valid_settings_cover_defaults_and_direct_read_keys() -> None:
    # VALID_SETTINGS 是"形如真实 settings.yaml"的样本，应覆盖所有键（含白名单）。
    assert _dotted_keys(VALID_SETTINGS) == set(DEFAULTS) | _DIRECT_READ_KEYS


# ---------------------------------------------------------------------------
# config_loader accessors
# ---------------------------------------------------------------------------


def test_get_str_returns_value() -> None:
    assert get_str({"paths": {"projects_dir": "work"}}, "paths.projects_dir") == "work"


def test_get_int_returns_value() -> None:
    assert get_int({"search": {"max_results": 25}}, "search.max_results") == 25


def test_get_int_rejects_bool() -> None:
    # True is not a valid count; fall back to the default (10).
    assert get_int({"search": {"max_results": True}}, "search.max_results") == DEFAULTS["search.max_results"]


def test_get_bool_returns_value() -> None:
    assert get_bool({"writing": {"bilingual_abstract": True}}, "writing.bilingual_abstract") is True


def test_get_float_default_is_zero_sentinel() -> None:
    # 默认值必须描述当前行为：0.0 = 不标注分歧（哨兵），确保未配置用户产物不变。
    assert get_float({}, "review.consensus_threshold") == 0.0


def test_get_float_returns_configured_value() -> None:
    assert get_float({"review": {"consensus_threshold": 0.8}}, "review.consensus_threshold") == 0.8


def test_get_float_preserves_fractional_value_not_truncated() -> None:
    # Regression guard: int(0.8) == 0 would silently erase a configured threshold;
    # get_float 必须返回 0.8，而不是被截断成 0。
    assert get_float({"review": {"consensus_threshold": 0.8}}, "review.consensus_threshold") == 0.8


def test_get_float_falls_back_on_string_value() -> None:
    settings = {"review": {"consensus_threshold": "0.8"}}
    assert get_float(settings, "review.consensus_threshold") == 0.0


def test_get_float_rejects_bool() -> None:
    settings = {"review": {"consensus_threshold": True}}
    assert get_float(settings, "review.consensus_threshold") == 0.0


def test_get_float_widens_int() -> None:
    assert get_float({"review": {"consensus_threshold": 1}}, "review.consensus_threshold") == 1.0


def test_get_float_unknown_key_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_float({}, "review.does_not_exist")


def test_get_str_list_returns_copy() -> None:
    settings = {"citation": {"export_formats": ["md"]}}
    result = get_str_list(settings, "citation.export_formats")

    assert result == ["md"]
    result.append("pdf")
    assert settings["citation"]["export_formats"] == ["md"]


def test_get_int_pair_returns_tuple() -> None:
    assert get_int_pair({"search": {"year_range": [2019, 2024]}}, "search.year_range") == (2019, 2024)


def test_get_mapping_returns_section() -> None:
    assert get_mapping({"review": {"reviewer_count": 2}}, "review") == {"reviewer_count": 2}


def test_get_mapping_missing_section_returns_empty() -> None:
    assert get_mapping({}, "review") == {}


@pytest.mark.parametrize(
    "accessor",
    [get_str, get_int, get_bool, get_float, get_str_list, get_int_pair],
)
def test_get_unknown_dotted_key_raises_key_error(accessor) -> None:
    with pytest.raises(KeyError):
        accessor({}, "search.does_not_exist")


@pytest.mark.parametrize(
    ("accessor", "dotted", "expected"),
    [
        (get_str, "paths.projects_dir", "projects"),
        (get_int, "search.max_results", 10),
        (get_bool, "review.include_devil_advocate", False),
        (get_str_list, "citation.export_formats", ["md", "pdf", "pptx"]),
        (get_int_pair, "search.year_range", (0, 0)),
    ],
)
def test_get_missing_key_falls_back_to_default(accessor, dotted: str, expected: object) -> None:
    # Empty settings: every accessor must return the DEFAULTS value, not raise.
    settings: dict = {}
    assert accessor(settings, dotted) == expected


def test_get_str_falls_back_on_wrong_type() -> None:
    assert get_str({"paths": {"projects_dir": 5}}, "paths.projects_dir") == DEFAULTS["paths.projects_dir"]


def test_get_str_list_falls_back_on_wrong_element_type() -> None:
    settings = {"citation": {"export_formats": ["md", 1]}}
    assert get_str_list(settings, "citation.export_formats") == DEFAULTS["citation.export_formats"]


def test_get_int_pair_falls_back_on_wrong_shape() -> None:
    settings = {"search": {"year_range": [2020]}}
    assert get_int_pair(settings, "search.year_range") == tuple(DEFAULTS["search.year_range"])  # type: ignore[arg-type]


def test_get_mapping_section_is_not_mapping_returns_empty() -> None:
    assert get_mapping({"review": "off"}, "review") == {}


def test_load_settings_reads_yaml(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        "search:\n  max_results: 42\n", encoding="utf-8"
    )

    settings = load_settings(tmp_path)

    assert settings["search"]["max_results"] == 42


def test_load_settings_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_settings(tmp_path) == {}
