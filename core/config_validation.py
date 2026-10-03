"""Validation for runtime settings loaded from config/settings.yaml.

The settings file is untrusted input: a typo or wrong type must be reported
loudly and early with the offending key named, rather than being silently
ignored or crashing deep inside a pipeline run. At the same time, validation
must not reject valid future configuration, so unknown keys are warnings and a
missing or unreadable file degrades to empty settings plus a warning.

Type mistakes are blocking errors; unknown keys inside a known section (or an
unknown top-level section) are non-blocking warnings.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeGuard

from core.config_loader import DEFAULTS, load_settings

# LLM provider values actually accepted by core.llm_client.build_llm_client.
# build_llm_client lowercases the provider before dispatch, so both the
# canonical names and the documented aliases are accepted here.
KNOWN_PROVIDERS = frozenset(
    {"codex_cli", "codex", "gemini", "openai_compatible", "openai", "compatible"}
)

#: Literature sources understood by the search layer.
KNOWN_SOURCES = frozenset({"crossref", "pubmed", "semantic_scholar", "arxiv"})

#: Output formats for the export layer (Markdown / PDF / PPTX).
KNOWN_EXPORT_FORMATS = frozenset({"md", "pdf", "pptx"})

#: PDF engines actually implemented by core.export_service.
KNOWN_PDF_ENGINES = frozenset({"auto", "pandoc", "reportlab"})

#: Figure output formats supported by the figure builder.
KNOWN_FIGURE_FORMATS = frozenset({"png", "pdf", "svg"})

#: Supported writing languages.
KNOWN_LANGUAGES = frozenset({"zh", "en"})

#: Standard logging levels (compared case-insensitively).
KNOWN_LOG_LEVELS = frozenset(
    {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
)

#: The only API key slots the project actually reads. Untracked slots such as
#: ``scopus_key`` / ``elsevier_key`` are deliberately not accepted.
KNOWN_API_KEYS = frozenset({"pubmed_email", "semantic_scholar_key", "crossref_email"})


class ConfigValidationError(ValueError):
    """Raised when settings cannot be used as configuration at all."""


@dataclass
class SettingsValidation:
    """Collected validation outcome: blocking errors and non-blocking warnings."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """True when no blocking error was found."""
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly snapshot with a deterministic shape."""
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _get(settings: Mapping[str, object], key: str) -> object:
    """Read one key while tolerating odd Mapping implementations."""
    try:
        return settings[key]
    except (KeyError, TypeError, IndexError):
        return None


def _is_int(value: object) -> TypeGuard[int]:
    """True for real integers, excluding bool (True/False are not counts)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> TypeGuard[float]:
    """True for int/float but not bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_str_list(value: object) -> TypeGuard[list[str] | tuple[str, ...]]:
    """True for a list/tuple whose elements are all strings."""
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, str) for item in value
    )


def _count(value: object) -> int:
    """Best-effort element count for diagnostics only."""
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _require_mapping(
    section: object, name: str, out: SettingsValidation
) -> Mapping[str, object] | None:
    """Validate that a section is a mapping; return it or record an error."""
    if not isinstance(section, Mapping):
        out.errors.append(f"{name} 必须是映射对象（section）")
        return None
    return section


def _warn_unknown(
    section: Mapping[str, object],
    prefix: str,
    known: set[str],
    out: SettingsValidation,
) -> None:
    """Warn (never fail) on keys outside the known set for a section."""
    for key in sorted(set(section) - known):
        out.warnings.append(f"未知配置项 {prefix}.{key}，已忽略")


def _validate_llm(section: object, out: SettingsValidation) -> None:
    """Validate llm.* keys; the section itself must be a mapping."""
    data = _require_mapping(section, "llm", out)
    if data is None:
        return

    provider = _get(data, "provider")
    if provider is not None and provider not in KNOWN_PROVIDERS:
        out.errors.append(f"llm.provider 必须是 {sorted(KNOWN_PROVIDERS)} 之一，当前为 {provider!r}")
    elif provider is not None and not isinstance(provider, str):
        # ``provider: true`` or ``provider: 1`` is a type mistake, not a typo.
        out.errors.append(f"llm.provider 必须是字符串，当前为 {provider!r}")

    timeout = _get(data, "timeout_seconds")
    if timeout is not None and (not _is_int(timeout) or timeout <= 0):
        out.errors.append(f"llm.timeout_seconds 必须是正整数，当前为 {timeout!r}")

    for key in ("model", "base_url"):
        value = _get(data, key)
        if value is not None and not isinstance(value, str):
            out.errors.append(f"llm.{key} 必须是字符串（留空表示使用默认值），当前为 {value!r}")

    _warn_unknown(data, "llm", {"provider", "timeout_seconds", "model", "base_url"}, out)


def _validate_paths(section: object, out: SettingsValidation) -> None:
    """Validate paths.* keys: every directory must be a non-empty string."""
    data = _require_mapping(section, "paths", out)
    if data is None:
        return

    for key in ("projects_dir", "output_dir", "scripts_dir"):
        value = _get(data, key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            out.errors.append(f"paths.{key} 必须是非空字符串，当前为 {value!r}")

    _warn_unknown(
        data,
        "paths",
        {"projects_dir", "output_dir", "scripts_dir"},
        out,
    )


def _validate_search(section: object, out: SettingsValidation) -> None:
    """Validate search.* keys: sources, result cap and year window."""
    data = _require_mapping(section, "search", out)
    if data is None:
        return

    sources = _get(data, "default_sources")
    if sources is not None:
        if not _is_str_list(sources):
            out.errors.append(f"search.default_sources 必须是字符串列表，当前为 {sources!r}")
        elif not sources:
            out.errors.append("search.default_sources 必须是非空列表")
        else:
            unknown = [item for item in sources if item not in KNOWN_SOURCES]
            if unknown:
                out.errors.append(
                    f"search.default_sources 含未知检索源 {unknown!r}，"
                    f"必须是 {sorted(KNOWN_SOURCES)} 之一"
                )

    max_results = _get(data, "max_results")
    if max_results is not None and (not _is_int(max_results) or max_results <= 0):
        out.errors.append(f"search.max_results 必须是正整数，当前为 {max_results!r}")

    year_range = _get(data, "year_range")
    if year_range is not None:
        if (
            not isinstance(year_range, (list, tuple))
            or len(year_range) != 2
            or not all(_is_int(item) for item in year_range)
        ):
            out.errors.append(
                f"search.year_range 必须是两个非负整数的列表 [起始年, 结束年]，当前为 {year_range!r}"
            )
        else:
            start, end = year_range
            if start < 0 or end < 0:
                out.errors.append(
                    f"search.year_range 的两个年份必须非负（0 表示不限），当前为 {year_range!r}"
                )
            elif start != 0 and end != 0 and start > end:
                out.errors.append(
                    f"search.year_range 起始年不得大于结束年，当前为 {year_range!r}"
                )

    _warn_unknown(
        data, "search", {"default_sources", "max_results", "year_range"}, out
    )


def _validate_citation(section: object, out: SettingsValidation) -> None:
    """Validate citation.* keys against the single source of truth.

    Supported styles and export formats are **code capability facts**, not user
    preferences: they come from :mod:`core.citation_styles` (``known_styles`` /
    ``validate_export_formats``), never from a second literal copy here.
    """
    from core.citation_styles import (
        known_export_formats,
        known_styles,
        validate_export_formats,
    )

    data = _require_mapping(section, "citation", out)
    if data is None:
        return

    supported_styles = known_styles()

    default_style = _get(data, "default_style")
    if default_style is not None:
        if not isinstance(default_style, str):
            out.errors.append(f"citation.default_style 必须是字符串，当前为 {default_style!r}")
        elif default_style not in supported_styles:
            out.errors.append(
                f"citation.default_style 必须是受支持的引用样式之一 "
                f"({list(supported_styles)}），当前为 {default_style!r}"
            )

    export_formats = _get(data, "export_formats")
    if export_formats is not None:
        if not _is_str_list(export_formats):
            out.errors.append(f"citation.export_formats 必须是字符串列表，当前为 {export_formats!r}")
        elif not export_formats:
            out.errors.append("citation.export_formats 必须是非空列表")
        else:
            unknown = validate_export_formats(list(export_formats))
            if unknown:
                out.errors.append(
                    f"citation.export_formats 含未知导出格式 {unknown!r}，"
                    f"当前仅支持 {list(known_export_formats())}"
                )

    _warn_unknown(data, "citation", {"default_style", "export_formats"}, out)


def _validate_writing(section: object, out: SettingsValidation) -> None:
    """Validate writing.* keys: paper type, language, abstract, style guide."""
    data = _require_mapping(section, "writing", out)
    if data is None:
        return

    # 留空 = 不追加该约束（沿用既有提示词）；非空时必须是合法的文体名。
    paper_type = _get(data, "default_paper_type")
    if paper_type is not None and not isinstance(paper_type, str):
        out.errors.append(f"writing.default_paper_type 必须是字符串（留空表示不追加约束），当前为 {paper_type!r}")

    # 留空 = 不追加语言约束；非空时必须是 zh / en。
    language = _get(data, "default_language")
    if language is not None and not isinstance(language, str):
        out.errors.append(f"writing.default_language 必须是字符串，当前为 {language!r}")
    elif language is not None and language not in KNOWN_LANGUAGES and language != "":
        out.errors.append(
            f"writing.default_language 必须留空或为 {sorted(KNOWN_LANGUAGES)} 之一，当前为 {language!r}"
        )

    bilingual = _get(data, "bilingual_abstract")
    if bilingual is not None and not isinstance(bilingual, bool):
        out.errors.append(f"writing.bilingual_abstract 必须是布尔值，当前为 {bilingual!r}")

    style_guide = _get(data, "style_guide")
    if style_guide is not None and not isinstance(style_guide, str):
        out.errors.append(f"writing.style_guide 必须是字符串（留空表示不启用），当前为 {style_guide!r}")

    _warn_unknown(
        data,
        "writing",
        {"default_paper_type", "default_language", "bilingual_abstract", "style_guide"},
        out,
    )


def _validate_export(section: object, out: SettingsValidation) -> None:
    """Validate export.* keys: default format, pdf engine, template, notes."""
    data = _require_mapping(section, "export", out)
    if data is None:
        return

    default_format = _get(data, "default_format")
    if default_format is not None and default_format not in KNOWN_EXPORT_FORMATS:
        out.errors.append(
            f"export.default_format 必须是 {sorted(KNOWN_EXPORT_FORMATS)} 之一，当前为 {default_format!r}"
        )
    elif default_format is not None and not isinstance(default_format, str):
        out.errors.append(f"export.default_format 必须是字符串，当前为 {default_format!r}")

    pdf_engine = _get(data, "pdf_engine")
    if pdf_engine is not None and pdf_engine not in KNOWN_PDF_ENGINES:
        out.errors.append(
            f"export.pdf_engine 必须是 {sorted(KNOWN_PDF_ENGINES)} 之一，当前为 {pdf_engine!r}"
        )

    template = _get(data, "pptx_template")
    if template is not None and not isinstance(template, str):
        out.errors.append(f"export.pptx_template 必须是字符串（留空表示使用默认模板），当前为 {template!r}")

    notes = _get(data, "include_speaker_notes")
    if notes is not None and not isinstance(notes, bool):
        out.errors.append(f"export.include_speaker_notes 必须是布尔值，当前为 {notes!r}")

    _warn_unknown(
        data,
        "export",
        {"default_format", "pdf_engine", "pptx_template", "include_speaker_notes"},
        out,
    )


def _validate_logging(section: object, out: SettingsValidation) -> None:
    """Validate logging.* keys: level, format string, optional file path."""
    data = _require_mapping(section, "logging", out)
    if data is None:
        return

    level = _get(data, "level")
    if level is not None:
        if not isinstance(level, str):
            out.errors.append(f"logging.level 必须是字符串，当前为 {level!r}")
        elif level.upper() not in KNOWN_LOG_LEVELS:
            out.errors.append(
                f"logging.level 必须是 {sorted(KNOWN_LOG_LEVELS)} 之一（大小写不敏感），当前为 {level!r}"
            )

    fmt = _get(data, "format")
    if fmt is not None and (not isinstance(fmt, str) or not fmt.strip()):
        out.errors.append(f"logging.format 必须是非空字符串，当前为 {fmt!r}")

    file = _get(data, "file")
    if file is not None and not isinstance(file, str):
        out.errors.append(f"logging.file 必须是字符串（留空表示不写文件），当前为 {file!r}")

    _warn_unknown(data, "logging", {"level", "format", "file"}, out)


def _validate_figures(section: object, out: SettingsValidation) -> None:
    """Validate figures.* keys; DPI/font size must be positive integers."""
    data = _require_mapping(section, "figures", out)
    if data is None:
        return

    dpi = _get(data, "default_dpi")
    if dpi is not None and (not _is_int(dpi) or dpi <= 0):
        out.errors.append(f"figures.default_dpi 必须是正整数，当前为 {dpi!r}")

    font_size = _get(data, "font_size_pt")
    if font_size is not None and (not _is_int(font_size) or font_size <= 0):
        out.errors.append(f"figures.font_size_pt 必须是正整数，当前为 {font_size!r}")

    default_format = _get(data, "default_format")
    if default_format is not None and default_format not in KNOWN_FIGURE_FORMATS:
        out.errors.append(
            f"figures.default_format 必须是 {sorted(KNOWN_FIGURE_FORMATS)} 之一，当前为 {default_format!r}"
        )

    for key in ("default_journal", "color_palette", "font_family"):
        value = _get(data, key)
        if value is not None and not isinstance(value, str):
            out.errors.append(f"figures.{key} 必须是字符串，当前为 {value!r}")

    _warn_unknown(
        data,
        "figures",
        {
            "default_dpi",
            "default_journal",
            "default_format",
            "color_palette",
            "font_family",
            "font_size_pt",
        },
        out,
    )


def _validate_review(section: object, out: SettingsValidation) -> None:
    """Validate review.* keys; reviewer_count accepts 0 to disable the stage."""
    data = _require_mapping(section, "review", out)
    if data is None:
        return

    count = _get(data, "reviewer_count")
    if count is not None and (not _is_int(count) or count < 0):
        out.errors.append(f"review.reviewer_count 必须是不小于 0 的整数，当前为 {count!r}")

    devil = _get(data, "include_devil_advocate")
    if devil is not None and not isinstance(devil, bool):
        out.errors.append(f"review.include_devil_advocate 必须是布尔值，当前为 {devil!r}")

    threshold = _get(data, "consensus_threshold")
    if threshold is not None:
        if not _is_number(threshold):
            out.errors.append(f"review.consensus_threshold 必须是数值，当前为 {threshold!r}")
        elif not 0.0 <= float(threshold) <= 1.0:
            out.errors.append(
                f"review.consensus_threshold 必须介于 0 和 1 之间，当前为 {threshold!r}"
            )

    score_scale = _get(data, "score_scale")
    if score_scale is not None and not isinstance(score_scale, str):
        out.errors.append(f"review.score_scale 必须是字符串，当前为 {score_scale!r}")

    _warn_unknown(
        data,
        "review",
        {"reviewer_count", "include_devil_advocate", "consensus_threshold", "score_scale"},
        out,
    )


def _validate_api_keys(section: object, out: SettingsValidation) -> None:
    """Validate api_keys.* values; empty string means "not configured"."""
    data = _require_mapping(section, "api_keys", out)
    if data is None:
        return

    for key in sorted(data):
        value = data[key]
        if value is None:
            continue
        if not isinstance(value, str):
            if value == "":
                # Some YAML writers emit null for an unset key; treat as empty.
                continue
            out.errors.append(f"api_keys.{key} 必须是字符串（留空表示未配置），当前为 {value!r}")

    _warn_unknown(data, "api_keys", set(KNOWN_API_KEYS), out)


# Fixed validation order keeps message ordering deterministic across runs.
_SECTION_VALIDATORS: tuple[
    tuple[str, Callable[[object, SettingsValidation], None]], ...
] = (
    ("paths", _validate_paths),
    ("search", _validate_search),
    ("citation", _validate_citation),
    ("writing", _validate_writing),
    ("export", _validate_export),
    ("logging", _validate_logging),
    ("figures", _validate_figures),
    ("review", _validate_review),
    ("llm", _validate_llm),
    ("api_keys", _validate_api_keys),
)

_KNOWN_SECTIONS = frozenset({name for name, _ in _SECTION_VALIDATORS})


def _validate_cross_section(settings: Mapping[str, object], out: SettingsValidation) -> None:
    """Validate consistency between sections that must agree.

    ``export.default_format`` must be one of ``citation.export_formats``;
    otherwise *every* export run fails at runtime with a configuration error.
    That is deterministically verifiable, so it is blocked at config time.
    """
    citation = _get(settings, "citation")
    export = _get(settings, "export")
    if not isinstance(citation, Mapping) or not isinstance(export, Mapping):
        return

    export_formats = _get(citation, "export_formats")
    default_format = _get(export, "default_format")
    # Only enforce when both are usable string values; type/shape problems are
    # already reported as errors by the per-section validators.
    if not _is_str_list(export_formats) or not isinstance(default_format, str):
        return
    if not export_formats:
        return
    if default_format not in export_formats:
        out.errors.append(
            f"export.default_format（当前为 {default_format!r}）必须是 "
            f"citation.export_formats 之一（{list(export_formats)}）——"
            "否则每次导出都会运行时失败"
        )


def _defaults_key_set() -> frozenset[str]:
    """The dotted key set declared in ``DEFAULTS`` (for drift detection)."""
    return frozenset(DEFAULTS)


def validate_settings(settings: object) -> SettingsValidation:
    """Validate a settings mapping, collecting errors and warnings.

    Unknown top-level and nested keys produce warnings so future configuration
    stays valid; unusable values produce errors that name the dotted key path.
    """
    out = SettingsValidation()
    if not isinstance(settings, Mapping):
        out.errors.append("settings 必须是映射对象")
        return out

    for name, validator in _SECTION_VALIDATORS:
        if name in settings:
            validator(_get(settings, name), out)

    _validate_cross_section(settings, out)

    for key in sorted(set(settings) - _KNOWN_SECTIONS):
        out.warnings.append(f"未知配置项 {key}，已忽略")

    return out


def load_validated_settings(base_dir: Path) -> tuple[dict, SettingsValidation]:
    """Load settings via core.config_loader and validate the result.

    A missing or unreadable file degrades to empty settings plus a warning; this
    never raises so callers keep the pre-validation behaviour.
    """
    settings = load_settings(base_dir)
    if not isinstance(settings, dict) or not settings:
        validation = SettingsValidation()
        validation.warnings.append(
            "未找到或无法读取 config/settings.yaml，使用空配置继续运行"
        )
        return {}, validation

    validation = validate_settings(settings)
    return settings, validation
