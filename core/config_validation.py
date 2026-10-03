"""Validation for runtime settings loaded from config/settings.yaml.

The settings file is untrusted input: a typo or wrong type must be reported
loudly and early with the offending key named, rather than being silently
ignored or crashing deep inside a pipeline run. At the same time, validation
must not reject valid future configuration, so unknown keys are warnings and a
missing or unreadable file degrades to empty settings plus a warning.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeGuard

from core.config_loader import load_settings

# LLM provider values actually accepted by core.llm_client.build_llm_client.
# build_llm_client lowercases the provider before dispatch, so both the
# canonical names and the documented aliases are accepted here.
KNOWN_PROVIDERS = frozenset(
    {"codex_cli", "codex", "gemini", "openai_compatible", "openai", "compatible"}
)

# Sources recognised by the real literature searchers.
KNOWN_SOURCES = frozenset({"crossref", "pubmed", "semantic_scholar", "arxiv"})

# Export formats supported by the exporter.
KNOWN_EXPORT_FORMATS = frozenset({"md", "pdf", "pptx"})


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
    """Read one top-level key while tolerating odd Mapping implementations."""
    try:
        return settings[key]
    except KeyError:
        return None


def _is_int(value: object) -> TypeGuard[int]:
    """True for real integers, excluding bool (True/False are not counts)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _count(value: object) -> int:
    """Best-effort element count for diagnostics only."""
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _validate_llm(section: object, out: SettingsValidation) -> None:
    """Validate llm.* keys; the section itself must be a mapping."""
    if not isinstance(section, Mapping):
        out.errors.append("llm 必须是映射对象（section）")
        return

    provider = _get(section, "provider")
    if provider is not None and provider not in KNOWN_PROVIDERS:
        out.errors.append(f"llm.provider 必须是 {sorted(KNOWN_PROVIDERS)} 之一，当前为 {provider!r}")
    elif not isinstance(provider, str):
        # ``provider: true`` or ``provider: 1`` is a type mistake, not a typo.
        out.errors.append(f"llm.provider 必须是字符串，当前为 {provider!r}")

    timeout = _get(section, "timeout_seconds")
    if timeout is not None and (not _is_int(timeout) or timeout <= 0):
        out.errors.append(f"llm.timeout_seconds 必须是正整数，当前为 {timeout!r}")

    for key in ("model", "base_url"):
        value = _get(section, key)
        if value is not None and not isinstance(value, str):
            out.errors.append(f"llm.{key} 必须是字符串（留空表示使用默认值），当前为 {value!r}")

    for key in sorted(set(section) - {"provider", "timeout_seconds", "model", "base_url"}):
        out.warnings.append(f"未知配置项 llm.{key}，已忽略")


def _validate_search(section: object, out: SettingsValidation) -> None:
    """Validate search.* keys; items of real_sources must be known sources."""
    if not isinstance(section, Mapping):
        out.errors.append("search 必须是映射对象（section）")
        return

    real_sources = _get(section, "real_sources")
    if real_sources is not None:
        if not isinstance(real_sources, list):
            out.errors.append(f"search.real_sources 必须是列表，当前为 {real_sources!r}")
        else:
            for index, source in enumerate(real_sources):
                if source not in KNOWN_SOURCES:
                    out.errors.append(
                        f"search.real_sources[{index}] 不是受支持的来源 {source!r}，"
                        f"可选值为 {sorted(KNOWN_SOURCES)}"
                    )

    for key in sorted(set(section) - {"real_sources"}):
        out.warnings.append(f"未知配置项 search.{key}，已忽略")


def _validate_citation(section: object, out: SettingsValidation) -> None:
    """Validate citation.* keys; default_style must be a declared style."""
    if not isinstance(section, Mapping):
        out.errors.append("citation 必须是映射对象（section）")
        return

    default_style = _get(section, "default_style")
    supported_styles = _get(section, "supported_styles")
    if (
        default_style is not None
        and supported_styles is not None
        and isinstance(supported_styles, list)
        and default_style not in supported_styles
    ):
        out.errors.append(
            f"citation.default_style {default_style!r} 不在 citation.supported_styles "
            f"{supported_styles!r} 中"
        )

    for key in sorted(set(section) - {"default_style", "supported_styles"}):
        out.warnings.append(f"未知配置项 citation.{key}，已忽略")


def _validate_figures(section: object, out: SettingsValidation) -> None:
    """Validate figures.* keys; DPI must be a positive integer."""
    if not isinstance(section, Mapping):
        out.errors.append("figures 必须是映射对象（section）")
        return

    dpi = _get(section, "default_dpi")
    if dpi is not None and (not _is_int(dpi) or dpi <= 0):
        out.errors.append(f"figures.default_dpi 必须是正整数，当前为 {dpi!r}")

    for key in sorted(set(section) - {"default_dpi"}):
        out.warnings.append(f"未知配置项 figures.{key}，已忽略")


def _validate_review(section: object, out: SettingsValidation) -> None:
    """Validate review.* keys; reviewer_count accepts 0 to disable the stage."""
    if not isinstance(section, Mapping):
        out.errors.append("review 必须是映射对象（section）")
        return

    count = _get(section, "reviewer_count")
    if count is not None and (not _is_int(count) or count < 0):
        out.errors.append(f"review.reviewer_count 必须是不小于 0 的整数，当前为 {count!r}")

    for key in sorted(set(section) - {"reviewer_count"}):
        out.warnings.append(f"未知配置项 review.{key}，已忽略")


def _validate_export(section: object, out: SettingsValidation) -> None:
    """Validate export.* keys; the default format must be exportable."""
    if not isinstance(section, Mapping):
        out.errors.append("export 必须是映射对象（section）")
        return

    default_format = _get(section, "default_format")
    if default_format is not None and default_format not in KNOWN_EXPORT_FORMATS:
        out.errors.append(
            f"export.default_format 必须是 {sorted(KNOWN_EXPORT_FORMATS)} 之一，"
            f"当前为 {default_format!r}"
        )

    for key in sorted(set(section) - {"default_format"}):
        out.warnings.append(f"未知配置项 export.{key}，已忽略")


def _validate_api_keys(section: object, out: SettingsValidation) -> None:
    """Validate api_keys.* values; empty string means "not configured"."""
    if not isinstance(section, Mapping):
        out.errors.append("api_keys 必须是映射对象（section）")
        return

    for key in sorted(section):
        value = section[key]
        if value is None:
            continue
        if not isinstance(value, str):
            if value == "":
                # Some YAML writers emit null for an unset key; treat as empty.
                continue
            out.errors.append(f"api_keys.{key} 必须是字符串（留空表示未配置），当前为 {value!r}")


# Fixed validation order keeps message ordering deterministic across runs.
_SECTION_VALIDATORS: tuple[
    tuple[str, Callable[[object, SettingsValidation], None]], ...
] = (
    ("llm", _validate_llm),
    ("search", _validate_search),
    ("citation", _validate_citation),
    ("figures", _validate_figures),
    ("review", _validate_review),
    ("export", _validate_export),
    ("api_keys", _validate_api_keys),
)

_KNOWN_SECTIONS = frozenset({name for name, _ in _SECTION_VALIDATORS})


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
