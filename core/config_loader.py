"""Runtime configuration loading and typed accessors.

This module is the **single source of truth for default values**: the
``DEFAULTS`` mapping below holds the default for every key read through the
``get_*`` helpers.  Every consumer reads settings through those helpers rather
than poking at raw dictionaries, so a missing key degrades to its documented
default instead of a crash.

``DEFAULTS`` is deliberately **not** the full set of ``config/settings.yaml``
keys: a few sections (``llm.*`` / ``api_keys.*``) are read directly as raw
dicts by ``ResearchService`` / ``build_llm_client``, so their unconfigured
behaviour is defined by **code**, not here (e.g. an unconfigured
``llm.provider`` falls back to ``gemini`` in ``core.llm_client``).  A test
enforces that every ``settings.yaml`` key is either in ``DEFAULTS`` or in an
explicit "read directly, not via ``get_*``" allow-list — i.e. every key has a
*traceable read path*.

The accessors raise :class:`KeyError` when asked for a dotted key that is not
part of ``DEFAULTS``.  This is intentional: a misspelled key name must fail
immediately and loudly rather than silently returning a default, because a
"looks-configured-but-does-nothing" typo is exactly the class of bug this
project keeps cleaning up.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Dotted key -> default value.  This is the **only** place default values live.
#: It must stay in lock-step with ``config/settings.yaml`` (pinned by tests).
DEFAULTS: dict[str, object] = {
    # --- paths ---------------------------------------------------------
    "paths.projects_dir": "projects",
    "paths.output_dir": "output",
    "paths.scripts_dir": "scripts",
    # --- search --------------------------------------------------------
    "search.default_sources": ["crossref", "pubmed", "semantic_scholar"],
    "search.max_results": 10,
    "search.year_range": [0, 0],
    # --- citation ------------------------------------------------------
    "citation.default_style": "numeric",
    "citation.export_formats": ["md", "pdf", "pptx"],
    # --- writing -------------------------------------------------------
    "writing.default_paper_type": "",
    "writing.default_language": "",
    "writing.bilingual_abstract": False,
    "writing.style_guide": "",
    # --- export --------------------------------------------------------
    "export.default_format": "md",
    "export.pdf_engine": "auto",
    "export.pptx_template": "",
    "export.include_speaker_notes": False,
    # --- logging -------------------------------------------------------
    "logging.level": "INFO",
    "logging.format": "%(asctime)s %(levelname)s %(name)s: %(message)s",
    "logging.file": "",
    # --- figures -------------------------------------------------------
    "figures.default_dpi": 300,
    "figures.default_journal": "",
    "figures.default_format": "png",
    "figures.color_palette": "default",
    "figures.font_family": "",
    "figures.font_size_pt": 10,
    # --- review --------------------------------------------------------
    "review.reviewer_count": 1,
    "review.include_devil_advocate": False,
    "review.consensus_threshold": 0.0,
    "review.score_scale": "0-100",
}


def load_settings(base_dir: Any) -> dict[str, Any]:
    """Load ``config/settings.yaml`` when available; return ``{}`` otherwise."""
    from pathlib import Path

    settings_path = Path(base_dir) / "config" / "settings.yaml"
    if not settings_path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _lookup(settings: Mapping[str, Any], dotted: str) -> Any:
    """Return the raw value for ``dotted`` or raise ``KeyError`` for unknown keys."""
    if dotted not in DEFAULTS:
        raise KeyError(f"未知配置键 {dotted}（不在 DEFAULTS 中，请检查拼写）")

    section, _, leaf = dotted.partition(".")
    container: Any = settings
    try:
        container = container[section]
    except (KeyError, TypeError, IndexError):
        return DEFAULTS[dotted]
    if not isinstance(container, Mapping):
        return DEFAULTS[dotted]
    try:
        return container[leaf]
    except (KeyError, TypeError, IndexError):
        return DEFAULTS[dotted]


def get_str(settings: Mapping[str, Any], dotted: str) -> str:
    """Return a string setting, falling back to ``DEFAULTS`` when absent."""
    value = _lookup(settings, dotted)
    return value if isinstance(value, str) else str(DEFAULTS[dotted])


def _as_int(value: object, fallback: int) -> int:
    """Coerce ``value`` to int, returning ``fallback`` when not a real int."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return fallback


def get_int(settings: Mapping[str, Any], dotted: str) -> int:
    """Return an integer setting, falling back to ``DEFAULTS`` when absent."""
    value = _lookup(settings, dotted)
    return _as_int(value, _as_int(DEFAULTS[dotted], 0))


def get_bool(settings: Mapping[str, Any], dotted: str) -> bool:
    """Return a boolean setting, falling back to ``DEFAULTS`` when absent."""
    value = _lookup(settings, dotted)
    if isinstance(value, bool):
        return value
    default = DEFAULTS[dotted]
    return default if isinstance(default, bool) else False


def _as_float(value: object, fallback: float) -> float:
    """Coerce ``value`` to float, returning ``fallback`` when not a real number."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return fallback


def get_float(settings: Mapping[str, Any], dotted: str) -> float:
    """Return a float setting, falling back to ``DEFAULTS`` when absent.

    ``bool`` is explicitly rejected (``True`` is not ``1.0``), and an int value
    such as ``1`` is accepted and widened to ``1.0``.
    """
    value = _lookup(settings, dotted)
    return _as_float(value, _as_float(DEFAULTS[dotted], 0.0))


def get_str_list(settings: Mapping[str, Any], dotted: str) -> list[str]:
    """Return a list-of-strings setting, falling back to ``DEFAULTS`` when absent."""
    value = _lookup(settings, dotted)
    if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        return list(value)
    default = DEFAULTS[dotted]
    if isinstance(default, (list, tuple)):
        return [item for item in default if isinstance(item, str)]
    return []


def get_int_pair(settings: Mapping[str, Any], dotted: str) -> tuple[int, int]:
    """Return a two-integer setting (e.g. ``search.year_range``)."""
    value = _lookup(settings, dotted)
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        return int(value[0]), int(value[1])
    default = DEFAULTS[dotted]
    if isinstance(default, (list, tuple)) and len(default) == 2:
        return _as_int(default[0], 0), _as_int(default[1], 0)
    return 0, 0


def get_mapping(settings: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    """Return a settings sub-section as a mapping, or an empty mapping."""
    try:
        value = settings[section]
    except (KeyError, TypeError, IndexError):
        return {}
    return value if isinstance(value, Mapping) else {}
