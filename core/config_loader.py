"""Runtime configuration loading with environment-variable precedence."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_settings(base_dir: Path) -> dict[str, Any]:
    """Load config/settings.yaml when available; return an empty mapping otherwise."""
    settings_path = base_dir / "config" / "settings.yaml"
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
