"""Shared data contracts for the real research pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PaperRecord:
    """Normalized metadata returned by a literature source."""

    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    journal: str = ""
    doi: str = ""
    abstract: str = ""
    url: str = ""
    source: str = ""
    citation_count: int | None = None
    external_id: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchReport:
    """Search output, including non-fatal upstream failures."""

    papers: list[PaperRecord]
    errors: list[str] = field(default_factory=list)
    sources_attempted: list[str] = field(default_factory=list)
    counts_by_source: dict[str, int] = field(default_factory=dict)
