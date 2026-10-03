"""Decide whether a quoted span actually occurs in the source it claims to quote.

Two gates in this project call themselves "evidence-bound":

* `core/claim_verifier.py` requires a verdict to quote the cited **abstract**;
* `core/peer_reviewer.py` requires a concern to quote the **manuscript**.

Checking only that a quote is *non-empty* does not deliver that guarantee — a
language model can invent a plausible-looking sentence, and a fabricated quote is
worse than no quote because it looks like evidence. This module turns the claim
into an actual check.

Comparison is deliberately tolerant of presentation and strict about content:
whitespace and common punctuation are ignored, case is ignored, and a quote that
stitches several fragments together with an ellipsis is accepted when **every**
fragment occurs in the source. Anything else is not grounded.
"""

from __future__ import annotations

#: Characters ignored when comparing a quote with its source.
_PUNCTUATION = frozenset(
    "，。、；：！？,.;:!?（）()「」《》\"'’‘“”·—–-"
)

#: Markers that let a quote join several non-adjacent fragments of the source.
_ELLIPSIS_MARKERS: tuple[str, ...] = ("...", "…", "[...]", "[…]")

#: Fragments shorter than this are too generic to prove grounding on their own.
_MIN_FRAGMENT_LENGTH = 8


def normalize_for_grounding(text: str) -> str:
    """Lowercase, drop all whitespace, and strip common punctuation.

    Used for both sides of the comparison so that a quote differing from the
    source only in spacing or punctuation is still recognised.
    """
    return "".join(
        character
        for character in text.lower()
        if not character.isspace() and character not in _PUNCTUATION
    )


def _split_fragments(quote: str) -> list[str]:
    """Split a quote on every supported ellipsis marker."""
    fragments = [quote]
    for marker in _ELLIPSIS_MARKERS:
        expanded: list[str] = []
        for fragment in fragments:
            expanded.extend(fragment.split(marker))
        fragments = expanded
    return [fragment for fragment in fragments if fragment.strip()]


def is_quote_grounded(quote: str, source: str) -> bool:
    """Return True when ``quote`` actually occurs in ``source``.

    A quote is grounded when, after normalization, it is a substring of the
    source, or when it joins at least two fragments with an ellipsis and every
    fragment that is long enough to be meaningful occurs in the source.

    An empty or whitespace-only quote is never grounded.
    """
    normalized_quote = normalize_for_grounding(quote)
    if not normalized_quote:
        return False

    normalized_source = normalize_for_grounding(source)
    if not normalized_source:
        return False

    if normalized_quote in normalized_source:
        return True

    fragments = _split_fragments(quote)
    if len(fragments) < 2:
        return False

    meaningful = [
        normalized
        for normalized in (normalize_for_grounding(f) for f in fragments)
        if len(normalized) >= _MIN_FRAGMENT_LENGTH
    ]
    if len(meaningful) < 2:
        return False

    return all(fragment in normalized_source for fragment in meaningful)
