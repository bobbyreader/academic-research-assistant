from __future__ import annotations

from core.quote_grounding import is_quote_grounded, normalize_for_grounding

SOURCE = (
    "Sleep improves memory consolidation in adults. "
    "Attention correlated with recall in the same cohort."
)


def test_exact_substring_is_grounded() -> None:
    assert is_quote_grounded("Sleep improves memory consolidation in adults.", SOURCE)


def test_case_and_punctuation_differences_are_tolerated() -> None:
    assert is_quote_grounded("sleep improves memory consolidation in adults", SOURCE)
    assert is_quote_grounded("Sleep,  improves   memory", SOURCE)


def test_fabricated_sentence_is_not_grounded() -> None:
    assert not is_quote_grounded("This paper proves causality beyond doubt.", SOURCE)


def test_empty_quote_is_never_grounded() -> None:
    assert not is_quote_grounded("", SOURCE)
    assert not is_quote_grounded("   \n ", SOURCE)


def test_empty_source_never_grounds_anything() -> None:
    assert not is_quote_grounded("anything", "")


def test_ellipsis_joining_two_present_fragments_is_grounded() -> None:
    quote = "Sleep improves memory consolidation ... Attention correlated with recall"
    assert is_quote_grounded(quote, SOURCE)


def test_ellipsis_with_one_absent_fragment_is_not_grounded() -> None:
    quote = "Sleep improves memory consolidation ... This proves causality"
    assert not is_quote_grounded(quote, SOURCE)


def test_unicode_ellipsis_is_supported() -> None:
    quote = "Sleep improves memory consolidation … Attention correlated with recall"
    assert is_quote_grounded(quote, SOURCE)


def test_quote_longer_than_source_is_not_grounded() -> None:
    assert not is_quote_grounded(SOURCE + " And much more text follows.", SOURCE)


def test_normalize_strips_whitespace_and_punctuation_and_lowercases() -> None:
    assert normalize_for_grounding("  Sleep, Improves！\n") == "sleepimproves"
