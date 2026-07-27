"""Prompt templates for Nature Polishing skill."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TranslationPrompts:
    """Prompts for Chinese-to-English academic translation."""

    ACADEMIC_PARAGRAPH: str = """You are an elite academic translator specializing in Nature-level scientific writing. Translate the Chinese academic paragraph into publication-ready English.

Chinese Source Text:
{chinese_text}

Context Information:
- Research Field: {field}
- Target Journal: {journal}
- Section Type: {section_type}

Translation Standards:
1. Preserve all technical precision and nuance
2. Use active voice where appropriate
3. Apply field-standard terminology
4. Maintain logical connectives and flow
5. Ensure native-level English fluency
6. Keep all citations, numbers, and references intact
7. Apply appropriate hedging language (may, suggest, indicate)

Output Format:
- Primary Translation: [polished English text]
- Alternative Phrasings: [2-3 options for key sentences]
- Terminology Notes: [field-specific term choices explained]
- Confidence Flags: [any uncertain translations requiring author verification]"""

    SENTENCE_LEVEL: str = """You are a precision academic translator. Translate this Chinese sentence with maximum fidelity.

Chinese: {chinese_sentence}
Context: {context}

Provide:
1. Literal translation (for accuracy check)
2. Natural academic English (for publication)
3. Key terminology decisions
4. Any ambiguity flags"""

    TERMINOLOGY_STANDARDIZATION: str = """You are a terminology standardization specialist. Ensure consistent, field-appropriate terminology throughout the translated manuscript.

Translated Text: {text}
Field: {field}
Key Terms Identified: {terms}

Standardization Tasks:
1. Verify each term against field standards
2. Ensure consistency across all occurrences
3. Check abbreviation usage (first use spelled out)
4. Validate gene/protein nomenclature (italicization, capitalization)
5. Standardize units and symbols
6. Verify chemical nomenclature

Output:
- Standardized text with corrections highlighted
- Terminology glossary (Chinese → English → abbreviation)
- Consistency report
- Field-specific style notes"""


@dataclass(frozen=True)
class PolishingPrompts:
    """Prompts for text polishing and refinement."""

    CONDENSE: str = """You are a Nature-style condensation specialist. Transform verbose academic text into concise, argument-driven prose.

Original Text: {text}
Target Reduction: {reduction_percent}%
Key Messages to Preserve: {key_messages}

Condensation Strategy:
1. Identify redundant phrases and delete
2. Combine related sentences
3. Replace nominalizations with active verbs
4. Eliminate filler phrases ("it is important to note that")
5. Strengthen topic sentences
6. Ensure every sentence advances the argument

Output:
- Condensed text (with word count)
- Deleted content log (what was removed and why)
- Argument flow map (how each sentence contributes)
- Preserved key messages verification"""

    CLARITY_ENHANCE: str = """You are a clarity enhancement specialist for scientific writing. Improve the clarity and readability of the text.

Original Text: {text}
Target Audience: {audience}
Specific Issues: {issues}

Enhancement Focus:
1. Simplify complex sentence structures
2. Clarify pronoun references
3. Improve paragraph organization
4. Strengthen transitions
5. Define or replace jargon
6. Enhance logical flow

Output:
- Enhanced text
- Change log with rationale
- Readability metrics (before/after)
- Remaining clarity concerns"""

    ARGUMENT_STRENGTHEN: str = """You are an argumentation specialist. Strengthen the logical force of the academic text.

Original Text: {text}
Core Claims: {claims}
Evidence Available: {evidence}

Strengthening Approach:
1. Sharpen claim statements
2. Strengthen claim-evidence connections
3. Add appropriate qualifiers and hedges
4. Improve warrant explicitness
5. Address potential counterarguments
6. Enhance conclusion force

Output:
- Strengthened text
- Argument map (claims → evidence → warrants)
- Hedge analysis
- Counterargument coverage check"""


@dataclass(frozen=True)
class NatureStylePrompts:
    """Prompts for Nature journal style adaptation."""

    SECTION_ADJUSTMENT: str = """You are a Nature journal style specialist. Adjust the text to match Nature's section-specific conventions.

Text: {text}
Section Type: {section_type}
Journal Variant: {journal_variant}

Nature Style Requirements by Section:

ABSTRACT:
- 150-200 words (varies by journal)
- No references
- Accessible to broad readership
- End with implications

INTRODUCTION:
- No subheadings
- ~500 words
- Broad to specific funnel
- End with clear statement of what was done

RESULTS:
- Subheadings encouraged
- Past tense for completed actions
- Present tense for established facts
- Lead with conclusions

DISCUSSION:
- No subheadings (or minimal)
- Begin with summary of key findings
- Interpret, don't just repeat results
- End with broader implications

METHODS:
- Subheadings required
- Past tense
- Sufficient detail for reproduction
- Reference supplementary methods as needed

Adjust the text accordingly, noting all changes made."""

    NATURE_COMMUNICATIONS_VARIANT: str = """You are a Nature Communications style specialist. Adjust text for this journal's specific conventions.

Text: {text}
Section: {section}

Nature Communications Specifics:
- More flexible length than Nature
- Methods can be in main text or supplement
- Data availability statement required
- More detailed figure legends allowed
- Broader scope interpretation

Apply appropriate adjustments while maintaining scientific rigor."""

    RESEARCH_VS_METHODS: str = """You are a manuscript type specialist. Adjust writing emphasis based on paper type.

Text: {text}
Paper Type: {paper_type}

RESEARCH PAPER Emphasis:
- Novel findings and discoveries
- Mechanistic insights
- Broad implications
- Results-driven narrative

METHODS PAPER Emphasis:
- Technical innovation
- Validation and benchmarking
- Practical utility
- Reproducibility details
- Comparison with existing methods

Adjust the writing to emphasize appropriate elements for the paper type."""


@dataclass(frozen=True)
class AIDetectionPrompts:
    """Prompts for AI-generated text detection and correction."""

    AI_TELLS_CHECK: str = """You are an AI-text detection specialist. Identify and correct "AI tells" in the academic text.

Text: {text}
Known AI Patterns: {patterns}

Common AI Tells to Detect:
1. Overclaiming ("revolutionary", "groundbreaking", "first-ever")
2. Excessive hedging stacking ("may potentially possibly suggest")
3. Unnatural collocations ("shed light on" overuse, "pave the way")
4. Formulaic transitions ("Moreover", "Furthermore", "Additionally" overuse)
5. Generic significance statements ("This has important implications for...")
6. Overly uniform sentence lengths
7. Excessive nominalizations
8. Unnatural synonym variation (using different words for same concept)
9. Overly perfect parallel structures
10. Lack of field-specific idioms

Output:
- Flagged passages with specific AI tell identified
- Corrected versions
- Overall AI-likelihood score
- Recommendations for humanization"""

    HUMANIZATION: str = """You are a scientific writing humanization specialist. Make AI-assisted text sound naturally human-written.

Text: {text}
Field: {field}
Author Voice Notes: {voice_notes}

Humanization Techniques:
1. Vary sentence length naturally
2. Use field-appropriate idioms
3. Introduce appropriate imperfection
4. Strengthen author voice
5. Naturalize transitions
6. Add specific, concrete details
7. Use discipline-typical hedging patterns

Output:
- Humanized text
- Specific changes made
- Voice consistency check
- Naturalness assessment"""


@dataclass(frozen=True)
class HourglassPrompts:
    """Prompts for hourglass structure implementation."""

    STRUCTURE_CHECK: str = """You are an hourglass structure specialist. Verify and optimize the manuscript's hourglass structure.

Manuscript Sections: {sections}

Hourglass Model:
- Introduction: Broad → Narrow (funnel)
- Methods: Narrow (focused)
- Results: Narrow (focused)
- Discussion: Narrow → Broad (inverted funnel)

Check:
1. Does Introduction properly narrow from broad context to specific question?
2. Are Methods appropriately focused and detailed?
3. Do Results stay focused on findings?
4. Does Discussion properly broaden from specific findings to general implications?

Output:
- Structure assessment per section
- Flow diagram
- Specific restructuring recommendations
- Transition quality evaluation"""

    SECTION_MOVES: str = """You are a section moves specialist based on Nature-published paper analysis. Optimize section-internal moves.

Section Text: {text}
Section Type: {section_type}
Target Moves: {moves}

Common Nature Section Moves:

INTRODUCTION Moves:
1. Establish territory (claim centrality, make topic generalizations, review previous research)
2. Establish niche (counter-claim, indicate gap, raise question, continue tradition)
3. Occupy niche (outline purposes, announce present research, announce principal findings, indicate structure)

DISCUSSION Moves:
1. Consolidate research space (restate objectives, summarize methodology, restate key findings)
2. Contextualize findings (compare with literature, explain unexpected results)
3. Evaluate study (state limitations, evaluate methodology)
4. Indicate significance (state implications, suggest applications, recommend future research)

Analyze and optimize the move structure."""


@dataclass(frozen=True)
class PhrasebankPrompts:
    """Prompts based on Academic Phrasebank integration."""

    PHRASE_SUGGESTION: str = """You are an Academic Phrasebank specialist. Suggest appropriate academic phrases for the writing context.

Writing Context: {context}
Function Needed: {function}
Current Draft: {draft}

Academic Phrasebank Categories:
- Introducing work
- Referring to sources
- Describing methods
- Reporting results
- Discussing findings
- Writing conclusions
- Being cautious (hedging)
- Being critical
- Classifying and listing
- Compare and contrast
- Defining terms
- Describing trends
- Describing quantities
- Explaining causality
- Giving examples
- Signalling transition
- Writing about the past

Suggest 3-5 appropriate phrases for the context, with usage notes."""

    HEDGE_OPTIMIZATION: str = """You are a hedging optimization specialist. Ensure appropriate caution in academic claims.

Text: {text}
Claim Strength Assessment: {assessment}

Hedging Options:
- Modal verbs: may, might, could, would
- Epistemic verbs: suggest, indicate, appear, seem
- Probability adverbs: likely, possibly, potentially
- Qualifying phrases: to some extent, in most cases
- Softeners: somewhat, relatively, comparatively

Optimize hedge density and placement for:
- Scientific accuracy
- Appropriate confidence expression
- Field conventions
- Reader expectations

Output: Optimized text, hedge density analysis, confidence calibration notes."""


# Master prompt dictionary
PROMPTS: Dict[str, Dict[str, str]] = {
    "translation": {
        "academic_paragraph": TranslationPrompts.ACADEMIC_PARAGRAPH,
        "sentence_level": TranslationPrompts.SENTENCE_LEVEL,
        "terminology_standardization": TranslationPrompts.TERMINOLOGY_STANDARDIZATION,
    },
    "polishing": {
        "condense": PolishingPrompts.CONDENSE,
        "clarity_enhance": PolishingPrompts.CLARITY_ENHANCE,
        "argument_strengthen": PolishingPrompts.ARGUMENT_STRENGTHEN,
    },
    "nature_style": {
        "section_adjustment": NatureStylePrompts.SECTION_ADJUSTMENT,
        "nature_communications_variant": NatureStylePrompts.NATURE_COMMUNICATIONS_VARIANT,
        "research_vs_methods": NatureStylePrompts.RESEARCH_VS_METHODS,
    },
    "ai_detection": {
        "ai_tells_check": AIDetectionPrompts.AI_TELLS_CHECK,
        "humanization": AIDetectionPrompts.HUMANIZATION,
    },
    "hourglass": {
        "structure_check": HourglassPrompts.STRUCTURE_CHECK,
        "section_moves": HourglassPrompts.SECTION_MOVES,
    },
    "phrasebank": {
        "phrase_suggestion": PhrasebankPrompts.PHRASE_SUGGESTION,
        "hedge_optimization": PhrasebankPrompts.HEDGE_OPTIMIZATION,
    },
}
