"""Prompt templates for Nature Writing skill."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class TitlePrompts:
    """Prompts for title construction."""

    GENERATE: str = """You are a Nature journal title specialist. Create compelling, accurate titles.

Research Content: {content}
Key Finding: {key_finding}
Target Journal: {journal}

Generate 5 title options following Nature conventions:
1. Declarative (states the finding)
2. Descriptive (describes the system/approach)
3. Question format (poses the research question)
4. Mechanistic (highlights the mechanism)
5. Broad impact (emphasizes significance)

For each title, provide:
- Character count
- Key strengths
- Potential concerns
- Recommended use case

Ensure titles are: specific, accessible to broad readership, free of jargon, and accurately represent the work."""

    REFINE: str = """You are a title refinement specialist. Improve the given title.

Current Title: {title}
Feedback: {feedback}
Constraints: {constraints}

Refine the title to:
- Enhance clarity and impact
- Improve accuracy
- Better match journal style
- Address specific feedback

Provide 3 refined options with rationale for each change."""


@dataclass(frozen=True)
class AbstractPrompts:
    """Prompts for abstract construction."""

    STRUCTURED: str = """You are a Nature abstract specialist. Write a structured abstract.

Research Content: {content}
Word Limit: {word_limit}

Write a structured abstract with these sections:
- Background (1-2 sentences)
- Methods (2-3 sentences)
- Results (3-4 sentences with key data)
- Conclusions (1-2 sentences)

Ensure: logical flow, appropriate hedging, precise language, and strict word limit compliance."""

    UNSTRUCTURED: str = """You are a Nature abstract specialist. Write a compelling unstructured abstract.

Research Content: {content}
Word Limit: {word_limit}

Write an unstructured abstract that:
- Opens with broad significance
- States the specific gap/question
- Describes the approach concisely
- Presents key results with effect sizes
- Concludes with implications

Maintain narrative flow while including all essential elements."""


@dataclass(frozen=True)
class IntroductionPrompts:
    """Prompts for introduction construction."""

    FULL: str = """You are a Nature introduction specialist. Write a complete introduction.

Research Context: {context}
Literature Gap: {gap}
Research Question: {question}
Contribution: {contribution}

Structure the introduction as:
1. Broad context (2-3 paragraphs) - establish importance
2. Narrowing focus (1-2 paragraphs) - identify specific gap
3. Research question (1 paragraph) - state clearly
4. Approach overview (1 paragraph) - brief preview
5. Contribution statement (1 paragraph) - explicit value

Ensure logical flow, appropriate citations, and compelling narrative."""

    LOGIC_CHAIN: str = """You are a logic chain specialist. Rebuild the introduction's logical structure.

Current Introduction: {introduction}
Identified Issues: {issues}

Rebuild the logic chain:
1. Background → What is known
2. Gap → What is unknown/missing
3. Question → What this study addresses
4. Contribution → What this study provides

Ensure each element flows logically to the next, with clear transitions and explicit connections."""

    PARAGRAPH_REBUILD: str = """You are a paragraph reconstruction specialist. Rebuild the introduction paragraph by paragraph.

Current Paragraphs: {paragraphs}
Target Structure: {structure}

For each paragraph:
- Identify the rhetorical move (context/gap/question/contribution)
- Assess effectiveness
- Reconstruct for clarity and impact
- Ensure smooth transitions

Output: Rebuilt paragraphs with move labels and transition notes."""


@dataclass(frozen=True)
class ResultsPrompts:
    """Prompts for results narrative construction."""

    NARRATIVE: str = """You are a results narrative specialist. Organize results into a compelling narrative.

Data Summary: {data}
Key Findings: {findings}
Figure Plan: {figures}

Structure the results narrative:
1. Opening paragraph - overview of approach
2. Primary findings (organized by question, not by method)
3. Secondary/supporting findings
4. Control/validation experiments
5. Summary paragraph

For each finding:
- Lead with the conclusion
- Support with specific data
- Reference appropriate figures/tables
- Use appropriate statistical language"""

    CLAIM_EVIDENCE: str = """You are a claim-evidence alignment specialist. Ensure every claim is properly supported.

Results Text: {text}
Data Available: {data}

Audit and align:
1. Extract all claims from the text
2. Map each claim to supporting evidence
3. Identify unsupported claims
4. Identify under-utilized evidence
5. Suggest reorganization for optimal claim-evidence flow

Output: Claim-evidence map, flagged issues, reorganization recommendations."""


@dataclass(frozen=True)
class DiscussionPrompts:
    """Prompts for discussion construction."""

    FULL: str = """You are a Nature discussion specialist. Write a comprehensive discussion.

Key Findings: {findings}
Literature Context: {literature}
Limitations: {limitations}
Implications: {implications}

Structure the discussion:
1. Summary of key findings (1 paragraph)
2. Interpretation and mechanism (2-3 paragraphs)
3. Comparison with prior work (1-2 paragraphs)
4. Limitations and caveats (1 paragraph)
5. Broader implications (1 paragraph)
6. Future directions (1 paragraph)
7. Conclusion (1 paragraph)

Maintain appropriate hedging and avoid overclaiming."""

    SIGNIFICANCE: str = """You are a significance paragraph specialist. Craft a compelling significance statement.

Research Finding: {finding}
Field Context: {context}
Potential Impact: {impact}

Write a significance paragraph that:
- States the advance clearly
- Explains why it matters to the field
- Identifies who will benefit
- Suggests new research directions enabled
- Maintains appropriate scope (avoid overclaiming)

Target: 100-150 words, accessible to non-specialists."""


@dataclass(frozen=True)
class TranslationPrompts:
    """Prompts for Chinese-to-English academic translation."""

    PARAGRAPH: str = """You are an academic translation specialist. Translate the Chinese research paragraph to publication-quality English.

Chinese Text: {chinese_text}
Context: {context}
Target Style: {style}

Translation requirements:
- Preserve all technical accuracy
- Use appropriate academic register
- Maintain logical flow
- Apply field-specific terminology correctly
- Ensure natural English expression
- Keep citations and references intact

Output: English translation, terminology notes, alternative phrasings for key terms."""

    TERMINOLOGY_CHECK: str = """You are a terminology consistency specialist. Ensure consistent terminology across the translated manuscript.

Translated Text: {text}
Key Terms: {terms}

Check and standardize:
1. Technical term consistency
2. Abbreviation usage
3. Nomenclature standards
4. Units and symbols
5. Gene/protein naming conventions

Output: Standardized text, terminology glossary, consistency report."""


@dataclass(frozen=True)
class SubmissionPrompts:
    """Prompts for submission package preparation."""

    COVER_LETTER: str = """You are a cover letter specialist. Write a compelling cover letter for journal submission.

Manuscript Title: {title}
Key Findings: {findings}
Journal: {journal}
Editors: {editors}

Cover letter structure:
1. Opening - manuscript title and submission type
2. Significance - why this matters to the journal's readership
3. Key findings - 2-3 sentence summary
4. Fit with journal scope
5. Suggested reviewers (with expertise)
6. Excluded reviewers (with reasons)
7. Declarations (no prior publication, no conflicts)
8. Closing - contact information

Tone: Professional, confident, concise."""

    TITLE_PAGE: str = """You are a title page specialist. Prepare a complete title page.

Manuscript Details: {details}
Author Information: {authors}
Affiliations: {affiliations}

Title page elements:
- Full title
- Running title (if required)
- All authors with affiliations
- Corresponding author contact
- Author contributions (if required at submission)
- Word counts (abstract, main text)
- Figure/table counts
- Supplementary material list"""

    HIGHLIGHTS: str = """You are a highlights specialist. Create compelling highlights for the manuscript.

Key Findings: {findings}
Innovations: {innovations}

Create 3-5 bullet points (85 characters max each) that:
- Highlight the most important findings
- Emphasize methodological innovations
- Note significant implications
- Are accessible to broad readership
- Avoid jargon and abbreviations"""

    AUTHOR_CONTRIBUTIONS: str = """You are an author contributions specialist. Prepare CRediT format contributions.

Author List: {authors}
Contribution Details: {contributions}

Prepare CRediT (Contributor Roles Taxonomy) statements:
- Conceptualization
- Data curation
- Formal analysis
- Funding acquisition
- Investigation
- Methodology
- Project administration
- Resources
- Software
- Supervision
- Validation
- Visualization
- Writing - original draft
- Writing - review & editing

Assign roles accurately based on actual contributions."""

    DATA_AVAILABILITY: str = """You are a data availability statement specialist. Prepare comprehensive data availability statements.

Data Types: {data_types}
Repository Information: {repositories}
Access Restrictions: {restrictions}

Prepare statements covering:
- What data are available
- Where data are deposited (with DOIs/accession numbers)
- Any access restrictions
- Code availability
- Materials availability
- Compliance with journal requirements"""


@dataclass(frozen=True)
class ReviewerRecommendationPrompts:
    """Prompts for reviewer recommendation."""

    SUGGEST_REVIEWERS: str = """You are a reviewer recommendation specialist. Suggest appropriate reviewers.

Manuscript Topic: {topic}
Key Methods: {methods}
Field: {field}
Excluded Individuals: {excluded}

Suggest 5-8 potential reviewers:
- Name and affiliation
- Expertise match
- Recent relevant publications
- Why appropriate for this manuscript
- Any potential conflicts to note

Ensure diversity (geographic, gender, career stage) and avoid close collaborators."""

    REVIEWER_MATRIX: str = """You are a submission matrix specialist. Prepare the complete reviewer recommendation matrix.

Manuscript: {manuscript}
Suggested Reviewers: {reviewers}
Opposed Reviewers: {opposed}

Prepare matrix with:
- Suggested reviewers (name, affiliation, expertise, rationale)
- Opposed reviewers (name, affiliation, reason for opposition)
- Editor suggestions (if applicable)
- Any special handling requests

Format for direct submission system entry."""


@dataclass(frozen=True)
class CompletenessCheckPrompts:
    """Prompts for pre-submission completeness check."""

    FULL_CHECK: str = """You are a submission completeness specialist. Perform a comprehensive pre-submission check.

Submission Package: {package}
Journal Requirements: {requirements}

Check all elements:
1. Manuscript file (format, length, structure)
2. Figures (resolution, format, captions)
3. Tables (format, captions)
4. Supplementary materials (completeness, format)
5. Title page (all required elements)
6. Cover letter
7. Highlights
8. Author information (complete, accurate)
9. Ethics statements
10. Data availability
11. Conflict of interest
12. Funding information
13. Author contributions
14. References (format, completeness)
15. Permissions (if applicable)

Output: Completeness checklist, missing items, format issues, recommendations."""

    FINAL_VERIFICATION: str = """You are a final verification specialist. Perform the last check before submission.

Complete Package: {package}
Submission System: {system}

Final verification:
1. All files present and correctly named
2. All forms completed
3. All declarations signed
4. Metadata matches manuscript
5. Author order confirmed
6. Corresponding author verified
7. Payment/ APC arrangements (if applicable)
8. Submission system entries complete

Output: Go/No-go recommendation, final checklist, submission instructions."""


# Master prompt dictionary
PROMPTS: Dict[str, Dict[str, str]] = {
    "title": {
        "generate": TitlePrompts.GENERATE,
        "refine": TitlePrompts.REFINE,
    },
    "abstract": {
        "structured": AbstractPrompts.STRUCTURED,
        "unstructured": AbstractPrompts.UNSTRUCTURED,
    },
    "introduction": {
        "full": IntroductionPrompts.FULL,
        "logic_chain": IntroductionPrompts.LOGIC_CHAIN,
        "paragraph_rebuild": IntroductionPrompts.PARAGRAPH_REBUILD,
    },
    "results": {
        "narrative": ResultsPrompts.NARRATIVE,
        "claim_evidence": ResultsPrompts.CLAIM_EVIDENCE,
    },
    "discussion": {
        "full": DiscussionPrompts.FULL,
        "significance": DiscussionPrompts.SIGNIFICANCE,
    },
    "translation": {
        "paragraph": TranslationPrompts.PARAGRAPH,
        "terminology_check": TranslationPrompts.TERMINOLOGY_CHECK,
    },
    "submission": {
        "cover_letter": SubmissionPrompts.COVER_LETTER,
        "title_page": SubmissionPrompts.TITLE_PAGE,
        "highlights": SubmissionPrompts.HIGHLIGHTS,
        "author_contributions": SubmissionPrompts.AUTHOR_CONTRIBUTIONS,
        "data_availability": SubmissionPrompts.DATA_AVAILABILITY,
    },
    "reviewer_recommendation": {
        "suggest_reviewers": ReviewerRecommendationPrompts.SUGGEST_REVIEWERS,
        "reviewer_matrix": ReviewerRecommendationPrompts.REVIEWER_MATRIX,
    },
    "completeness_check": {
        "full_check": CompletenessCheckPrompts.FULL_CHECK,
        "final_verification": CompletenessCheckPrompts.FINAL_VERIFICATION,
    },
}
