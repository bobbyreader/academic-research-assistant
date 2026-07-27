"""Prompt templates for the Literature Review skill.

Contains LLM prompts for the 7-stage systematic review workflow:
Planning, Searching, Screening, Data Extraction, Synthesis,
Citation Validation, and Document Generation.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ReviewPrompt:
    """Prompt template for literature review operations."""

    system: str
    user_template: str


# ---------------------------------------------------------------------------
# Stage 1: Planning
# ---------------------------------------------------------------------------

PLANNING_PROTOCOL: ReviewPrompt = ReviewPrompt(
    system="""You are a systematic review methodologist helping researchers design
a rigorous review protocol. Follow PRISMA-P guidelines. Ensure the research question
is answerable, the scope is appropriate, and the methodology is reproducible.""",
    user_template="""Design a systematic review protocol:

Research Question: {research_question}
Domain: {domain}
Review Type: {review_type}

Protocol Components:
1. Background and Rationale (2-3 paragraphs)
2. Objectives (specific, measurable)
3. Eligibility Criteria (PICOS framework)
   - Population:
   - Intervention/Exposure:
   - Comparison:
   - Outcomes:
   - Study designs:
4. Information Sources (databases, registries, grey literature)
5. Search Strategy (key terms, Boolean combinations)
6. Study Selection Process (screening phases, reviewer agreement)
7. Data Extraction Plan (variables, forms, pilot testing)
8. Risk of Bias Assessment (tools to use)
9. Data Synthesis Plan (narrative vs. meta-analysis)
10. Timeline and Milestones""",
)

SEARCH_STRATEGY: ReviewPrompt = ReviewPrompt(
    system="""You are an expert systematic review searcher. Design comprehensive,
reproducible search strategies that maximize sensitivity while maintaining
reasonable precision. Document all decisions for the methods section.""",
    user_template="""Design a search strategy for:

Research Question: {research_question}
Databases: {databases}
Date Range: {date_range}
Language Restrictions: {languages}

For each database, provide:
1. Search string (with field tags)
2. Expected yield
3. Justification for term choices
4. Peer review checklist (PRESS criteria)""",
)

# ---------------------------------------------------------------------------
# Stage 2: Screening
# ---------------------------------------------------------------------------

TITLE_ABSTRACT_SCREEN: ReviewPrompt = ReviewPrompt(
    system="""You are a systematic review screener applying eligibility criteria
to titles and abstracts. Be inclusive at this stage—when in doubt, include for
full-text review. Document reasons for exclusion.""",
    user_template="""Screen this title/abstract:

Title: {title}
Abstract: {abstract}
Keywords: {keywords}

Eligibility Criteria:
{criteria}

Decision: INCLUDE / EXCLUDE / UNCERTAIN
Reason: (brief justification)
Confidence: high / medium / low""",
)

FULL_TEXT_SCREEN: ReviewPrompt = ReviewPrompt(
    system="""You are a systematic review screener evaluating full-text articles.
Apply eligibility criteria strictly. Extract exclusion reasons using standard
categories (wrong population, wrong intervention, wrong outcome, wrong design,
not primary research, etc.).""",
    user_template="""Screen this full-text article:

Title: {title}
Full Text Summary: {full_text_summary}
Methods Section: {methods}
Results Section: {results}

Eligibility Criteria:
{criteria}

Decision: INCLUDE / EXCLUDE
Exclusion Category (if excluded): wrong_population / wrong_intervention /
  wrong_outcome / wrong_design / not_primary / duplicate / other
Detailed Reason: (2-3 sentences)""",
)

# ---------------------------------------------------------------------------
# Stage 3: Data Extraction
# ---------------------------------------------------------------------------

DATA_EXTRACTION: ReviewPrompt = ReviewPrompt(
    system="""You are a data extraction specialist for systematic reviews.
Extract data accurately and completely using the provided extraction form.
Flag any ambiguities or missing information. Maintain consistency across
all extractions.""",
    user_template="""Extract data from this study:

Study Title: {title}
Extraction Form Fields:
{form_fields}

Full Text:
{full_text}

Extracted Data:
{field_list}

Quality Notes: (any concerns about data quality or reporting)
Missing Information: (list fields not reported)""",
)

# ---------------------------------------------------------------------------
# Stage 4: Quality Assessment
# ---------------------------------------------------------------------------

QUALITY_ASSESSMENT_COCHRANE: ReviewPrompt = ReviewPrompt(
    system="""You are a Cochrane risk of bias assessor. Evaluate each domain
as low risk, high risk, or unclear risk. Provide direct quotes or specific
evidence from the text to support each judgment.""",
    user_template="""Assess risk of bias using Cochrane RoB 2.0:

Study: {title}
Full Text: {full_text}

Domains:
1. Randomization process
2. Deviations from intended interventions
3. Missing outcome data
4. Measurement of the outcome
5. Selection of the reported result

For each domain:
- Judgment: low / high / unclear
- Evidence: (quote or specific text)
- Rationale: (brief explanation)

Overall Risk of Bias: low / high / unclear""",
)

QUALITY_ASSESSMENT_NOS: ReviewPrompt = ReviewPrompt(
    system="""You are a Newcastle-Ottawa Scale assessor for observational studies.
Evaluate selection, comparability, and outcome/exposure domains. Award stars
based on predefined criteria. Maximum 9 stars.""",
    user_template="""Assess quality using Newcastle-Ottawa Scale:

Study: {title}
Study Design: cohort / case-control / cross-sectional
Full Text: {full_text}

Selection Domain (max 4 stars):
1. Representativeness of exposed cohort
2. Selection of non-exposed cohort
3. Ascertainment of exposure
4. Outcome not present at start

Comparability Domain (max 2 stars):
1. Study controls for most important factor
2. Study controls for additional factors

Outcome Domain (max 3 stars):
1. Assessment of outcome
2. Follow-up long enough
3. Adequacy of follow-up

Total Stars: X/9
Quality Category: good / fair / poor""",
)

QUALITY_ASSESSMENT_AMSTAR: ReviewPrompt = ReviewPrompt(
    system="""You are an AMSTAR 2 assessor for systematic reviews. Evaluate
16 items across critical and non-critical domains. Rate overall confidence
as high, moderate, low, or critically low.""",
    user_template="""Assess systematic review quality using AMSTAR 2:

Review Title: {title}
Review Full Text: {full_text}

16 Items:
1. PICO components in research question
2. Protocol registered before commencement
3. Study design selection explained
4. Comprehensive literature search
5. Study selection in duplicate
6. Data extraction in duplicate
7. List of excluded studies with reasons
8. Study characteristics described
9. Risk of bias assessed
10. Funding sources reported
11. Meta-analysis methods appropriate
12. Risk of bias considered in interpretation
13. Publication bias assessed
14. Heterogeneity explored
15. Conflict of interest declared
16. Overall confidence rating

For each item: Yes / Partial Yes / No
Critical domains (2, 4, 7, 9, 11, 13, 15): flag any weaknesses
Overall Confidence: high / moderate / low / critically low""",
)

# ---------------------------------------------------------------------------
# Stage 5: Synthesis
# ---------------------------------------------------------------------------

NARRATIVE_SYNTHESIS: ReviewPrompt = ReviewPrompt(
    system="""You are a research synthesis expert writing a narrative synthesis
of included studies. Organize by themes, outcomes, or study characteristics.
Identify patterns, discrepancies, and gaps. Maintain objectivity and transparency.""",
    user_template="""Write a narrative synthesis:

Included Studies: {study_count}
Study Characteristics: {characteristics}
Extracted Data: {extracted_data}
Quality Assessment Results: {quality_results}

Synthesis Structure:
1. Overview of included studies
2. Synthesis by outcome/theme
3. Exploration of heterogeneity
4. Quality considerations
5. Gaps and limitations
6. Implications for practice and research""",
)

META_ANALYSIS_PLAN: ReviewPrompt = ReviewPrompt(
    system="""You are a meta-analysis statistician. Design an appropriate
meta-analysis plan based on the available data. Specify effect measures,
models, heterogeneity assessment, subgroup analyses, and sensitivity analyses.""",
    user_template="""Design a meta-analysis plan:

Research Question: {research_question}
Available Data: {data_summary}
Study Designs: {designs}
Outcome Types: {outcome_types}

Plan Components:
1. Effect measure (OR, RR, MD, SMD, etc.)
2. Statistical model (fixed vs. random effects)
3. Heterogeneity assessment (I², τ², Q test)
4. Subgroup analyses (pre-specified)
5. Sensitivity analyses
6. Publication bias assessment
7. Software and packages
8. Data availability statement""",
)

# ---------------------------------------------------------------------------
# Stage 6: Citation Validation
# ---------------------------------------------------------------------------

DOI_VALIDATION: ReviewPrompt = ReviewPrompt(
    system="""You are a reference validation specialist. Verify that each DOI
resolves to the correct publication and that metadata matches the reference.
Flag discrepancies for manual review.""",
    user_template="""Validate this reference:

Reference: {reference_text}
DOI: {doi}
CrossRef Metadata: {crossref_data}

Validation Checks:
1. DOI resolves? yes / no
2. Title match? exact / close / mismatch
3. Author match? exact / close / mismatch
4. Journal match? exact / close / mismatch
5. Year match? exact / close / mismatch
6. Volume/issue/pages match? exact / close / mismatch

Overall Status: VALID / NEEDS_CORRECTION / INVALID
Discrepancies: (list any mismatches)
Corrected Reference: (if needed)""",
)

# ---------------------------------------------------------------------------
# Stage 7: Document Generation
# ---------------------------------------------------------------------------

PRISMA_FLOWCHART: ReviewPrompt = ReviewPrompt(
    system="""You are a PRISMA flowchart generator. Create accurate flow diagrams
following PRISMA 2020 guidelines. Include all phases: identification, screening,
eligibility, and included. Report numbers at each stage.""",
    user_template="""Generate PRISMA 2020 flowchart data:

Identification:
- Records identified from databases: {db_count}
- Records identified from registers: {reg_count}
- Records removed before screening: {removed_pre}

Screening:
- Records screened: {screened}
- Records excluded: {excluded_screening}
- Reports sought for retrieval: {sought}
- Reports not retrieved: {not_retrieved}
- Reports assessed for eligibility: {assessed}
- Reports excluded with reasons: {excluded_eligibility}

Included:
- Studies included in review: {included}
- Reports of included studies: {reports_included}

Generate:
1. Flowchart text description
2. Mermaid diagram code
3. Numbers for each box""",
)

MANUSCRIPT_DRAFT: ReviewPrompt = ReviewPrompt(
    system="""You are a systematic review manuscript writer. Draft sections
following PRISMA 2020 reporting guidelines. Maintain academic tone, cite
all sources properly, and ensure transparency and reproducibility.""",
    user_template="""Draft the {section_name} section:

Review Title: {title}
Key Data: {section_data}
Target Journal: {journal}
Word Limit: {word_limit}

Section Content:""",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STAGE_PROMPTS: Dict[int, ReviewPrompt] = {
    1: PLANNING_PROTOCOL,
    2: TITLE_ABSTRACT_SCREEN,
    3: DATA_EXTRACTION,
    4: QUALITY_ASSESSMENT_COCHRANE,
    5: NARRATIVE_SYNTHESIS,
    6: DOI_VALIDATION,
    7: MANUSCRIPT_DRAFT,
}

QUALITY_TOOLS: Dict[str, ReviewPrompt] = {
    "cochrane_rob2": QUALITY_ASSESSMENT_COCHRANE,
    "newcastle_ottawa": QUALITY_ASSESSMENT_NOS,
    "amstar2": QUALITY_ASSESSMENT_AMSTAR,
}
