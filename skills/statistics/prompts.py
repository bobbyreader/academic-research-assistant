"""Prompt templates for the Statistics skill.

Contains LLM prompts for statistical methods review, replication analysis,
pseudo-replication detection, multiple comparisons correction, and
reviewer response generation.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class StatsPrompt:
    """Prompt template for statistics review operations."""

    system: str
    user_template: str


# ---------------------------------------------------------------------------
# Statistical Methods Review
# ---------------------------------------------------------------------------

METHODS_COMPLETENESS: StatsPrompt = StatsPrompt(
    system="""You are a statistical reviewer for a high-impact journal. Evaluate
the completeness and appropriateness of statistical methods reporting. Check
against SAMPL (Statistical Analyses and Methods in the Published Literature)
guidelines and journal-specific requirements.""",
    user_template="""Review the statistical methods section:

Manuscript Section:
{methods_text}

Study Design: {study_design}
Field: {field}

Checklist:
1. Sample size determination / power analysis
2. Randomization method
3. Blinding procedures
4. Inclusion/exclusion criteria
5. Primary and secondary outcomes defined
6. Statistical tests named and justified
7. Alpha level specified
8. Software and version reported
9. Data availability statement
10. Code availability statement

For each item: Present / Absent / Inadequate / Not Applicable
Overall Completeness Score: 0-100
Critical Missing Elements: (list)
Recommendations: (specific improvements)""",
)

TEST_JUSTIFICATION: StatsPrompt = StatsPrompt(
    system="""You are a statistical consultant evaluating the appropriateness
of chosen statistical tests. Consider data type, distribution, sample size,
study design, and research question. Suggest alternatives when tests are
inappropriate.""",
    user_template="""Evaluate the statistical test choice:

Research Question: {research_question}
Data Type: {data_type}
Sample Size: {sample_size}
Groups: {groups}
Chosen Test: {chosen_test}
Assumptions Checked: {assumptions}

Evaluation:
1. Is the test appropriate for the data type? yes / no / marginal
2. Are assumptions adequately checked? yes / no / partially
3. Is the test powered adequately? yes / no / unclear
4. Alternative tests to consider: (list)
5. Recommended test: (if different)
6. Justification: (2-3 sentences)""",
)

# ---------------------------------------------------------------------------
# Replication Analysis
# ---------------------------------------------------------------------------

REPLICATE_DISTINCTION: StatsPrompt = StatsPrompt(
    system="""You are a statistical methods expert distinguishing between
biological and technical replicates. This distinction is critical for
proper statistical inference. Misclassification leads to pseudo-replication
and inflated significance.""",
    user_template="""Classify the replicates in this study:

Study Description: {study_description}
Experimental Design: {experimental_design}
Measurements: {measurements}

Definitions:
- Biological replicate: Independent biological samples (different animals,
  cell cultures from different passages, different patients)
- Technical replicate: Repeated measurements of the same biological sample
  (same sample run multiple times, same RNA sequenced twice)

Classification:
1. Are biological replicates present? yes / no / unclear
   - How many?
   - What is the unit of biological replication?
2. Are technical replicates present? yes / no / unclear
   - How many per biological replicate?
   - How are they handled in analysis?
3. Is the distinction clear in the manuscript? yes / no
4. Risk of pseudo-replication: high / medium / low
5. Recommendations: (specific)""",
)

PSEUDOREPLICATION_DETECTION: StatsPrompt = StatsPrompt(
    system="""You are a statistical reviewer detecting pseudo-replication.
Pseudo-replication occurs when technical replicates are treated as biological
replicates, or when non-independent observations are analyzed as independent.
This inflates sample size and produces spurious significance.""",
    user_template="""Detect pseudo-replication in this analysis:

Study Design: {study_design}
Data Structure: {data_structure}
Statistical Analysis: {analysis_description}
Sample Size Claimed: {claimed_n}
True Biological Replicates: {true_n}

Analysis:
1. Is pseudo-replication present? yes / no / likely
2. Type: technical_as_biological / temporal_as_independent /
   spatial_as_independent / nested_as_independent
3. Inflation factor: claimed_n / true_n = X
4. Impact on p-values: severe / moderate / minimal
5. Correct analysis approach: (describe)
6. Required re-analysis: yes / no
7. Manuscript revision needed: major / minor / none""",
)

NESTED_DATA_ANALYSIS: StatsPrompt = StatsPrompt(
    system="""You are a statistical expert in hierarchical/nested data analysis.
Identify nested structures and recommend appropriate mixed-effects models
or other hierarchical approaches. Ignoring nesting leads to incorrect
standard errors and inflated Type I error.""",
    user_template="""Analyze nested data structure:

Study Design: {study_design}
Grouping Factors: {grouping_factors}
Measurements per Group: {measurements_per_group}
Current Analysis: {current_analysis}

Nested Structure:
1. Identify nesting levels: (e.g., cells within mice, students within schools)
2. Crossed vs. nested factors: (classify)
3. Recommended model: (mixed-effects / GEE / cluster-robust SE / other)
4. Random effects structure: (random intercepts / slopes / both)
5. Software implementation: (R/lme4, SAS, Python/statsmodels)
6. Sample size considerations: (effective sample size, design effect)""",
)

# ---------------------------------------------------------------------------
# Multiple Comparisons & Significance
# ---------------------------------------------------------------------------

MULTIPLE_COMPARISONS: StatsPrompt = StatsPrompt(
    system="""You are a statistical reviewer specializing in multiple comparisons
correction. Identify all comparisons being made, assess whether correction is
needed, and recommend appropriate methods. Balance Type I and Type II error
concerns.""",
    user_template="""Review multiple comparisons in this study:

Number of Primary Outcomes: {primary_outcomes}
Number of Secondary Outcomes: {secondary_outcomes}
Number of Subgroups: {subgroups}
Number of Timepoints: {timepoints}
Total Comparisons: {total_comparisons}
Current Correction: {current_correction}

Assessment:
1. Is correction needed? yes / no
2. Family definition: (what constitutes a family of tests)
3. Recommended method: Bonferroni / Holm / FDR / hierarchical / gatekeeping
4. Justification: (2-3 sentences)
5. Impact on current results: (which findings survive correction)
6. Reporting recommendation: (how to present in manuscript)""",
)

SIGNIFICANCE_ABUSE: StatsPrompt = StatsPrompt(
    system="""You are a statistical reviewer detecting significance chasing
and p-hacking. Identify practices like HARKing (Hypothesizing After Results
are Known), cherry-picking outcomes, optional stopping, and inappropriate
subgroup analyses. Recommend transparent reporting practices.""",
    user_template="""Detect significance abuse in this study:

Study Design: {study_design}
Analysis Plan: {analysis_plan}
Reported Results: {reported_results}
Discrepancies: {discrepancies}

Red Flags:
1. HARKing: yes / no / suspected
2. Outcome switching: yes / no / suspected
3. Selective reporting: yes / no / suspected
4. Optional stopping: yes / no / unclear
5. P-value just below 0.05: yes / no
6. Subgroup fishing: yes / no / suspected

Severity: high / medium / low / none
Recommendations:
1. Pre-registration needed: yes / no
2. Analysis plan amendment: yes / no
3. Replication required: yes / no
4. Transparent reporting additions: (list)""",
)

# ---------------------------------------------------------------------------
# Reviewer Response Generation
# ---------------------------------------------------------------------------

REVIEWER_RESPONSE: StatsPrompt = StatsPrompt(
    system="""You are an experienced researcher responding to statistical
reviewer comments. Be professional, evidence-based, and constructive.
Acknowledge valid concerns, provide additional analyses when requested,
and respectfully disagree with justification when appropriate.""",
    user_template="""Draft a response to this statistical reviewer comment:

Reviewer Comment: {reviewer_comment}
Manuscript Section: {manuscript_section}
Current Analysis: {current_analysis}
Additional Data Available: {additional_data}

Response Structure:
1. Thank the reviewer
2. Summarize the concern
3. Provide point-by-point response
4. Describe any additional analyses performed
5. Reference specific manuscript changes
6. If disagreeing, provide statistical justification

Draft Response:""",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REVIEW_PROMPTS: Dict[str, StatsPrompt] = {
    "methods_completeness": METHODS_COMPLETENESS,
    "test_justification": TEST_JUSTIFICATION,
    "replicate_distinction": REPLICATE_DISTINCTION,
    "pseudoreplication": PSEUDOREPLICATION_DETECTION,
    "nested_data": NESTED_DATA_ANALYSIS,
    "multiple_comparisons": MULTIPLE_COMPARISONS,
    "significance_abuse": SIGNIFICANCE_ABUSE,
    "reviewer_response": REVIEWER_RESPONSE,
}
