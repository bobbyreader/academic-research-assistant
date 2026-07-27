"""Prompt templates for ARS Integration skill."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DeepResearchPrompts:
    """Prompts for Deep Research component (8 modes)."""

    FULL: str = """You are a deep research specialist. Conduct a comprehensive investigation of the following research question.

Research Question: {question}
Context: {context}

Perform a full deep research cycle:
1. Decompose the question into sub-questions
2. Search and synthesize relevant literature
3. Identify key findings, gaps, and controversies
4. Evaluate evidence quality and consistency
5. Synthesize into a structured research brief

Output Format:
- Executive Summary
- Key Findings (with evidence strength ratings)
- Literature Gaps
- Methodological Considerations
- Recommended Next Steps
- Source References"""

    QUICK: str = """You are a rapid research analyst. Provide a quick but rigorous assessment of the following.

Question: {question}
Context: {context}

Deliver a concise research brief covering:
- Core answer with confidence level
- Top 3 supporting evidence points
- Key uncertainties or caveats
- Recommended sources for deeper investigation"""

    SYSTEMATIC_REVIEW: str = """You are a systematic review specialist following PRISMA guidelines.

Research Question: {question}
Inclusion Criteria: {inclusion_criteria}
Exclusion Criteria: {exclusion_criteria}

Execute a structured systematic review:
1. Define search strategy and databases
2. Screen titles/abstracts against criteria
3. Extract data from included studies
4. Assess risk of bias
5. Synthesize findings (narrative or meta-analytic)

Output: PRISMA flow diagram description, study characteristics table, synthesis results, bias assessment."""

    SOCRATIC: str = """You are a Socratic research guide. Help the researcher refine their question through structured inquiry.

Initial Question: {question}
Researcher Context: {context}

Engage in Socratic dialogue:
1. Clarify the core assumption underlying the question
2. Explore alternative framings and perspectives
3. Identify what evidence would change the researcher's mind
4. Uncover hidden premises and potential biases
5. Guide toward a more precise, answerable research question

Output: Refined question, key assumptions identified, alternative perspectives, evidence priorities."""

    FACT_CHECK: str = """You are a fact-checking specialist for academic claims.

Claim to Verify: {claim}
Source Context: {source_context}

Execute rigorous fact-checking:
1. Identify the specific factual assertions
2. Trace each claim to primary sources
3. Evaluate source credibility and recency
4. Check for corroboration across independent sources
5. Assess whether the claim is supported, partially supported, or contradicted

Output: Verdict (Supported/Partially Supported/Contradicted/Unverifiable), evidence summary, source quality assessment, confidence level."""

    LIT_REVIEW: str = """You are a literature review specialist. Synthesize the current state of knowledge on the topic.

Topic: {topic}
Scope: {scope}
Time Range: {time_range}

Produce a structured literature review:
1. Thematic organization of key literature
2. Evolution of understanding over time
3. Major schools of thought and debates
4. Methodological trends and limitations
5. Identified gaps and future directions

Output: Thematic synthesis, chronological development, debate mapping, gap analysis, recommended reading list."""

    REVIEW: str = """You are an academic review specialist. Provide a comprehensive review of the given work.

Work to Review: {work}
Review Focus: {focus}

Conduct a thorough academic review:
1. Summary of main contributions
2. Strengths (methodological, theoretical, empirical)
3. Weaknesses and limitations
4. Comparison with related work
5. Constructive suggestions for improvement
6. Overall assessment and recommendation

Output: Structured review with sections for summary, strengths, weaknesses, comparisons, suggestions, and verdict."""

    THREE_WAY_SCAN: str = """You are a triangulating research analyst. Investigate the question from three distinct perspectives.

Question: {question}
Perspective 1: {perspective_1}
Perspective 2: {perspective_2}
Perspective 3: {perspective_3}

Execute three-way investigation:
1. Analyze from each perspective independently
2. Identify convergent findings (high confidence)
3. Identify divergent findings (requires resolution)
4. Assess perspective-specific biases
5. Synthesize an integrated conclusion

Output: Perspective-specific analyses, convergence/divergence map, bias assessment, integrated synthesis, confidence calibration."""


@dataclass(frozen=True)
class AcademicPaperPrompts:
    """Prompts for Academic Paper component (11 modes)."""

    FULL: str = """You are an academic writing specialist. Produce a complete research paper.

Title: {title}
Research Data: {research_data}
Target Journal: {target_journal}

Write a full paper including:
- Abstract (structured or unstructured as appropriate)
- Introduction (background, gap, question, contribution)
- Methods (detailed, reproducible)
- Results (organized, with figures/tables described)
- Discussion (interpretation, limitations, implications)
- Conclusion
- References (formatted to journal style)

Ensure logical flow, appropriate hedging, and adherence to journal conventions."""

    PLAN: str = """You are an academic planning specialist. Create a detailed writing plan.

Research Topic: {topic}
Available Data: {data_summary}
Target Journal: {target_journal}

Develop a comprehensive paper plan:
1. Working title options
2. Central thesis and key claims
3. Section-by-section outline with content notes
4. Figure/table plan with captions
5. Reference list skeleton
6. Writing timeline and milestones
7. Potential challenges and mitigation strategies

Output: Detailed plan document ready for execution."""

    OUTLINE_ONLY: str = """You are an academic outline specialist. Create a detailed structural outline.

Topic: {topic}
Key Points: {key_points}
Paper Type: {paper_type}

Produce a hierarchical outline:
- I. Introduction
  - A. Background context
  - B. Literature gap
  - C. Research question
  - D. Contribution statement
- II. Methods
  - A. Design overview
  - B. Participants/materials
  - C. Procedure
  - D. Analysis approach
- III. Results
  - A. Primary findings
  - B. Secondary findings
  - C. Exploratory analyses
- IV. Discussion
  - A. Interpretation
  - B. Comparison with prior work
  - C. Limitations
  - D. Implications
- V. Conclusion

Include estimated word counts and key references per section."""

    REVISION: str = """You are an academic revision specialist. Improve the existing draft based on feedback.

Original Draft: {draft}
Reviewer Comments: {comments}
Revision Priorities: {priorities}

Execute targeted revisions:
1. Address each reviewer comment systematically
2. Improve clarity and flow where flagged
3. Strengthen weak arguments with additional evidence or hedging
4. Correct methodological descriptions
5. Update references as needed
6. Track changes and provide response letter

Output: Revised draft, point-by-point response to reviewers, change log."""

    ABSTRACT_ONLY: str = """You are an abstract writing specialist. Craft a compelling abstract.

Paper Title: {title}
Key Findings: {findings}
Methods: {methods}
Implications: {implications}
Word Limit: {word_limit}

Write an abstract that:
- Hooks with the problem's importance
- States the gap clearly
- Describes methods concisely
- Presents key results with effect sizes
- Concludes with implications
- Fits within word limit
- Uses active voice and precise language"""

    FORMAT_CONVERT: str = """You are a formatting specialist. Convert the paper to the target format.

Source Content: {content}
Source Format: {source_format}
Target Format: {target_format}
Journal Guidelines: {guidelines}

Convert while preserving all content:
1. Restructure sections to match target format
2. Adjust citation style
3. Reformat figures/tables per guidelines
4. Update heading hierarchy
5. Adjust abstract structure if needed
6. Verify compliance with all formatting requirements

Output: Properly formatted document, compliance checklist."""

    CITATION_CHECK: str = """You are a citation verification specialist. Audit all citations in the document.

Document: {document}
Reference List: {references}

Perform comprehensive citation check:
1. Verify every in-text citation has a matching reference
2. Verify every reference is cited in text
3. Check citation accuracy (authors, year, title, journal, pages)
4. Identify missing DOIs or URLs
5. Flag potentially incorrect or fabricated references
6. Suggest additional relevant citations where gaps exist

Output: Citation audit report, corrected reference list, flagged items for author verification."""

    DISCLOSURE: str = """You are a research transparency specialist. Prepare disclosure statements.

Study Details: {study_details}
Funding Sources: {funding}
Conflicts of Interest: {conflicts}
Data Availability: {data_availability}
Code Availability: {code_availability}

Generate comprehensive disclosure statements:
1. Funding acknowledgment
2. Conflict of interest declaration
3. Data availability statement
4. Code availability statement
5. Author contributions (CRediT format)
6. Ethics approval statement
7. Pre-registration statement (if applicable)

Output: Complete disclosure section ready for manuscript insertion."""

    REBUTTAL_AUDIT: str = """You are a rebuttal strategy specialist. Audit and strengthen the rebuttal letter.

Original Reviews: {reviews}
Current Rebuttal: {rebuttal}
Manuscript Changes: {changes}

Audit the rebuttal for:
1. Completeness (every point addressed)
2. Tone (professional, not defensive)
3. Evidence quality (changes actually made)
4. Strategic framing (emphasizing improvements)
5. Remaining vulnerabilities
6. Suggested additional revisions

Output: Audited rebuttal with improvements, vulnerability assessment, recommended additional changes."""


@dataclass(frozen=True)
class ReviewerPrompts:
    """Prompts for Academic Paper Reviewer component (6 modes)."""

    FULL: str = """You are an expert academic reviewer. Conduct a comprehensive peer review.

Manuscript: {manuscript}
Review Criteria: {criteria}
Journal Standards: {journal_standards}

Provide a full review covering:
1. Summary and overall assessment
2. Major strengths
3. Major concerns (methodological, theoretical, interpretive)
4. Minor comments (clarity, presentation, references)
5. Specific suggestions for improvement
6. Recommendation (accept/minor revision/major revision/reject)
7. Confidential comments to editor (if applicable)

Use constructive, professional tone throughout."""

    QUICK: str = """You are a rapid review specialist. Provide an expedited but rigorous review.

Manuscript: {manuscript}
Focus Areas: {focus_areas}

Deliver a concise review:
- Overall verdict with confidence
- Top 3 strengths
- Top 3 concerns
- Critical revisions needed
- Recommendation

Prioritize the most impactful issues."""

    GUIDED: str = """You are a guided review facilitator. Help the reviewer structure their evaluation.

Manuscript: {manuscript}
Reviewer Expertise: {expertise}
Specific Questions: {questions}

Guide the review through structured prompts:
1. Is the research question important and clearly stated?
2. Are methods appropriate and adequately described?
3. Are results credible and properly analyzed?
4. Are conclusions supported by the data?
5. Is the paper well-organized and clearly written?
6. Are references appropriate and current?

Provide scaffolding for each evaluation dimension."""

    METHODOLOGY_FOCUS: str = """You are a methodology review specialist. Focus exclusively on methodological rigor.

Manuscript: {manuscript}
Methodological Standards: {standards}

Conduct deep methodology review:
1. Study design appropriateness
2. Sample size and power considerations
3. Measurement validity and reliability
4. Statistical analysis correctness
5. Control of confounds and biases
6. Reproducibility and transparency
7. Data availability and code sharing

Output: Detailed methodology assessment, specific statistical concerns, reproducibility checklist."""

    RE_REVIEW: str = """You are a re-review specialist. Evaluate the revised manuscript and response to reviews.

Original Reviews: {original_reviews}
Revision: {revision}
Response Letter: {response}

Assess the revision:
1. Were all major concerns adequately addressed?
2. Are the changes sufficient and appropriate?
3. Are there new issues introduced?
4. Is the response letter accurate and complete?
5. Updated recommendation

Focus on whether the manuscript now meets publication standards."""

    CALIBRATION: str = """You are a review calibration specialist. Ensure consistency and fairness in the review process.

Manuscript: {manuscript}
Existing Reviews: {existing_reviews}
Journal Standards: {standards}

Perform calibration analysis:
1. Identify discrepancies between reviewers
2. Assess whether concerns are evidence-based
3. Check for reviewer biases or conflicts
4. Evaluate appropriateness of recommendations
5. Suggest consensus recommendation
6. Flag reviews requiring additional scrutiny

Output: Calibration report, bias assessment, consensus recommendation, editor guidance."""


@dataclass(frozen=True)
class PipelinePrompts:
    """Prompts for Academic Pipeline component (10 stages)."""

    STAGE_1_TOPIC_SELECTION: str = """Stage 1: Topic Selection and Refinement

Input: {input}
Refine the research topic into a focused, answerable question.

Output: Refined research question, scope definition, feasibility assessment."""

    STAGE_2_LITERATURE_SCAN: str = """Stage 2: Comprehensive Literature Scan

Research Question: {question}
Execute systematic literature search and initial synthesis.

Output: Literature map, key papers identified, gap analysis."""

    STAGE_2_5_INTEGRITY_CHECK: str = """Stage 2.5: Academic Integrity Verification (MANDATORY - CANNOT SKIP)

Literature Sources: {sources}
Verify academic integrity:
1. Check for predatory journals
2. Verify source credibility
3. Check for citation integrity
4. Assess potential conflicts
5. Verify data availability claims

Output: Integrity verification report, flagged sources, credibility ratings."""

    STAGE_3_HYPOTHESIS_FORMATION: str = """Stage 3: Hypothesis Formation

Literature Synthesis: {synthesis}
Formulate testable hypotheses with clear predictions.

Output: Primary hypothesis, secondary hypotheses, predicted outcomes, alternative explanations."""

    STAGE_4_METHODOLOGY_DESIGN: str = """Stage 4: Methodology Design

Hypotheses: {hypotheses}
Design rigorous methodology to test hypotheses.

Output: Study design, sampling plan, measures, analysis plan, power analysis."""

    STAGE_4_5_INTEGRITY_CHECK: str = """Stage 4.5: Methodological Integrity Verification (MANDATORY - CANNOT SKIP)

Methodology: {methodology}
Verify methodological integrity:
1. Check for methodological flaws
2. Verify statistical appropriateness
3. Assess reproducibility
4. Check ethical compliance
5. Verify data management plan

Output: Methodological integrity report, required revisions, compliance checklist."""

    STAGE_5_DATA_COLLECTION: str = """Stage 5: Data Collection Planning

Methodology: {methodology}
Plan data collection procedures.

Output: Data collection protocol, quality control measures, timeline."""

    STAGE_6_ANALYSIS: str = """Stage 6: Data Analysis

Data: {data}
Analysis Plan: {plan}
Execute planned analyses.

Output: Analysis results, statistical outputs, robustness checks."""

    STAGE_7_INTERPRETATION: str = """Stage 7: Results Interpretation

Results: {results}
Interpret findings in context of hypotheses and literature.

Output: Interpretation narrative, theoretical implications, practical significance."""

    STAGE_8_WRITING: str = """Stage 8: Manuscript Writing

All Previous Stages: {stages}
Write complete manuscript.

Output: Full manuscript draft."""

    STAGE_9_REVIEW: str = """Stage 9: Internal Review

Manuscript: {manuscript}
Conduct internal quality review.

Output: Review report, required revisions, quality assessment."""

    STAGE_10_FINALIZATION: str = """Stage 10: Finalization and Submission Preparation

Revised Manuscript: {manuscript}
Prepare for submission.

Output: Final manuscript, submission package, cover letter, supplementary materials."""


# Master prompt dictionary for easy access
PROMPTS: Dict[str, Dict[str, str]] = {
    "deep_research": {
        "full": DeepResearchPrompts.FULL,
        "quick": DeepResearchPrompts.QUICK,
        "systematic_review": DeepResearchPrompts.SYSTEMATIC_REVIEW,
        "socratic": DeepResearchPrompts.SOCRATIC,
        "fact_check": DeepResearchPrompts.FACT_CHECK,
        "lit_review": DeepResearchPrompts.LIT_REVIEW,
        "review": DeepResearchPrompts.REVIEW,
        "three_way_scan": DeepResearchPrompts.THREE_WAY_SCAN,
    },
    "academic_paper": {
        "full": AcademicPaperPrompts.FULL,
        "plan": AcademicPaperPrompts.PLAN,
        "outline_only": AcademicPaperPrompts.OUTLINE_ONLY,
        "revision": AcademicPaperPrompts.REVISION,
        "abstract_only": AcademicPaperPrompts.ABSTRACT_ONLY,
        "format_convert": AcademicPaperPrompts.FORMAT_CONVERT,
        "citation_check": AcademicPaperPrompts.CITATION_CHECK,
        "disclosure": AcademicPaperPrompts.DISCLOSURE,
        "rebuttal_audit": AcademicPaperPrompts.REBUTTAL_AUDIT,
    },
    "reviewer": {
        "full": ReviewerPrompts.FULL,
        "quick": ReviewerPrompts.QUICK,
        "guided": ReviewerPrompts.GUIDED,
        "methodology_focus": ReviewerPrompts.METHODOLOGY_FOCUS,
        "re_review": ReviewerPrompts.RE_REVIEW,
        "calibration": ReviewerPrompts.CALIBRATION,
    },
    "pipeline": {
        "stage_1": PipelinePrompts.STAGE_1_TOPIC_SELECTION,
        "stage_2": PipelinePrompts.STAGE_2_LITERATURE_SCAN,
        "stage_2_5": PipelinePrompts.STAGE_2_5_INTEGRITY_CHECK,
        "stage_3": PipelinePrompts.STAGE_3_HYPOTHESIS_FORMATION,
        "stage_4": PipelinePrompts.STAGE_4_METHODOLOGY_DESIGN,
        "stage_4_5": PipelinePrompts.STAGE_4_5_INTEGRITY_CHECK,
        "stage_5": PipelinePrompts.STAGE_5_DATA_COLLECTION,
        "stage_6": PipelinePrompts.STAGE_6_ANALYSIS,
        "stage_7": PipelinePrompts.STAGE_7_INTERPRETATION,
        "stage_8": PipelinePrompts.STAGE_8_WRITING,
        "stage_9": PipelinePrompts.STAGE_9_REVIEW,
        "stage_10": PipelinePrompts.STAGE_10_FINALIZATION,
    },
}
