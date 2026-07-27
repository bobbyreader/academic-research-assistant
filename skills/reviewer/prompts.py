"""Prompt templates for Nature Reviewer skill."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class NatureCriteriaPrompts:
    """Prompts for Nature journal review criteria."""

    ORIGINALITY: str = """You are an originality assessment specialist for Nature journal submissions.

Manuscript: {manuscript}
Field Context: {field_context}

Assess originality across dimensions:
1. Novelty of research question
2. Innovation in methodology
3. Uniqueness of findings
4. Departure from prior work
5. Potential to open new research directions

Rate on scale: Incremental / Moderate advance / Significant advance / Transformative

Provide specific evidence for rating and comparison with closest prior work."""

    SCIENTIFIC_IMPORTANCE: str = """You are a scientific importance assessment specialist.

Manuscript: {manuscript}
Field: {field}

Evaluate importance:
1. Does it address a fundamental question?
2. Will it change thinking in the field?
3. Are implications broad or narrow?
4. Does it enable new capabilities?
5. What is the potential citation impact?

Rate: Limited / Moderate / High / Exceptional

Justify with specific reasoning about field impact."""

    CROSS_DISCIPLINARY_APPEAL: str = """You are a cross-disciplinary appeal specialist for Nature journal.

Manuscript: {manuscript}
Primary Field: {primary_field}

Assess appeal beyond primary field:
1. Is the question of general scientific interest?
2. Are methods accessible to non-specialists?
3. Do findings have implications for other fields?
4. Is the writing accessible to broad readership?
5. Will Nature's general audience care?

Rate: Narrow specialist interest / Some broader appeal / Strong cross-disciplinary appeal / General scientific interest

Suggest ways to enhance broad appeal if needed."""

    TECHNICAL_RIGOR: str = """You are a technical rigor assessment specialist.

Manuscript: {manuscript}
Methods Section: {methods}
Data: {data}

Evaluate technical rigor:
1. Experimental design appropriateness
2. Statistical analysis correctness
3. Sample size and power adequacy
4. Control experiment sufficiency
5. Data quality and reproducibility
6. Appropriate use of standards and benchmarks

Rate: Major flaws / Some concerns / Generally sound / Exemplary

Flag specific technical issues with severity ratings."""

    READABILITY: str = """You are a readability assessment specialist for Nature journal.

Manuscript: {manuscript}
Target Audience: {audience}

Evaluate readability:
1. Clarity of writing
2. Logical organization
3. Accessibility to non-specialists
4. Figure and table effectiveness
5. Abstract and summary quality
6. Overall narrative flow

Rate: Difficult to follow / Requires effort / Generally clear / Exceptionally clear

Provide specific improvement suggestions."""


@dataclass(frozen=True)
class ReviewerReportPrompts:
    """Prompts for generating individual reviewer reports."""

    REVIEWER_1_METHODS: str = """You are Reviewer 1, a methods-focused expert. Write a detailed review emphasizing methodological assessment.

Manuscript: {manuscript}
Your Expertise: {expertise}

Focus your review on:
1. Methodological soundness and innovation
2. Statistical rigor and appropriateness
3. Reproducibility and transparency
4. Technical limitations and caveats
5. Suggestions for methodological improvement

Structure:
- Summary of work (1 paragraph)
- Major methodological strengths
- Major methodological concerns (numbered, with specific locations)
- Minor technical comments
- Recommendation

Be thorough but constructive. Cite specific page/line numbers for all comments."""

    REVIEWER_2_IMPACT: str = """You are Reviewer 2, an impact and significance expert. Write a review emphasizing novelty and importance.

Manuscript: {manuscript}
Field Context: {field_context}

Focus your review on:
1. Originality and novelty assessment
2. Scientific importance and potential impact
3. Fit with journal scope and readership
4. Comparison with competing approaches
5. Potential follow-up directions

Structure:
- Summary and overall assessment
- Novelty evaluation (with specific comparisons)
- Impact assessment (field and broader)
- Concerns about significance claims
- Recommendation

Be direct about significance concerns while remaining constructive."""

    REVIEWER_3_CLARITY: str = """You are Reviewer 3, a clarity and communication expert. Write a review emphasizing presentation and accessibility.

Manuscript: {manuscript}
Target Audience: {audience}

Focus your review on:
1. Writing clarity and organization
2. Figure and table effectiveness
3. Accessibility to broad readership
4. Logical flow and narrative
5. Completeness of information

Structure:
- Overall communication assessment
- Major clarity issues (with specific examples)
- Figure/table critique
- Suggestions for improving accessibility
- Recommendation

Provide specific, actionable suggestions for improvement."""

    CROSS_REVIEW_SYNTHESIS: str = """You are a cross-review synthesis specialist. Integrate the three reviewer reports into a coherent synthesis.

Reviewer 1 Report: {review_1}
Reviewer 2 Report: {review_2}
Reviewer 3 Report: {review_3}

Synthesize:
1. Points of consensus (all reviewers agree)
2. Points of disagreement (reviewers differ)
3. Unique insights from each reviewer
4. Overall assessment integration
5. Priority-ordered revision list
6. Final recommendation with confidence level

Identify any contradictions and suggest resolution."""


@dataclass(frozen=True)
class IssueFlaggingPrompts:
    """Prompts for flagging specific issues in manuscripts."""

    UNSUPPORTED_CLAIMS: str = """You are an unsupported claim detection specialist.

Manuscript Text: {text}
Available Evidence: {evidence}

Identify all claims that lack adequate support:
1. Extract each factual claim
2. Map to supporting evidence
3. Flag claims with:
   - No evidence provided
   - Insufficient evidence
   - Contradicted evidence
   - Overgeneralized from evidence

For each flagged claim:
- Quote the claim
- Location (section, paragraph)
- Evidence gap description
- Severity (minor/moderate/major)
- Suggested fix (add evidence/hedge/remove)

Output: Claim-evidence gap report with specific locations."""

    TECHNICAL_FLAWS: str = """You are a technical flaw detection specialist.

Manuscript: {manuscript}
Methods Details: {methods}
Data Analysis: {analysis}

Detect technical flaws:
1. Statistical errors (wrong test, violated assumptions, multiple comparisons)
2. Experimental design flaws (missing controls, confounds, bias)
3. Measurement issues (validity, reliability, precision)
4. Analysis errors (data processing, visualization, interpretation)
5. Reproducibility barriers (missing details, unavailable materials)

For each flaw:
- Description
- Location
- Severity (critical/major/minor)
- Impact on conclusions
- Required correction

Output: Technical flaw report with severity ratings."""

    EVIDENCE_CHAIN_BREAKS: str = """You are an evidence chain analysis specialist.

Manuscript: {manuscript}
Claim Structure: {claims}

Analyze the evidence chain for each major conclusion:
1. Identify the claim
2. Trace supporting evidence
3. Identify weak links:
   - Missing intermediate steps
   - Correlation presented as causation
   - Extrapolation beyond data
   - Alternative explanations not addressed

For each break:
- Claim affected
- Break location and type
- Severity
- Suggested strengthening

Output: Evidence chain map with break points flagged."""

    COMPREHENSION_BARRIERS: str = """You are a comprehension barrier identification specialist.

Manuscript: {manuscript}
Target Reader: {reader_profile}

Identify barriers to understanding:
1. Undefined jargon or abbreviations
2. Missing logical connections
3. Unclear figure legends or labels
4. Assumed background knowledge
5. Complex sentence structures
6. Poor paragraph organization

For each barrier:
- Location
- Type
- Affected reader population
- Suggested clarification

Output: Comprehension barrier report with accessibility improvements."""


@dataclass(frozen=True)
class TechnicalChecklistPrompts:
    """Prompts for 12-axis technical checklist."""

    AXIS_1_RESEARCH_DESIGN: str = """Axis 1: Research Design Assessment

Evaluate:
- Study type appropriateness for question
- Experimental vs observational design
- Control group adequacy
- Randomization and blinding
- Confounding variable control

Rate: Pass / Concern / Fail
Provide specific justification."""

    AXIS_2_SAMPLE_SIZE: str = """Axis 2: Sample Size and Power

Evaluate:
- Power analysis reported
- Sample size justification
- Effect size assumptions
- Attrition handling
- Multiple comparison correction

Rate: Pass / Concern / Fail"""

    AXIS_3_MEASUREMENT: str = """Axis 3: Measurement Validity

Evaluate:
- Construct validity
- Measurement reliability
- Instrument calibration
- Observer bias control
- Measurement error handling

Rate: Pass / Concern / Fail"""

    AXIS_4_STATISTICAL_METHODS: str = """Axis 4: Statistical Methods

Evaluate:
- Test selection appropriateness
- Assumption verification
- Effect size reporting
- Confidence intervals
- P-value interpretation

Rate: Pass / Concern / Fail"""

    AXIS_5_DATA_QUALITY: str = """Axis 5: Data Quality

Evaluate:
- Missing data handling
- Outlier treatment
- Data transformation appropriateness
- Quality control procedures
- Raw data availability

Rate: Pass / Concern / Fail"""

    AXIS_6_REPRODUCIBILITY: str = """Axis 6: Reproducibility

Evaluate:
- Method detail sufficiency
- Code availability
- Data availability
- Material availability
- Protocol registration

Rate: Pass / Concern / Fail"""

    AXIS_7_ETHICS: str = """Axis 7: Ethical Compliance

Evaluate:
- IRB/ethics approval
- Informed consent
- Animal welfare compliance
- Data privacy protection
- Conflict of interest disclosure

Rate: Pass / Concern / Fail"""

    AXIS_8_REPORTING: str = """Axis 8: Reporting Standards

Evaluate:
- Guideline compliance (CONSORT, STROBE, etc.)
- Complete outcome reporting
- Adverse event reporting
- Protocol deviation reporting
- Selective reporting check

Rate: Pass / Concern / Fail"""

    AXIS_9_FIGURES_TABLES: str = """Axis 9: Figures and Tables

Evaluate:
- Data-ink ratio
- Appropriate visualization type
- Error bar representation
- Sample size indication
- Statistical test indication

Rate: Pass / Concern / Fail"""

    AXIS_10_REFERENCES: str = """Axis 10: Reference Quality

Evaluate:
- Currency and relevance
- Primary source preference
- Appropriate citation density
- Self-citation appropriateness
- Reference accuracy

Rate: Pass / Concern / Fail"""

    AXIS_11_INTERPRETATION: str = """Axis 11: Interpretation Appropriateness

Evaluate:
- Conclusion-data alignment
- Causal claim appropriateness
- Generalization limits
- Alternative explanation consideration
- Speculation labeling

Rate: Pass / Concern / Fail"""

    AXIS_12_NOVELTY: str = """Axis 12: Novelty Verification

Evaluate:
- Prior art search adequacy
- Novelty claim accuracy
- Incremental vs transformative
- Methodological innovation
- Conceptual advance

Rate: Pass / Concern / Fail"""


@dataclass(frozen=True)
class ClaimPointerPrompts:
    """Prompts for claim pointer and evidence location binding."""

    CLAIM_EXTRACTION: str = """You are a claim extraction specialist. Extract all verifiable claims from the text.

Text: {text}
Section: {section}

For each claim:
1. Exact claim text
2. Claim type (empirical, theoretical, methodological)
3. Location (section, paragraph, sentence)
4. Required evidence type
5. Verifiability status

Output: Structured claim inventory with pointers."""

    EVIDENCE_MAPPING: str = """You are an evidence mapping specialist. Map each claim to its supporting evidence.

Claims: {claims}
Manuscript Evidence: {evidence}

For each claim:
1. Identify all cited evidence
2. Locate evidence in manuscript (figure, table, text)
3. Assess evidence adequacy
4. Identify evidence gaps
5. Flag unsupported claims

Output: Claim-evidence map with location pointers and adequacy ratings."""

    VERIFICATION_GUIDE: str = """You are a verification guide specialist. Create a guide for verifying each claim.

Claims with Evidence: {mapped_claims}

For each claim, provide:
1. Claim restatement
2. Evidence location (figure/table/section)
3. Verification method
4. Expected result if claim is true
5. Potential falsification indicators

Output: Verification guide for reviewers and editors."""


@dataclass(frozen=True)
class ConsensusPrompts:
    """Prompts for reviewer consensus and repetition checking."""

    REPETITION_CHECK: str = """You are a reviewer report repetition specialist. Check for redundancy across the three reviews.

Review 1: {review_1}
Review 2: {review_2}
Review 3: {review_3}

Analyze:
1. Issues raised by multiple reviewers (consensus concerns)
2. Issues raised by only one reviewer (unique perspectives)
3. Contradictory assessments
4. Coverage gaps (important aspects not addressed)

Classify each issue:
- Consensus (2+ reviewers): High priority
- Single reviewer: Consider for author response
- Contradictory: Requires editor adjudication

Output: Repetition analysis, consensus issue list, unique issue list."""

    CONSENSUS_SYNTHESIS: str = """You are a consensus synthesis specialist. Generate the final consensus review.

Individual Reviews: {reviews}
Repetition Analysis: {repetition}

Synthesize:
1. Summary of consensus concerns (must address)
2. Summary of unique concerns (should consider)
3. Overall assessment integration
4. Recommendation with confidence
5. Priority-ordered revision list
6. Timeline estimate for revisions

Ensure all consensus concerns are clearly actionable."""


# Master prompt dictionary
PROMPTS: Dict[str, Dict[str, str]] = {
    "nature_criteria": {
        "originality": NatureCriteriaPrompts.ORIGINALITY,
        "scientific_importance": NatureCriteriaPrompts.SCIENTIFIC_IMPORTANCE,
        "cross_disciplinary_appeal": NatureCriteriaPrompts.CROSS_DISCIPLINARY_APPEAL,
        "technical_rigor": NatureCriteriaPrompts.TECHNICAL_RIGOR,
        "readability": NatureCriteriaPrompts.READABILITY,
    },
    "reviewer_reports": {
        "reviewer_1_methods": ReviewerReportPrompts.REVIEWER_1_METHODS,
        "reviewer_2_impact": ReviewerReportPrompts.REVIEWER_2_IMPACT,
        "reviewer_3_clarity": ReviewerReportPrompts.REVIEWER_3_CLARITY,
        "cross_review_synthesis": ReviewerReportPrompts.CROSS_REVIEW_SYNTHESIS,
    },
    "issue_flagging": {
        "unsupported_claims": IssueFlaggingPrompts.UNSUPPORTED_CLAIMS,
        "technical_flaws": IssueFlaggingPrompts.TECHNICAL_FLAWS,
        "evidence_chain_breaks": IssueFlaggingPrompts.EVIDENCE_CHAIN_BREAKS,
        "comprehension_barriers": IssueFlaggingPrompts.COMPREHENSION_BARRIERS,
    },
    "technical_checklist": {
        "axis_1_research_design": TechnicalChecklistPrompts.AXIS_1_RESEARCH_DESIGN,
        "axis_2_sample_size": TechnicalChecklistPrompts.AXIS_2_SAMPLE_SIZE,
        "axis_3_measurement": TechnicalChecklistPrompts.AXIS_3_MEASUREMENT,
        "axis_4_statistical_methods": TechnicalChecklistPrompts.AXIS_4_STATISTICAL_METHODS,
        "axis_5_data_quality": TechnicalChecklistPrompts.AXIS_5_DATA_QUALITY,
        "axis_6_reproducibility": TechnicalChecklistPrompts.AXIS_6_REPRODUCIBILITY,
        "axis_7_ethics": TechnicalChecklistPrompts.AXIS_7_ETHICS,
        "axis_8_reporting": TechnicalChecklistPrompts.AXIS_8_REPORTING,
        "axis_9_figures_tables": TechnicalChecklistPrompts.AXIS_9_FIGURES_TABLES,
        "axis_10_references": TechnicalChecklistPrompts.AXIS_10_REFERENCES,
        "axis_11_interpretation": TechnicalChecklistPrompts.AXIS_11_INTERPRETATION,
        "axis_12_novelty": TechnicalChecklistPrompts.AXIS_12_NOVELTY,
    },
    "claim_pointers": {
        "claim_extraction": ClaimPointerPrompts.CLAIM_EXTRACTION,
        "evidence_mapping": ClaimPointerPrompts.EVIDENCE_MAPPING,
        "verification_guide": ClaimPointerPrompts.VERIFICATION_GUIDE,
    },
    "consensus": {
        "repetition_check": ConsensusPrompts.REPETITION_CHECK,
        "consensus_synthesis": ConsensusPrompts.CONSENSUS_SYNTHESIS,
    },
}
