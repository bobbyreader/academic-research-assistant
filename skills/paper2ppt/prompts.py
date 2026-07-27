"""Prompt templates for Paper to PPT skill."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ContentExtractionPrompts:
    """Prompts for extracting key content from papers."""

    RESEARCH_QUESTION: str = """You are a research question extraction specialist. Identify and articulate the core research question.

Paper Content: {content}
Paper Type: {paper_type}

Extract:
1. Primary research question (one sentence)
2. Secondary questions (if any)
3. Hypotheses being tested
4. Why this question matters (significance)

Format for PPT slide:
- Clear, concise question statement
- Context in 1-2 bullet points
- Significance indicator"""

    KEY_CLAIMS: str = """You are a key claims extraction specialist. Identify the paper's main claims and findings.

Paper Content: {content}
Results Section: {results}

Extract:
1. Primary claim (main finding)
2. Supporting claims (2-4 secondary findings)
3. Novel contributions
4. Unexpected results

For each claim:
- One-sentence statement
- Evidence strength indicator
- Figure/table reference
- Implication note

Prioritize claims by importance and evidence strength."""

    CORE_EVIDENCE: str = """You are a core evidence extraction specialist. Identify the most compelling evidence.

Paper Content: {content}
Figures and Tables: {figures_tables}

Extract:
1. Key quantitative results (with effect sizes)
2. Critical qualitative observations
3. Comparison results (vs baseline/prior work)
4. Validation evidence

For each evidence item:
- Type (quantitative/qualitative/comparative)
- Strength assessment
- Best visualization reference
- Presentation priority"""

    LIMITATIONS: str = """You are a limitations extraction specialist. Identify and frame study limitations.

Paper Content: {content}
Discussion Section: {discussion}

Extract:
1. Methodological limitations
2. Generalizability constraints
3. Data limitations
4. Interpretation caveats
5. Future work directions

Frame constructively for presentation:
- Acknowledge without undermining
- Connect to future opportunities
- Maintain credibility"""

    REUSABLE_VALUE: str = """You are a reusable value assessment specialist. Identify what others can reuse from this work.

Paper Content: {content}
Methods Section: {methods}

Extract:
1. Reusable methods/protocols
2. Available datasets
3. Code/tools released
4. Theoretical frameworks
5. Benchmarking standards

For each reusable element:
- What it is
- How to access it
- Potential applications
- Citation requirements"""


@dataclass(frozen=True)
class SlideDesignPrompts:
    """Prompts for PPT slide design and content organization."""

    SLIDE_STRUCTURE: str = """You are a scientific presentation specialist. Design the slide structure for this paper.

Paper Summary: {summary}
Target Slide Count: {slide_count}
Presentation Context: {context}

Design 10-16 slide structure:
1. Title slide (title, authors, affiliation, date)
2. Background/motivation (1-2 slides)
3. Research question (1 slide)
4. Approach/methods (1-2 slides)
5. Key results (3-5 slides, one per major finding)
6. Comparison/validation (1 slide)
7. Limitations (1 slide)
8. Implications/applications (1 slide)
9. Summary/conclusions (1 slide)
10. Acknowledgments/funding (optional)

For each slide:
- Title
- Key message (one sentence)
- Content type (text/figure/table/mixed)
- Speaker notes focus"""

    FIGURE_SELECTION: str = """You are a figure selection specialist for presentations. Choose the most impactful figures.

Available Figures: {figures}
Key Messages: {messages}
Slide Constraints: {constraints}

Select figures that:
1. Support key messages directly
2. Are readable at presentation scale
3. Tell a clear story
4. Balance quantitative and qualitative

For each selected figure:
- Slide assignment
- Cropping/zoom recommendations
- Annotation suggestions
- Alternative if original is too dense"""

    DENSE_FIGURE_SPLIT: str = """You are a figure optimization specialist. Split dense figures for presentation clarity.

Original Figure: {figure_description}
Information Density: {density}
Target Slide: {slide}

Split strategy:
1. Identify logical sub-components
2. Determine split orientation (horizontal/vertical)
3. Preserve critical context
4. Plan annotation overlays
5. Ensure readability at presentation scale

Output:
- Split plan (how many slides, what each shows)
- Cropping coordinates/description
- Annotation guide
- Transition narrative between slides"""


@dataclass(frozen=True)
class SpeakerNotesPrompts:
    """Prompts for generating speaker notes."""

    COMPREHENSIVE_NOTES: str = """You are a speaker notes specialist. Generate comprehensive speaker notes for each slide.

Slide Content: {slide_content}
Slide Purpose: {purpose}
Presentation Context: {context}
Time Allocation: {time}

Generate notes covering:
1. Opening statement (grab attention)
2. Key points to emphasize
3. Transition to next slide
4. Anticipated questions
5. Timing guidance
6. Delivery tips (pace, emphasis)

Tone: Professional but conversational
Length: 100-200 words per slide"""

    Q_A_PREPARATION: str = """You are a Q&A preparation specialist. Anticipate and prepare for questions.

Paper Content: {content}
Likely Audience: {audience}
Controversial Aspects: {controversial}

Prepare for:
1. Methodology questions
2. Interpretation challenges
3. Comparison requests
4. Application inquiries
5. Limitation probes

For each anticipated question:
- Likely question phrasing
- Key points for response
- Supporting evidence to reference
- Potential follow-up questions"""


@dataclass(frozen=True)
class QAReportPrompts:
    """Prompts for quality assurance reporting."""

    CONTENT_ACCURACY: str = """You are a content accuracy QA specialist. Verify presentation accuracy against source paper.

Presentation Content: {presentation}
Source Paper: {paper}

Check:
1. Data accuracy (numbers, statistics)
2. Claim accuracy (no overstatement)
3. Figure accuracy (correct labeling, no distortion)
4. Citation accuracy
5. Context preservation

Flag any discrepancies with severity ratings."""

    PRESENTATION_QUALITY: str = """You are a presentation quality QA specialist. Assess presentation effectiveness.

Presentation: {presentation}
Best Practices: {practices}

Evaluate:
1. Slide readability (font size, contrast, density)
2. Logical flow
3. Time allocation appropriateness
4. Figure quality and relevance
5. Speaker notes completeness
6. Overall narrative coherence

Provide improvement suggestions with priority ratings."""

    ACCESSIBILITY_CHECK: str = """You are an accessibility QA specialist. Ensure presentation accessibility.

Presentation: {presentation}
Accessibility Standards: {standards}

Check:
1. Color contrast ratios
2. Font sizes and readability
3. Alt text for figures
4. Colorblind-friendly palettes
5. Text alternatives for visual content
6. Reading order logic

Provide accessibility compliance report and fixes."""


@dataclass(frozen=True)
class ScenarioPrompts:
    """Prompts for different presentation scenarios."""

    GROUP_MEETING: str = """You are a group meeting presentation specialist. Adapt the presentation for a lab/group meeting.

Paper: {paper}
Audience: {audience}
Time Limit: {time_limit}

Adaptations:
1. Emphasize methodology details
2. Include more technical depth
3. Focus on implications for ongoing work
4. Encourage discussion points
5. Connect to group research themes

Adjust slide content and speaker notes accordingly."""

    LITERATURE_REVIEW: str = """You are a literature review presentation specialist. Adapt for journal club/literature review.

Paper: {paper}
Related Literature: {literature}
Discussion Goals: {goals}

Adaptations:
1. Emphasize context within literature
2. Highlight methodological contributions
3. Prepare critical evaluation points
4. Connect to related papers
5. Frame discussion questions

Adjust for critical analysis and discussion facilitation."""

    THESIS_DEFENSE: str = """You are a thesis defense presentation specialist. Adapt for defense context.

Paper: {paper}
Thesis Context: {thesis}
Committee: {committee}

Adaptations:
1. Emphasize original contributions
2. Prepare for methodological scrutiny
3. Connect to thesis narrative
4. Anticipate committee expertise areas
5. Highlight future directions

Adjust for formal defense format and rigorous questioning."""

    CONFERENCE_PRESENTATION: str = """You are a conference presentation specialist. Adapt for conference talk.

Paper: {paper}
Conference: {conference}
Session Type: {session_type}
Time Limit: {time_limit}

Adaptations:
1. Maximize visual impact
2. Minimize text density
3. Emphasize broad significance
4. Prepare for diverse audience
5. Include memorable takeaways

Adjust for professional conference standards and networking goals."""


# Master prompt dictionary
PROMPTS: Dict[str, Dict[str, str]] = {
    "content_extraction": {
        "research_question": ContentExtractionPrompts.RESEARCH_QUESTION,
        "key_claims": ContentExtractionPrompts.KEY_CLAIMS,
        "core_evidence": ContentExtractionPrompts.CORE_EVIDENCE,
        "limitations": ContentExtractionPrompts.LIMITATIONS,
        "reusable_value": ContentExtractionPrompts.REUSABLE_VALUE,
    },
    "slide_design": {
        "slide_structure": SlideDesignPrompts.SLIDE_STRUCTURE,
        "figure_selection": SlideDesignPrompts.FIGURE_SELECTION,
        "dense_figure_split": SlideDesignPrompts.DENSE_FIGURE_SPLIT,
    },
    "speaker_notes": {
        "comprehensive_notes": SpeakerNotesPrompts.COMPREHENSIVE_NOTES,
        "q_a_preparation": SpeakerNotesPrompts.Q_A_PREPARATION,
    },
    "qa_report": {
        "content_accuracy": QAReportPrompts.CONTENT_ACCURACY,
        "presentation_quality": QAReportPrompts.PRESENTATION_QUALITY,
        "accessibility_check": QAReportPrompts.ACCESSIBILITY_CHECK,
    },
    "scenarios": {
        "group_meeting": ScenarioPrompts.GROUP_MEETING,
        "literature_review": ScenarioPrompts.LITERATURE_REVIEW,
        "thesis_defense": ScenarioPrompts.THESIS_DEFENSE,
        "conference_presentation": ScenarioPrompts.CONFERENCE_PRESENTATION,
    },
}
