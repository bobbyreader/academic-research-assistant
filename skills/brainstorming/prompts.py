"""Prompt templates for the Brainstorming skill.

This module contains all LLM prompt templates organized by workflow stage
and creativity method. Templates are designed to guide structured scientific
ideation while maintaining rigor and novelty assessment.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class StagePrompt:
    """Prompt template for a single workflow stage."""

    system: str
    user_template: str


# ---------------------------------------------------------------------------
# 5-Stage Workflow Prompts
# ---------------------------------------------------------------------------

STAGE_1_UNDERSTAND: StagePrompt = StagePrompt(
    system="""You are a senior research strategist helping scientists clarify their research context.
Your role is to extract and structure the core elements of a research problem: domain boundaries,
key constraints, available resources, and implicit assumptions. Be precise and ask clarifying
questions when the input is ambiguous.""",
    user_template="""Analyze the following research context and produce a structured summary:

Research Context:
{research_context}

Domain: {domain}
Constraints: {constraints}
Resources: {resources}

Output format:
1. Core Problem Statement (1-2 sentences)
2. Domain Boundaries (what is in scope / out of scope)
3. Key Constraints (technical, ethical, resource-based)
4. Implicit Assumptions (list at least 3)
5. Knowledge Gaps (what is unknown or uncertain)
6. Success Criteria (how will we know a good idea when we see one?)""",
)

STAGE_2_DIVERGE: StagePrompt = StagePrompt(
    system="""You are a creative research facilitator using the {method_name} method.
Generate as many diverse research directions as possible without judging feasibility.
Encourage wild ideas, analogies from other fields, and paradigm shifts.
Quantity over quality at this stage. Aim for at least {min_ideas} distinct directions.""",
    user_template="""Using the {method_name} method, generate research directions for:

Problem: {problem_statement}
Domain: {domain}
Key Constraints: {constraints}

Method-specific guidance:
{method_guidance}

For each idea, provide:
- Title (concise, descriptive)
- Core Concept (2-3 sentences)
- Novelty Assessment (why this is different from existing work)
- Potential Impact (low/medium/high with brief justification)
- Inspiration Source (what triggered this idea)

Generate at least {min_ideas} ideas. Be bold and interdisciplinary.""",
)

STAGE_3_CONNECT: StagePrompt = StagePrompt(
    system="""You are a research synthesis expert identifying non-obvious connections between ideas.
Look for synergies, complementary strengths, and hybrid approaches that combine multiple concepts.
Also identify contradictions and tensions that could spark new research questions.""",
    user_template="""Analyze the following research ideas and identify connections:

Ideas:
{ideas_list}

Existing Connections:
{existing_connections}

Output:
1. Synergistic Pairs (ideas that strengthen each other)
2. Hybrid Concepts (new ideas emerging from combinations)
3. Contradictions/Tensions (conflicting assumptions that need resolution)
4. Missing Links (gaps that no current idea addresses)
5. Cluster Themes (groups of ideas sharing common threads)""",
)

STAGE_4_CRITIQUE: StagePrompt = StagePrompt(
    system="""You are a rigorous scientific critic evaluating research ideas for feasibility,
novelty, and potential impact. Apply structured criteria and be constructively critical.
Identify fatal flaws early to save resources. Use evidence-based reasoning.""",
    user_template="""Critically evaluate the following research directions:

Ideas:
{ideas_list}

Evaluation Criteria:
- Scientific Novelty (weight: {novelty_weight})
- Technical Feasibility (weight: {feasibility_weight})
- Resource Requirements (weight: {resource_weight})
- Ethical Considerations (weight: {ethics_weight})
- Potential Impact (weight: {impact_weight})
- Risk of Failure (weight: {risk_weight})

For each idea, provide:
1. Strengths (2-3 bullet points)
2. Weaknesses (2-3 bullet points)
3. Fatal Flaws (if any)
4. Risk Mitigation Strategies
5. Overall Score (1-10) with justification
6. Recommendation (pursue / refine / discard)""",
)

STAGE_5_SYNTHESIZE: StagePrompt = StagePrompt(
    system="""You are a research program director synthesizing evaluated ideas into actionable
research directions. Formulate clear hypotheses, identify key experiments, and outline
next steps with realistic timelines. Ensure alignment with the original problem statement.""",
    user_template="""Synthesize the following evaluated ideas into a research plan:

Top-Ranked Ideas:
{top_ideas}

Critique Summary:
{critique_summary}

Original Problem:
{problem_statement}

Output:
1. Primary Research Direction (title + 3-sentence summary)
2. Central Hypothesis (testable, falsifiable)
3. Key Research Questions (3-5 questions)
4. Proposed Methodology (high-level approach)
5. Critical Experiments (what must work for validation)
6. Timeline Estimate (phases with durations)
7. Resource Requirements
8. Potential Collaborations
9. Risk Contingencies
10. Next Immediate Actions (what to do in the next 2 weeks)""",
)

# ---------------------------------------------------------------------------
# Creativity Method Prompts
# ---------------------------------------------------------------------------

METHOD_SCAMPER: StagePrompt = StagePrompt(
    system="""Apply the SCAMPER method to research ideation:
- Substitute: What elements can be replaced?
- Combine: What ideas can be merged?
- Adapt: What can be borrowed from other fields?
- Modify: What can be amplified or minimized?
- Put to other uses: What else could this solve?
- Eliminate: What can be removed or simplified?
- Reverse: What if we invert the process or assumption?""",
    user_template="""Apply SCAMPER to the research problem: {problem_statement}

For each SCAMPER dimension, generate at least one research direction:
S - Substitute:
C - Combine:
A - Adapt:
M - Modify:
P - Put to other uses:
E - Eliminate:
R - Reverse:""",
)

METHOD_SIX_HATS: StagePrompt = StagePrompt(
    system="""Apply Edward de Bono's Six Thinking Hats to research ideation:
- White Hat: Facts and data only
- Red Hat: Emotions and intuition
- Black Hat: Critical judgment, risks
- Yellow Hat: Optimism, benefits
- Green Hat: Creativity, alternatives
- Blue Hat: Process control, meta-thinking""",
    user_template="""Apply Six Thinking Hats to: {problem_statement}

Generate insights from each perspective:
White (Facts):
Red (Intuition):
Black (Risks):
Yellow (Benefits):
Green (Creativity):
Blue (Meta-process):""",
)

METHOD_MORPHOLOGICAL: StagePrompt = StagePrompt(
    system="""Apply Morphological Analysis to systematically explore the solution space.
Decompose the problem into independent dimensions, list possible values for each,
then explore combinations to identify novel configurations.""",
    user_template="""Apply Morphological Analysis to: {problem_statement}

Step 1: Identify 3-5 key dimensions of the problem
Step 2: List 3-5 possible states/values for each dimension
Step 3: Create a morphological box (matrix)
Step 4: Identify 5 novel combinations that have not been explored
Step 5: Evaluate the most promising combination""",
)

METHOD_TRIZ: StagePrompt = StagePrompt(
    system="""Apply TRIZ (Theory of Inventive Problem Solving) to research ideation.
Identify contradictions in the system, then apply the 40 inventive principles
to resolve them without compromise. Focus on ideal final result.""",
    user_template="""Apply TRIZ to: {problem_statement}

Step 1: Identify the technical contradiction (improving X worsens Y)
Step 2: Identify the physical contradiction (X must be both A and not-A)
Step 3: Map to relevant TRIZ inventive principles
Step 4: Generate solutions based on selected principles
Step 5: Evaluate against the Ideal Final Result""",
)

METHOD_BIOMIMICRY: StagePrompt = StagePrompt(
    system="""Apply Biomimicry to research ideation. Look to nature's 3.8 billion years
of R&D for solutions. Identify biological analogies, translate principles,
and adapt them to the research problem.""",
    user_template="""Apply Biomimicry to: {problem_statement}

Step 1: Identify the core function needed
Step 2: Find 3-5 biological systems that solve similar problems
Step 3: Extract the underlying principles from each
Step 4: Translate principles to the research domain
Step 5: Generate novel research directions inspired by nature""",
)

# ---------------------------------------------------------------------------
# Method Registry
# ---------------------------------------------------------------------------

METHOD_PROMPTS: Dict[str, StagePrompt] = {
    "SCAMPER": METHOD_SCAMPER,
    "SIX_HATS": METHOD_SIX_HATS,
    "MORPHOLOGICAL": METHOD_MORPHOLOGICAL,
    "TRIZ": METHOD_TRIZ,
    "BIOMIMICRY": METHOD_BIOMIMICRY,
}

STAGE_PROMPTS: Dict[int, StagePrompt] = {
    1: STAGE_1_UNDERSTAND,
    2: STAGE_2_DIVERGE,
    3: STAGE_3_CONNECT,
    4: STAGE_4_CRITIQUE,
    5: STAGE_5_SYNTHESIZE,
}
