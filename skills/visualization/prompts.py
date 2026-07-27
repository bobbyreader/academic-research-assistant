"""Prompt templates for the Visualization skill.

Contains LLM prompts for chart design, multi-panel layout, uncertainty
representation, accessibility review (WCAG 2.2), and journal export planning.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class VizPrompt:
    """Prompt template for visualization operations."""

    system: str
    user_template: str


# ---------------------------------------------------------------------------
# Chart Design
# ---------------------------------------------------------------------------

CHART_TYPE_SELECTION: VizPrompt = VizPrompt(
    system="""You are a data visualization expert recommending chart types.
Consider data structure, comparison goals, audience, and publication venue.
Follow Cleveland's hierarchy of graphical perception and Tufte's principles.""",
    user_template="""Recommend chart types for this data:

Data Structure: {data_structure}
Variables: {variables}
Comparison Goal: {comparison_goal}
Audience: {audience}
Journal: {journal}

Recommendations:
1. Primary chart type: (with justification)
2. Alternative chart types: (2-3 options)
3. Chart types to avoid: (with reasons)
4. Encoding channels: (position, length, angle, area, color, etc.)
5. Interactive vs. static recommendation""",
)

MULTI_PANEL_LAYOUT: VizPrompt = VizPrompt(
    system="""You are a scientific figure designer specializing in multi-panel
layouts. Design clear, balanced compositions that guide the reader's eye
through a logical narrative. Follow journal guidelines for figure sizing
and panel labeling.""",
    user_template="""Design a multi-panel figure layout:

Number of Panels: {n_panels}
Content per Panel: {panel_contents}
Journal Requirements: {journal_requirements}
Narrative Flow: {narrative_flow}

Layout Design:
1. Grid arrangement: (rows x columns)
2. Panel sizes: (relative proportions)
3. Panel labels: (A, B, C... positioning)
4. Shared elements: (axes, legends, colorbars)
5. White space balance
6. Reading order: (Z-pattern / F-pattern / custom)
7. Caption structure: (brief title + detailed description)""",
)

# ---------------------------------------------------------------------------
# Uncertainty & Missing Data
# ---------------------------------------------------------------------------

UNCERTAINTY_REPRESENTATION: VizPrompt = VizPrompt(
    system="""You are a visualization expert in uncertainty representation.
Choose appropriate visual encodings for different types of uncertainty
(statistical, model, measurement, missing data). Balance informativeness
with clarity to avoid overwhelming the viewer.""",
    user_template="""Design uncertainty representation for:

Data Type: {data_type}
Uncertainty Type: {uncertainty_type}
Sample Size: {sample_size}
Chart Type: {chart_type}

Recommendations:
1. Primary uncertainty encoding: (error bars / bands / violin / gradient / etc.)
2. Confidence level display: (95% CI / SE / SD / prediction interval)
3. Missing data handling: (gap / indicator / imputation note)
4. Sample size encoding: (point size / transparency / annotation)
5. Legend/annotation requirements
6. Common pitfalls to avoid""",
)

MISSING_DATA_DISPLAY: VizPrompt = VizPrompt(
    system="""You are a data visualization specialist in missing data representation.
Design honest, informative displays that distinguish between different types
of missingness (MCAR, MAR, MNAR) and avoid misleading visual continuity.""",
    user_template="""Design missing data display for:

Dataset: {dataset_description}
Missing Pattern: {missing_pattern}
Missing Mechanism: {missing_mechanism}
Analysis Method: {analysis_method}

Display Strategy:
1. Missing data indicators: (color / symbol / gap / annotation)
2. Imputation visualization: (if applicable)
3. Sensitivity display: (complete case vs. imputed)
4. Legend and caption requirements
5. Warning labels: (if MNAR suspected)""",
)

# ---------------------------------------------------------------------------
# Accessibility & Color Review (WCAG 2.2)
# ---------------------------------------------------------------------------

COLOR_ACCESSIBILITY: VizPrompt = VizPrompt(
    system="""You are an accessibility expert reviewing scientific visualizations
against WCAG 2.2 guidelines. Evaluate color contrast, colorblind safety,
and non-color redundancies. Recommend specific color palette improvements.""",
    user_template="""Review color accessibility for this visualization:

Current Colors: {current_colors}
Chart Type: {chart_type}
Data Categories: {n_categories}
Background: {background_color}

WCAG 2.2 Assessment:
1. Contrast ratios: (calculate for each color pair)
2. Colorblind simulation: (protanopia / deuteranopia / tritanopia)
3. Non-color redundancies: (shape / pattern / label / line style)
4. Grayscale readability: (test conversion)
5. Recommended palette: (specific hex codes)
6. Alternative encodings: (if color alone insufficient)""",
)

CONTRAST_REVIEW: VizPrompt = VizPrompt(
    system="""You are a WCAG 2.2 compliance reviewer for scientific figures.
Evaluate text, graphical objects, and UI components against contrast
requirements. Distinguish between normal text, large text, and graphical
objects with different thresholds.""",
    user_template="""Review contrast compliance:

Elements:
{elements_list}

WCAG 2.2 Requirements:
- Normal text: 4.5:1 (AA) / 7:1 (AAA)
- Large text: 3:1 (AA) / 4.5:1 (AAA)
- Graphical objects: 3:1 against adjacent colors

For each element:
1. Current contrast ratio
2. Compliance level: AAA / AA / Fail
3. Recommended adjustment: (if failing)
4. Alternative approach: (if contrast cannot be improved)""",
)

# ---------------------------------------------------------------------------
# Journal Export Planning
# ---------------------------------------------------------------------------

JOURNAL_REQUIREMENTS: VizPrompt = VizPrompt(
    system="""You are a publication specialist familiar with figure requirements
for major scientific journals. Provide specific technical specifications
for figure preparation including dimensions, resolution, file formats,
fonts, and color modes.""",
    user_template="""Provide figure specifications for:

Target Journal: {journal}
Figure Type: {figure_type}
Content: {content_description}

Specifications:
1. Dimensions: (single column / double column / custom)
2. Resolution: (DPI for raster, vector requirements)
3. File format: (TIFF / EPS / PDF / PNG / SVG)
4. Color mode: (RGB / CMYK / grayscale)
5. Font requirements: (family, size, embedding)
6. Line weights: (minimum pt)
7. Panel labeling: (format, size, position)
8. File naming convention
9. Supplementary figure requirements
10. Common rejection reasons to avoid""",
)

EXPORT_CHECKLIST: VizPrompt = VizPrompt(
    system="""You are a pre-submission checklist generator for scientific figures.
Create comprehensive checklists ensuring all technical and aesthetic
requirements are met before journal submission.""",
    user_template="""Generate pre-submission checklist for:

Journal: {journal}
Figure Files: {figure_files}
Manuscript Section: {manuscript_section}

Checklist Categories:
1. Technical specifications
2. Content accuracy
3. Aesthetic quality
4. Accessibility compliance
5. Reproducibility documentation
6. File organization

For each category, list specific check items with pass/fail criteria.""",
)

# ---------------------------------------------------------------------------
# Code Generation
# ---------------------------------------------------------------------------

MATPLOTLIB_CODE: VizPrompt = VizPrompt(
    system="""You are a Matplotlib expert generating publication-quality
visualization code. Follow best practices for figure sizing, DPI, font
management, and export settings. Include comments for reproducibility.""",
    user_template="""Generate Matplotlib code for:

Chart Type: {chart_type}
Data: {data_description}
Style Requirements: {style_requirements}
Export Format: {export_format}

Code Requirements:
1. Figure and axis setup
2. Data plotting with proper encodings
3. Labels, titles, and annotations
4. Legend placement
5. Grid and spine styling
6. Export with correct DPI and format
7. Color palette (accessible)
8. Font settings""",
)

SEABORN_CODE: VizPrompt = VizPrompt(
    system="""You are a Seaborn expert generating statistical visualization
code. Leverage Seaborn's statistical estimation, color palettes, and
multi-plot grids. Ensure publication-quality output.""",
    user_template="""Generate Seaborn code for:

Chart Type: {chart_type}
Data: {data_description}
Statistical Requirements: {statistical_requirements}
Style: {style}

Code Requirements:
1. Data preparation and validation
2. Figure-level vs. axes-level function selection
3. Statistical estimation (CI, regression, etc.)
4. Color palette selection (colorblind-safe)
5. Facet/grid layout (if multi-panel)
6. Customization beyond defaults
7. Export settings""",
)

PLOTLY_CODE: VizPrompt = VizPrompt(
    system="""You are a Plotly expert generating interactive visualization
code. Balance interactivity with static export quality. Ensure figures
work both in notebooks and as static exports for publication.""",
    user_template="""Generate Plotly code for:

Chart Type: {chart_type}
Data: {data_description}
Interactivity Requirements: {interactivity_requirements}
Static Export Needs: {static_export_needs}

Code Requirements:
1. Figure construction (graph_objects vs. express)
2. Interactive features (hover, zoom, pan, select)
3. Layout and styling
4. Static export configuration (Kaleido)
5. Accessibility features
6. Performance optimization (if large data)
7. Embedding options""",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DESIGN_PROMPTS: Dict[str, VizPrompt] = {
    "chart_type": CHART_TYPE_SELECTION,
    "multi_panel": MULTI_PANEL_LAYOUT,
    "uncertainty": UNCERTAINTY_REPRESENTATION,
    "missing_data": MISSING_DATA_DISPLAY,
}

ACCESSIBILITY_PROMPTS: Dict[str, VizPrompt] = {
    "color": COLOR_ACCESSIBILITY,
    "contrast": CONTRAST_REVIEW,
}

EXPORT_PROMPTS: Dict[str, VizPrompt] = {
    "journal_requirements": JOURNAL_REQUIREMENTS,
    "checklist": EXPORT_CHECKLIST,
}

CODE_PROMPTS: Dict[str, VizPrompt] = {
    "matplotlib": MATPLOTLIB_CODE,
    "seaborn": SEABORN_CODE,
    "plotly": PLOTLY_CODE,
}
