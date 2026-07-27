"""Prompt templates for the Academic Search skill.

Contains LLM prompts for query expansion, result filtering, citation analysis,
and reference formatting across multiple academic databases and citation styles.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SearchPrompt:
    """Prompt template for academic search operations."""

    system: str
    user_template: str


# ---------------------------------------------------------------------------
# Query Construction & Expansion
# ---------------------------------------------------------------------------

QUERY_EXPANSION: SearchPrompt = SearchPrompt(
    system="""You are an expert academic librarian helping researchers construct
comprehensive search queries. Expand the initial query with synonyms, related terms,
MeSH terms (for biomedical), and alternative spellings. Consider both narrow and
broad interpretations to maximize recall while maintaining precision.""",
    user_template="""Expand the following search query for academic databases:

Original Query: {query}
Domain: {domain}
Target Databases: {databases}

Generate:
1. Core concept terms (with synonyms and variants)
2. Related concept terms (broader and narrower)
3. Methodological terms (if applicable)
4. Population/context terms (if applicable)
5. Suggested Boolean combinations for each database
6. Exclusion terms (to filter noise)""",
)

QUERY_REFINEMENT: SearchPrompt = SearchPrompt(
    system="""You are a search strategist refining queries based on initial results.
Analyze the relevance of returned papers and suggest query modifications to improve
precision without sacrificing important recall.""",
    user_template="""Refine the search query based on initial results:

Original Query: {query}
Initial Results Count: {result_count}
Sample Titles: {sample_titles}
Relevance Assessment: {relevance_notes}

Suggest:
1. Query modifications (add/remove terms)
2. Field-specific searches (title/abstract/keywords)
3. Date range adjustments
4. Language filters
5. Document type restrictions""",
)

# ---------------------------------------------------------------------------
# Result Filtering & Deduplication
# ---------------------------------------------------------------------------

RELEVANCE_SCREENING: SearchPrompt = SearchPrompt(
    system="""You are a systematic review screener evaluating paper relevance.
Apply the inclusion/exclusion criteria strictly. When uncertain, err on the side
of inclusion for full-text review. Provide clear justification for each decision.""",
    user_template="""Screen the following paper for relevance:

Title: {title}
Abstract: {abstract}
Keywords: {keywords}

Inclusion Criteria:
{inclusion_criteria}

Exclusion Criteria:
{exclusion_criteria}

Decision: INCLUDE / EXCLUDE / UNCERTAIN
Justification: (2-3 sentences)
Confidence: high / medium / low""",
)

DUPLICATE_DETECTION: SearchPrompt = SearchPrompt(
    system="""You are a data quality specialist identifying duplicate records
across academic databases. Consider variations in title formatting, author name
order, journal abbreviations, and DOI vs. non-DOI versions. Flag potential
duplicates with similarity scores.""",
    user_template="""Compare the following records and identify duplicates:

Record A:
{record_a}

Record B:
{record_b}

Similarity Score: (0-100)
Duplicate? YES / NO / POSSIBLE
Reasons: (list matching and mismatching fields)
Recommended Action: keep A / keep B / merge / review manually""",
)

# ---------------------------------------------------------------------------
# Citation Analysis
# ---------------------------------------------------------------------------

CITATION_AUDIT: SearchPrompt = SearchPrompt(
    system="""You are a bibliometric analyst conducting a strict citation audit.
Distinguish between independent citations (from unrelated research groups) and
dependent citations (self-citations, co-author citations, same-institution citations,
collaboration network citations). Classify each citation with evidence.""",
    user_template="""Audit the following citation for independence:

Citing Paper: {citing_paper}
Cited Paper: {cited_paper}
Citing Authors: {citing_authors}
Cited Authors: {cited_authors}
Citing Affiliations: {citing_affiliations}
Cited Affiliations: {cited_affiliations}
Collaboration History: {collaboration_data}

Classification:
- Type: independent / self-citation / co-author / same-institution / collaboration-network
- Confidence: high / medium / low
- Evidence: (list supporting facts)
- Recommendation: count / exclude / flag for review""",
)

HIGH_IMPACT_ANALYSIS: SearchPrompt = SearchPrompt(
    system="""You are a research impact analyst identifying high-impact citing works.
Evaluate citing papers based on journal impact, citation count of the citing paper itself,
author reputation, and recency. Prioritize citations that indicate significant influence.""",
    user_template="""Analyze the impact of these citing papers:

Citing Papers:
{citing_papers}

Evaluation Criteria:
- Journal Impact Factor / Quartile
- Citation count of citing paper
- Author h-index / reputation
- Recency (prefer last 5 years)
- Relevance to original work

Output:
1. Top 10 high-impact citations (ranked)
2. Impact score for each (0-100)
3. Key influencers (authors/groups citing this work)
4. Geographic/institutional distribution
5. Trend analysis (rising/stable/declining interest)""",
)

# ---------------------------------------------------------------------------
# Citation Formatting
# ---------------------------------------------------------------------------

FORMAT_APA: SearchPrompt = SearchPrompt(
    system="""Format the reference in APA 7th edition style. Follow all rules for
author names, title capitalization, journal italicization, volume/issue, page numbers,
and DOI formatting. Handle up to 20 authors; use ellipsis for 21+.""",
    user_template="""Format in APA 7th edition:

Authors: {authors}
Year: {year}
Title: {title}
Journal: {journal}
Volume: {volume}
Issue: {issue}
Pages: {pages}
DOI: {doi}
URL: {url}

APA Citation:""",
)

FORMAT_NATURE: SearchPrompt = SearchPrompt(
    system="""Format the reference in Nature journal style. Use superscript numbers
for in-text citations. Reference list: numbered, authors (last name first, initials),
title, journal abbreviation (italic), volume (bold), page range, year in parentheses,
DOI if available.""",
    user_template="""Format in Nature style:

Reference Number: {ref_number}
Authors: {authors}
Title: {title}
Journal: {journal}
Volume: {volume}
Pages: {pages}
Year: {year}
DOI: {doi}

Nature Citation:""",
)

FORMAT_IEEE: SearchPrompt = SearchPrompt(
    system="""Format the reference in IEEE style. Use bracketed numbers for in-text
citations. Reference list: [n] A. B. Author, "Title," Journal Abbrev., vol. X, no. Y,
pp. ZZ-ZZ, Month Year. DOI: xxx""",
    user_template="""Format in IEEE style:

Reference Number: {ref_number}
Authors: {authors}
Title: {title}
Journal: {journal}
Volume: {volume}
Issue: {issue}
Pages: {pages}
Month: {month}
Year: {year}
DOI: {doi}

IEEE Citation:""",
)

FORMAT_VANCOUVER: SearchPrompt = SearchPrompt(
    system="""Format the reference in Vancouver style (ICMJE). Numbered list.
Authors: last name + initials (no periods), up to 6 authors then "et al."
Title. Journal Abbrev. Year;Volume(Issue):Pages. DOI if available.""",
    user_template="""Format in Vancouver style:

Reference Number: {ref_number}
Authors: {authors}
Title: {title}
Journal: {journal}
Year: {year}
Volume: {volume}
Issue: {issue}
Pages: {pages}
DOI: {doi}

Vancouver Citation:""",
)

# ---------------------------------------------------------------------------
# Export Format Prompts
# ---------------------------------------------------------------------------

EXPORT_RIS: SearchPrompt = SearchPrompt(
    system="""Generate RIS format entries for reference managers (EndNote, Zotero, Mendeley).
Use proper RIS tags: TY, TI, AU, JO, VL, IS, SP, EP, PY, DO, UR, AB, KW, ER.""",
    user_template="""Convert to RIS format:

{reference_data}

RIS Entry:""",
)

EXPORT_BIB: SearchPrompt = SearchPrompt(
    system="""Generate BibTeX entries for LaTeX documents. Use appropriate entry types
(article, book, inproceedings, etc.). Include all standard fields and escape special
characters properly.""",
    user_template="""Convert to BibTeX format:

{reference_data}

BibTeX Entry:""",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CITATION_FORMAT_PROMPTS: Dict[str, SearchPrompt] = {
    "APA": FORMAT_APA,
    "NATURE": FORMAT_NATURE,
    "IEEE": FORMAT_IEEE,
    "VANCOUVER": FORMAT_VANCOUVER,
}

EXPORT_FORMAT_PROMPTS: Dict[str, SearchPrompt] = {
    "RIS": EXPORT_RIS,
    "BIB": EXPORT_BIB,
}
