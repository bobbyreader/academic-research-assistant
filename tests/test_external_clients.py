from __future__ import annotations

from typing import Any

from core.external_clients import (
    ArxivSource,
    CrossrefSource,
    LiteratureSearcher,
    PaperRecord,
    PubMedSource,
    SemanticScholarSource,
)


class FakeTransport:
    def __init__(self, json_responses: dict[str, Any]) -> None:
        self.json_responses = json_responses

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        del params, headers
        response = self.json_responses[url]
        if isinstance(response, Exception):
            raise response
        return response

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        del url, params, headers
        return ""


class XMLTransport(FakeTransport):
    def __init__(self, json_responses: dict[str, Any], text_responses: dict[str, str]) -> None:
        super().__init__(json_responses)
        self.text_responses = text_responses

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str:
        del params, headers
        return self.text_responses[url]


def test_crossref_source_normalizes_external_metadata() -> None:
    transport = FakeTransport(
        {
            "https://api.crossref.org/works": {
                "message": {
                    "items": [
                        {
                            "title": ["A real paper"],
                            "author": [
                                {"given": "Ada", "family": "Lovelace"},
                            ],
                            "published": {"date-parts": [[2025, 2, 1]]},
                            "container-title": ["Research Journal"],
                            "DOI": "10.1234/example",
                            "abstract": "<jats:p>Evidence summary.</jats:p>",
                            "URL": "https://doi.org/10.1234/example",
                        }
                    ]
                }
            }
        }
    )

    paper = CrossrefSource(transport).search("research topic", 5)[0]

    assert paper.title == "A real paper"
    assert paper.authors == ["Ada Lovelace"]
    assert paper.year == 2025
    assert paper.doi == "10.1234/example"
    assert paper.abstract == "Evidence summary."
    assert paper.source == "crossref"


def test_searcher_deduplicates_by_doi_and_keeps_source_errors() -> None:
    class DuplicateSource:
        name = "duplicate"

        def search(self, query: str, max_results: int) -> list[PaperRecord]:
            del query, max_results
            return [
                PaperRecord(
                    title="Same paper",
                    authors=["A Author"],
                    year=2024,
                    doi="10.1234/same",
                    source="duplicate",
                )
            ]

    class BrokenSource:
        name = "broken"

        def search(self, query: str, max_results: int) -> list[PaperRecord]:
            del query, max_results
            raise RuntimeError("upstream unavailable")

    report = LiteratureSearcher(
        {"duplicate": DuplicateSource(), "broken": BrokenSource()}
    ).search("topic", ["duplicate", "broken"], 5)

    assert len(report.papers) == 1
    assert report.papers[0].doi == "10.1234/same"
    assert report.errors == ["broken: upstream unavailable"]


def test_semantic_scholar_source_reads_external_ids() -> None:
    transport = FakeTransport(
        {
            "https://api.semanticscholar.org/graph/v1/paper/search": {
                "data": [
                    {
                        "paperId": "abc",
                        "title": "Semantic paper",
                        "authors": [{"name": "A Researcher"}],
                        "year": 2023,
                        "venue": "Open Journal",
                        "abstract": "An abstract",
                        "externalIds": {"DOI": "10.5678/semantic"},
                        "url": "https://www.semanticscholar.org/paper/abc",
                        "citationCount": 12,
                    }
                ]
            }
        }
    )

    paper = SemanticScholarSource(transport).search("topic", 3)[0]

    assert paper.doi == "10.5678/semantic"
    assert paper.citation_count == 12
    assert paper.url.endswith("/abc")


def test_pubmed_source_parses_xml_metadata() -> None:
    transport = XMLTransport(
        {
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi": {
                "esearchresult": {"idlist": ["123"]}
            }
        },
        {
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi": """
            <PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID>
              <Article><ArticleTitle>PubMed paper</ArticleTitle>
                <Journal><Title>Medical Journal</Title><JournalIssue><PubDate><Year>2022</Year></PubDate></JournalIssue></Journal>
                <AuthorList><Author><ForeName>Grace</ForeName><LastName>Hopper</LastName></Author></AuthorList>
                <Abstract><AbstractText>Clinical evidence.</AbstractText></Abstract>
              </Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1111/pubmed</ArticleId></ArticleIdList></PubmedData>
            </PubmedArticle></PubmedArticleSet>
            """,
        },
    )

    paper = PubMedSource(transport).search("topic", 1)[0]

    assert paper.title == "PubMed paper"
    assert paper.year == 2022
    assert paper.doi == "10.1111/pubmed"
    assert paper.url.endswith("/123/")


def test_arxiv_source_parses_atom_feed() -> None:
    transport = XMLTransport(
        {},
        {
            "https://export.arxiv.org/api/query": """
            <feed xmlns="http://www.w3.org/2005/Atom"><entry>
              <id>http://arxiv.org/abs/1234.5678</id><title>Arxiv paper</title>
              <summary>Summary text.</summary><published>2021-01-01T00:00:00Z</published>
              <author><name>Alan Turing</name></author>
            </entry></feed>
            """,
        },
    )

    paper = ArxivSource(transport).search("topic", 1)[0]

    assert paper.title == "Arxiv paper"
    assert paper.year == 2021
    assert paper.external_id.endswith("1234.5678")
