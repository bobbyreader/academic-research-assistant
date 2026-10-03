"""Real literature API adapters and a resilient multi-source searcher."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from core.http_client import HttpClientError, UrllibTransport
from core.research_models import PaperRecord, SearchReport


class Transport(Protocol):
    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any: ...

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> str: ...


class LiteratureSource(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[PaperRecord]: ...


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _first(values: Any) -> str:
    if isinstance(values, list) and values:
        return _text(values[0])
    return ""


def _year(value: Any) -> int | None:
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0][0] if value[0] else None
    try:
        parsed = int(value)
        return parsed if 1000 <= parsed <= datetime.now(UTC).year + 2 else None
    except (TypeError, ValueError):
        return None


def _strip_markup(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value).strip()


def _normalize_doi(value: Any) -> str:
    doi = _text(value)
    doi = re.sub(r"^https?://doi.org/", "", doi, flags=re.IGNORECASE)
    return doi.removesuffix(".").strip()


def _authors(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for author in value:
        if isinstance(author, dict):
            name = " ".join(
                part for part in (_text(author.get("given")), _text(author.get("family"))) if part
            )
            if not name:
                name = _text(author.get("name"))
        else:
            name = _text(author)
        if name:
            result.append(name)
    return result


class CrossrefSource:
    name = "crossref"

    def __init__(self, transport: Transport | None = None, email: str = "") -> None:
        self.transport = transport or UrllibTransport()
        self.email = email

    def search(self, query: str, max_results: int) -> list[PaperRecord]:
        params: dict[str, Any] = {
            "query.bibliographic": query,
            "rows": max_results,
            "select": "DOI,title,author,published,container-title,abstract,URL",
        }
        if self.email:
            params["mailto"] = self.email
        payload = self.transport.get_json("https://api.crossref.org/works", params=params)
        items = payload.get("message", {}).get("items", [])
        if not isinstance(items, list):
            raise HttpClientError("Crossref 返回的 items 不是列表")

        results = []
        for item in items:
            if not isinstance(item, dict) or not _first(item.get("title")):
                continue
            results.append(
                PaperRecord(
                    title=_first(item.get("title")),
                    authors=_authors(item.get("author")),
                    year=_year(item.get("published", {}).get("date-parts")),
                    journal=_first(item.get("container-title")),
                    doi=_normalize_doi(item.get("DOI")),
                    abstract=_strip_markup(_text(item.get("abstract"))),
                    url=_text(item.get("URL")),
                    source=self.name,
                    raw_data=item,
                )
            )
        return results


class PubMedSource:
    name = "pubmed"
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, transport: Transport | None = None, email: str = "") -> None:
        self.transport = transport or UrllibTransport()
        self.email = email

    def search(self, query: str, max_results: int) -> list[PaperRecord]:
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
        }
        if self.email:
            params["email"] = self.email
        search_payload = self.transport.get_json(self.search_url, params=params)
        ids = search_payload.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        fetch_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
        if self.email:
            fetch_params["email"] = self.email
        xml = self.transport.get_text(self.fetch_url, params=fetch_params)
        return self._parse_xml(xml)

    def _parse_xml(self, xml: str) -> list[PaperRecord]:
        root = ET.fromstring(xml)
        results = []
        for article in root.findall(".//PubmedArticle"):
            citation = article.find("MedlineCitation")
            article_node = citation.find("Article") if citation is not None else None
            if article_node is None:
                continue
            title = "".join(article_node.findtext("ArticleTitle", default="")).strip()
            journal = article_node.findtext("Journal/Title", default="").strip()
            abstract = " ".join(
                "".join(node.itertext()).strip()
                for node in article_node.findall("Abstract/AbstractText")
            )
            authors = []
            for author in article_node.findall("AuthorList/Author"):
                name = " ".join(
                    part
                    for part in (
                        author.findtext("ForeName", default=""),
                        author.findtext("LastName", default=""),
                    )
                    if part
                ).strip() or author.findtext("CollectiveName", default="").strip()
                if name:
                    authors.append(name)
            year_text = (
                article_node.findtext("Journal/JournalIssue/PubDate/Year")
                or article_node.findtext("Journal/JournalIssue/PubDate/MedlineDate", default="")[:4]
            )
            doi = ""
            for identifier in article.findall(".//ArticleId"):
                if identifier.attrib.get("IdType") == "doi":
                    doi = _normalize_doi(identifier.text)
                    break
            pmid = citation.findtext("PMID", default="") if citation is not None else ""
            if title:
                results.append(
                    PaperRecord(
                        title=title,
                        authors=authors,
                        year=_year(year_text),
                        journal=journal,
                        doi=doi,
                        abstract=abstract,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                        source=self.name,
                        external_id=pmid,
                    )
                )
        return results


class SemanticScholarSource:
    name = "semantic_scholar"
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, transport: Transport | None = None, api_key: str = "") -> None:
        self.transport = transport or UrllibTransport()
        self.api_key = api_key

    def search(self, query: str, max_results: int) -> list[PaperRecord]:
        fields = "title,authors,year,venue,abstract,externalIds,url,citationCount"
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        payload = self.transport.get_json(
            self.url,
            params={"query": query, "limit": max_results, "fields": fields},
            headers=headers,
        )
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise HttpClientError("Semantic Scholar 返回的 data 不是列表")
        results = []
        for item in data:
            if not isinstance(item, dict) or not _text(item.get("title")):
                continue
            external_ids = item.get("externalIds") or {}
            results.append(
                PaperRecord(
                    title=_text(item.get("title")),
                    authors=_authors(item.get("authors")),
                    year=_year(item.get("year")),
                    journal=_text(item.get("venue")),
                    doi=_normalize_doi(external_ids.get("DOI")),
                    abstract=_text(item.get("abstract")),
                    url=_text(item.get("url")),
                    source=self.name,
                    citation_count=item.get("citationCount"),
                    external_id=_text(item.get("paperId")),
                    raw_data=item,
                )
            )
        return results


class ArxivSource:
    name = "arxiv"
    url = "https://export.arxiv.org/api/query"

    def __init__(self, transport: Transport | None = None) -> None:
        self.transport = transport or UrllibTransport()

    def search(self, query: str, max_results: int) -> list[PaperRecord]:
        xml = self.transport.get_text(
            self.url,
            params={"search_query": f"all:{query}", "max_results": max_results},
        )
        root = ET.fromstring(xml)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results = []
        for entry in root.findall("atom:entry", ns):
            title = " ".join(entry.findtext("atom:title", default="", namespaces=ns).split())
            if not title:
                continue
            published = entry.findtext("atom:published", default="", namespaces=ns)
            results.append(
                PaperRecord(
                    title=title,
                    authors=[
                        name.text.strip()
                        for name in entry.findall("atom:author/atom:name", ns)
                        if name.text
                    ],
                    year=_year(published[:4]),
                    abstract=" ".join(
                        entry.findtext("atom:summary", default="", namespaces=ns).split()
                    ),
                    url=entry.findtext("atom:id", default="", namespaces=ns),
                    source=self.name,
                    external_id=entry.findtext("atom:id", default="", namespaces=ns),
                )
            )
        return results


@dataclass
class LiteratureSearcher:
    """Run configured sources independently and merge normalized records."""

    sources: dict[str, LiteratureSource]

    @classmethod
    def from_config(
        cls,
        *,
        pubmed_email: str = "",
        crossref_email: str = "",
        semantic_scholar_key: str = "",
        transport: Transport | None = None,
    ) -> LiteratureSearcher:
        return cls(
            {
                "crossref": CrossrefSource(transport, email=crossref_email),
                "pubmed": PubMedSource(transport, email=pubmed_email),
                "semantic_scholar": SemanticScholarSource(
                    transport, api_key=semantic_scholar_key
                ),
                "arxiv": ArxivSource(transport),
            }
        )

    def search(
        self,
        query: str,
        sources: list[str],
        max_results: int,
    ) -> SearchReport:
        if not query.strip():
            raise ValueError("研究主题不能为空")
        if max_results < 1:
            raise ValueError("max_results 必须大于 0")

        papers: list[PaperRecord] = []
        errors: list[str] = []
        counts: dict[str, int] = {}
        for source_name in sources:
            source = self.sources.get(source_name)
            if source is None:
                errors.append(f"{source_name}: 未配置该检索源")
                continue
            try:
                found = source.search(query, max_results)
                counts[source_name] = len(found)
                papers.extend(found)
            except (
                HttpClientError,
                RuntimeError,
                ET.ParseError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                errors.append(f"{source_name}: {exc}")

        deduplicated: list[PaperRecord] = []
        seen: set[str] = set()
        for paper in papers:
            key = paper.doi.lower() if paper.doi else re.sub(
                r"\W+", " ", paper.title.lower()
            ).strip()
            if key and key not in seen:
                seen.add(key)
                deduplicated.append(paper)

        return SearchReport(
            papers=deduplicated,
            errors=errors,
            sources_attempted=sources,
            counts_by_source=counts,
        )
