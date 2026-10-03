"""Real literature API adapters and a resilient multi-source searcher."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from core.http_client import HttpClientError, UrllibTransport
from core.research_models import PaperRecord, SearchReport
from core.retrieval import build_query, deduplicate, normalize_doi, rank_candidates


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
    """委托给 :func:`core.retrieval.normalize_doi`（唯一权威实现）。

    刻意不在本文件保留第二份实现：去重键与源层解析必须用**同一套**归一化，
    两处各写一份必然漂移（``https://doi.org/10.1234/x`` vs ``10.1234/x`` 会被
    判成两篇）。
    """
    return normalize_doi(value)


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


def _normalize_year_range(
    year_range: tuple[int, int] | list[int] | None,
) -> tuple[int, int] | None:
    """归一化年份范围；``None`` 或 ``[0, 0]`` 表示**不过滤**。

    语义（与 ``search.year_range`` 配置一致）：

    * ``None`` / ``[0, 0]`` → 返回 ``None``（不过滤，与改动前逐字节一致）；
    * 单边为 ``0`` → 该边**不限**（例如 ``[2020, 0]`` 表示"2020 年及以后"，
      ``[0, 2020]`` 表示"2020 年及以前"）；
    * 两端都非零 → 必须满足 ``起始 <= 结束``；若反了按**空范围**处理
      （返回 ``(start, end)`` 原样，过滤器会给所有"年份已知"的文献判为出界，
      这是明确可见的结果，而不是静默忽略）。

    只接受 2 元序列；其他长度一律抛 ``ValueError``——配置层已做类型校验，
    这里再严格一次，避免"看起来生效、实际没接上"的第三份语义。
    """
    if year_range is None:
        return None
    try:
        values = list(year_range)
    except TypeError as exc:  # 传入非可迭代对象：明确的编程错误
        raise ValueError(f"year_range 必须是二元序列，当前为 {year_range!r}") from exc
    if len(values) != 2:
        raise ValueError(f"year_range 必须是二元序列，当前为 {year_range!r}")
    start, end = values
    if not isinstance(start, int) or isinstance(start, bool):
        raise TypeError(f"year_range 的起始年必须是整数，当前为 {start!r}")
    if not isinstance(end, int) or isinstance(end, bool):
        raise TypeError(f"year_range 的结束年必须是整数，当前为 {end!r}")
    if start == 0 and end == 0:
        return None
    return (start, end)


def _in_year_range(year: int | None, bounds: tuple[int, int] | None) -> bool:
    """``year`` 是否落在 ``bounds`` 内。

    **关键取舍：``year`` 为 ``None`` 时返回 ``True``（保留）。**

    年份缺失是上游元数据不全（预印本、早期记录、接口未回填），**不是**
    "该文献不在范围内"的证据。过滤掉它们会因一个缺失字段而**静默丢弃真实
    证据**，这与 Phase 7"宁可多留也不静默丢弃"的原则冲突。因此范围过滤只
    约束**年份已知**的文献；未知年份的文献被保留，并在
    :attr:`SearchReport.ranking_reasons` 里显式说明，绝不让用户误以为
    "范围内只有这些"。

    单边为 0 表示该边不限（见 :func:`_normalize_year_range`）。
    """
    if bounds is None:
        return True
    if year is None:
        return True
    start, end = bounds
    if start and year < start:
        return False
    return not (end and year > end)


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
        year_range: tuple[int, int] | list[int] | None = None,
    ) -> SearchReport:
        """检索多个来源、去重、按确定性规则排序，并把总量裁剪到 ``max_results``。

        **行为变更（刻意）：``max_results`` 现在是"最终收录的总量上限"**，
        而不是过去那种"每个来源各自的上限"。例如 ``--max-results 10`` 配 3 个
        来源，旧行为最多返回 30 篇，新行为**最多 10 篇**——这与用户对这个名字的
        直觉一致。

        单源预算是 :func:`math.ceil`\\ ``(max_results / len(sources))``，至少为 1；
        这样即使上游返回很多，也不会因为先到先得而让某个来源独占全部名额。

        裁剪**绝不静默**：被裁掉的条数进入 :attr:`SearchReport.dropped_by_limit`，
        原始条数、去重后条数进入 :attr:`SearchReport.total_found` /
        :attr:`SearchReport.deduplicated_count`，排序依据进入
        :attr:`SearchReport.ranking_reasons`。调用方据此可以如实告诉用户
        "找到了多少、留下了多少、为什么"。

        查询串按来源适配（见 :func:`core.retrieval.build_query`）：``pubmed`` /
        ``arxiv`` 用 ``AND`` 连接词项，其余来源用空格连接；词项提取为空时回退
        为原主题。

        **重名来源会被保序去重**（见下方 ``unique_sources``）：``sources`` 里若
        同一名称出现两次，只会向其发起一次检索，避免 ``counts_by_source`` 覆盖与
        ``total_found`` 重复累加导致二者不自洽。``sources_attempted`` 也记录去重后
        的列表。

        去重与**字段合并**见 :func:`core.retrieval.deduplicate`：同一篇文献的多条
        记录会合并（摘要取最长、引用取最大、年份取最早非空……），因此结果与
        ``sources`` 顺序无关。"标题归一化后相同但旁证不足"的记录**保留为多条**，
        并记入 :attr:`SearchReport.ranking_reasons`（绝不静默丢弃）。

        **年份过滤（``year_range``）**：``[起始年, 结束年]``，``0`` 表示**该边
        不限**，``None`` / ``[0, 0]`` 表示**不过滤**（此时行为与引入该参数之前
        逐字节一致）。语义细节见 :func:`_in_year_range`：

        * **过滤发生在排序之后、总量裁剪之前**。理由：排序是去重候选**集合**
          的纯函数，与年份无关；把过滤放在排序之后，既保证默认（不过滤）时
          排序结果逐字节不变，也保证过滤后的保留顺序仍是"排序结果的子序列"
          ——即文献集合的纯函数，与来源顺序无关。若放在排序之前，虽然结果
          相同，但会让"排序输入规模"随范围变化，掩盖 Phase 7 已有的确定性保证。
        * **``year`` 为 ``None`` 的文献被保留**（见 :func:`_in_year_range`）：
          年份缺失不等于"不在范围内"，丢弃它们会因一个缺失字段而静默丢证据。
        * 被排除的条数进入 :attr:`SearchReport.excluded_by_year`（绝不静默），
          并在排序依据里说明；单边/无年份保留情况也会显式记入。

        既有契约不变：:class:`SearchReport` 的 ``papers`` / ``errors`` /
        ``sources_attempted`` / ``counts_by_source`` 语义与类型保持原样，新增字段
        均带默认值。
        """
        if not query.strip():
            raise ValueError("研究主题不能为空")
        if max_results < 1:
            raise ValueError("max_results 必须大于 0")
        if len(sources) == 0:
            raise ValueError("sources 不能为空")

        # 年份范围在此处归一化并**尽早失败**：非法形状/类型是编程错误，不应在
        # 完成一次真实联网检索之后才暴露。
        year_bounds = _normalize_year_range(year_range)

        # 保序去重来源名：重名不再重复检索，保证 counts_by_source 与 total_found 自洽
        # （否则同名键互相覆盖，sum(counts) != total_found）。
        unique_sources = list(dict.fromkeys(sources))

        # 单源预算：向上取整，至少 1。保证总量上限按设计成立。
        per_source_budget = max(1, math.ceil(max_results / len(unique_sources)))

        # 收集 (source_name, paper) 对；来源顺序不影响最终结果（去重/合并/排序都
        # 是集合级确定性的，见 core.retrieval）。
        collected: list[tuple[str, PaperRecord]] = []
        errors: list[str] = []
        counts: dict[str, int] = {}
        total_found = 0
        for source_name in unique_sources:
            source = self.sources.get(source_name)
            if source is None:
                errors.append(f"{source_name}: 未配置该检索源")
                continue
            try:
                source_query = build_query(query, source_name)
                found = source.search(source_query, per_source_budget)
                counts[source_name] = len(found)
                total_found += len(found)
                collected.extend((source_name, paper) for paper in found)
            except (
                HttpClientError,
                RuntimeError,
                ET.ParseError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                errors.append(f"{source_name}: {exc}")

        dedup = deduplicate(collected)
        candidates = list(dedup.candidates)

        ranked = rank_candidates(candidates, topic=query)

        # 年份过滤发生在**排序之后、总量裁剪之前**：排序仍是"去重候选集合"的
        # 纯函数（与年份无关），过滤只是从有序序列里剔除不满足年份谓词的条目，
        # 保留顺序因此仍是文献集合的纯函数，与来源顺序无关。bounds 为 None 时
        # 该分支不改变任何行为（逐字节与改动前一致）。
        if year_bounds is None:
            in_range = ranked
        else:
            in_range = [
                item
                for item in ranked
                if _in_year_range(item.paper.year, year_bounds)
            ]
        excluded_by_year = len(ranked) - len(in_range)

        kept = in_range[:max_results]
        dropped_by_limit = len(in_range) - len(kept)

        ranking_reasons: list[str] = []
        ranking_reasons.extend(dedup.kept_separate)
        if year_bounds is not None:
            start, end = year_bounds
            if excluded_by_year:
                ranking_reasons.append(
                    f"年份范围过滤：范围 [{start or '不限'}, {end or '不限'}] 排除了 "
                    f"{excluded_by_year} 篇**年份已知且落在范围外**的文献（绝不静默丢弃）。"
                )
            kept_unknown = sum(1 for item in in_range if item.paper.year is None)
            if kept_unknown:
                ranking_reasons.append(
                    f"年份范围过滤：保留 {kept_unknown} 篇**年份未知**（``year`` 为空）"
                    "的文献——年份缺失不是'不在范围内'的证据，丢弃它们会因缺失字段"
                    "而丢证据。"
                )
        if in_range:
            ranking_reasons.append(
                f"排序依据（高→低）：多源命中数、主题词项重合、引用数、年份新近；"
                f"共 {len(in_range)} 篇范围内候选，保留前 {len(kept)} 篇"
                f"（总量上限 {max_results}，单源预算 {per_source_budget}）。"
            )
            for position, item in enumerate(kept, start=1):
                ranking_reasons.append(
                    f"#{position} {item.paper.title}：得分 {item.score}；"
                    + "；".join(item.reasons)
                )

        return SearchReport(
            papers=[item.paper for item in kept],
            errors=errors,
            sources_attempted=list(unique_sources),
            counts_by_source=counts,
            total_found=total_found,
            deduplicated_count=len(candidates),
            dropped_by_limit=dropped_by_limit,
            excluded_by_year=excluded_by_year,
            ranking_reasons=ranking_reasons,
        )
