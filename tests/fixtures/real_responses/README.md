# 真实 API 原始响应回归样本

本目录保存的是从 crossref / pubmed / arxiv 三源**真实抓取**的原始响应字节，
**未经任何手工"美化"**（个别为压缩体积做了裁剪，均在下文标注）。它们用于
`tests/test_real_api_shapes.py`，以检验 `core/external_clients.py` 的解析/归一化层
以及 `core/retrieval.py` 的去重/排序层在**真实世界形状**上的行为——而不是干净的假件。

采集时间：2026-10-04。采集方式：系统 `python3`（3.13.9）+ 标准库 `urllib`，
User-Agent 为 `AcademicResearchAssistant/1.0`。全部为**正常的一次性查询**，
未对任何服务进行高频请求以刻意触发限流。

## 样本清单与来源

| 文件 | 源 | 采集时 HTTP 状态 | 场景 |
| --- | --- | --- | --- |
| `crossref_normal.json` | crossref | 200 | 正常查询 `sleep memory consolidation`（5 条） |
| `crossref_no_results.json` | crossref | 200 | **无结果**：`total-results: 0`，`items: []` |
| `crossref_chinese.json` | crossref | 200 | **中文主题** `深度学习 医学图像`，含 `<jats:p>` 包裹的中文摘要 |
| `crossref_long_query.json` | crossref | 200 | 超长查询（60 次重复 `machine learning`） |
| `crossref_many.json` | crossref | 200 | 结果较多（20 条，11259 字节），含**缺作者/缺年份/缺摘要/缺期刊**的记录，以及**期刊名带 HTML 实体 `&amp;`** |
| `crossref_special_chars.json` | crossref | **429** | 含特殊字符的查询 `CRISPR/Cas9 & "off-target" (2020)`；**429 响应体为空（0 字节）** |
| `crossref_no_select.json` | crossref | **400** | 非法参数（`select=` 为空）；响应体是 **validation-failure JSON**，其 `message` 字段是**列表**而非对象 |
| `crossref_bad_filter.json` | crossref | **400** | 非法 filter（`filter=notafield:5`），同为 validation-failure JSON |
| `crossref_html_404.html` | crossref | **404** | 错误路径返回**纯文本** `Resource not found.`（非 JSON） |
| `crossref_bad_rows.json` | crossref | **429** | 非法 `rows=-1`，服务端返回 429 且**空响应体（0 字节）** |
| `pubmed_search_normal.json` | pubmed | 200 | 正常 esearch `sleep memory consolidation`（5 个 PMID） |
| `pubmed_fetch_normal.xml` | pubmed | 200 | 对应的 efetch XML（90478 字节，含真实作者/摘要/DOI） |
| `pubmed_search_no_results.json` | pubmed | 200 | **无结果**：`count: "0"`，`idlist: []`，`warninglist.outputmessages=["No items found."]` |
| `pubmed_search_chinese.json` | pubmed | 200 | **中文主题** `深度学习` → `count: "0"`，`querytranslation` 被清空为空白 |
| `pubmed_search_special.json` | pubmed | 200 | 特殊字符 `CRISPR/Cas9 AND "off-target"` |
| `pubmed_search_long.json` | pubmed | 200 | 超长 AND 查询（40 次重复），`querytranslation` 极长 |
| `pubmed_search_letter.json` | pubmed | 200 | `Comment on article[pt]` → 0 条，但带 `errorlist.phrasesnotfound` |
| `pubmed_bad_request.json` | pubmed | **200** | **HTTP 200 但语义错误**：`esearchresult.ERROR = "retmax is not a positive number"`，**没有 `idlist` 字段** |
| `arxiv_normal.xml` | arxiv | 200 | 正常查询（5 条 Atom entry） |
| `arxiv_no_results.xml` | arxiv | 200 | **无结果**：`opensearch:totalResults = 0`，无 entry |
| `arxiv_chinese.xml` | arxiv | 200 | **中文主题** `深度学习` → 0 条 |
| `arxiv_special.xml` | arxiv | 200 | 特殊字符查询，返回 4 条 |
| `arxiv_long.xml` | arxiv | **500**（首次），正文为 **Atom 错误 feed** | 超长查询触发服务端错误；正文是一个**可被 XML 解析的 Atom feed**，其中 `<entry>` 的 `<id>` 为 `https://arxiv.org/api/errors`、`<title>` 为 `Error`、`<author><name>` 为 `arXiv api core` |
| `arxiv_error_feed.xml` | arxiv | **200** | 同样的超长查询，服务端这次返回 **200 但正文是被拼坏的查询串**（非错误 feed），其中仍含 5 条真实结果——用于说明"200 不代表内容正确" |
| `semantic_scholar_429.json` | semantic_scholar | **429** | 无 API key 时的限流响应体：`{"message": "Too Many Requests...", "code": "429"}` |

## 响应头

`headers_manifest.json` 记录了若干端点的**响应头**实测值，要点：

- `semantic_scholar` 无 key 时**稳定返回 429**，响应头**不含** `Retry-After` / `X-RateLimit-*`；
- `pubmed` 的 200 响应**确实带有** `X-Ratelimit-Limit` / `X-Ratelimit-Remaining` 头；
- `crossref` 429 的响应头不含限流提示字段。

## 明确**未能**抓到的情形（如实声明，未伪造）

1. **带 `Retry-After` 的 429**：`semantic_scholar` 与 `crossref` 的 429 响应头均不含
   `Retry-After`，无法给出真实样本。
2. **带 `X-RateLimit-*` 的 429**：只在 pubmed 的 **200** 响应上观察到
   `X-Ratelimit-*`，未在 429 上观察到。
3. **超时（Timeout）**：未构造出真实超时，未伪造。
4. **HTML 错误页**：crossref 的错误路径返回的是 `text/plain` 的
   `Resource not found.`（`crossref_html_404.html`），未取得典型 HTML 错误页。
5. **pubmed 的 429 / 5xx**：本次未能自然触发，未伪造。

## `manifest.json` 与 `headers_manifest.json`

`manifest.json` 由采集脚本自动生成，逐条记录 `name / url / status / bytes / file`，
`url` 即**当时的完整请求 URL**。`headers_manifest.json` 记录带响应头的探测结果。
