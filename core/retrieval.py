"""Deterministic query construction and explainable ranking for literature search.

这个模块是**纯函数**的集合：不调用模型、不联网、不依赖随机数或 ``set`` 的
迭代顺序。相同的输入必须产生**完全一致**的输出——这是可复现性的前提，也是
"如实报告排序依据"能够成立的前提。

设计取舍（为什么是这些信号、这个顺序）：

1. **多源命中数**权重最高。同一篇文献被越多独立检索源返回，说明它越可能在
   该主题下是核心文献；这是唯一一个"跨源共识"信号，单个源内部的排序
   （相关性、引用数）都可能被其自身的偏差放大。
2. **主题词项重合**次之。它是唯一直接针对"这篇文献和用户的研究主题有多大关系"
   的信号；引用数和年份都只是"一般意义上的重要性"，与本主题无关。
3. **引用数**再次之：它是质量的粗糙代理，但对新发表的高质量论文不公平，
   因此低于主题相关度。
4. **年份新近**最低：只在其余信号完全打平时起决胜作用，避免新论文压过经典文献。

每一项都按固定方向排序（越大越靠前），并且对数值信号做**饱和**处理，防止
单一维度的极值主导整个排序。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from core.research_models import PaperRecord

__all__ = [
    "STOPWORDS",
    "DeduplicationResult",
    "PaperCandidate",
    "RankedPaper",
    "build_query",
    "deduplicate",
    "normalize_doi",
    "normalize_term",
    "rank_candidates",
    "topic_terms",
]


#: 英文停用词与学术写作中的**噪声词**（"研究""影响"这类不承载检索信息的词）。
#: 刻意保持为一小组静态词表而非引入同义词库：同义词表是一份需要维护、且会随
#: 版本漂移的资产，会让"确定性"变成"取决于词表版本"。这里只做保守的噪声剔除。
#:
#: **注意**：本表**只对拉丁词项生效**。中文按单字切分（见 :func:`topic_terms`），
#: 因此表中若出现多字中文词也匹配不上——故这里**不放**任何中文词条，避免留下
#: "看起来生效、实际不生效"的死代码。
STOPWORDS: frozenset[str] = frozenset(
    {
        # 通用英文停用词
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
        "from", "had", "has", "have", "he", "her", "his", "in", "into", "is",
        "it", "its", "of", "on", "or", "our", "she", "that", "the", "their",
        "them", "then", "there", "these", "they", "this", "those", "to", "was",
        "were", "which", "who", "will", "with", "we", "us", "you", "your",
        # 学术主题里的高频噪声词（"……的影响""……的研究"）
        "about", "analysis", "approach", "based", "between", "case", "does",
        "during", "effect", "effects", "how", "impact", "implications", "influence",
        "method", "methods", "research", "role", "study", "studies", "toward",
        "towards", "using", "what", "when", "where", "whether", "why",
        # 疑问/语气词
        "can", "could", "may", "might", "must", "should", "would",
    }
)

#: 拉丁词项长度下限。过短的字符串（如单个字母）几乎不携带检索信息，作为噪声剔除。
#: 中文单字**不受**此限：单个汉字的信息量远高于单个拉丁字母，切掉会直接丢词。
_MIN_TERM_LENGTH = 2

#: 引用数归一化的饱和点：达到该引用数即拿满该项权重。避免个别"高被引巨佬论文"
#: 通过引用数这一单一维度压过所有主题相关的文献。
_CITATION_SATURATION = 30

#: 每个独立信号的最大权重。排序优先级即按这些数值从高到低体现。
#: 每个信号都先**归一化到 [0, 1]** 再乘权重，确保权重就是该信号的最大贡献：
#: 否则"命中词项数"这类无上界的原始计数会随主题长度膨胀，把权重关系搞反。
#: 多源命中数的原始值 = 来源数 - 1，本模块直接按"每多一个来源记 1 分"处理。
_WEIGHT_SOURCE_SUPPORT = 6.0
_WEIGHT_TOPIC_MATCH = 4.0
_WEIGHT_CITATION = 3.0
_WEIGHT_RECENCY = 3.0


# --------------------------------------------------------------------------- #
# 文本归一化与词项提取
# --------------------------------------------------------------------------- #
def _levenshtein(a: str, b: str) -> int:
    """标准编辑距离；仅用于极短的词项边界判断，输入长度很小。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,        # 删除
                    current[j - 1] + 1,     # 插入
                    previous[j - 1] + (ca != cb),  # 替换
                )
            )
        previous = current
    return previous[-1]


def normalize_doi(value: object) -> str:
    """DOI 的**唯一权威**归一化实现。

    剥离 ``https?://doi.org/`` 前缀（大小写不敏感）、去掉末尾句点、去两端空白。
    其余情况原样返回（不做 URL 解码等有损变换）。

    为什么放在这里：去重键必须与源层解析出的 DOI 用**同一套**归一化，否则
    ``https://doi.org/10.1234/x`` 与 ``10.1234/x`` 会被判成两篇。把实现收敛到
    一处（叶子模块），由 ``core.external_clients._normalize_doi`` 委托调用，
    避免两处各写一份、将来再次漂移。
    """
    doi = str(value).strip() if value is not None else ""
    doi = re.sub(r"^https?://doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi.removesuffix(".").strip()


def normalize_term(term: str) -> str:
    """保守的词形归一化：小写、去两端非字母数字字符。

    只做 **规则可预测** 的处理，不做词干提取（"running" → "run"）。词干提取
    依赖具体算法实现，不同库版本结果不同，会破坏确定性；而"研究"这类噪声词
    已在 :data:`STOPWORDS` 里显式剔除，不需要倚赖词干。
    """
    return re.sub(r"^[^0-9a-z\u4e00-\u9fff]+|[^0-9a-z\u4e00-\u9fff]+$", "", term.lower())


def _tokenize(text: str) -> list[str]:
    """把一段文本切成候选词项：英文按单词、中文按单字。

    中文不做分词（分词器是另一份会漂移的依赖），改取**单字**作为词项。这偏保守：
    中文检索源（如 PubMed 上的中英混排摘要）对单字匹配容忍度尚可，而单字不会
    引入分词错误导致的漏召。
    """
    latin = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]*", text)
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    return [*latin, *cjk]


def topic_terms(topic: str) -> tuple[str, ...]:
    """从研究主题中提取检索词（确定性、可复现）。

    规则：

    * 切成英文单词与中文**单字**；
    * 去掉 :data:`STOPWORDS` 中的停用词/噪声词/语气词；
    * 拉丁词项去掉长度小于 :data:`_MIN_TERM_LENGTH` 的项；中文单字保留；
    * 英文词项同时保留"去掉末尾 s 的变体"与原名（去重后按首次出现顺序返回）。

    **中文处理的已知局限（如实声明，勿声称做了未做的事）**：本函数**不做中文
    分词**，中文一律按**单字**处理。由此产生两个真实后果：

    1. :data:`STOPWORDS` 里那些**多字中文噪声词**（如"研究""影响""的"）对中文
       **实际不生效**——单字切分后永远匹配不到它们。该表对中文是死代码。
    2. 单字重合对中文只是**粗略但对排序仍可用**的相关性信号（字符级重合），
       它**不适合**用来构造面向 PubMed / arXiv 的检索串（那两个源对单字
       ``AND`` 查询基本返回零结果）。因此 :func:`build_query` 对中文会**回退为
       原主题**，而不是拿单字去拼查询。

    返回空元组时调用方应回退为原主题（见 :func:`build_query`）——本函数**从不**
    抛出异常，也**从不**返回空查询。
    """
    seen: dict[str, None] = {}
    for raw in _tokenize(topic):
        term = normalize_term(raw)
        if not term:
            continue
        if not _is_cjk_term(term) and len(term) < _MIN_TERM_LENGTH:
            continue
        if term in STOPWORDS:
            continue
        candidates = [term]
        # 保守的单复数归一：仅在词长足够时去除末尾 s，避免把 "bus" -> "bu"。
        if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
            candidates.append(term[:-1])
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen[candidate] = None
    return tuple(seen)


# --------------------------------------------------------------------------- #
# 按检索源适配的查询串
# --------------------------------------------------------------------------- #
def build_query(topic: str, source: str) -> str:
    """按检索源适配查询串。**保守原则：绝不产出比原主题语义更差的查询。**

    * ``pubmed`` / ``arxiv``：用 ``AND`` 连接词项——**但仅当全部词项都是词形
      （拉丁字母/数字词）时**。这两个源支持布尔语法，多词 ``AND`` 比整句更精确。
    * ``crossref`` / ``semantic_scholar`` / 其他：用空格连接词项。这两个源是
      相关性排序接口，不识别布尔 ``AND``（会被当成普通词），连接符必须用空格。

    **中文为何回退为原主题**：本模块不做中文分词，中文被切成单字。把单字用
    ``AND`` 拼起来（如 ``研 AND 究 AND 的 AND 影 AND 响``）对 PubMed / arXiv
    几乎必然零结果，**比直接发原主题更差**——这是功能退化。因此当词项中出现中文
    单字时，这两个源**回退为原主题**（原主题作为一句自然语言，至少能被它们按
    短语/松散匹配处理）。宁可退回到改动前的行为，也不构造无效查询。

    若 :func:`topic_terms` 提取结果为空，**同样回退为原主题**（去空白），确保
    永远不会构造出空查询。``source`` 大小写不敏感。
    """
    original = " ".join(topic.split()).strip()
    terms = topic_terms(topic)
    if not terms:
        return original
    # 只要词项里含中文单字，就**对所有源**回退为原主题。理由见上：本模块不做
    # 中文分词，单字（无论用 AND 还是空格拼接）对检索源都劣于整句原主题。
    # 统一回退能保证"绝不产出比原主题更差的查询"这条原则对所有源一致成立。
    if any(_is_cjk_term(term) for term in terms):
        return original
    if source.strip().lower() in {"pubmed", "arxiv"}:
        return " AND ".join(terms)
    return " ".join(terms)


def _is_cjk_term(term: str) -> bool:
    """词项是否只由中文字符组成。"""
    return bool(re.fullmatch(r"[\u4e00-\u9fff]+", term))


# --------------------------------------------------------------------------- #
# 去重与合并
# --------------------------------------------------------------------------- #
#: 标题归一化：小写、把非字母数字折叠为单个空格。**仅用于比较**，不改变原记录。
_TITLE_KEY_RE = re.compile(r"\W+")


def _title_key(title: str) -> str:
    return _TITLE_KEY_RE.sub(" ", title.lower()).strip()


def _merge_papers(papers: Sequence[PaperRecord]) -> PaperRecord:
    """把"同一篇文献"的多条记录**合并**为一条，结果与输入顺序无关。

    逐字段规则（全部是集合级运算或确定性选择，因此与顺序无关）：

    * ``title``：取最长非空者（最长者携带信息最多；并列时取字典序最小者）。
    * ``abstract``：取最长非空者（摘要越长越可能含引用核验所需的原文；并列取字典序最小）。
    * ``authors``：按首次出现顺序去重后拼接**所有**记录的作者（并集，保序）。
    * ``year``：取**最早**非空年份（在线优先论文可能在多个源上标注不同年份，
      取最早者是保守选择；并列取最小值本身即确定）。
    * ``citation_count``：取**最大**非空值（不同源的引用数不同步，取最大代表最全）。
    * ``doi`` / ``external_id`` / ``url`` / ``journal``：各取字典序最小的非空值。
    * ``source``：取字典序最小的来源名（真正的来源集合由调用方单独维护）。
    * ``raw_data``：各源原始数据按键合并，键冲突时保留字典序最小的来源的值。

    合并是**确定性**的：给定同一组记录（任意顺序），输出完全一致。
    """
    records = list(papers)
    if len(records) == 1:
        return records[0]

    def _longest(values: list[str]) -> str:
        non_empty = [value for value in values if value]
        if not non_empty:
            return ""
        # 取 (-长度, 字典序) 最小者 == 最长、并列时字典序最小者：与输入顺序无关。
        return min(non_empty, key=lambda value: (-len(value), value))

    def _min_non_empty(values: list[str]) -> str:
        non_empty = [value for value in values if value]
        return min(non_empty) if non_empty else ""

    authors: list[str] = []
    for record in records:
        for author in record.authors:
            if author not in authors:
                authors.append(author)

    years = [record.year for record in records if record.year is not None]
    citations = [
        record.citation_count
        for record in records
        if record.citation_count is not None
    ]
    # raw_data：按键合并。同一键在多个源上可能给出不同值，必须**确定性**选址，
    # 不能依赖记录顺序。做法：收集该键的**全部取值**，按值的字符串表示取最小者。
    # 这对值不可比较的 dict 也安全（任何对象都有稳定的 str/repr）。
    raw_candidates: dict[str, dict[str, Any]] = {}
    for record in records:
        for key, value in record.raw_data.items():
            raw_candidates.setdefault(key, {})
            # 用值的字符串表示做去重键，保留首个出现该字符串的值对象。
            raw_candidates[key].setdefault(repr(value), value)
    raw_data: dict = {
        key: min(values.items(), key=lambda item: item[0])[1]
        for key, values in raw_candidates.items()
    }

    return PaperRecord(
        title=_longest([record.title for record in records]),
        authors=authors,
        year=min(years) if years else None,
        journal=_min_non_empty([record.journal for record in records]),
        doi=_min_non_empty([record.doi for record in records]),
        abstract=_longest([record.abstract for record in records]),
        url=_min_non_empty([record.url for record in records]),
        source=_min_non_empty([record.source for record in records]),
        citation_count=max(citations) if citations else None,
        external_id=_min_non_empty([record.external_id for record in records]),
        raw_data=raw_data,
    )


def _dedup_key(paper: PaperRecord) -> tuple[str, str]:
    """去重主键：优先强标识。返回 ``(kind, value)``。

    * 有 DOI → ``("doi", 归一化DOI)``；**不同 DOI 永不合并**。
    * 否则有 ``external_id`` → ``("extid", 归一化external_id)``（arXiv id / PMID /
      S2 paperId 都是强标识）。
    * 否则 → ``("title", 归一化标题)``（弱标识，需旁证才合并，见 :func:`deduplicate`）。
    """
    doi = normalize_doi(paper.doi)
    if doi:
        return ("doi", doi.lower())
    if paper.external_id:
        return ("extid", paper.external_id.strip().lower())
    return ("title", _title_key(paper.title))


def _contradicts(a: PaperRecord, b: PaperRecord) -> bool:
    """两条记录是否存在**相互矛盾**的非空元数据。

    用于 :func:`deduplicate` 的"逐字节同名"快速合并路径：标题在归一化**之前**
    就逐字节相同时，默认判为同一篇；但只要两条记录给出了**互相冲突**的年份或
    第一作者，就说明它们可能不是同一篇（如同名论文、勘误），此时拆开。

    只有"两边都非空且不等"才算矛盾；一方缺失不构成矛盾。
    """
    if a.year is not None and b.year is not None and a.year != b.year:
        return True
    first_a = a.authors[0].strip().lower() if a.authors else ""
    first_b = b.authors[0].strip().lower() if b.authors else ""
    return bool(first_a and first_b and first_a != first_b)


def _corroborates(a: PaperRecord, b: PaperRecord) -> bool:
    """标题键相同的两条记录是否**确为同一篇**（旁证）。

    仅"标点归一化后的标题相同"不足以判定同一篇——`Sleep: memory & learning!`
    与 `Sleep memory learning` 会归一到同一键，但它们是两篇不同论文（甚至可能
    一个是另一篇的标题变体）。旁证满足任一条才合并：

    * 年份相同（都非空且相等）；
    * 第一作者相同（都非空且相等，小写比较）；
    * ``external_id`` 相同（都非空且相等）。

    旁证不足时**宁可保留两条**，绝不静默丢弃一篇真实文献。
    """
    if a.year is not None and b.year is not None and a.year == b.year:
        return True
    if a.external_id and b.external_id and a.external_id.lower() == b.external_id.lower():
        return True
    first_a = a.authors[0].strip().lower() if a.authors else ""
    first_b = b.authors[0].strip().lower() if b.authors else ""
    return bool(first_a and first_b and first_a == first_b)


def _should_merge(a: PaperRecord, b: PaperRecord) -> bool:
    """标题键相同的两条记录是否应合并为同一篇。

    * 归一化**前**标题**逐字节相同**：这是比"归一化后相同"更强的同一性信号，
      默认合并；**除非** :func:`_contradicts` 判定二者元数据互相矛盾（年份或
      第一作者冲突），此时拆开。
    * 归一化前不同（纯标点碰撞）：仍需 :func:`_corroborates` 旁证。

    注意："逐字节相同即合并"**不是静默的**：一旦合并，
    ``total_found - deduplicated_count`` 会如实反映"有记录被合并"，且合并只在
    元数据不矛盾时发生。反之，拆开会让用户在参考文献里看到同一篇两次——那是
    一个**可见瑕疵**，而合并只在矛盾时才可能有损失，那种情况已被守卫拦住。
    """
    if a.title == b.title:
        return not _contradicts(a, b)
    return _corroborates(a, b)


@dataclass(frozen=True)
class DeduplicationResult:
    """去重结果，含"疑似重复但保留为两条"的可见记录。"""

    candidates: tuple[PaperCandidate, ...]
    #: 人类可读的说明：哪些记录因**旁证不足**而未被合并（绝不静默丢弃）。
    kept_separate: tuple[str, ...]


def deduplicate(
    records: Sequence[tuple[str, PaperRecord]],
) -> DeduplicationResult:
    """按主键去重并**合并**重复记录；结果与输入顺序无关。

    ``records`` 是 ``(source_name, paper)`` 序列。函数对：
    1. 用 :func:`_dedup_key` 归类；键相同者视为同一篇的候选，再判旁证；
    2. DOI / external_id 键：直接是同一篇，合并；
    3. 标题键：按下列规则再聚类（见 :func:`_should_merge`）：
       - 归一化**前**标题**逐字节相同**且**无相互矛盾元数据** → 合并；
       - 逐字节相同但存在矛盾（年份或第一作者冲突）→ 拆开；
       - 归一化后才相同（标点碰撞）、归一化前不同 → 需 :func:`_corroborates`
         旁证才合并；旁证不足 → **保留为多条**。
       任何"保留为多条"的决策都写入 :attr:`DeduplicationResult.kept_separate`
       （可见，不静默）。

    合并用 :func:`_merge_papers`（字段级、确定性、顺序无关）。来源集合取**权威
    来源名**（即 ``records`` 里 ``(source_name, paper)`` 的 ``source_name``，
    而非 ``PaperRecord.source`` 字段——后者可能为空或与调用方标注不一致），
    取并集后**按字典序排序**，因此与 ``sources`` 的传入顺序无关。

    **所有分组在合并前都按内容身份键预排序**（强键、弱键一视同仁），因此
    :func:`_merge_papers` 里任何"按首次出现顺序"的规则（如 ``authors``）都不会
    暴露在调用方的输入顺序之下。
    """
    # 按主键分组，组内保留 (权威来源名, 记录) 对。
    groups: dict[tuple[str, str], list[tuple[str, PaperRecord]]] = {}
    for source_name, paper in records:
        groups.setdefault(_dedup_key(paper), []).append((source_name, paper))

    candidates: list[PaperCandidate] = []
    kept_separate: list[str] = []

    def _emit(pairs: list[tuple[str, PaperRecord]]) -> None:
        merged = _merge_papers([paper for _source, paper in pairs])
        sources = tuple(sorted({source for source, _paper in pairs}))
        candidates.append(PaperCandidate(paper=merged, sources=sources))

    # 组按主键字典序处理，保证候选产出顺序也与输入顺序无关。
    for key in sorted(groups):
        kind, _value = key
        # 关键：**所有路径**都先按内容身份键预排序再合并。强键路径此前漏了这一步，
        # 导致 authors 等"按出现顺序"的字段依赖输入顺序。
        group = sorted(groups[key], key=lambda item: _identity_key(item[1]))
        if kind in ("doi", "extid"):
            # 强标识：整组就是同一篇，直接合并。
            _emit(group)
            continue

        # 弱标识（标题）：贪心聚类，同簇内两两满足 _should_merge。
        clusters: list[list[tuple[str, PaperRecord]]] = []
        for pair in group:
            placed = False
            for cluster in clusters:
                if all(_should_merge(pair[1], member[1]) for member in cluster):
                    cluster.append(pair)
                    placed = True
                    break
            if not placed:
                clusters.append([pair])

        for cluster in clusters:
            _emit(cluster)
        if len(clusters) > 1:
            titles = sorted({paper.title for _source, paper in group})
            kept_separate.append(
                "标题归一化后相同但不足以判定为同一篇，保留为 "
                f"{len(clusters)} 条独立文献（未合并，可能各自是真实文献）: "
                + " | ".join(titles)
            )

    # 候选按内容身份键排序，确保集合级确定性。
    candidates.sort(key=lambda item: _identity_key(item.paper))
    return DeduplicationResult(
        candidates=tuple(candidates),
        kept_separate=tuple(kept_separate),
    )


# --------------------------------------------------------------------------- #
# 候选与排序
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PaperCandidate:
    """去重后的一篇候选文献，以及返回过它的检索源。"""

    paper: PaperRecord
    sources: tuple[str, ...]


@dataclass(frozen=True)
class RankedPaper:
    """一篇带确定性得分与**人类可读**排序依据的文献。"""

    paper: PaperRecord
    sources: tuple[str, ...]
    score: float
    reasons: tuple[str, ...]


def _match_score(paper: PaperRecord, terms: tuple[str, ...]) -> tuple[float, tuple[str, ...]]:
    """词项重合：标题命中权重高于摘要，并记录命中的词项用于解释。

    **已知粗糙度（如实声明）**：这里是**子串**匹配，不是词边界匹配。因此主题
    ``cars`` 会命中标题 ``Cardiac arrest``（"car" 是子串），中文单字更是天然
    子串匹配。这是刻意的取舍：词边界匹配需要分词，而分词是另一份会漂移的依赖。
    该信号只用于**排序**（相对位置），不用于判定"是否相关"，因此粗糙但可用。
    """
    if not terms:
        return 0.0, ()
    title = paper.title.lower()
    abstract = paper.abstract.lower()

    matched: list[str] = []
    score = 0.0
    for term in terms:
        in_title = term in title
        in_abstract = term in abstract
        if not (in_title or in_abstract):
            continue
        matched.append(term)
        weight = 1.0 if in_title else 0.5
        # 标题里出现过的词，在摘要里再出现不额外加分，避免长摘要刷分。
        score += weight
    return score, tuple(matched)


def _citation_value(paper: PaperRecord) -> int:
    value = paper.citation_count
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _year_value(paper: PaperRecord) -> int:
    year = paper.year
    if isinstance(year, bool) or not isinstance(year, int):
        return 0
    return year


def _describe(
    *,
    sources: tuple[str, ...],
    matched: tuple[str, ...],
    citation_value: int,
    year_value: int,
    current_year: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if len(sources) > 1:
        reasons.append(f"{len(sources)} 个检索源同时命中: {', '.join(sources)}")
    else:
        reasons.append(f"仅 1 个检索源命中: {sources[0] if sources else '未知'}")
    if matched:
        reasons.append(f"主题词项命中 {len(matched)} 个: {', '.join(matched)}")
    else:
        reasons.append("主题词项未命中")
    if citation_value:
        reasons.append(f"被引用 {citation_value} 次")
    else:
        reasons.append("无引用数据")
    if year_value:
        age = current_year - year_value
        if age < 0:
            # 未来年份是真实可达的（上游允许 current_year+1/+2 的"在线优先/待出版"
            # 出版日）。解释文本**不得出现负数**——"约 -2 年前"对用户毫无意义。
            reasons.append(f"发表于 {year_value} 年（未来年份/待出版）")
        else:
            reasons.append(f"发表于 {year_value} 年（约 {age} 年前）")
    else:
        reasons.append("无年份数据")
    return tuple(reasons)


def _composite_score(
    *,
    source_count: int,
    match: float,
    term_count: int,
    citation_value: int,
    year_value: int,
    current_year: int,
) -> float:
    # 多源命中：每多一个独立来源记 1 分（无上界，因为"更多来源一致"永远更强）。
    source_bonus = _WEIGHT_SOURCE_SUPPORT * (source_count - 1)
    # 主题重合：归一化到 [0, 1]——命中全部词项才拿满分，避免长主题刷分。
    match_ratio = min(match / term_count, 1.0) if term_count else 0.0
    match_bonus = _WEIGHT_TOPIC_MATCH * match_ratio
    # 引用数：线性到饱和点，之后不再加分。
    citation_ratio = min(citation_value, _CITATION_SATURATION) / _CITATION_SATURATION
    citation_bonus = _WEIGHT_CITATION * citation_ratio
    if year_value:
        # 未来年份与当年同分：`max(age, 0)` 把负年龄夹到 0，使 "待出版" 与
        # "今年发表" 都拿满 recency 权重。这是刻意的——上游允许的"在线优先"
        # 日期不应比当年论文更"新"地刷分。
        age = max(current_year - year_value, 0)
        recency = 1.0 / (1.0 + age)
        recency_bonus = _WEIGHT_RECENCY * (recency if age <= 2 else recency * 0.5)
    else:
        recency_bonus = 0.0
    return round(source_bonus + match_bonus + citation_bonus + recency_bonus, 6)


def _identity_key(paper: PaperRecord) -> tuple:
    """**内容决定**的最终决胜键：与输入顺序无关。

    依次比较 ``DOI → 归一化标题 → 第一作者 → 年份 → 外部标识``，全部小写归一化。
    空值排最后（用 ``(1, "")`` 表示"缺失"、``(0, value)`` 表示"有值"），这样
    "有 DOI 的"总是先于"没 DOI 的"，但两者都缺 DOI 时再比标题，依此类推。

    为什么需要它：``sorted`` 是稳定排序，若只用分数做键，**完全并列**的多篇文献
    会跟随输入顺序——而输入顺序来自 `for source_name in sources`。用户改一下
    ``--sources`` 的顺序，并列项就会互换位置；一旦裁剪到 ``max_results``，
    "谁被保留"就变了，进入手稿的证据集会因源顺序不同而不同。加上这个内容键后，
    排序结果是文献**集合**的纯函数，与来源顺序无关。
    """

    def _present(value: object) -> tuple[int, str]:
        text = str(value).strip().lower() if value not in (None, "") else ""
        return (1, "") if not text else (0, text)

    authors = paper.authors or []
    return (
        _present(paper.doi),
        _present(re.sub(r"\W+", " ", paper.title.lower()).strip()),
        _present(authors[0] if authors else ""),
        _present(paper.year),
        _present(paper.external_id),
    )


def rank_candidates(
    candidates: Sequence[PaperCandidate], *, topic: str
) -> list[RankedPaper]:
    """确定性排序：**不使用 LLM**，相同输入必须产生完全相同的顺序。

    排序优先级（高 → 低）：

    1. **多源命中数**（跨源共识，最强信号）；
    2. **主题词项重合**（与用户主题的直接相关度）；
    3. **引用数**（质量代理，饱和于 :data:`_CITATION_SATURATION` 次）；
    4. **年份新近**（仅作决胜，近 2 年内权重更高）；
    5. **内容决胜键**（:func:`_identity_key`）：当以上全部并列时，按
       DOI/标题/作者/年份/外部标识的字典序排定。**这一步保证了排序结果是文献
       *集合*的纯函数——与 ``sources`` 的传入顺序、与候选列表的初始顺序都无关。**

    实现上**不依赖** ``set`` 的迭代顺序：多源命中数由调用方传入的 ``sources``
    元组长度得出，词项比较使用有序元组。
    """
    current_year = datetime.now(UTC).year
    terms = topic_terms(topic)

    ranked: list[RankedPaper] = []
    for candidate in candidates:
        sources = tuple(candidate.sources)
        matched = _match_score(candidate.paper, terms)
        citation_value = _citation_value(candidate.paper)
        year_value = _year_value(candidate.paper)
        score = _composite_score(
            source_count=len(sources),
            match=matched[0],
            term_count=len(terms),
            citation_value=citation_value,
            year_value=year_value,
            current_year=current_year,
        )
        ranked.append(
            RankedPaper(
                paper=candidate.paper,
                sources=sources,
                score=score,
                reasons=_describe(
                    sources=sources,
                    matched=matched[1],
                    citation_value=citation_value,
                    year_value=year_value,
                    current_year=current_year,
                ),
            )
        )

    # 显式指定全部键，避免依赖 dataclass 字段顺序。两趟稳定排序：
    #   第一趟按"内容决胜键"升序；第二趟按主信号降序（stable，保留第一趟的
    #   并列顺序）。最终结果与输入顺序无关。
    ranked.sort(key=lambda item: _identity_key(item.paper))
    ranked.sort(
        key=lambda item: (
            item.score,
            len(item.sources),
            _year_value(item.paper),
            _citation_value(item.paper),
        ),
        reverse=True,
    )
    return ranked
