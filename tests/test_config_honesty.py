"""配置诚实性审计：`settings.yaml` 里宣称可配置的每个键，都必须真的被代码读取。

本文件**独立编写**，不复用其他测试的断言，只驱动真实链路并观察
`core.config_loader` 的读取接口实际被请求过哪些键。

为什么需要它
------------
本阶段的准入规则是"只保留当前有代码读取的键——未被读取的键会误导用户，让他们
以为自己能配置某件事"。`tests/test_config_validation.py` 用**静态**方式保证"每个键
都有可追溯的读取路径"（键集关系 + 直读白名单），但它**无法发现**这种缺陷：

    `DEFAULTS` 里存在一个没有任何代码读取的键。

实际发生过：`llm.*` 与 `api_keys.*` 共 7 个键曾被放进 `DEFAULTS` 却无人读取，其中
`llm.provider` 的取值（`codex_cli`）还与代码真实回退（`gemini`，见
`core/llm_client.py`）**不符**。它当时不可见，只因为没有任何代码读它；一旦有人把它
改走 `get_*`，默认行为就会**静默变化**，而没有任何测试会发现。

本文件用**运行时插桩**补上这个缺口：跑一遍真实链路，记录实际被读取的键。

插桩的坑（本文件最容易做错的地方）
----------------------------------
消费者写的是 `from core.config_loader import get_str`，因此只补丁
`core.config_loader.get_str` **完全无效**——已导入的模块持有的是原函数对象。必须
同时替换所有模块里已导入的同名引用。下面 `_install_spies` 遍历 `sys.modules` 完成
这件事，并在测试结束后逐一还原。
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from core import config_loader
from core.external_clients import PaperRecord, SearchReport

#: 需要插桩的读取接口。`get_mapping` 也在内：按节读取同样是"读取该节"的证据。
_READERS = (
    "get_str",
    "get_int",
    "get_bool",
    "get_float",
    "get_str_list",
    "get_int_pair",
    "get_mapping",
)

_DATASET = "group,score\nA,1\nA,2\nB,3\nB,4\n"


class _FakeSearcher:
    """代替四个真实文献 API。"""

    def search(
        self,
        query: str,
        sources: list[str],
        max_results: int,
        year_range: object = None,
    ) -> SearchReport:
        return SearchReport(
            papers=[
                PaperRecord(
                    title="Climate adaptation evidence",
                    authors=["A Author"],
                    year=2024,
                    journal="Research Journal",
                    doi="10.1234/climate",
                    abstract="A finding.",
                    source="crossref",
                )
            ],
            errors=[],
            sources_attempted=list(sources),
            counts_by_source={name: 1 for name in sources},
        )


class _FakeResolver:
    """代替 Crossref DOI 查询。"""

    def resolve(self, doi: str) -> tuple[bool, str | None]:
        return True, None


class _FakeLLM:
    """代替 Codex CLI / Gemini / OpenAI 兼容客户端，按系统提示词路由回复。"""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        if "事实核查员" in system_prompt:
            return {
                "verdicts": [
                    {
                        "claim_index": 0,
                        "claim": "Adaptation is context-dependent",
                        "citation_id": "P1",
                        "verdict": "supports",
                        "quote": "A finding.",
                        "rationale": "摘要直接支持该论断。",
                    }
                ]
            }
        if "审稿人" in system_prompt:
            return {
                "summary": "结构清晰。",
                "strengths": ["主题明确"],
                "concerns": [],
                "recommendation": "minor_revision",
                "score": 65,
            }
        return {
            "research_question": "How does climate adaptation work?",
            "key_findings": [
                {"claim": "Adaptation is context-dependent", "citation_ids": ["P1"]}
            ],
            "research_gaps": ["More longitudinal evidence is needed"],
            "proposed_methods": ["Compare cohorts over time"],
        }

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "# Draft\n\nA difference was observed [P1] (p = 0.001).\n"


def _install_spies() -> tuple[list[str], dict[str, object], dict[str, object]]:
    """替换读取接口。

    返回 `(记录列表, 已安装的 wrapper, 原始函数)`。必须返回 wrapper 本身，因为还原
    时要靠**身份比对**把它们找回来（见 `_restore_spies`）。
    """
    reads: list[str] = []
    originals = {name: getattr(config_loader, name) for name in _READERS}

    def make(original):
        def wrapper(settings, dotted, *args, **kwargs):
            reads.append(dotted)
            return original(settings, dotted, *args, **kwargs)

        return wrapper

    patched = {name: make(original) for name, original in originals.items()}

    # 关键：消费者持有的是"已导入的函数对象"，必须逐一替换（只补丁原模块无效）。
    for module in list(sys.modules.values()):
        if module is None or module is config_loader:
            continue
        for name, original in originals.items():
            if getattr(module, name, None) is original:
                setattr(module, name, patched[name])
    for name in originals:
        setattr(config_loader, name, patched[name])
    return reads, patched, originals


def _restore_spies(patched: dict[str, object], originals: dict[str, object]) -> None:
    """还原全部替换。

    必须**重新扫描** `sys.modules`，而不能只还原"安装时记录过的位置"：被审计的链路
    会在插桩**之后**才导入某些模块（如 `orchestrator`），这些模块在导入时就把当时已
    插桩的函数对象绑定下来了。漏掉它们会留下指向**已废弃记录列表**的陈旧 wrapper，
    使下一次插桩的身份比对失败——表现为"同一测试单独跑通过、连跑却失败"的顺序依赖。
    """
    for module in list(sys.modules.values()):
        if module is None:
            continue
        for name, wrapper in patched.items():
            if getattr(module, name, None) is wrapper:
                setattr(module, name, originals[name])


@pytest.fixture
def spy_reads() -> Iterator[list[str]]:
    """插桩 `config_loader` 的读取接口，产出"被请求过的点号键"列表。"""
    reads, patched, originals = _install_spies()
    try:
        yield reads
    finally:
        _restore_spies(patched, originals)


def _drive_real_chain(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """驱动真实链路，只把外部世界（检索 API / LLM / DOI）换成假件。"""
    from core import citation_verifier, research_service
    from orchestrator import Orchestrator, configure_logging
    from web_app import create_app

    # 入口层：`logging.*` 只在应用入口被读取。
    configure_logging(root)
    # 构造调度器：`paths.*`、`export.*`、`citation.export_formats`。
    orchestrator = Orchestrator(root)
    orchestrator.init_project("honesty", "hybrid")

    data = root / "data.csv"
    data.write_text(_DATASET, encoding="utf-8")

    monkeypatch.setattr(
        research_service.LiteratureSearcher,
        "from_config",
        staticmethod(lambda **kwargs: _FakeSearcher()),
    )
    monkeypatch.setattr(
        research_service, "build_llm_client", lambda **kwargs: _FakeLLM()
    )
    monkeypatch.setattr(citation_verifier, "CrossrefDoiResolver", _FakeResolver)

    # 服务层：`search.*`、`citation.*`、`writing.*`、`review.*`、`figures.*`。
    orchestrator.run_real_research(
        "honesty",
        "climate adaptation",
        sources=["crossref"],
        max_results=2,
        data_path=data,
    )
    # 导出：`export.*` 与 `citation.export_formats` 的白名单校验。
    orchestrator.export("honesty", "pptx")
    # Web 入口：`create_app` 也会读 `paths.*` 与 `search.default_sources` 等。
    create_app(root)


def test_every_declared_config_key_is_actually_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spy_reads: list[str]
) -> None:
    """`DEFAULTS` 里的每个键都必须在真实链路上被读取过。

    未被读取的键 = "宣称可配置、实则无效"，正是本阶段存在的理由。这条断言能抓到
    静态检查抓不到的缺陷：键存在于 `DEFAULTS`，却没有任何代码读取它。
    """
    _drive_real_chain(tmp_path, monkeypatch)

    declared = set(config_loader.DEFAULTS)
    never_read = sorted(declared - set(spy_reads))

    # 插桩自检：若替换失效（消费者持有原函数对象），读取数会极少——那会让本测试
    # 以"全部键都没读"的形式误报，必须与真实缺陷区分开。
    assert len(spy_reads) >= 20, (
        f"插桩未生效：仅记录到 {len(spy_reads)} 次读取。"
        "请检查是否遗漏了替换已导入的模块属性。"
    )
    assert not never_read, (
        "以下配置键被宣称可配置，但整条真实链路从未读取它们——"
        f"它们会误导用户以为自己能配置某件事：{never_read}"
    )


def test_spy_installation_is_reversible() -> None:
    """插桩必须可还原，否则会污染其他测试（这是本文件的主要风险）。"""
    original = config_loader.get_str
    reads, patched, originals = _install_spies()
    try:
        assert config_loader.get_str is not original
        assert reads == []
        # 用一个真实存在的键，确认插桩记录了调用并正确转发到原实现。
        assert config_loader.get_str(object(), "figures.default_format") == "png"
        assert reads == ["figures.default_format"]
    finally:
        _restore_spies(patched, originals)
    assert config_loader.get_str is original


def test_restore_covers_modules_imported_after_installation() -> None:
    """还原必须覆盖"插桩之后才被导入"的模块。

    被审计的链路会在插桩之后导入 `orchestrator` 等模块，它们在导入时绑定当时的
    wrapper。若还原只处理"安装时记录的位置"，就会留下陈旧 wrapper，导致**同一测试
    单独跑通过、连跑却失败**的顺序依赖。本测试固定该行为。
    """
    reads, patched, originals = _install_spies()
    try:
        module = ModuleType("late_import_probe")
        # 模拟"插桩之后才导入"：该模块从 config_loader 取到的是当时的 wrapper。
        module.get_str = config_loader.get_str  # type: ignore[attr-defined]
        sys.modules[module.__name__] = module
        assert module.get_str is patched["get_str"]  # type: ignore[attr-defined]
    finally:
        _restore_spies(patched, originals)
        sys.modules.pop("late_import_probe", None)
    assert module.get_str is originals["get_str"]  # type: ignore[attr-defined]
    assert reads == []


def test_spies_reach_consumers_not_just_the_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spy_reads: list[str]
) -> None:
    """插桩必须作用到**消费者**，而不只是 `config_loader` 自身。

    消费者写的是 `from core.config_loader import get_str`，持有的是已导入的函数对象。
    若只补丁原模块，读取记录会接近空——本测试用"服务层独有的读取"来区分这两种情况，
    否则一次失效的插桩会被误读成"所有键都没人读"（假阳性）。
    """
    _drive_real_chain(tmp_path, monkeypatch)

    # 这些键只在服务层/入口层被读取，能证明替换确实穿透到了消费者。
    assert "review.reviewer_count" in spy_reads
    assert "figures.default_dpi" in spy_reads
    assert "paths.projects_dir" in spy_reads
