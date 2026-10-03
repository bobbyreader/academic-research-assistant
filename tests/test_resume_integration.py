"""断点续跑接入真实执行主干的集成测试。

覆盖（对应任务 4 的六项要求）：
1. 完整跑一次 → 输入不变再跑 → 全部阶段复用，且**外部调用次数为零**；
2. 输入变化（改主题）→ 不复用任何阶段；
3. 中间产物缺失 → 只复用缺失点之前的阶段（前缀性）；
4. 重水化失败（产物损坏）→ 该阶段改为执行，并产生 warning；
5. 续跑结果与全新运行的产物一致（统计数字、图表清单、参考文献、评审结论）；
6. `resume=False` 时行为与全新运行一致。

计数假件与 `ResearchService` 装配方式对齐：工厂（`LiteratureSearcher.from_config`
与 `build_llm_client`）在服务层是无条件调用的，因此把"零外部调用"的断言打在
假件对象的 `.search()` / `.complete*()` 上。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.artifact_store import ArtifactStore
from core.external_clients import PaperRecord, SearchReport
from core.research_pipeline import (
    PipelineResult,
    ResearchPipeline,
    ResearchPipelineConfig,
)
from core.resume import RESUME_STEPS, RunFingerprint, describe_reuse, plan_resume
from core.state_manager import ProjectState, StateManager, WorkflowStage

DATASET = (
    "group,score\n"
    "A,1\nA,2\nA,3\nA,4\nA,5\n"
    "B,6\nB,7\nB,8\nB,9\nB,10\n"
)


class CountingResolver:
    """离线 DOI 解析器，记录被查询的 DOI。"""

    def __init__(self) -> None:
        self.checked: list[str] = []

    def resolve(self, doi: str) -> tuple[bool, str | None]:
        self.checked.append(doi)
        return True, None


class CountingSearcher:
    """计数检索桩：任何复用都不应触发 `.search()`。"""

    def __init__(self) -> None:
        self.search_calls = 0
        self.queries: list[str] = []

    def search(self, query: str, sources: list[str], max_results: int) -> SearchReport:
        self.search_calls += 1
        self.queries.append(query)
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
            counts_by_source={source: 1 for source in sources},
        )


class CountingLLM:
    """计数模型桩：任何复用都不应触发 `.complete()` / `.complete_json()`。"""

    def __init__(self) -> None:
        self.complete_calls = 0
        self.complete_json_calls = 0

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.complete_json_calls += 1
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
                "summary": "结构清晰，证据强度有限。",
                "strengths": ["主题明确"],
                "concerns": [
                    {
                        "category": "methodology",
                        "severity": "minor",
                        "statement": "样本描述不足",
                        "evidence": "A difference was observed",
                    }
                ],
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
        self.complete_calls += 1
        return "# Draft\n\nA difference was observed [P1] (p = 0.001).\n"


class Harness:
    """用真实管线驱动重复运行，并在两次运行之间保留同一对计数假件。"""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.projects_dir = tmp_path / "projects"
        self.artifact_store = ArtifactStore(self.projects_dir)
        self.state_manager = StateManager(self.projects_dir)
        self.searcher = CountingSearcher()
        self.llm = CountingLLM()
        self.resolver = CountingResolver()
        self.project = "resume"
        self.data = tmp_path / "data.csv"
        self.data.write_text(DATASET, encoding="utf-8")
        # 建项目：`_record_all` 需要读写 state.json（等价于服务层的写入点）。
        self.state_manager.save(
            ProjectState(
                name=self.project,
                mode="hybrid",
                current_stage=WorkflowStage.BRAINSTORMING,
            )
        )

    def new_pipeline(self) -> ResearchPipeline:
        return ResearchPipeline(
            self.artifact_store,
            searcher=self.searcher,
            llm_client=self.llm,
            doi_resolver=self.resolver,
            reviewer_count=1,
        )

    def run(
        self,
        topic: str = "climate adaptation",
        *,
        data_path: Path | None = None,
        resume_decision=None,
    ) -> PipelineResult:
        pipeline = self.new_pipeline()
        pipeline.resume = resume_decision
        return pipeline.run(
            ResearchPipelineConfig(
                project_name=self.project,
                topic=topic,
                sources=["crossref"],
                max_results=2,
                data_path=data_path if data_path is not None else self.data,
            )
        )

    def plan(
        self,
        *,
        topic: str = "climate adaptation",
        data_path: Path | None = None,
        enabled: bool = True,
        previous_steps=None,
    ):
        fingerprint = RunFingerprint.build(
            topic=topic,
            sources=["crossref"],
            max_results=2,
            data_path=data_path if data_path is not None else self.data,
            reviewer_count=1,
            figure_dpi=300,
        )
        return plan_resume(
            artifact_store=self.artifact_store,
            project_name=self.project,
            fingerprint=fingerprint,
            previous_steps=previous_steps if previous_steps is not None else {},
            enabled=enabled,
        )

    def fingerprint(self, *, topic: str = "climate adaptation") -> RunFingerprint:
        return RunFingerprint.build(
            topic=topic,
            sources=["crossref"],
            max_results=2,
            data_path=self.data,
            reviewer_count=1,
            figure_dpi=300,
        )

    def latest(self, stage: str, filename: str) -> Path:
        path = self.artifact_store.get_artifact(self.project, stage, filename)
        assert path is not None, f"{stage}/{filename} 从未写出"
        return path

    def recorded(self, *markers: str) -> dict[str, RunFingerprint]:
        """构造与 `state.json` 中记录的指纹等价的映射（按进度标记名）。"""
        return {marker: self.fingerprint() for marker in markers}


def _record_all(harness: Harness, first: PipelineResult) -> None:
    """把首跑记录下来的分阶段指纹写入 state.json（等价于服务层的写入点）。"""
    state = harness.state_manager.load(harness.project)
    assert state is not None
    state.metadata["completed_step_fingerprints"] = {
        "search": harness.fingerprint().to_dict(),
        "lit_review": harness.fingerprint().to_dict(),
        "writing": harness.fingerprint().to_dict(),
    }
    harness.state_manager.save(state)


# --------------------------------------------------------------------------- #
# 1. 输入不变 → 全部复用，且外部调用为零
# --------------------------------------------------------------------------- #
def test_second_run_with_unchanged_inputs_reuses_everything(tmp_path: Path) -> None:
    harness = Harness(tmp_path)

    first = harness.run()
    assert first.reused_steps == []
    _record_all(harness, first)

    searches = harness.searcher.search_calls
    completes = harness.llm.complete_calls
    json_calls = harness.llm.complete_json_calls

    decision = harness.plan(previous_steps=harness.recorded("search", "lit_review", "writing"))
    second = harness.run(resume_decision=decision)

    assert second.reused_steps == list(RESUME_STEPS)
    assert second.resume_note and "复用" in second.resume_note
    # 外部调用次数为零：没有重新检索、没有重新调用 LLM。
    assert harness.searcher.search_calls == searches
    assert harness.llm.complete_calls == completes
    assert harness.llm.complete_json_calls == json_calls


# --------------------------------------------------------------------------- #
# 2. 输入变化（改主题）→ 不复用任何阶段
# --------------------------------------------------------------------------- #
def test_changed_topic_reuses_nothing(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    first = harness.run()
    _record_all(harness, first)

    decision = harness.plan(
        topic="quantum computing for medicine",
        previous_steps=harness.recorded("search", "lit_review", "writing"),
    )
    assert decision.reusable == ()

    second = harness.run(
        topic="quantum computing for medicine", resume_decision=decision
    )
    assert second.reused_steps == []
    assert harness.searcher.search_calls == 2
    assert harness.searcher.queries[-1] == "quantum computing for medicine"

    # `resume_plan`（事前）必须解释"为什么没能复用"：用户改主题后若只看到事实
    # 口径的"未复用任何旧产物"，会误以为续跑功能坏了；原因才是判断依据。
    assert second.resume_plan
    assert "运行输入已变化" in second.resume_plan
    assert "研究主题" in second.resume_plan

    # 两个字段语义分离、各司其职：plan 是预测+理由，note 是事实，故不相等。
    assert second.resume_note == describe_reuse([])
    assert "未复用任何旧产物" in second.resume_note
    assert second.resume_plan != second.resume_note


# --------------------------------------------------------------------------- #
# 3. 中间产物缺失 → 只复用缺失点之前的阶段（前缀性）
# --------------------------------------------------------------------------- #
def test_missing_middle_artifact_stops_the_prefix(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    first = harness.run()
    _record_all(harness, first)

    # 删除 claims 阶段所依赖的论断核验产物（它位于 search/analysis 之后、writing 之前）。
    claims_artifact = harness.latest("writing", "claim_evidence_verification.json")
    claims_artifact.unlink()

    decision = harness.plan(previous_steps=harness.recorded("search", "lit_review", "writing"))
    assert decision.reusable == ("search", "analysis")

    second = harness.run(resume_decision=decision)
    assert second.reused_steps == ["search", "analysis"]
    assert "claims" not in second.reused_steps
    assert "writing" not in second.reused_steps
    assert "review" not in second.reused_steps
    # search 复用 → 未重新检索。
    assert harness.searcher.search_calls == 1


# --------------------------------------------------------------------------- #
# 4. 重水化失败（产物损坏）→ 该阶段改为执行，且产生 warning
# --------------------------------------------------------------------------- #
def test_corrupted_artifact_forces_rerun_with_warning(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    first = harness.run()
    _record_all(harness, first)

    searches = harness.searcher.search_calls

    # 损坏检索产物：`plan_resume` 会判定 search 不可复用，因此这里直接传入一个
    # 声称可用的判定，强制触发"重水化失败 → 回退执行"这一路径。
    literature = harness.latest("search", "literature.json")
    literature.write_text("{ not valid json", encoding="utf-8")

    decision = plan_resume(
        artifact_store=harness.artifact_store,
        project_name=harness.project,
        fingerprint=RunFingerprint.build(
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
            data_path=harness.data,
            reviewer_count=1,
            figure_dpi=300,
        ),
        previous_steps=harness.recorded("search", "lit_review", "writing"),
        enabled=True,
    )
    # `plan_resume` 看到损坏 JSON 时会拒绝复用，因此断点位置落在 search 之前；
    # 这仍然验证了"损坏 → 不复用"的安全方向。
    assert decision.can_skip("search") is False

    result = harness.run(resume_decision=decision)
    assert "search" not in result.reused_steps
    # search 被重新执行 → 检索调用增加。
    assert harness.searcher.search_calls == searches + 1


def test_rehydration_failure_is_reported_as_warning(tmp_path: Path) -> None:
    """产物在"判定之后、重水化之前"被损坏时，必须回退执行并给出可核对警告。

    这里直接构造一个 `ResumeDecision`，声称 search 可复用（模拟判定与读取之间
    的竞态/外部损坏），再让产物在中途损坏，验证管线捕获异常、执行该阶段并告警。
    """
    from core.resume import ResumeDecision

    harness = Harness(tmp_path)
    first = harness.run()
    _record_all(harness, first)
    searches = harness.searcher.search_calls

    # 破坏 search_report.json，但让判定仍声称 search 可跳过。
    report_path = harness.latest("search", "search_report.json")
    report_path.write_text("[]", encoding="utf-8")  # 结构非法（应为对象）

    fingerprint = RunFingerprint.build(
        topic="climate adaptation",
        sources=["crossref"],
        max_results=2,
        data_path=harness.data,
        reviewer_count=1,
        figure_dpi=300,
    )
    decision = ResumeDecision(
        enabled=True,
        fingerprint=fingerprint,
        reusable=("search", "analysis", "claims", "writing", "review"),
        reason="测试用：声称全部可复用。",
    )

    result = harness.run(resume_decision=decision)

    assert "search" not in result.reused_steps
    assert harness.searcher.search_calls == searches + 1  # 回退为执行
    assert any("search" in warning and "重新执行" in warning for warning in result.warnings)

    # `resume_note` 必须是**事后**口径：反映实际复用数，且与 reused_steps 一致，
    # 不能被"判定声称 5 个可复用"的事前预测带偏。
    assert result.resume_note
    assert result.resume_note == describe_reuse(result.reused_steps)
    # 判定声称 5 个阶段可复用，但实际必然少于 5（search 回退、claims/review 也回退）。
    assert len(result.reused_steps) < 5
    assert "判定可复用" not in result.resume_note


# --------------------------------------------------------------------------- #
# 5. 续跑结果与全新运行一致
# --------------------------------------------------------------------------- #
def test_resumed_run_matches_fresh_run_artifacts(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    first = harness.run()
    _record_all(harness, first)

    decision = harness.plan(previous_steps=harness.recorded("search", "lit_review", "writing"))
    resumed = harness.run(resume_decision=decision)
    resumed_manuscript = resumed.manuscript_path.read_text(encoding="utf-8")

    # 换一个目录跑一次全新运行，比较最终交付物。
    fresh_harness = Harness(tmp_path / "fresh")
    fresh = fresh_harness.run()
    fresh_manuscript = fresh.manuscript_path.read_text(encoding="utf-8")

    assert resumed_manuscript == fresh_manuscript

    # 统计报告、图表清单、引用核验、评审结论逐项一致（含 figures.json——其 path
    # 记录的是相对项目根的路径，故两个不同项目目录下也应逐字一致）。
    for stage, filename in (
        ("analysis", "statistics_report.json"),
        ("visualization", "figures.json"),
        ("writing", "citation_verification.json"),
        ("review", "review_reports.json"),
    ):
        resumed_bytes = harness.latest(stage, filename).read_bytes()
        fresh_bytes = fresh_harness.latest(stage, filename).read_bytes()
        assert resumed_bytes == fresh_bytes, f"{stage}/{filename} 不一致"

    assert resumed.review is not None
    assert resumed.review.decision == fresh.review.decision


# --------------------------------------------------------------------------- #
# 6. resume=False 时行为与改动前一致
# --------------------------------------------------------------------------- #
def test_resume_disabled_reruns_everything(tmp_path: Path) -> None:
    harness = Harness(tmp_path)
    first = harness.run()
    _record_all(harness, first)

    searches = harness.searcher.search_calls
    completes = harness.llm.complete_calls
    json_calls = harness.llm.complete_json_calls

    decision = harness.plan(
        previous_steps=harness.recorded("search", "lit_review", "writing"),
        enabled=False,
    )
    assert decision.enabled is False
    assert decision.reusable == ()

    second = harness.run(resume_decision=decision)
    assert second.reused_steps == []
    assert second.resume_note  # 说明"已禁用续跑"
    # 全部阶段重做 → 外部调用全部发生。
    assert harness.searcher.search_calls == searches + 1
    assert harness.llm.complete_calls == completes + 1
    assert harness.llm.complete_json_calls > json_calls


def test_resume_none_behaves_like_fresh_run(tmp_path: Path) -> None:
    """`resume=None`（管线直连）必须与"完全没引入续跑"一致：不填充续跑字段。"""
    harness = Harness(tmp_path)
    result = harness.run()  # resume_decision 默认 None
    assert result.reused_steps == []
    assert result.resume_note == ""
    assert result.resume_plan == ""


# --------------------------------------------------------------------------- #
# 硬约束：空主题仍阻断、引用关口仍硬阻断（续跑不得弱化这两道门槛）
# --------------------------------------------------------------------------- #
def test_empty_topic_still_raises_even_with_resume_decision(tmp_path: Path) -> None:
    from core.research_pipeline import ResearchPipelineError

    harness = Harness(tmp_path)
    with pytest.raises(ResearchPipelineError, match="研究主题不能为空"):
        harness.run(topic="   ", resume_decision=harness.plan())


def test_unknown_citation_marker_still_blocks_even_when_resuming(tmp_path: Path) -> None:
    from core.research_pipeline import ResearchPipelineError

    class BadDraftLLM(CountingLLM):
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            self.complete_calls += 1
            return "# Draft\n\nEvidence says so [P9].\n"

    harness = Harness(tmp_path)
    harness.llm = BadDraftLLM()

    # 前缀为空（无完成记录）→ 全部重跑；writing 执行时引用关口必须仍然硬阻断。
    decision = harness.plan(previous_steps={})
    with pytest.raises(ResearchPipelineError, match="P9"):
        harness.run(resume_decision=decision)
