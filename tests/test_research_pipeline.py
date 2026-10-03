from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.artifact_store import ArtifactStore
from core.config_validation import ConfigValidationError
from core.external_clients import PaperRecord, SearchReport
from core.llm_client import build_llm_client
from core.research_pipeline import (
    ResearchPipeline,
    ResearchPipelineConfig,
    ResearchPipelineError,
)
from core.research_service import ResearchService
from core.state_manager import ProjectState, StateManager, WorkflowStage


class FakeResolver:
    """Offline DOI resolver that records every DOI it is asked about."""

    def __init__(self, resolved: bool = True) -> None:
        self.resolved = resolved
        self.checked: list[str] = []

    def resolve(self, doi: str) -> tuple[bool, str | None]:
        self.checked.append(doi)
        return (self.resolved, None if self.resolved else "Crossref 未返回该 DOI 的元数据")


class FakeSearcher:
    def search(
        self, query: str, sources: list[str], max_results: int
    ) -> SearchReport:
        assert query == "climate adaptation"
        assert sources == ["crossref"]
        assert max_results == 2
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
        )


class FakeLLM:
    def __init__(self, body: str = "# Climate adaptation\n\n## Abstract\nA grounded draft.") -> None:
        self.body = body

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
                "summary": "稿件结构清晰，但证据强度不足。",
                "strengths": ["主题明确"],
                "concerns": [
                    {
                        "category": "methodology",
                        "severity": "minor",
                        "statement": "样本描述不足",
                        "evidence": "A grounded draft.",
                    }
                ],
                "recommendation": "minor_revision",
                "score": 65,
            }
        assert "untrusted literature" in system_prompt
        assert "Climate adaptation evidence" in user_prompt
        return {
            "research_question": "How does climate adaptation work?",
            "key_findings": [
                {"claim": "Adaptation is context-dependent", "citation_ids": ["P1"]}
            ],
            "research_gaps": ["More longitudinal evidence is needed"],
            "proposed_methods": ["Compare cohorts over time"],
        }

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        assert "P1" in user_prompt
        return self.body


class FailingReviewLLM(FakeLLM):
    """Analysis succeeds but every simulated reviewer errors out."""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        if "审稿人" in system_prompt:
            raise RuntimeError("reviewer unavailable")
        return super().complete_json(system_prompt, user_prompt)


def test_pipeline_saves_search_analysis_and_manuscript_artifacts(
    tmp_path: Path,
) -> None:
    resolver = FakeResolver()
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=FakeLLM(),
        doi_resolver=resolver,
    )

    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="climate",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )

    assert result.manuscript_path is not None
    assert result.manuscript_path.exists()
    assert result.manuscript_path.read_text(encoding="utf-8").startswith(
        "# Climate adaptation"
    )

    search_path = tmp_path / "projects/climate/artifacts/search/literature.json.v1"
    assert json.loads(search_path.read_text(encoding="utf-8"))[0]["doi"] == (
        "10.1234/climate"
    )

    analysis_path = tmp_path / (
        "projects/climate/artifacts/analysis/research_analysis.json.v1"
    )
    assert json.loads(analysis_path.read_text(encoding="utf-8"))["research_question"] == (
        "How does climate adaptation work?"
    )

    verification_path = tmp_path / (
        "projects/climate/artifacts/writing/citation_verification.json.v1"
    )
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    assert verification["passed"] is True
    assert verification["unknown_markers"] == []
    assert resolver.checked == ["10.1234/climate"]


def test_pipeline_blocks_when_draft_cites_unknown_reference(tmp_path: Path) -> None:
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=FakeLLM(body="# Draft\n\nEvidence says so [P9].\n"),
        doi_resolver=FakeResolver(),
    )

    with pytest.raises(ResearchPipelineError, match="P9"):
        pipeline.run(
            ResearchPipelineConfig(
                project_name="fabricated",
                topic="climate adaptation",
                sources=["crossref"],
                max_results=2,
            )
        )

    writing_dir = tmp_path / "projects/fabricated/artifacts/writing"
    verification = json.loads(
        (writing_dir / "citation_verification.json.v1").read_text(encoding="utf-8")
    )
    assert verification["passed"] is False
    assert verification["unknown_markers"] == ["P9"]
    # A draft with untraceable citations must never be persisted as the deliverable.
    assert not (writing_dir / "manuscript.md.v1").exists()


def test_pipeline_reports_unresolved_doi_as_warning(tmp_path: Path) -> None:
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=FakeLLM(body="# Draft\n\nEvidence says so [P1].\n"),
        doi_resolver=FakeResolver(resolved=False),
    )

    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="unresolved",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )

    assert result.manuscript_path.exists()
    assert any("DOI" in warning for warning in result.warnings)


def test_missing_llm_key_marks_project_blocked(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("ARS_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ARS_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    projects_dir = tmp_path / "projects"
    state_manager = StateManager(projects_dir)
    state_manager.save(
        ProjectState(
            name="blocked",
            mode="hybrid",
            current_stage=WorkflowStage.BRAINSTORMING,
        )
    )

    with pytest.raises(RuntimeError, match="未配置 Gemini API 密钥"):
        ResearchService(
            projects_dir,
            state_manager,
            ArtifactStore(projects_dir),
        ).run(
            "blocked",
            "a topic",
            sources=["crossref"],
            max_results=1,
        )

    state = state_manager.load("blocked")
    assert state is not None
    assert state.stage_status[WorkflowStage.SEARCH] == "blocked"


def test_pipeline_with_dataset_produces_statistics_figures_and_traceability(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data.csv"
    data.write_text(
        "group,score\nA,1\nA,2\nA,3\nA,4\nA,5\nB,6\nB,7\nB,8\nB,9\nB,10\n",
        encoding="utf-8",
    )
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=FakeLLM(body="# Draft\n\nA difference was observed [P1] (p = 0.001).\n"),
        doi_resolver=FakeResolver(),
    )

    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="dataset",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
            data_path=data,
        )
    )

    artifacts = tmp_path / "projects/dataset/artifacts"
    stats = json.loads(
        (artifacts / "analysis/statistics_report.json.v1").read_text(encoding="utf-8")
    )
    assert stats["tests"][0]["test_name"] == "Welch t-test"
    assert stats["tests"][0]["effect_size_name"] == "Cohen's d"

    figures = json.loads(
        (artifacts / "visualization/figures.json.v1").read_text(encoding="utf-8")
    )
    assert figures["figures"], "expected at least one figure for a numeric column"
    figure_types = {figure["figure_type"] for figure in figures["figures"]}
    assert "distribution" in figure_types
    # `path` 必须指向项目内真实保存的图件，绝不记录运行期临时目录的悬空路径
    # （临时目录在运行结束时已被删除，写进持久化产物等于记录一个保证失效的位置）。
    # 记录的是相对项目根目录的路径，以保证产物可移植且确定。
    project_dir = tmp_path / "projects/dataset"
    for figure in figures["figures"]:
        stored = Path(figure["path"])
        assert not stored.is_absolute(), f"figures.json 记录了绝对路径: {stored}"
        assert "research-figures-" not in str(stored), (
            f"figures.json 记录了临时目录路径: {stored}"
        )
        real = project_dir / stored
        assert real.is_file(), f"figures.json 记录了不存在的图件路径: {real}"
        assert real == artifacts / "visualization" / f"{figure['filename']}.v1"

    manuscript = result.manuscript_path.read_text(encoding="utf-8")
    assert "# 推断统计分析报告" in manuscript  # statistics injected by the system
    assert "# 图表清单" in manuscript  # figure list injected by the system
    assert "## References" in manuscript

    verification = json.loads(
        (artifacts / "writing/statistics_verification.json.v1").read_text(encoding="utf-8")
    )
    assert verification["passed"] is True
    assert verification["computed_p_values"], "computed p-values must be recorded"
    # Phase 6 的新能力必须真的生效，而不只是签名换了：统计关口现在同时看效应量与
    # 样本量，产物里必须能看到这两类「本次分析产生」的数值。
    assert verification["computed_effect_sizes"], (
        "the statistics gate must record the computed effect size, not just p-values"
    )
    assert verification["computed_sample_sizes"] == [10], (
        "the statistics gate must record the computed sample sizes, not just p-values"
    )
    # 标称的检验名/效应量名与 analysis/statistics_report.json 一致，证明这些数值
    # 确实来自本次分析而非签名更换后的空转。
    assert stats["tests"][0]["n"] == 10
    assert (
        stats["tests"][0]["effect_size"]
        in verification["computed_effect_sizes"]
    )

    # 正文级论断核验（Phase 6）：产物必须真实落盘且非空。
    manuscript_claims_path = artifacts / "writing/manuscript_claim_verification.json.v1"
    assert manuscript_claims_path.is_file()
    manuscript_claims = json.loads(manuscript_claims_path.read_text(encoding="utf-8"))
    assert manuscript_claims["claim_count"] >= 1
    assert set(manuscript_claims) >= {
        "passed",
        "claim_count",
        "unsupported_count",
        "claims_without_evidence_count",
        "warnings",
        "author_checks",
        "claims",
    }
    manuscript_claims_md = artifacts / "writing/manuscript_claim_verification.md.v1"
    assert manuscript_claims_md.is_file() and manuscript_claims_md.stat().st_size > 0


def test_two_fresh_runs_produce_byte_identical_artifacts(tmp_path: Path) -> None:
    """确定性：相同输入、相同种子，全新运行的产物必须**逐字节**一致。

    这要求产物里不出现任何随运行变化的量（例如运行期临时目录的绝对路径）。
    若某处退回记录临时路径，本测试会立刻失败。
    """
    import hashlib

    data = tmp_path / "data.csv"
    data.write_text(
        "group,score\nA,1\nA,2\nA,3\nA,4\nA,5\nB,6\nB,7\nB,8\nB,9\nB,10\n",
        encoding="utf-8",
    )
    body = "# Draft\n\nA difference was observed [P1] (p = 0.001).\n"

    def run_into(project_name: str) -> dict[str, str]:
        pipeline = ResearchPipeline(
            ArtifactStore(tmp_path / "projects"),
            searcher=FakeSearcher(),
            llm_client=FakeLLM(body=body),
            doi_resolver=FakeResolver(),
        )
        pipeline.run(
            ResearchPipelineConfig(
                project_name=project_name,
                topic="climate adaptation",
                sources=["crossref"],
                max_results=2,
                data_path=data,
            )
        )
        base = tmp_path / "projects" / project_name / "artifacts"
        digests: dict[str, str] = {}
        for path in sorted(base.rglob("*.v*")):
            if path.name.endswith(".meta"):
                continue
            digests[str(path.relative_to(base))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        return digests

    first = run_into("determinism_a")
    second = run_into("determinism_b")

    assert first, "expected the run to produce artifacts"
    assert first == second, (
        "两次全新运行的产物不逐字节一致："
        f"{sorted(set(first) ^ set(second)) or [k for k in first if first[k] != second.get(k)]}"
    )


def test_pipeline_flags_invented_p_value_when_dataset_present(tmp_path: Path) -> None:
    data = tmp_path / "data.csv"
    data.write_text(
        "group,score\nA,1\nA,2\nA,3\nA,4\nA,5\nB,6\nB,7\nB,8\nB,9\nB,10\n",
        encoding="utf-8",
    )
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=FakeLLM(body="# Draft\n\nThe effect was decisive [P1] (p = 0.0001).\n"),
        doi_resolver=FakeResolver(),
    )

    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="invented",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
            data_path=data,
        )
    )

    verification = json.loads(
        (tmp_path / "projects/invented/artifacts/writing/statistics_verification.json.v1")
        .read_text(encoding="utf-8")
    )
    assert verification["passed"] is False
    assert verification["unmatched_count"] == 1
    assert any("统计陈述" in warning for warning in result.warnings)


def test_pipeline_flags_invented_effect_size_and_sample_size(tmp_path: Path) -> None:
    """Phase 6: 效应量与样本量也要可追溯——伪造的 d 与 n 必须被标为未匹配。

    这证明统计关口的新能力真的生效（不只是签名换了）：正文声称 d=1.2、n=500，
    而本次分析只产生 d≈-3.16、n=10。关口必须报告 kind 为 effect_size / sample_size
    的 unmatched 陈述，并产生 warning，**但绝不阻断运行**。
    """
    data = tmp_path / "data.csv"
    data.write_text(
        "group,score\nA,1\nA,2\nA,3\nA,4\nA,5\nB,6\nB,7\nB,8\nB,9\nB,10\n",
        encoding="utf-8",
    )
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=FakeLLM(
            body=(
                "# Draft\n\n"
                "The effect was large [P1] (Cohen's d = 1.2), n = 500.\n"
            )
        ),
        doi_resolver=FakeResolver(),
    )

    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="invented_stats",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
            data_path=data,
        )
    )

    # 顾问级：运行完成、手稿产出，绝不阻断。
    assert result.manuscript_path.exists()

    verification = json.loads(
        (
            tmp_path
            / "projects/invented_stats/artifacts/writing/statistics_verification.json.v1"
        ).read_text(encoding="utf-8")
    )
    assert verification["passed"] is False
    kinds = {claim["kind"] for claim in verification["claims"]}
    assert "effect_size" in kinds, "a fabricated effect size must be extracted"
    assert "sample_size" in kinds, "a fabricated sample size must be extracted"
    unmatched_kinds = {
        claim["kind"] for claim in verification["claims"] if not claim["matched"]
    }
    assert {"effect_size", "sample_size"} <= unmatched_kinds
    assert any("统计陈述" in warning for warning in result.warnings)


def test_manuscript_claim_gate_flags_citation_that_does_not_support(tmp_path: Path) -> None:
    """Phase 6: 正文里「句子—被引文献」配对缺乏摘要支持时，报告该配对不为
    supports 并产生 warning，**绝不阻断运行**。
    """
    class UnrelatedCitationLLM(FakeLLM):
        def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
            if "事实核查员" in system_prompt:
                return {
                    "verdicts": [
                        {
                            "claim_index": 0,
                            "claim": "A difference was observed [P1]",
                            "citation_id": "P1",
                            "verdict": "unsupported",
                            "quote": "A finding.",
                            "rationale": "摘要与被引句子的主张无关。",
                        }
                    ]
                }
            return super().complete_json(system_prompt, user_prompt)

    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=UnrelatedCitationLLM(
            body="# Draft\n\nA difference was observed [P1].\n"
        ),
        doi_resolver=FakeResolver(),
    )

    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="unrelated",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )

    # 顾问级：运行完成、手稿产出。
    assert result.manuscript_path.exists()

    payload = json.loads(
        (
            tmp_path
            / "projects/unrelated/artifacts/writing/manuscript_claim_verification.json.v1"
        ).read_text(encoding="utf-8")
    )
    assert payload["claim_count"] >= 1
    verdicts = [
        evidence["verdict"]
        for claim in payload["claims"]
        for evidence in claim["evidence"]
    ]
    assert verdicts, "the manuscript gate must judge at least one pairing"
    assert all(verdict != "supports" for verdict in verdicts)
    assert any("正文级论断核验" in warning for warning in result.warnings)


def test_manuscript_claim_gate_makes_no_extra_model_call_without_markers(
    tmp_path: Path,
) -> None:
    """Phase 6: 正文没有引用标识时，正文级核验贡献 **0** 次模型调用。

    成本可预期是硬要求。这里用「有标识 vs 无标识」两次运行的**差值**来隔离正文级
    关口的贡献：同一条链路其它阶段（分析、论断级核验）调用次数相同，因此差值恰好
    等于正文级核验的调用次数。
    """

    class CountingLLM(FakeLLM):
        def __init__(self, body: str) -> None:
            super().__init__(body=body)
            self.complete_json_calls = 0

        def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
            self.complete_json_calls += 1
            return super().complete_json(system_prompt, user_prompt)

    def run_with(project_name: str, body: str) -> tuple[int, Path]:
        llm = CountingLLM(body=body)
        pipeline = ResearchPipeline(
            ArtifactStore(tmp_path / "projects"),
            searcher=FakeSearcher(),
            llm_client=llm,
            doi_resolver=FakeResolver(),
            reviewer_count=0,
        )
        pipeline.run(
            ResearchPipelineConfig(
                project_name=project_name,
                topic="climate adaptation",
                sources=["crossref"],
                max_results=2,
            )
        )
        return llm.complete_json_calls, (
            tmp_path / f"projects/{project_name}/artifacts/writing"
        )

    without_markers, no_marker_dir = run_with(
        "nomarkers", "# Draft\n\nNo citation markers here at all.\n"
    )
    with_markers, _ = run_with(
        "withmarkers", "# Draft\n\nA difference was observed [P1].\n"
    )

    # 有标识恰好一次正文级调用；无标识 0 次 → 差值恰为 1。
    assert with_markers - without_markers == 1, (
        "the manuscript gate must make exactly one model call when markers are "
        f"present and none when absent (delta={with_markers - without_markers})"
    )

    payload = json.loads(
        (no_marker_dir / "manuscript_claim_verification.json.v1").read_text(
            encoding="utf-8"
        )
    )
    assert payload["claims"] == []
    assert payload["passed"] is True
    assert payload["warnings"], "no-marker runs must explain that nothing was checked"
    # 产物仍然落盘（resume 依赖它作为 writing 完成的证据）。
    assert (no_marker_dir / "manuscript_claim_verification.md.v1").is_file()


def test_manuscript_claim_gate_failure_is_advisory_and_never_blocks(tmp_path: Path) -> None:
    """Phase 6: 正文级核验抛异常时必须转为 warning 并继续，绝不阻断运行。

    两层防线都验证：
    1. 关口内部（`verify_claims`）把模型失败降级为 unclear + warning；
    2. 管线层（`_run_manuscript_claim_verification` 的 try/except）兜住任何从关口
       逃逸的异常，转为 warning。这里直接把关口替换成会抛异常的实现来触发第 2 层。
    """
    from core import research_pipeline as rp

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("manuscript gate model unavailable")

    original = rp.verify_manuscript_claims
    rp.verify_manuscript_claims = boom  # type: ignore[assignment]
    try:
        pipeline = ResearchPipeline(
            ArtifactStore(tmp_path / "projects"),
            searcher=FakeSearcher(),
            llm_client=FakeLLM(body="# Draft\n\nA difference was observed [P1].\n"),
            doi_resolver=FakeResolver(),
        )
        result = pipeline.run(
            ResearchPipelineConfig(
                project_name="gateboom",
                topic="climate adaptation",
                sources=["crossref"],
                max_results=2,
            )
        )
    finally:
        rp.verify_manuscript_claims = original  # type: ignore[assignment]

    # 运行完成、手稿产出，且 warning 中有说明。
    assert result.manuscript_path.exists()
    assert any("正文级论断核验未执行" in warning for warning in result.warnings)
    # 关口失败时仍不留残缺产物：json/md 均未落盘。
    writing = tmp_path / "projects/gateboom/artifacts/writing"
    assert not (writing / "manuscript_claim_verification.json.v1").exists()


def test_manuscript_claim_gate_model_failure_degrades_with_warning(
    tmp_path: Path,
) -> None:
    """Phase 6: 模型在正文级核验里报错时，关口内部降级为 unclear + warning，
    运行照常完成、产物照常落盘。"""

    class ExplodingManuscriptGateLLM(FakeLLM):
        def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
            # 正文级核验与论断级核验共用「事实核查员」提示词；只要 user_prompt
            # 呈现的是「句子—被引文献」配对（正文级），就让它失败。
            if "事实核查员" in system_prompt and "待核验句子" in user_prompt:
                raise RuntimeError("manuscript gate model unavailable")
            if "事实核查员" in system_prompt:
                raise RuntimeError("manuscript gate model unavailable")
            return super().complete_json(system_prompt, user_prompt)

    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=ExplodingManuscriptGateLLM(
            body="# Draft\n\nA difference was observed [P1].\n"
        ),
        doi_resolver=FakeResolver(),
    )

    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="gatedegrade",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )

    assert result.manuscript_path.exists()
    assert any(
        "论断证据核验的模型调用失败" in warning for warning in result.warnings
    )
    payload = json.loads(
        (
            tmp_path
            / "projects/gatedegrade/artifacts/writing/manuscript_claim_verification.json.v1"
        ).read_text(encoding="utf-8")
    )
    verdicts = [
        evidence["verdict"]
        for claim in payload["claims"]
        for evidence in claim["evidence"]
    ]
    assert verdicts and all(verdict == "unclear" for verdict in verdicts)


def test_pipeline_runs_advisory_peer_review(tmp_path: Path) -> None:
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=FakeLLM(),
        doi_resolver=FakeResolver(),
    )

    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="reviewed",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )

    assert result.review is not None
    assert result.review.decision == "minor_revision"
    assert len(result.review.reports) == 1  # default reviewer_count is 1

    review_dir = tmp_path / "projects/reviewed/artifacts/review"
    payload = json.loads((review_dir / "review_reports.json.v1").read_text(encoding="utf-8"))
    assert payload["decision"] == "minor_revision"
    assert (review_dir / "review_reports.md.v1").exists()

    # A review is advisory: a non-blocking verdict must never block the run.
    assert result.manuscript_path.exists()


def test_peer_review_failure_does_not_break_the_pipeline(tmp_path: Path) -> None:
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=FailingReviewLLM(),
        doi_resolver=FakeResolver(),
    )

    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="noreview",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )

    assert result.review is None
    assert result.manuscript_path.exists()
    assert any("模拟同行评审未执行" in warning for warning in result.warnings)


def test_peer_review_can_be_disabled(tmp_path: Path) -> None:
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=FailingReviewLLM(),
        doi_resolver=FakeResolver(),
        reviewer_count=0,
    )

    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="noreview0",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )

    assert result.review is None
    assert not any("模拟同行评审未执行" in warning for warning in result.warnings)
    assert not (tmp_path / "projects/noreview0/artifacts/review").exists()


def test_invalid_settings_fail_fast_before_any_work(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        "review:\n  reviewer_count: three\n", encoding="utf-8"
    )

    projects_dir = tmp_path / "projects"
    state_manager = StateManager(projects_dir)
    state_manager.save(
        ProjectState(
            name="badcfg",
            mode="hybrid",
            current_stage=WorkflowStage.BRAINSTORMING,
        )
    )

    with pytest.raises(ConfigValidationError, match="reviewer_count"):
        ResearchService(projects_dir, state_manager, ArtifactStore(projects_dir)).run(
            "badcfg", "a topic", sources=["crossref"], max_results=1
        )

    # Failing fast means no stage was mutated and no network work was attempted.
    state = state_manager.load("badcfg")
    assert state is not None
    assert state.stage_status[WorkflowStage.SEARCH] == "pending"


def test_pipeline_runs_claim_evidence_gate(tmp_path: Path) -> None:
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=FakeLLM(),
        doi_resolver=FakeResolver(),
    )

    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="claims",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )

    writing = tmp_path / "projects/claims/artifacts/writing"
    payload = json.loads(
        (writing / "claim_evidence_verification.json.v1").read_text(encoding="utf-8")
    )
    assert payload["claim_count"] == 1
    assert payload["unsupported_count"] == 0
    assert payload["passed"] is True
    assert (writing / "claim_evidence_verification.md.v1").is_file()
    assert result.review is not None


def test_unsupported_claim_is_advisory_and_reaches_the_review(tmp_path: Path) -> None:
    """An unsupported claim must surface as a warning — never abort the run."""

    class UnsupportedLLM(FakeLLM):
        def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
            if "事实核查员" in system_prompt:
                return {
                    "verdicts": [
                        {
                            "claim_index": 0,
                            "claim": "Adaptation is context-dependent",
                            "citation_id": "P1",
                            "verdict": "unsupported",
                            "quote": "A finding.",
                            "rationale": "摘要与该论断无关。",
                        }
                    ]
                }
            return super().complete_json(system_prompt, user_prompt)

    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=UnsupportedLLM(),
        doi_resolver=FakeResolver(),
    )

    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="unsupported",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )

    # Advisory by design: the run completes and the manuscript is still produced.
    assert result.manuscript_path.exists()

    payload = json.loads(
        (
            tmp_path
            / "projects/unsupported/artifacts/writing/claim_evidence_verification.json.v1"
        ).read_text(encoding="utf-8")
    )
    assert payload["passed"] is False
    assert payload["unsupported_count"] == 1
    assert any("论断—证据核验" in warning for warning in result.warnings)

    # The cross-module contract: the peer review must pick the failure up.
    assert result.review is not None
    concerns = [
        concern for report in result.review.reports for concern in report.concerns
    ]
    assert any("unsupported_count" in concern.evidence for concern in concerns)


def _legacy_state_payload() -> dict:
    """历史 state.json 的等效载荷：含已删除阶段，且停在已删除的 current_stage。"""
    return {
        "name": "legacy",
        "mode": "hybrid",
        "current_stage": "polishing",  # 已删除的阶段
        "stage_status": {
            "brainstorming": "completed",
            "search": "completed",
            "lit_review": "completed",
            "statistics": "pending",
            "visualization": "pending",
            "writing": "in_progress",
            "polishing": "pending",  # 已删除的阶段
            "review": "pending",  # 已删除的阶段
            "export": "pending",
        },
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "metadata": {},
    }


def test_load_tolerates_removed_stage_names(tmp_path: Path) -> None:
    """旧 state.json 含已删除阶段名时仍可加载，且 current_stage 永不落空。

    删除 WorkflowStage 成员是数据迁移：历史项目文件里可能残留
    statistics/visualization/polishing/review，甚至当前阶段就停在已删除的
    阶段上。加载必须跳过未知阶段并确定性地收敛 current_stage，否则
    `state.current_stage.value`（CLI 与 Web）会在“加载成功”后崩溃。
    """
    projects_dir = tmp_path / "projects"
    project_dir = projects_dir / "legacy"
    project_dir.mkdir(parents=True)
    (project_dir / "state.json").write_text(
        json.dumps(_legacy_state_payload(), ensure_ascii=False), encoding="utf-8"
    )

    state = StateManager(projects_dir).load("legacy")

    assert state is not None
    # current_stage 必须落在有效成员上（当前阶段是已删除的 polishing）。
    assert state.current_stage is WorkflowStage.WRITING
    # 每个有效阶段都存在，且有效阶段的原始状态被保留。
    assert set(state.stage_status) == set(WorkflowStage)
    assert state.stage_status[WorkflowStage.BRAINSTORMING] == "completed"
    assert state.stage_status[WorkflowStage.SEARCH] == "completed"
    assert state.stage_status[WorkflowStage.LIT_REVIEW] == "completed"
    assert state.stage_status[WorkflowStage.WRITING] == "in_progress"
    assert state.stage_status[WorkflowStage.EXPORT] == "pending"


def test_current_stage_falls_back_to_export_when_everything_completed(
    tmp_path: Path,
) -> None:
    """未知 current_stage 且无 in_progress/未完成阶段时，收敛到 EXPORT。"""
    projects_dir = tmp_path / "projects"
    project_dir = projects_dir / "done"
    project_dir.mkdir(parents=True)
    payload = _legacy_state_payload()
    payload["current_stage"] = "polishing"
    payload["stage_status"] = {key: "completed" for key in payload["stage_status"]}
    (project_dir / "state.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    state = StateManager(projects_dir).load("done")

    assert state is not None
    assert state.current_stage is WorkflowStage.EXPORT


def test_build_llm_client_honours_timeout_for_every_provider(tmp_path: Path) -> None:
    """`llm.timeout_seconds` 必须对全部 provider 生效，而不只是默认的那几个。

    codex_cli 是 settings.yaml 与 README 中的默认 provider；若它的分支忽略
    timeout，则该配置项对多数用户形同虚设。
    """
    for provider in ("openai_compatible", "gemini", "codex_cli"):
        client = build_llm_client(
            provider=provider, api_key="test-key", timeout=42, workspace_dir=tmp_path
        )
        assert client.settings.timeout == 42, provider
