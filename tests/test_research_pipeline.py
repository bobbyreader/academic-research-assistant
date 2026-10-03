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
    ResearchPipelineCancelled,
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
        self,
        query: str,
        sources: list[str],
        max_results: int,
        year_range: tuple[int, int] | list[int] | None = None,
    ) -> SearchReport:
        assert query == "climate adaptation"
        assert sources == ["crossref"]
        assert max_results == 2
        # 缺省年份窗口 (0, 0) 必须原样透传（= 不过滤，与改动前一致）。
        assert year_range == (0, 0)
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
    # Phase 7 修复：关口抛错时**也必须落盘**，且产物如实记录 ran=False。
    # 旧行为（不落盘）会让 resume 永久无法复用 writing——一次瞬时故障之后每次都
    # 重新付费，而运行却显示“成功”。
    writing = tmp_path / "projects/gateboom/artifacts/writing"
    payload = json.loads(
        (writing / "manuscript_claim_verification.json.v1").read_text(encoding="utf-8")
    )
    assert payload["ran"] is False
    assert payload["not_run_reason"]
    # ran=False 时 passed 必须为 False：「未执行」绝不能被读作「检查通过」。
    assert payload["passed"] is False
    assert (writing / "manuscript_claim_verification.md.v1").is_file()


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


# --------------------------------------------------------------------------- #
# P5.6 任务 1：协作式取消（阶段边界生效）
# --------------------------------------------------------------------------- #

#: 含引用标识的正文：让正文级论断核验真的发起一次模型调用（否则调用次数会因
#: 正文有无 [P1] 而不同，掩盖"用量统计是否改变调用次数"的判定）。
BODY = "# Draft\n\nA difference was observed [P1].\n"


class CountingSearcher(FakeSearcher):
    """记录 `search()` 被调用的次数，用于断言阶段是否被进入。"""

    def __init__(self) -> None:
        self.calls = 0

    def search(
        self,
        query: str,
        sources: list[str],
        max_results: int,
        year_range: tuple[int, int] | list[int] | None = None,
    ) -> SearchReport:
        self.calls += 1
        return super().search(query, sources, max_results, year_range=year_range)


class CountingLLM(FakeLLM):
    """记录模型调用次数，用于断言取消/用量统计不改变调用次数。"""

    def __init__(self, body: str = BODY) -> None:
        super().__init__(body=body)
        self.complete_calls = 0
        self.complete_json_calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.complete_calls += 1
        return super().complete(system_prompt, user_prompt)

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.complete_json_calls += 1
        return super().complete_json(system_prompt, user_prompt)


def test_should_continue_false_stops_before_analysis(tmp_path: Path) -> None:
    """`should_continue` 返回 False → 抛 `ResearchPipelineCancelled`。

    取消在**阶段边界**生效：这里让它恰好在 search 边界后返回 False，于是 search
    已经跑完（并落盘），但 analysis 阶段**绝不被进入**——用计数桩证明 analysis 的
    模型调用为 0。

    Phase 7：相关性关口在 **search 阶段内**执行，恰好一次 `complete_json`；这**不是**
    “进入了 analysis”。因此这里断言的是「相对 search 结束时不再新增调用」，而不是
    「调用总数为 0」。
    """
    searcher = CountingSearcher()
    llm = CountingLLM()
    calls = {"n": 0}

    def should_continue() -> bool:
        # 第一次边界（search 之前）放行；第二次边界（analysis 之前）取消。
        calls["n"] += 1
        return calls["n"] < 2

    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=searcher,
        llm_client=llm,
        doi_resolver=FakeResolver(),
        should_continue=should_continue,
    )

    with pytest.raises(ResearchPipelineCancelled):
        pipeline.run(
            ResearchPipelineConfig(
                project_name="cancel1",
                topic="climate adaptation",
                sources=["crossref"],
                max_results=2,
            )
        )

    assert searcher.calls == 1, "search 阶段应已执行"
    # search 阶段内的相关性关口恰好一次模型调用（Phase 7）；analysis 未进入，
    # 因此总数不应超过这一次。
    assert llm.complete_json_calls == 1, (
        "取消后只应有 search 阶段内的相关性关口调用；analysis 不得进入"
    )
    assert llm.complete_calls == 0, "取消后不得进入 writing"
    # 取消不留下半成品：已完成的 search 阶段产物仍在磁盘上。
    literature = tmp_path / "projects/cancel1/artifacts/search/literature.json.v1"
    assert literature.is_file()
    # 相关性关口在 search 阶段内已落盘（续跑据此可复用 search）。
    relevance = tmp_path / "projects/cancel1/artifacts/search/relevance_check.json.v1"
    assert relevance.is_file()
    # 未进入的阶段没有产物。
    assert not (tmp_path / "projects/cancel1/artifacts/analysis").exists()


def test_cancellation_is_observable_through_service_as_cancelled(
    tmp_path: Path, monkeypatch
) -> None:
    """服务层：取消时状态为 `"cancelled"` 而非 `"blocked"`，并原样重新抛出。

    取消是用户**主动行为**，不是错误——把它记成 `blocked` 会让用户以为运行失败。
    """
    monkeypatch.setenv("ARS_LLM_PROVIDER", "codex_cli")
    projects_dir = tmp_path / "projects"
    state_manager = StateManager(projects_dir)
    state_manager.save(
        ProjectState(
            name="cancel-svc",
            mode="hybrid",
            current_stage=WorkflowStage.BRAINSTORMING,
        )
    )
    service = ResearchService(projects_dir, state_manager, ArtifactStore(projects_dir))

    # 第一次边界放行，之后取消：运行会在 search 阶段内继续（search 无法取消），
    # 但在进入 analysis 前停止。
    calls = {"n": 0}

    def should_continue() -> bool:
        calls["n"] += 1
        return calls["n"] < 2

    # 用假管线替代真实管线，避免真实外部调用；只验证服务层的取消处理契约。
    from core import research_service as rs

    class StubPipeline:
        def __init__(self, *args: object, **kwargs: object) -> None:
            assert kwargs.get("should_continue") is should_continue

        def run(self, config: object) -> object:
            raise ResearchPipelineCancelled("cancelled at boundary")

    monkeypatch.setattr(rs, "ResearchPipeline", StubPipeline)

    with pytest.raises(ResearchPipelineCancelled):
        service.run(
            "cancel-svc",
            "a topic",
            sources=["crossref"],
            max_results=1,
            should_continue=should_continue,
        )

    state = state_manager.load("cancel-svc")
    assert state is not None
    assert state.stage_status[WorkflowStage.SEARCH] == "cancelled"
    assert state.stage_status[WorkflowStage.SEARCH] != "blocked"


def test_cancellation_does_not_block_resume_of_completed_prefix(
    tmp_path: Path,
) -> None:
    """取消绝不阻断续跑：取消 → 再次运行 → 前段被复用。

    取消发生在阶段边界，已完成的阶段产物完整且可复用。第二次运行时 search 阶段
    必须被复用（`reused_steps` 含 "search"），且外部检索调用数为 0。
    """
    from core.resume import RunFingerprint, plan_resume
    from core.state_manager import WorkflowStage as _WS  # noqa: F401

    store = ArtifactStore(tmp_path / "projects")
    searcher = CountingSearcher()

    # 第一次运行：在 search 之后取消。
    calls = {"n": 0}

    def stop_after_search() -> bool:
        calls["n"] += 1
        return calls["n"] < 2

    first = ResearchPipeline(
        store,
        searcher=searcher,
        llm_client=CountingLLM(),
        doi_resolver=FakeResolver(),
        should_continue=stop_after_search,
    )
    with pytest.raises(ResearchPipelineCancelled):
        first.run(
            ResearchPipelineConfig(
                project_name="resume-cancel",
                topic="climate adaptation",
                sources=["crossref"],
                max_results=2,
            )
        )
    assert searcher.calls == 1

    # 第二次运行：复用已完成的 search 阶段（指纹一致、产物齐备）。
    fingerprint = RunFingerprint.build(
        topic="climate adaptation",
        sources=["crossref"],
        max_results=2,
        data_path=None,
        reviewer_count=1,
        figure_dpi=300,
    )
    decision = plan_resume(
        artifact_store=store,
        project_name="resume-cancel",
        fingerprint=fingerprint,
        previous_steps={"search": fingerprint},
        enabled=True,
    )
    assert decision.can_skip("search"), "取消后 search 产物应可复用"

    llm2 = CountingLLM()
    second = ResearchPipeline(
        store,
        searcher=searcher,
        llm_client=llm2,
        doi_resolver=FakeResolver(),
        resume=decision,
    )
    result = second.run(
        ResearchPipelineConfig(
            project_name="resume-cancel",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )

    assert "search" in result.reused_steps
    assert searcher.calls == 1, "复用 search 阶段不得新增外部检索调用"
    # 前段复用后，后续阶段从未产出；本次运行把它们补齐。
    assert result.manuscript_path.exists()


# --------------------------------------------------------------------------- #
# P5.6 任务 2：用量落盘（成功与取消两条路径）
# --------------------------------------------------------------------------- #


class _FakeUsageReport:
    """最小化的 `UsageReport` 鸭子类型：只提供管线会读取的三个方法。"""

    def __init__(self, note: str, warnings: list[str], payload: dict) -> None:
        self._note = note
        self._warnings = warnings
        self._payload = payload

    def note(self) -> str:
        return self._note

    def warnings(self) -> list[str]:
        return list(self._warnings)

    def to_dict(self) -> dict:
        return dict(self._payload)


class UsageTrackingFakeLLM(CountingLLM):
    """在完整转发的同时暴露一个 `report` 属性，模拟 UsageTrackingClient。"""

    def __init__(self, report: _FakeUsageReport, body: str = BODY) -> None:
        super().__init__(body=body)
        self.report = report


def _usage_report() -> _FakeUsageReport:
    return _FakeUsageReport(
        note="本次运行共 3 次模型调用，2 次上报用量（prompt=10, completion=5, total=15）。",
        warnings=["1 次调用未上报用量，已如实标注，未做估算。"],
        # `records` 才是逐次调用明细（`calls` 是计数，是整数）。
        payload={
            "calls": 3,
            "records": [
                {"operation": "complete_json", "reported": True, "total_tokens": 15},
                {"operation": "complete", "reported": True, "total_tokens": 12},
                {"operation": "complete_json", "reported": False, "total_tokens": None},
            ],
        },
    )


def test_usage_report_is_persisted_on_success(tmp_path: Path) -> None:
    """成功路径：用量落盘为 artifacts/run/usage_report.json 与 .md，且注意一致。"""
    report = _usage_report()
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=CountingSearcher(),
        llm_client=UsageTrackingFakeLLM(report),
        doi_resolver=FakeResolver(),
    )

    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="usage-ok",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )

    run_dir = tmp_path / "projects/usage-ok/artifacts/run"
    assert (run_dir / "usage_report.json.v1").is_file()
    assert (run_dir / "usage_report.md.v1").is_file()
    payload = json.loads((run_dir / "usage_report.json.v1").read_text(encoding="utf-8"))
    assert payload == report.to_dict()
    markdown = (run_dir / "usage_report.md.v1").read_text(encoding="utf-8")
    assert report.note() in markdown
    # 逐次调用明细必须来自 `records` 列表（`calls` 是整数计数，不是列表）。
    assert "## 逐次调用明细" in markdown
    assert "operation=complete_json" in markdown
    # **不得静默跳过**：marginal 明细条数必须等于 records 条数。若将来键名再漂移
    # （例如退回读 `calls`），这里会失败，而不是悄悄少写一大块内容。
    detail_lines = [
        line for line in markdown.splitlines() if line.startswith("- 调用 ")
    ]
    assert len(detail_lines) == len(report.to_dict()["records"]), (
        f"逐次明细条数({len(detail_lines)})！= records 条数"
        f"({len(report.to_dict()['records'])})：markdown 静默丢数据"
    )
    # `usage_note` 必须与 `UsageReport.note()` 逐字一致（绝不加工）。
    assert result.usage_note == report.note()
    # 用量的 warnings 透传进结果 warnings。
    assert any("未上报用量" in warning for warning in result.warnings)


def test_usage_report_is_persisted_on_cancellation(tmp_path: Path) -> None:
    """取消路径：用量同样落盘——部分运行的用量是一笔真实开销，不能丢。"""
    report = _usage_report()
    llm = UsageTrackingFakeLLM(report)
    calls = {"n": 0}

    def stop_after_search() -> bool:
        calls["n"] += 1
        return calls["n"] < 2

    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=CountingSearcher(),
        llm_client=llm,
        doi_resolver=FakeResolver(),
        should_continue=stop_after_search,
    )

    with pytest.raises(ResearchPipelineCancelled):
        pipeline.run(
            ResearchPipelineConfig(
                project_name="usage-cancel",
                topic="climate adaptation",
                sources=["crossref"],
                max_results=2,
            )
        )

    run_dir = tmp_path / "projects/usage-cancel/artifacts/run"
    assert (run_dir / "usage_report.json.v1").is_file(), (
        "取消路径也必须落盘用量"
    )
    assert (run_dir / "usage_report.md.v1").is_file()
    payload = json.loads((run_dir / "usage_report.json.v1").read_text(encoding="utf-8"))
    assert payload == report.to_dict()


def test_usage_tracking_does_not_change_model_call_counts(tmp_path: Path) -> None:
    """用量统计是**纯观测**：包不包装，模型调用次数必须完全一致。"""
    def run_with(project_name: str, client: object) -> tuple[int, int]:
        pipeline = ResearchPipeline(
            ArtifactStore(tmp_path / "projects"),
            searcher=CountingSearcher(),
            llm_client=client,  # type: ignore[arg-type]
            doi_resolver=FakeResolver(),
        )
        pipeline.run(
            ResearchPipelineConfig(
                project_name=project_name,
                topic="climate adaptation",
                sources=["crossref"],
                max_results=2,
            )
        )
        counted = client
        assert isinstance(counted, CountingLLM)
        return counted.complete_calls, counted.complete_json_calls

    plain = CountingLLM()
    plain_counts = run_with("count-plain", plain)

    tracked = UsageTrackingFakeLLM(_usage_report())
    tracked_counts = run_with("count-tracked", tracked)

    assert plain_counts == tracked_counts, (
        "用量统计改变了模型调用次数："
        f"plain={plain_counts} tracked={tracked_counts}"
    )


def test_malformed_usage_records_are_not_silently_skipped(tmp_path: Path) -> None:
    """`records` 存在但不是列表 → 视作**结构异常**，显式失败而非静默降级。

    这直接对应"绝不静默丢弃"的项目标准：把结构损坏当成"没有明细"悄悄吞掉，会让
    产物看起来正常却少了一大块内容。渲染器必须在落盘阶段就暴露问题。
    """
    broken = _FakeUsageReport(
        note="用量摘要",
        warnings=[],
        payload={"calls": 1, "records": "not-a-list"},
    )
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=CountingSearcher(),
        llm_client=UsageTrackingFakeLLM(broken),
        doi_resolver=FakeResolver(),
    )

    with pytest.raises(ResearchPipelineError, match="records"):
        pipeline.run(
            ResearchPipelineConfig(
                project_name="usage-broken",
                topic="climate adaptation",
                sources=["crossref"],
                max_results=2,
            )
        )


def test_no_usage_report_is_silent_and_behavior_unchanged(tmp_path: Path) -> None:
    """无 `report`（裸客户端）时：不落盘用量、`usage_note` 为空、行为不变。"""
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=CountingSearcher(),
        llm_client=FakeLLM(),
        doi_resolver=FakeResolver(),
    )
    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="nounit",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )
    assert result.usage_note == ""
    assert not (tmp_path / "projects/nounit/artifacts/run").exists()


# --------------------------------------------------------------------------- #
# Phase 7：相关性关口（顾问级）——在 search 阶段内、且在 progress 之前落盘
# --------------------------------------------------------------------------- #
class RelevanceRecordingLLM(CountingLLM):
    """记录「每次 complete_json 收到的提示词」，用于精确断言调用来自哪个关口。"""

    def __init__(self, body: str = BODY) -> None:
        super().__init__(body=body)
        self.system_prompts: list[str] = []
        self.user_prompts: list[str] = []

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.system_prompts.append(system_prompt)
        self.user_prompts.append(user_prompt)
        if "相关性评审员" in system_prompt:
            return {
                "verdicts": [
                    {
                        "citation_id": "P1",
                        "relevance": "relevant",
                        "rationale": "标题与主题直接相关。",
                    }
                ]
            }
        return super().complete_json(system_prompt, user_prompt)


def test_relevance_artifacts_persist_before_search_completed_signal(
    tmp_path: Path,
) -> None:
    """相关性产物必须在 `progress("search","completed")` **之前**已落盘。

    这是本阶段最容易做错的地方：`completed` 是 P5.5 记录"该阶段产物已落盘"的时刻。
    这里在 on_progress 回调里、**那一刻**检查文件已在磁盘上；若落盘晚于该回调，本
    测试必然失败——从而锁死"指纹已记录、产物还不存在"这个窗口。
    """
    observed: dict[str, bool] = {}

    def on_progress(stage: str, status: str) -> None:
        if stage == "search" and status == "completed":
            search_dir = tmp_path / "projects/relseq/artifacts/search"
            observed["json"] = (search_dir / "relevance_check.json.v1").is_file()
            observed["md"] = (search_dir / "relevance_check.md.v1").is_file()

    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=CountingSearcher(),
        llm_client=RelevanceRecordingLLM(),
        doi_resolver=FakeResolver(),
        progress=on_progress,
    )
    pipeline.run(
        ResearchPipelineConfig(
            project_name="relseq",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )

    assert observed.get("json") is True, (
        "相关性 JSON 必须在 progress('search','completed') 之前落盘"
    )
    assert observed.get("md") is True, (
        "相关性 Markdown 必须在 progress('search','completed') 之前落盘"
    )


def test_relevance_gate_exactly_one_model_call(tmp_path: Path) -> None:
    """相关性关口恰好贡献 **1** 次模型调用（非空文献时），检索阶段内完成。"""
    llm = RelevanceRecordingLLM()
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=CountingSearcher(),
        llm_client=llm,
        doi_resolver=FakeResolver(),
        reviewer_count=0,
    )
    pipeline.run(
        ResearchPipelineConfig(
            project_name="relcost",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )

    relevance_calls = [
        prompt for prompt in llm.system_prompts if "相关性评审员" in prompt
    ]
    assert len(relevance_calls) == 1, "相关性关口必须恰好一次模型调用"


def test_relevance_gate_never_deletes_papers(tmp_path: Path) -> None:
    """相关性关口**只标记、绝不删除文献**：文献数与不带该关口时一致。"""
    llm = RelevanceRecordingLLM()
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=CountingSearcher(),
        llm_client=llm,
        doi_resolver=FakeResolver(),
    )
    result = pipeline.run(
        ResearchPipelineConfig(
            project_name="relkeep",
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
        )
    )

    # 检索到的 1 篇文献必须原样保留（无论相关判定如何）。
    literature = json.loads(
        (
            tmp_path / "projects/relkeep/artifacts/search/literature.json.v1"
        ).read_text(encoding="utf-8")
    )
    assert len(literature) == 1
    assert len(result.papers) == 1


def test_advisory_gate_failure_still_persists_and_enables_resume(
    tmp_path: Path,
) -> None:
    """核心修复：顾问级关口抛错时**仍落盘**，且随后续跑能复用 writing。

    旧缺陷：关口抛错 → 不落盘 → `core/resume.py` 因缺产物而永久拒绝复用 writing，
    一次瞬时故障之后每次都重新付费，而运行仍显示“成功”。本测试同时锁定：
    1. 抛错时产物仍存在，且 `ran=False` + `not_run_reason` 非空；
    2. 再次运行（无故障）时 writing 可被复用（外部模型调用不再增加）。
    """
    from core import research_pipeline as rp

    original = rp.verify_manuscript_claims

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("manuscript gate model unavailable")

    # --- 第一次：关口故障，但必须落盘并完成运行 -----------------------------
    rp.verify_manuscript_claims = boom  # type: ignore[assignment]
    try:
        first = ResearchPipeline(
            ArtifactStore(tmp_path / "projects"),
            searcher=CountingSearcher(),
            llm_client=RelevanceRecordingLLM(),
            doi_resolver=FakeResolver(),
        ).run(
            ResearchPipelineConfig(
                project_name="gatepersist",
                topic="climate adaptation",
                sources=["crossref"],
                max_results=2,
            )
        )
    finally:
        rp.verify_manuscript_claims = original  # type: ignore[assignment]

    assert first.manuscript_path.exists()
    writing = tmp_path / "projects/gatepersist/artifacts/writing"
    payload = json.loads(
        (writing / "manuscript_claim_verification.json.v1").read_text(encoding="utf-8")
    )
    assert payload["ran"] is False
    assert payload["not_run_reason"]
    assert payload["passed"] is False
    assert (writing / "manuscript_claim_verification.md.v1").is_file()

    # --- 第二次：无故障，writing 必须能被复用 ------------------------------
    from core.resume import RESUME_STEPS, RunFingerprint, plan_resume

    store = ArtifactStore(tmp_path / "projects")
    decision = plan_resume(
        artifact_store=store,
        project_name="gatepersist",
        fingerprint=RunFingerprint.build(
            topic="climate adaptation",
            sources=["crossref"],
            max_results=2,
            data_path=None,
            reviewer_count=1,
            figure_dpi=300,
        ),
        previous_steps={
            marker: RunFingerprint.build(
                topic="climate adaptation",
                sources=["crossref"],
                max_results=2,
                data_path=None,
                reviewer_count=1,
                figure_dpi=300,
            )
            for marker in ("search", "lit_review", "writing")
        },
        enabled=True,
    )
    assert decision.can_skip("writing"), (
        "关口抛错后 writing 阶段仍必须可复用（产物已如实落盘）"
    )
    assert set(decision.reusable) == set(RESUME_STEPS)


# --------------------------------------------------------------------------- #
# Phase 8 交付层：writing.* / review.* / export.* / citation_style 真实生效
# --------------------------------------------------------------------------- #

#: 引入参数化之前的写作系统提示词逐字文本——所有"缺省不变"断言的基线。
_BASELINE_WRITING_SYSTEM_PROMPT = (
    "你是学术写作助手。请写一份基于文献的研究综述/研究计划草稿，而不是"
    "冒充已经完成的实验论文。不得编造数据、结果、引用或 DOI。正文引用只能使用"
    "[P1]、[P2] 这样的标识，并且只能引用给定资料。明确区分已发表证据、推断和待验证方案。"
    "如果系统提供了已计算的统计结果，只能原样引用其中的数值（p 值、效应量、n），"
    "不得自行计算、四舍五入或改写。不要自行生成 References、统计表或图表清单章节，"
    "这些内容由系统附加。请只输出 Markdown 正文。"
)


def _config(**overrides: object) -> ResearchPipelineConfig:
    base: dict[str, object] = {
        "project_name": "cfg",
        "topic": "climate adaptation",
        "sources": ["crossref"],
        "max_results": 2,
    }
    base.update(overrides)
    return ResearchPipelineConfig(**base)  # type: ignore[arg-type]


def test_writing_prompt_is_byte_identical_when_keys_default() -> None:
    """所有 writing.* 键缺省时，写作提示词必须与改动前**逐字节**一致。"""
    prompt = ResearchPipeline._writing_system_prompt(_config())

    assert prompt == _BASELINE_WRITING_SYSTEM_PROMPT


def test_writing_paper_type_language_abstract_style_change_prompt() -> None:
    """四个 writing.* 键都必须真实改变提示词，而非"读了没用"。"""
    prompt = ResearchPipeline._writing_system_prompt(
        _config(
            writing_paper_type="review",
            writing_language="en",
            writing_bilingual_abstract=True,
            writing_style_guide="避免使用第一人称",
        )
    )

    assert prompt != _BASELINE_WRITING_SYSTEM_PROMPT
    assert "综述" in prompt  # paper type
    assert "英文" in prompt  # language
    assert "中英双语摘要" in prompt  # bilingual abstract
    assert "避免使用第一人称" in prompt  # style guide
    # 完整性约束一个字都不能弱化。
    assert "不得编造数据、结果、引用或 DOI" in prompt
    assert "只能引用给定资料" in prompt


def test_writing_system_prompt_reaches_the_model(tmp_path: Path) -> None:
    """写入配置后，真实链路里传给模型的系统提示词必须真的带上这些约束。"""
    seen: dict[str, str] = {}

    class RecordingLLM(FakeLLM):
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            seen["system"] = system_prompt
            return super().complete(system_prompt, user_prompt)

    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=RecordingLLM(),
        doi_resolver=FakeResolver(),
        reviewer_count=0,
    )
    pipeline.run(
        _config(
            project_name="writing-cfg",
            writing_paper_type="letter",
            writing_bilingual_abstract=True,
        )
    )

    assert "快报" in seen["system"]
    assert "中英双语摘要" in seen["system"]


def test_review_devil_advocate_instruction_appears_in_prompt(tmp_path: Path) -> None:
    """include_devil_advocate=True 时审稿人提示词必须要求一条明确的反对意见。"""
    prompts: list[str] = []

    class RecordingLLM(FakeLLM):
        def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
            if "审稿人" in system_prompt:
                prompts.append(user_prompt)
            return super().complete_json(system_prompt, user_prompt)

    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=RecordingLLM(),
        doi_resolver=FakeResolver(),
        reviewer_count=1,
    )
    pipeline.run(
        _config(project_name="devil", review_include_devil_advocate=True)
    )

    assert prompts, "the reviewer must have been called"
    assert "反对意见" in prompts[0]


def test_review_default_has_no_devil_advocate_text() -> None:
    """缺省时提示词中不得出现反对意见要求（逐字节一致的必要条件）。"""
    from core.peer_reviewer import _build_user_prompt

    prompt = _build_user_prompt(
        role="methodology",
        topic="t",
        manuscript_body="body",
        citation_verification={},
        statistics_verification={},
        statistics_report={},
        claim_verification={},
    )
    assert "反对意见" not in prompt
    assert '"score": 0-100' in prompt


def test_review_consensus_threshold_flags_divergence(tmp_path: Path) -> None:
    """consensus_threshold 必须真实生效：分歧时综合意见里标注"评审意见分歧"。"""
    class DivergingLLM(FakeLLM):
        def __init__(self, body: str | None = None) -> None:
            super().__init__(body=body or "# Climate adaptation\n\n## Abstract\nA grounded draft.")

        def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
            payload = super().complete_json(system_prompt, user_prompt)
            # 第 3 位角色（novelty）给出不同的建议 → 3 人中 2 人一致 = 0.67。
            if "审稿人" in system_prompt and "你的审稿角色：novelty" in user_prompt:
                payload = dict(payload, recommendation="reject")
            return payload

    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=DivergingLLM(),
        doi_resolver=FakeResolver(),
        reviewer_count=3,
    )
    result = pipeline.run(
        _config(project_name="diverge", review_consensus_threshold=0.9)
    )

    assert result.review is not None
    assert "评审意见分歧" in result.review.synthesis


def test_review_consensus_threshold_default_does_not_flag(tmp_path: Path) -> None:
    """缺省阈值 0.0 时永不标注分歧——保证综合意见与改动前逐字节一致。"""
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=FakeLLM(),
        doi_resolver=FakeResolver(),
        reviewer_count=3,
    )
    result = pipeline.run(_config(project_name="nodiverge"))

    assert result.review is not None
    assert "评审意见分歧" not in result.review.synthesis


def test_review_score_scale_appears_in_prompt_and_artifact(tmp_path: Path) -> None:
    """score_scale 必须同时影响提示词刻度与落盘 Markdown 的刻度标注。"""
    prompts: list[str] = []

    class RecordingLLM(FakeLLM):
        def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
            if "审稿人" in system_prompt:
                prompts.append(user_prompt)
            return super().complete_json(system_prompt, user_prompt)

    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=RecordingLLM(),
        doi_resolver=FakeResolver(),
        reviewer_count=1,
    )
    pipeline.run(_config(project_name="scale", review_score_scale="0-10"))

    assert prompts and '"score": 0-10' in prompts[0]
    markdown = (
        tmp_path / "projects/scale/artifacts/review/review_reports.md.v1"
    ).read_text(encoding="utf-8")
    assert "评分: 65/10" in markdown

    payload = json.loads(
        (tmp_path / "projects/scale/artifacts/review/review_reports.json.v1").read_text(
            encoding="utf-8"
        )
    )
    assert payload["score_scale"] == "0-10"


def test_review_default_score_scale_markdown_is_byte_identical(tmp_path: Path) -> None:
    """缺省 score_scale 时评审 Markdown 的评分标注仍为 ``/100``。"""
    pipeline = ResearchPipeline(
        ArtifactStore(tmp_path / "projects"),
        searcher=FakeSearcher(),
        llm_client=FakeLLM(),
        doi_resolver=FakeResolver(),
        reviewer_count=1,
    )
    pipeline.run(_config(project_name="scale-default"))

    markdown = (
        tmp_path / "projects/scale-default/artifacts/review/review_reports.md.v1"
    ).read_text(encoding="utf-8")
    assert "评分: 65/100" in markdown


# ------------------------------------------------------------------------------------- #
# export.* 生效
# ------------------------------------------------------------------------------------- #
def test_export_resolve_format_prefers_requested_then_default() -> None:
    from core.export_service import ExportError, resolve_export_format

    assert resolve_export_format("pdf", "md") == "pdf"
    assert resolve_export_format("", "pptx") == "pptx"
    assert resolve_export_format("", "") == "md"  # 历史行为
    with pytest.raises(ExportError):
        resolve_export_format("docx", "md")


def test_export_whitelist_rejects_format_outside_citation_export_formats() -> None:
    """`citation.export_formats: [pdf]` + 请求 `pptx` → 抛错且点名叫出该配置。"""
    from core.export_service import ExportError, resolve_export_format

    with pytest.raises(ExportError) as excinfo:
        resolve_export_format("pptx", "md", allowed_formats=["pdf"])

    message = str(excinfo.value)
    assert "citation.export_formats" in message, "错误消息必须点名叫出该配置项"
    assert "pptx" in message, "错误消息必须点名被拒的具体格式"


def test_export_whitelist_permits_format_in_list() -> None:
    """白名单内的格式正常通过；空列表视作不限制。"""
    from core.export_service import resolve_export_format

    assert resolve_export_format("pdf", "md", allowed_formats=["pdf", "md"]) == "pdf"
    # 空列表不构成白名单（避免"配置成空 = 全部禁止"这种反直觉行为）。
    assert resolve_export_format("pdf", "md", allowed_formats=[]) == "pdf"


def test_export_whitelist_none_does_not_constrain_and_is_unchanged() -> None:
    """未配置（allowed_formats=None）时不施加任何限制，行为与改动前一致。"""
    from core.export_service import resolve_export_format

    assert resolve_export_format("pdf", "md", allowed_formats=None) == "pdf"
    assert resolve_export_format("pptx", "") == "pptx"  # 缺省参数不限制
    assert resolve_export_format("", "pdf") == "pdf"


def test_export_pdf_engine_reportlab_forces_python_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """engine=reportlab 必须跳过 pandoc，即便它在 PATH 中。"""
    from core import export_service

    called: list[str] = []
    monkeypatch.setattr(export_service.shutil, "which", lambda _n: "/usr/bin/pandoc")

    def fake_run(*args: object, **kwargs: object) -> object:
        called.append("pandoc")
        raise AssertionError("engine=reportlab 不得调用 pandoc")

    monkeypatch.setattr(export_service.subprocess, "run", fake_run)
    markdown = tmp_path / "m.md"
    markdown.write_text("# T\n\n- a\n", encoding="utf-8")

    pdf = export_service.export_pdf(markdown, tmp_path / "o.pdf", engine="reportlab")

    assert called == []
    assert pdf.read_bytes().startswith(b"%PDF")


def test_export_pdf_engine_pandoc_fails_loudly_when_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """engine=pandoc 且工具缺失时必须显式失败，绝不静默回退 reportlab。"""
    from core import export_service

    monkeypatch.setattr(export_service.shutil, "which", lambda _n: None)
    markdown = tmp_path / "m.md"
    markdown.write_text("# T\n", encoding="utf-8")

    with pytest.raises(export_service.ExportError, match="pandoc"):
        export_service.export_pdf(markdown, tmp_path / "o.pdf", engine="pandoc")


def test_export_pdf_default_engine_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺省 engine=auto 时行为与改动前一致：无系统后端时 reportlab 兜底。"""
    from core import export_service

    monkeypatch.setattr(export_service.shutil, "which", lambda _n: None)
    markdown = tmp_path / "m.md"
    markdown.write_text("# T\n\n- a\n", encoding="utf-8")

    pdf = export_service.export_pdf(markdown, tmp_path / "o.pdf")

    assert pdf.read_bytes().startswith(b"%PDF")


def test_export_pptx_template_and_speaker_notes_effective(tmp_path: Path) -> None:
    """pptx_template 与 include_speaker_notes 必须真实进入导出脚本调用。"""
    from core.export_service import export_pptx

    script = tmp_path / "fake_pptx.py"
    script.write_text(
        "import argparse, pathlib, sys\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('input'); p.add_argument('--output'); p.add_argument('--template')\n"
        "a = p.parse_args()\n"
        "pathlib.Path(a.output).write_bytes(b'PK')\n"
        "pathlib.Path(str(a.output) + '.args').write_text(\n"
        "    (str(a.template) if a.template else '') + '|' + pathlib.Path(a.input).read_text(encoding='utf-8')\n"
        ", encoding='utf-8')\n",
        encoding="utf-8",
    )
    markdown = tmp_path / "outline.md"
    markdown.write_text("# Slide\n- point one\n- point two\n---\n# Second\n- x\n", encoding="utf-8")
    template = tmp_path / "t.pptx"
    template.write_bytes(b"PK")

    out = export_pptx(
        markdown,
        tmp_path / "o.pptx",
        script,
        template_path=template,
        include_speaker_notes=True,
    )

    args_file = Path(str(out) + ".args").read_text(encoding="utf-8")
    template_part, outline_part = args_file.split("|", 1)
    assert template_part == str(template)
    assert "Notes:" in outline_part


def test_export_pptx_defaults_are_unchanged(tmp_path: Path) -> None:
    """缺省（无模板、无备注）时传给导出脚本的大纲与源文件逐字节一致。"""
    from core.export_service import export_pptx

    script = tmp_path / "fake_pptx.py"
    script.write_text(
        "import argparse, pathlib\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('input'); p.add_argument('--output'); p.add_argument('--template')\n"
        "a = p.parse_args()\n"
        "pathlib.Path(a.output).write_bytes(b'PK')\n"
        "pathlib.Path(str(a.output) + '.args').write_text(\n"
        "    (str(a.template) if a.template else '') + '|' + pathlib.Path(a.input).read_text(encoding='utf-8')\n"
        ", encoding='utf-8')\n",
        encoding="utf-8",
    )
    content = "# Slide\n- point one\n"
    markdown = tmp_path / "outline.md"
    markdown.write_text(content, encoding="utf-8")

    out = export_pptx(markdown, tmp_path / "o.pptx", script)

    args_file = Path(str(out) + ".args").read_text(encoding="utf-8")
    template_part, outline_part = args_file.split("|", 1)
    assert template_part == ""
    assert outline_part == content
    assert "Notes:" not in outline_part


def test_citation_style_default_references_byte_identical(tmp_path: Path) -> None:
    """citation_style 缺省时，References 与内联基线逐字节一致。"""
    from core.research_pipeline import ResearchPipeline as RP

    papers = [
        PaperRecord(
            title="Climate adaptation evidence",
            authors=[f"A{i} Author" for i in range(7)],
            year=2024,
            journal="Research Journal",
            doi="10.1234/climate",
            source="crossref",
        ),
        PaperRecord(
            title="No DOI paper",
            authors=["Solo Writer"],
            year=2020,
            journal="",
            url="https://example.org/x",
            source="pubmed",
        ),
    ]

    lines = RP._render_reference_lines(papers, "")

    assert lines[0] == (
        "[P1] A0 Author, A1 Author, A2 Author, A3 Author, A4 Author et al.. "
        "Climate adaptation evidence. *Research Journal*. (2024). "
        "https://doi.org/10.1234/climate"
    )
    assert lines[1] == "[P2] Solo Writer. No DOI paper. (2020). https://example.org/x"


def test_citation_style_empty_and_numeric_are_byte_identical() -> None:
    """``citation_style`` 为 ``""`` 与显式 ``"numeric"`` 都必须与现状逐字节一致。

    真实运行时配置项默认值是 ``"numeric"``（非空），因此两条路径都要覆盖。
    """
    from core.research_pipeline import ResearchPipeline as RP

    papers = [
        PaperRecord(
            title="Climate adaptation evidence",
            authors=["A Author"],
            year=2024,
            journal="Research Journal",
            doi="10.1234/climate",
            source="crossref",
        ),
        PaperRecord(
            title="No DOI paper",
            authors=["Solo Writer"],
            year=2020,
            journal="",
            url="https://example.org/x",
            source="pubmed",
        ),
        # **空作者**边界：现状装配器输出 ``[P3] . Title.``（作者段为空仍留 ". "）。
        # 引用样式渲染器必须复刻这一字节序列，否则"缺省即与改动前逐字节一致"不成立。
        # 该样例故意保留：在 figure-citation-dev 把 numeric 空作者对齐现状之前，本
        # 测试会保持红色——这是**刻意的可见性**，不得为了让测试变绿而删除。
        PaperRecord(
            title="Lost authors paper",
            authors=[],
            year=2021,
            journal="Journal X",
            doi="10.9999/noauth",
            source="crossref",
        ),
    ]

    baseline = RP._inline_reference_lines(papers)

    assert RP._render_reference_lines(papers, "") == baseline
    assert RP._render_reference_lines(papers, "numeric") == baseline


def test_citation_style_author_year_changes_output() -> None:
    """``citation_style="author_year"`` 必须真实改变参考文献渲染（若模块可用）。"""
    from core.research_pipeline import ResearchPipeline as RP

    papers = [
        PaperRecord(
            title="Climate adaptation evidence",
            authors=["A Author"],
            year=2024,
            journal="Research Journal",
            doi="10.1234/climate",
            source="crossref",
        )
    ]

    baseline = RP._inline_reference_lines(papers)
    rendered = RP._render_reference_lines(papers, "author_year")

    try:
        from core import citation_styles

        assert hasattr(citation_styles, "render_references")
    except ImportError:
        assert rendered == baseline, "缺少 citation_styles 时必须回退到内联基线"
    else:
        assert rendered != baseline, "author_year 必须产生与 numeric 不同的条目"


# ------------------------------------------------------------------------------------- #
# figures.* 接线：缺省不传任何样式参数，非缺省才真实进入 build_figures
# ------------------------------------------------------------------------------------- #
def test_figure_style_kwargs_default_is_empty() -> None:
    """figures.* 全缺省时必须返回空 dict —— 调用与改动前逐字节一致。"""
    from core.research_pipeline import ResearchPipeline as RP

    assert RP._figure_style_kwargs(_config()) == {}


def test_figure_style_kwargs_maps_config_to_builder_params() -> None:
    """figures.* 非缺省时必须映射为 build_figures 的关键字参数。"""
    from core.research_pipeline import ResearchPipeline as RP

    style = RP._figure_style_kwargs(
        _config(
            figures_color_palette="colorblind",
            figures_default_format="pdf",
            figures_font_family="DejaVu Sans",
            figures_default_journal="Nature",
            figures_font_size_pt=12,
        )
    )

    assert style == {
        "palette": "colorblind",
        "default_format": "pdf",
        "font_family": "DejaVu Sans",
        "default_journal": "Nature",
        "font_size_pt": 12,
    }


def test_figure_style_kwargs_drops_config_default_palette_and_size() -> None:
    """配置默认值 ``"default"`` 配色不得作为 palette 传入（否则会新增告警）。"""
    from core.research_pipeline import ResearchPipeline as RP

    style = RP._figure_style_kwargs(
        _config(figures_color_palette="default", figures_font_size_pt=10)
    )

    # 字号 10 是后端默认，可传可不传；关键是 "default" 配色必须被丢弃。
    assert "palette" not in style


def test_figure_default_format_changes_real_figure_filenames(tmp_path: Path) -> None:
    """figures_default_format 必须真实改变图件文件名（不是读了没用）。"""
    data = tmp_path / "data.csv"
    data.write_text(
        "group,score\nA,1\nA,2\nA,3\nA,4\nA,5\nB,6\nB,7\nB,8\nB,9\nB,10\n",
        encoding="utf-8",
    )

    def figures_for(project: str, default_format: str) -> dict:
        pipeline = ResearchPipeline(
            ArtifactStore(tmp_path / "projects"),
            searcher=FakeSearcher(),
            llm_client=FakeLLM(body="# Draft\n\nA difference was observed [P1] (p = 0.001).\n"),
            doi_resolver=FakeResolver(),
            reviewer_count=0,
        )
        pipeline.run(
            _config(
                project_name=project,
                data_path=data,
                figures_default_format=default_format,
            )
        )
        return json.loads(
            (
                tmp_path
                / f"projects/{project}/artifacts/visualization/figures.json.v1"
            ).read_text(encoding="utf-8")
        )

    png = figures_for("figfmt-png", "png")
    pdf = figures_for("figfmt-pdf", "pdf")

    assert png["figures"] and pdf["figures"]
    assert all(f["filename"].endswith(".png") for f in png["figures"])
    assert all(f["filename"].endswith(".pdf") for f in pdf["figures"])


def test_end_to_end_delivery_config_produces_complete_manuscript(tmp_path: Path) -> None:
    """端到端：写入写作/评审/引用配置后，真实链路仍产出完整手稿与评审产物。"""
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
        reviewer_count=3,
    )
    result = pipeline.run(
        _config(
            project_name="delivery-e2e",
            data_path=data,
            writing_paper_type="research_article",
            writing_language="zh",
            writing_bilingual_abstract=True,
            writing_style_guide="保持客观",
            citation_style="",
            review_include_devil_advocate=True,
            review_consensus_threshold=0.6,
            review_score_scale="0-100",
        )
    )

    manuscript = result.manuscript_path.read_text(encoding="utf-8")
    assert "# 推断统计分析报告" in manuscript
    assert "## References" in manuscript
    assert result.review is not None
    assert len(result.review.reports) == 3
    assert (
        tmp_path / "projects/delivery-e2e/artifacts/review/review_reports.md.v1"
    ).is_file()
