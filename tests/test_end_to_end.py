"""End-to-end chain audit: every handoff must carry data through intact.

This test exists because a green unit-test suite can still hide a broken chain.
It walks the whole product **the way a user does** —
`Orchestrator -> ResearchService -> ResearchPipeline -> gates -> review -> export
-> CLI surface -> web surface` — and asserts at each handoff that the next stage
actually received usable data.

Only the outside world is faked (literature API, LLM, DOI resolver). Everything
else is the real implementation, so a break anywhere in the chain fails here.
Driving the service rather than the pipeline directly matters: project state is
owned by `ResearchService`, so bypassing it would silently skip every state
transition the user depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.artifact_store import ArtifactStore
from core.config_validation import ConfigValidationError
from core.external_clients import PaperRecord, SearchReport
from core.research_pipeline import ResearchPipelineCancelled
from core.research_service import ResearchService
from core.state_manager import ProjectState, StateManager, WorkflowStage
from orchestrator import Orchestrator
from web_app import create_app

DATASET = (
    "group,score\n"
    "A,1\nA,2\nA,3\nA,4\nA,5\n"
    "B,6\nB,7\nB,8\nB,9\nB,10\n"
)


class FakeSearcher:
    """Stands in for the four real literature APIs."""

    def search(self, query: str, sources: list[str], max_results: int) -> SearchReport:
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


class FakeResolver:
    """Stands in for the Crossref DOI lookup."""

    def resolve(self, doi: str) -> tuple[bool, str | None]:
        return True, None


class FakeLLM:
    """Stands in for Codex CLI / Gemini / OpenAI-compatible.

    Routes on the system prompt: the analysis prompt and the reviewer prompt
    each get a shape-appropriate reply.
    """

    def __init__(
        self, body: str = "# Draft\n\nA difference was observed [P1] (p = 0.001).\n"
    ) -> None:
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
        return self.body


def _run_user_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Orchestrator:
    """Drive the real user chain, faking only the outside world."""
    orchestrator = Orchestrator(tmp_path)
    orchestrator.init_project("e2e", "hybrid")

    data = tmp_path / "data.csv"
    data.write_text(DATASET, encoding="utf-8")

    monkeypatch.setattr(
        "core.research_service.LiteratureSearcher.from_config",
        lambda **kwargs: FakeSearcher(),
    )
    monkeypatch.setattr("core.research_service.build_llm_client", lambda **kwargs: FakeLLM())
    monkeypatch.setattr("core.citation_verifier.CrossrefDoiResolver", FakeResolver)

    orchestrator.run_real_research(
        "e2e",
        "climate adaptation",
        sources=["crossref"],
        max_results=2,
        data_path=data,
    )
    return orchestrator


def test_every_handoff_carries_data_through_the_whole_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_user_chain(tmp_path, monkeypatch)
    base = tmp_path / "projects/e2e/artifacts"

    # --- search -> analysis -------------------------------------------------
    literature = json.loads((base / "search/literature.json.v1").read_text(encoding="utf-8"))
    assert literature, "search produced no literature for the analysis stage"
    assert literature[0]["doi"] == "10.1234/climate"

    # --- dataset -> statistics (inferential, not just descriptive) ----------
    stats = json.loads(
        (base / "analysis/statistics_report.json.v1").read_text(encoding="utf-8")
    )
    assert stats["tests"], "dataset reached the statistics engine but produced no test"
    assert stats["tests"][0]["test_name"] == "Welch t-test"
    assert stats["tests"][0]["effect_size_name"] == "Cohen's d"

    # --- dataset -> figures -------------------------------------------------
    figures = json.loads(
        (base / "visualization/figures.json.v1").read_text(encoding="utf-8")
    )
    assert figures["figures"], "dataset reached the figure builder but produced no figure"
    for spec in figures["figures"]:
        saved = base / "visualization" / f"{spec['filename']}.v1"
        assert saved.is_file(), f"figure {spec['filename']} was registered but never stored"
        assert saved.stat().st_size > 0

    # --- analysis -> writing ------------------------------------------------
    assert (base / "analysis/research_analysis.md.v1").is_file()

    # --- analysis claims -> claim-evidence gate -----------------------------
    claims = json.loads(
        (base / "writing/claim_evidence_verification.json.v1").read_text(encoding="utf-8")
    )
    assert claims["claim_count"] >= 1, "the analysis claims never reached the claim gate"
    assert claims["passed"] is True
    assert claims["claims"][0]["evidence"][0]["verdict"] == "supports"
    assert claims["claims"][0]["evidence"][0]["quote"], (
        "a verdict was accepted without a verbatim quote from the abstract"
    )

    # --- writing -> citation integrity gate ---------------------------------
    citations = json.loads(
        (base / "writing/citation_verification.json.v1").read_text(encoding="utf-8")
    )
    assert citations["passed"] is True
    assert citations["unknown_markers"] == []

    # --- writing -> statistics integrity gate -------------------------------
    statistics = json.loads(
        (base / "writing/statistics_verification.json.v1").read_text(encoding="utf-8")
    )
    assert statistics["computed_p_values"], (
        "the statistics gate ran without seeing the computed p-values"
    )
    # Phase 6: 效应量与样本量也必须随链路到达关口。
    assert statistics["computed_effect_sizes"], (
        "the statistics gate ran without seeing the computed effect sizes"
    )
    assert statistics["computed_sample_sizes"], (
        "the statistics gate ran without seeing the computed sample sizes"
    )

    # --- writing -> manuscript claim integrity gate (Phase 6) ---------------
    manuscript_claims = json.loads(
        (base / "writing/manuscript_claim_verification.json.v1").read_text(
            encoding="utf-8"
        )
    )
    assert manuscript_claims["claim_count"] >= 1, (
        "the manuscript body's citation pairings never reached the claim gate"
    )
    assert (
        base / "writing/manuscript_claim_verification.md.v1"
    ).is_file(), "the manuscript claim report was never rendered"

    # --- gate results -> manuscript ----------------------------------------
    manuscript = (base / "writing/manuscript.md.v1").read_text(encoding="utf-8")
    assert "# 推断统计分析报告" in manuscript, "computed statistics never reached the manuscript"
    assert "# 图表清单" in manuscript, "figure list never reached the manuscript"
    assert "## References" in manuscript, "references never reached the manuscript"

    # --- writing -> peer review --------------------------------------------
    review = json.loads(
        (base / "review/review_reports.json.v1").read_text(encoding="utf-8")
    )
    assert review["reports"], "peer review produced no report"
    assert review["decision"]

    # --- writing -> communication ------------------------------------------
    outline = (base / "communication/presentation_outline.md.v1").read_text(encoding="utf-8")
    assert "# 统计结果" in outline, "statistics never reached the presentation outline"
    assert "# 图表" in outline, "figures never reached the presentation outline"


def test_state_transitions_reach_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chain must leave the project in a truthful, finished state."""
    orchestrator = _run_user_chain(tmp_path, monkeypatch)

    state = orchestrator.state_manager.load("e2e")
    assert state is not None
    assert state.current_stage == WorkflowStage.EXPORT
    assert state.stage_status[WorkflowStage.SEARCH] == "completed"
    assert state.stage_status[WorkflowStage.WRITING] == "completed"
    assert state.stage_status[WorkflowStage.EXPORT] == "ready_with_author_checks"


def test_phase6_gates_land_in_the_real_chain_and_export_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 6: 两个新关口走真实链路落盘，且导出全链路不断裂。

    这条测试走的是 `Orchestrator -> ResearchService -> ResearchPipeline`，只伪造
    外部世界。它证明新关口不是孤立单元：产物真实落盘（存在且非空）、手稿仍可导出
    为 MD/PDF/PPTX。
    """
    orchestrator = _run_user_chain(tmp_path, monkeypatch)
    base = tmp_path / "projects/e2e/artifacts/writing"

    claim_json = base / "manuscript_claim_verification.json.v1"
    claim_md = base / "manuscript_claim_verification.md.v1"
    stats_json = base / "statistics_verification.json.v1"
    assert claim_json.is_file() and claim_json.stat().st_size > 0
    assert claim_md.is_file() and claim_md.stat().st_size > 0
    assert stats_json.is_file() and stats_json.stat().st_size > 0

    # 正文级报告与统计关口都带上了 Phase 6 的新能力。
    claim_payload = json.loads(claim_json.read_text(encoding="utf-8"))
    assert claim_payload["claim_count"] >= 1
    stats_payload = json.loads(stats_json.read_text(encoding="utf-8"))
    assert stats_payload["computed_effect_sizes"]
    assert stats_payload["computed_sample_sizes"]

    # 手稿仍可导出为三种格式（新关口绝不阻断导出）。
    markdown = orchestrator.export("e2e", "md")
    assert markdown.is_file() and markdown.stat().st_size > 0
    pdf = orchestrator.export("e2e", "pdf")
    assert pdf.read_bytes()[:4] == b"%PDF"
    pptx = orchestrator.export("e2e", "pptx")
    assert pptx.read_bytes()[:2] == b"PK"


def test_all_three_export_formats_produce_valid_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = _run_user_chain(tmp_path, monkeypatch)

    markdown = orchestrator.export("e2e", "md")
    assert markdown.is_file() and markdown.stat().st_size > 0

    pdf = orchestrator.export("e2e", "pdf")
    assert pdf.read_bytes()[:4] == b"%PDF", "PDF export did not produce a real PDF"

    pptx = orchestrator.export("e2e", "pptx")
    assert pptx.stat().st_size > 0
    assert pptx.read_bytes()[:2] == b"PK", "PPTX is not a valid zip container"


def test_cli_surface_serves_the_finished_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = _run_user_chain(tmp_path, monkeypatch)

    # status/list must survive a fully-populated real project.
    orchestrator.show_status("e2e")
    orchestrator.list_projects()


def test_web_surface_serves_and_downloads_the_finished_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator = _run_user_chain(tmp_path, monkeypatch)
    orchestrator.export("e2e", "md")

    client = create_app(tmp_path).test_client()

    listing = client.get("/api/projects")
    assert listing.status_code == 200
    assert listing.get_json()["projects"][0]["name"] == "e2e"

    detail = client.get("/api/projects/e2e")
    assert detail.status_code == 200
    assert detail.get_json()["current_stage"] == "export"

    download = client.get("/api/projects/e2e/download/md")
    assert download.status_code == 200
    assert len(download.data) > 0


def test_legacy_project_schema_still_drives_every_surface(tmp_path: Path) -> None:
    """A project saved under the pre-Phase-5 schema must keep working everywhere.

    This is the regression guard for the migration: removing the dead enum
    members must never turn an existing project into a crash.
    """
    legacy = tmp_path / "projects" / "legacy"
    legacy.mkdir(parents=True)
    (legacy / "state.json").write_text(
        json.dumps(
            {
                "name": "legacy",
                "mode": "hybrid",
                "current_stage": "polishing",
                "stage_status": {
                    "brainstorming": "completed",
                    "search": "completed",
                    "lit_review": "completed",
                    "statistics": "pending",
                    "visualization": "pending",
                    "writing": "in_progress",
                    "polishing": "pending",
                    "review": "pending",
                    "export": "pending",
                },
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )

    state = StateManager(tmp_path / "projects").load("legacy")
    assert state is not None
    assert state.current_stage is not None
    assert set(state.stage_status) == set(WorkflowStage)

    orchestrator = Orchestrator(tmp_path)
    orchestrator.show_status("legacy")
    orchestrator.list_projects()

    assert create_app(tmp_path).test_client().get("/api/projects").status_code == 200


def test_invalid_config_fails_fast_and_breaks_nothing(tmp_path: Path) -> None:
    """A broken settings file must fail before any work, leaving state untouched."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        "review:\n  reviewer_count: three\n", encoding="utf-8"
    )

    projects_dir = tmp_path / "projects"
    state_manager = StateManager(projects_dir)
    state_manager.save(
        ProjectState(
            name="badcfg", mode="hybrid", current_stage=WorkflowStage.BRAINSTORMING
        )
    )

    with pytest.raises(ConfigValidationError, match="reviewer_count"):
        ResearchService(projects_dir, state_manager, ArtifactStore(projects_dir)).run(
            "badcfg", "a topic", sources=["crossref"], max_results=1
        )

    reloaded = state_manager.load("badcfg")
    assert reloaded is not None
    assert reloaded.stage_status[WorkflowStage.SEARCH] == "pending"


# ==========================================================================
# Resume (断点续跑): the user's real chain must skip only what is provably safe
# ==========================================================================
#
# These tests walk the same real path as the rest of this file
# (Orchestrator -> ResearchService -> ResearchPipeline). Only the outside world
# is faked, and the fakes count their own calls, so "no search, no model call"
# is asserted on the actual methods the pipeline would invoke.

RESUME_STEPS = ("search", "analysis", "claims", "writing", "review")


class CountingSearcher:
    """A literature API that records how many times it was actually queried."""

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
            counts_by_source={name: 1 for name in sources},
        )


class CountingLLM:
    """A model client that records every call, split by kind."""

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


class ResumeHarness:
    """Drive the real chain repeatedly, sharing one pair of counting fakes.

    The service constructs its searcher and model client on every run, so the
    fakes must be *reused* across runs for their counts to mean anything. Patching
    the factory functions to return these shared objects does exactly that.
    """

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.searcher = CountingSearcher()
        self.llm = CountingLLM()
        self.orchestrator = Orchestrator(tmp_path)
        self.orchestrator.init_project("resume", "hybrid")
        self.data = tmp_path / "data.csv"
        self.data.write_text(DATASET, encoding="utf-8")

        monkeypatch.setattr(
            "core.research_service.LiteratureSearcher.from_config",
            lambda **kwargs: self.searcher,
        )
        monkeypatch.setattr(
            "core.research_service.build_llm_client", lambda **kwargs: self.llm
        )
        monkeypatch.setattr("core.citation_verifier.CrossrefDoiResolver", FakeResolver)

    def run(self, topic: str = "climate adaptation", *, resume: bool = True):
        return self.orchestrator.run_real_research(
            "resume",
            topic,
            sources=["crossref"],
            max_results=2,
            data_path=self.data,
            resume=resume,
        )

    def latest(self, stage: str, filename: str) -> Path:
        path = self.orchestrator.artifact_store.get_artifact(
            "resume", stage, filename
        )
        assert path is not None, f"{stage}/{filename} was never written"
        return path


def test_resume_reruns_nothing_when_inputs_are_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core promise: identical inputs => no literature API call, no model call.

    This is the whole point of resuming. If it ever regressed, the user would pay
    twice for the same run without noticing.
    """
    harness = ResumeHarness(tmp_path, monkeypatch)

    first = harness.run()
    assert harness.searcher.search_calls == 1
    assert harness.llm.complete_calls == 1
    assert harness.llm.complete_json_calls >= 1  # analysis (+ claim gate + review)
    assert first.reused_steps == [], "the first run must not reuse anything"

    searches_after_first = harness.searcher.search_calls
    completes_after_first = harness.llm.complete_calls
    json_after_first = harness.llm.complete_json_calls

    manuscript_v1 = harness.latest("writing", "manuscript.md")
    manuscript_text = manuscript_v1.read_text(encoding="utf-8")

    second = harness.run()

    # --- the outside world was not touched again ---------------------------
    assert harness.searcher.search_calls == searches_after_first, (
        "a resumed run re-queried the literature API"
    )
    assert harness.llm.complete_calls == completes_after_first, (
        "a resumed run re-called the language model"
    )
    assert harness.llm.complete_json_calls == json_after_first

    # --- everything was reused, and the result says so ---------------------
    assert second.reused_steps == list(RESUME_STEPS), (
        f"expected all stages reused, got {second.reused_steps}"
    )
    assert second.resume_note, "a resumed run must explain itself to the user"
    assert "复用" in second.resume_note

    # --- the artifacts were reused, not silently rewritten -----------------
    assert harness.latest("writing", "manuscript.md") == manuscript_v1, (
        "resuming wrote a new manuscript version instead of reusing the existing one"
    )
    assert (
        harness.latest("writing", "manuscript.md").read_text(encoding="utf-8")
        == manuscript_text
    )


def test_resume_after_crash_reuses_prefix_and_redoes_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing late artifact must redrive only the late stages.

    Reuse is a contiguous prefix: once ``writing`` cannot be reused, ``review``
    cannot either, even though its artifacts still exist on disk.
    """
    harness = ResumeHarness(tmp_path, monkeypatch)
    harness.run()

    # Simulate a crash after the earlier stages: the manuscript never landed.
    manuscript = harness.latest("writing", "manuscript.md")
    manuscript.unlink()

    searches_before = harness.searcher.search_calls

    result = harness.run()

    # Search was reused (no new API call), but writing was redone.
    assert harness.searcher.search_calls == searches_before, (
        "the crash reran search even though its artifacts were intact"
    )
    assert result.reused_steps == ["search", "analysis", "claims"], (
        f"expected the prefix to stop before writing, got {result.reused_steps}"
    )
    assert "writing" not in result.reused_steps
    assert "review" not in result.reused_steps

    # The final deliverable is complete again.
    restored = harness.latest("writing", "manuscript.md")
    assert restored.is_file() and restored.stat().st_size > 0
    body = restored.read_text(encoding="utf-8")
    assert "## References" in body and "# 图表清单" in body


def test_changing_the_topic_reuses_no_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression guard against answering a *new* question with *old* artifacts.

    This is the dangerous failure mode: a reused search would still satisfy every
    downstream gate, producing a manuscript that looks fully verified yet answers
    the previous topic. Reuse must be refused outright.
    """
    harness = ResumeHarness(tmp_path, monkeypatch)
    first = harness.run("climate adaptation")
    assert first.reused_steps == []

    second = harness.run("quantum computing for medicine")

    assert second.reused_steps == [], (
        "a changed topic reused old artifacts — the run would answer the wrong question"
    )
    assert harness.searcher.search_calls == 2, (
        "a changed topic must re-query the literature API"
    )
    assert harness.searcher.queries[-1] == "quantum computing for medicine"

    assert second.resume_plan, "a changed input must be explained, not just felt"
    assert "运行输入已变化" in second.resume_plan, (
        "the plan must name the reason nothing could be reused"
    )
    assert "研究主题" in second.resume_plan

    # The CLI must present the reason *and* the fact, each labelled, together.
    Orchestrator(tmp_path).report_resume(second)
    out = capsys.readouterr().out
    assert "[续跑] 判定：" in out and "运行输入已变化" in out
    assert "[续跑] 实际：" in out and "未复用任何旧产物" in out


def test_resume_matches_a_fresh_run_and_no_resume_forces_reruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed manuscript must equal a fresh one; ``--no-resume`` must rerun all.

    Two independent guarantees: resuming is not allowed to change the deliverable,
    and forcing a fresh run is not allowed to silently reuse anything.
    """
    harness = ResumeHarness(tmp_path, monkeypatch)

    harness.run()
    fresh_text = harness.latest("writing", "manuscript.md").read_text(encoding="utf-8")

    second = harness.run()
    assert second.reused_steps == list(RESUME_STEPS)
    resumed_text = harness.latest("writing", "manuscript.md").read_text(
        encoding="utf-8"
    )
    assert resumed_text == fresh_text, (
        "resuming produced a manuscript different from a fresh run"
    )

    searches_before = harness.searcher.search_calls
    completes_before = harness.llm.complete_calls

    forced = harness.run(resume=False)

    assert forced.reused_steps == [], "--no-resume still reused artifacts"
    assert harness.searcher.search_calls == searches_before + 1, (
        "--no-resume did not re-query the literature API"
    )
    assert harness.llm.complete_calls == completes_before + 1, (
        "--no-resume did not re-call the model"
    )


# ==========================================================================
# Cancellation (P5.6): cancel at a stage boundary, keep artifacts, resume after
# ==========================================================================
#
# This walks the same real chain as the rest of the file
# (Orchestrator -> ResearchService -> ResearchPipeline). Cancellation is
# cooperative: the callback is consulted at each stage boundary, so flipping the
# flag from the real ``on_progress`` callback stops the run *after* the stage
# that just finished — proving cancellation and resume compose correctly.


class CancellingHarness(ResumeHarness):
    """A resume harness that can request cancellation from the progress callback.

    The flag is flipped when ``search`` reports ``completed`` — i.e. its
    artifacts are already on disk (the pipeline saves them before that signal).
    The run must therefore stop at the next boundary (before ``analysis``) and
    leave the search artifacts intact.
    """

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__(tmp_path, monkeypatch)
        self.cancel_flag = False
        self.stages_seen: list[tuple[str, str]] = []

    def run_cancelling(self):
        def on_progress(stage: str, status: str) -> None:
            self.stages_seen.append((stage, status))
            if stage == "search" and status == "completed":
                self.cancel_flag = True

        return self.orchestrator.run_real_research(
            "resume",
            "climate adaptation",
            sources=["crossref"],
            max_results=2,
            data_path=self.data,
            resume=True,
            on_progress=on_progress,
            should_continue=lambda: not self.cancel_flag,
        )


def test_cancel_stops_at_boundary_keeps_artifacts_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel must be honest AND composable with resume.

    Three guarantees:
    1. The run stops with ``ResearchPipelineCancelled`` at a stage boundary
       (after ``search``, before ``analysis``).
    2. The artifacts already produced are still on disk — cancellation never
       destroys work — and the **analysis** stage was not entered.
    3. A subsequent run reuses the preserved prefix (``search``) without
       re-querying the literature API, then finishes the manuscript.

    Phase 7: the relevance gate legitimately makes ONE ``complete_json`` call
    **inside** the search stage. So the property to lock is "analysis was not
    entered", not "no model call happened at all".
    """
    harness = CancellingHarness(tmp_path, monkeypatch)

    with pytest.raises(ResearchPipelineCancelled):
        harness.run_cancelling()

    # --- 1. it stopped at the boundary, not mid-stage -----------------------
    assert ("search", "completed") in harness.stages_seen
    assert ("lit_review", "completed") not in harness.stages_seen
    assert harness.searcher.search_calls == 1, "search ran exactly once"
    # search 阶段内的相关性关口恰好一次调用；analysis 未进入，故总数不超过它。
    assert harness.llm.complete_json_calls == 1, (
        "only the search-stage relevance gate should have called the model; "
        "analysis must not have started before the boundary"
    )
    # 产物层面（最强）：analysis 阶段根本没被创建。
    assert not (tmp_path / "projects/resume/artifacts/analysis").exists(), (
        "analysis artifacts must not exist after cancelling at the boundary"
    )

    # --- 2. the search artifacts survive cancellation -----------------------
    literature = harness.latest("search", "literature.json")
    assert literature.is_file() and literature.stat().st_size > 0
    # Phase 7：相关性产物也在 search 阶段内落盘，续跑据此可复用 search。
    relevance = harness.latest("search", "relevance_check.json")
    assert relevance.is_file() and relevance.stat().st_size > 0

    # Project state records the honest cancellation (not 'blocked').
    state = harness.orchestrator.state_manager.load("resume")
    assert state is not None
    assert "cancelled" in state.stage_status.values(), (
        "the cancelled stage must be recorded as 'cancelled'"
    )
    assert "blocked" not in state.stage_status.values(), (
        "cancellation must not be recorded as a failure"
    )

    # --- 3. a follow-up run reuses the prefix and finishes ------------------
    searches_before = harness.searcher.search_calls
    result = harness.run()

    assert "search" in result.reused_steps, (
        "the cancelled run's search artifacts must be reused, not thrown away"
    )
    assert harness.searcher.search_calls == searches_before, (
        "resuming after a cancel must not re-query the literature API"
    )
    manuscript = harness.latest("writing", "manuscript.md")
    assert manuscript.is_file() and manuscript.stat().st_size > 0


def test_usage_note_travels_the_real_chain_to_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run's ``usage_note`` must survive the real chain onto the result.

    Whatever the pipeline measured (or could not measure) must reach the caller
    verbatim; the CLI is the user's only view of cost, so a fabricated or dropped
    note would be a truth defect. This fake client reports no token usage, so the
    note must honestly say "not reported" — never a made-up number.
    """
    harness = ResumeHarness(tmp_path, monkeypatch)
    result = harness.run()

    note = result.usage_note
    assert isinstance(note, str) and note, (
        "a completed run must carry a usage note describing what was measured"
    )
    # The fake LLM reports no usage, so the note must admit that plainly...
    assert "未获得用量" in note, (
        "unreported usage must be stated as such, not silently dropped"
    )
    # ...and must never invent a number that was not reported.
    assert "估算" in note, (
        "the note must say it does not estimate when usage is unavailable"
    )

    # The CLI passes the note through unchanged (verbatim, labelled 用量).
    Orchestrator(tmp_path).report_usage(result)
    out = capsys.readouterr().out
    assert f"[用量] {note}" in out, (
        "the CLI must print the usage note verbatim, with no recomputation"
    )


# ==========================================================================
# Phase 7: the relevance gate + honest search reporting on the real chain
# ==========================================================================


class LimitSearcher:
    """Stands in for the searcher when ``max_results`` is a **total** cap.

    Returns more deduplicated candidates than ``max_results`` and reports the
    Phase 7 accounting fields (total_found / deduplicated_count /
    dropped_by_limit / ranking_reasons), as the real searcher now does.
    """

    def search(self, query: str, sources: list[str], max_results: int) -> SearchReport:
        papers = [
            PaperRecord(
                title=f"Candidate paper {index}",
                authors=["A Author"],
                year=2024,
                journal="Research Journal",
                doi=f"10.1234/candidate-{index}",
                abstract="A finding.",
                source="crossref",
            )
            for index in range(5)
        ]
        kept = papers[:max_results]
        return SearchReport(
            papers=kept,
            errors=[],
            sources_attempted=list(sources),
            counts_by_source={name: 5 for name in sources},
            total_found=5,
            deduplicated_count=5,
            dropped_by_limit=5 - len(kept),
            ranking_reasons=["仅 1 个检索源命中: crossref", "主题词项命中 1 个: climate"],
        )


def _run_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    searcher: object,
    max_results: int,
    project: str = "phase7",
) -> Orchestrator:
    orchestrator = Orchestrator(tmp_path)
    orchestrator.init_project(project, "hybrid")
    monkeypatch.setattr(
        "core.research_service.LiteratureSearcher.from_config",
        lambda **kwargs: searcher,
    )
    monkeypatch.setattr("core.research_service.build_llm_client", lambda **kwargs: FakeLLM())
    monkeypatch.setattr("core.citation_verifier.CrossrefDoiResolver", FakeResolver)
    orchestrator.run_real_research(
        project,
        "climate adaptation",
        sources=["crossref"],
        max_results=max_results,
    )
    return orchestrator


def test_phase7_relevance_gate_lands_in_the_real_chain_and_export_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 7: 相关性关口走真实链路落盘，手稿仍可导出，且 `max_results` 是总量上限。"""
    orchestrator = _run_chain(
        tmp_path,
        monkeypatch,
        searcher=FakeSearcher(),
        max_results=2,
    )
    base = tmp_path / "projects/phase7/artifacts/search"

    relevance_json = base / "relevance_check.json.v1"
    relevance_md = base / "relevance_check.md.v1"
    assert relevance_json.is_file() and relevance_json.stat().st_size > 0
    assert relevance_md.is_file() and relevance_md.stat().st_size > 0

    payload = json.loads(relevance_json.read_text(encoding="utf-8"))
    assert payload["ran"] is True
    assert set(payload) >= {
        "passed",
        "ran",
        "not_run_reason",
        "verdict_count",
        "irrelevant_count",
        "unclear_count",
        "verdicts",
    }
    # 只标记、绝不删除：检索到的每一篇都必须仍在 literature.json 中。
    literature = json.loads(
        (base / "literature.json.v1").read_text(encoding="utf-8")
    )
    assert payload["verdict_count"] == len(literature)

    # 手稿仍可导出（新关口绝不阻断导出）。
    markdown = orchestrator.export("phase7", "md")
    assert markdown.is_file() and markdown.stat().st_size > 0
    pdf = orchestrator.export("phase7", "pdf")
    assert pdf.read_bytes()[:4] == b"%PDF"


def test_phase7_max_results_is_a_total_cap_and_drop_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`max_results` 是**总量上限**：最终文献数 ≤ max_results，且被裁掉必须可见。"""
    orchestrator = _run_chain(
        tmp_path,
        monkeypatch,
        searcher=LimitSearcher(),
        max_results=2,
    )
    base = tmp_path / "projects/phase7/artifacts/search"
    literature = json.loads(
        (base / "literature.json.v1").read_text(encoding="utf-8")
    )
    assert len(literature) <= 2, "max_results must cap the TOTAL number of papers"

    report = json.loads((base / "search_report.json.v1").read_text(encoding="utf-8"))
    assert set(report) >= {
        "total_found",
        "deduplicated_count",
        "dropped_by_limit",
        "ranking_reasons",
    }
    # 算术自洽：去重条数 - 最终收录条数 == 被裁条数。
    assert report["deduplicated_count"] - len(literature) == report["dropped_by_limit"]
    assert report["dropped_by_limit"] == 3

    # CLI 必须如实、可见地报告被裁掉的文献。
    orchestrator.report_literature_limit("phase7")
    out = capsys.readouterr().out
    assert "[检索]" in out and "3 篇" in out and "总量上限" in out
