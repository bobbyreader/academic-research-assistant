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
