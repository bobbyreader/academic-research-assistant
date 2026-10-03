from __future__ import annotations

import time
from pathlib import Path

from orchestrator import Orchestrator
from web_app import DEFAULT_WEB_PORT, create_app


def test_web_home_and_health_are_available(tmp_path: Path) -> None:
    client = create_app(tmp_path).test_client()

    home = client.get("/")
    health = client.get("/api/health")

    assert home.status_code == 200
    assert "研究工作台" in home.get_data(as_text=True)
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert health.status_code == 200
    assert health.get_json()["status"] == "ok"


def test_web_home_offers_local_codex_engine(tmp_path: Path) -> None:
    page = create_app(tmp_path).test_client().get("/").get_data(as_text=True)

    assert 'name="provider"' in page
    assert 'value="codex_cli"' in page
    assert "本机 Codex" in page


def test_template_has_direct_file_guidance_and_launcher() -> None:
    project_root = Path(__file__).resolve().parents[1]
    template = (project_root / "web" / "templates" / "index.html").read_text(
        encoding="utf-8"
    )
    launcher = project_root / "启动研究助手.command"

    assert "direct-file-preview" in template
    assert "启动研究助手.command" in template
    assert launcher.is_file()
    launcher_text = launcher.read_text(encoding="utf-8")
    assert DEFAULT_WEB_PORT == 5050
    assert "CANDIDATE_PORTS=(5050" in launcher_text
    assert "SELECTED_PORT" in launcher_text


def test_research_endpoint_validates_topic_and_sources(tmp_path: Path) -> None:
    client = create_app(tmp_path).test_client()

    missing_topic = client.post("/api/research", json={})
    bad_source = client.post(
        "/api/research",
        json={"topic": "A useful topic", "sources": ["unknown"]},
    )

    assert missing_topic.status_code == 400
    assert "研究主题" in missing_topic.get_json()["error"]
    assert bad_source.status_code == 400
    assert "检索源" in bad_source.get_json()["error"]


def test_projects_endpoint_reads_existing_project(tmp_path: Path) -> None:
    orchestrator = Orchestrator(tmp_path)
    orchestrator.init_project("demo", "hybrid")

    response = create_app(tmp_path).test_client().get("/api/projects")

    assert response.status_code == 200
    assert response.get_json()["projects"][0]["name"] == "demo"


def test_research_job_runs_in_background_with_injected_runner(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, base_dir: Path) -> None:
            self.base_dir = base_dir
            self.delegate = Orchestrator(base_dir)

        @property
        def state_manager(self):
            return self.delegate.state_manager

        def init_project(self, name: str, mode: str) -> None:
            self.delegate.init_project(name, mode)

        def run_real_research(self, project_name: str, topic: str, **options: object) -> None:
            captured.update(options)
            callback = options["on_progress"]
            callback("search", "completed")
            callback("lit_review", "completed")
            callback("writing", "completed")

        def export(self, project_name: str, format: str) -> Path:
            output = self.base_dir / "projects" / project_name / "exports" / f"{project_name}_final.{format}"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("test export", encoding="utf-8")
            return output

    app = create_app(tmp_path, orchestrator_factory=FakeRunner)
    client = app.test_client()
    response = client.post(
        "/api/research",
        json={
            "topic": "A useful topic",
            "project_name": "demo_job",
            "sources": ["crossref"],
            "exports": ["md"],
        },
    )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    for _ in range(20):
        snapshot = client.get(f"/api/jobs/{job_id}").get_json()
        if snapshot["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)

    assert snapshot["status"] == "completed"
    assert snapshot["project_name"] == "demo_job"
    assert snapshot["artifacts"][0]["format"] == "md"
    assert captured["provider"] == "codex_cli"


def test_multipart_request_keeps_multiple_sources(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, base_dir: Path) -> None:
            self.delegate = Orchestrator(base_dir)

        @property
        def state_manager(self):
            return self.delegate.state_manager

        def init_project(self, name: str, mode: str) -> None:
            self.delegate.init_project(name, mode)

        def run_real_research(self, project_name: str, topic: str, **options: object) -> None:
            captured.update(options)

        def export(self, project_name: str, format: str) -> Path:
            output = self.delegate.base_dir / "projects" / project_name / "exports" / f"{project_name}_final.{format}"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("test export", encoding="utf-8")
            return output

    client = create_app(tmp_path, orchestrator_factory=FakeRunner).test_client()
    response = client.post(
        "/api/research",
        data={
            "topic": "A multipart topic",
            "project_name": "multipart_job",
            "sources": ["crossref", "arxiv"],
            "exports": ["md", "pptx"],
        },
    )

    assert response.status_code == 202
    job_id = response.get_json()["job_id"]
    for _ in range(20):
        snapshot = client.get(f"/api/jobs/{job_id}").get_json()
        if snapshot["status"] in {"completed", "failed"}:
            break
        time.sleep(0.02)

    assert snapshot["status"] == "completed"
    assert captured["sources"] == ["crossref", "arxiv"]
    assert {artifact["format"] for artifact in snapshot["artifacts"]} == {"md", "pptx"}
