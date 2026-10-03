"""Project-level orchestration for the real external research pipeline."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from core.artifact_store import ArtifactStore
from core.config_validation import ConfigValidationError, load_validated_settings
from core.external_clients import LiteratureSearcher
from core.llm_client import build_llm_client
from core.research_pipeline import (
    PipelineResult,
    ResearchPipeline,
    ResearchPipelineCancelled,
    ResearchPipelineConfig,
)
from core.resume import (
    COMPLETED_STEPS_KEY,
    RunFingerprint,
    parse_recorded_steps,
    plan_resume,
)
from core.state_manager import StateManager, WorkflowStage
from core.usage import UsageTrackingClient


class ResearchService:
    """Connect project state, external clients, and the research pipeline."""

    def __init__(
        self,
        projects_dir: Path,
        state_manager: StateManager,
        artifact_store: ArtifactStore,
    ) -> None:
        self.projects_dir = projects_dir
        self.state_manager = state_manager
        self.artifact_store = artifact_store

    def run(
        self,
        project_name: str,
        topic: str,
        *,
        sources: list[str],
        max_results: int,
        data_path: Path | None = None,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        on_progress: Callable[[str, str], None] | None = None,
        resume: bool = True,
        should_continue: Callable[[], bool] | None = None,
    ) -> PipelineResult:
        # Validate configuration before touching any state or network resource:
        # a broken settings file must fail in under a second with the offending
        # key named, not halfway through an expensive pipeline run.
        runtime_settings, validation = load_validated_settings(self.projects_dir.parent)
        if not validation.valid:
            details = "；".join(validation.errors)
            raise ConfigValidationError(
                f"config/settings.yaml 配置无效，请修正后再运行：{details}"
            )

        state = self.state_manager.load(project_name)
        if state is None:
            raise FileNotFoundError(f"项目 '{project_name}' 不存在")

        state.metadata.update(
            {
                "research_topic": topic,
                "pipeline": "external_api",
                "sources": sources,
                "max_results": max_results,
                "data_path": str(data_path) if data_path else None,
                "llm_provider": provider,
                "llm_model": model,
            }
        )
        self.state_manager.save(state)
        llm_settings = runtime_settings.get("llm", {})
        api_settings = runtime_settings.get("api_keys", {})
        if not isinstance(api_settings, dict):
            api_settings = {}
        review_settings = runtime_settings.get("review", {})
        reviewer_count = (
            int(review_settings.get("reviewer_count", 1))
            if isinstance(review_settings, dict)
            else 1
        )
        figure_settings = runtime_settings.get("figures", {})
        figure_dpi = (
            int(figure_settings.get("default_dpi", 300))
            if isinstance(figure_settings, dict)
            else 300
        )
        configured_timeout = (
            llm_settings.get("timeout_seconds") if isinstance(llm_settings, dict) else None
        )
        llm_timeout = int(configured_timeout) if configured_timeout is not None else None
        configured_provider = llm_settings.get("provider") if isinstance(llm_settings, dict) else None
        selected_provider = provider or os.getenv("ARS_LLM_PROVIDER") or configured_provider
        configured_base_url = (
            llm_settings.get("base_url") if isinstance(llm_settings, dict) else None
        )
        configured_model = llm_settings.get("model") if isinstance(llm_settings, dict) else None
        selected_model = model or (
            os.getenv("ARS_LLM_MODEL")
            if selected_provider in {"openai", "openai_compatible", "compatible"}
            else os.getenv("CODEX_MODEL")
            if selected_provider in {"codex", "codex_cli"}
            else os.getenv("GEMINI_MODEL")
        ) or configured_model
        selected_base_url = base_url or os.getenv("ARS_LLM_BASE_URL") or configured_base_url
        active_stage = WorkflowStage.SEARCH
        state.stage_status[WorkflowStage.BRAINSTORMING] = "completed"
        state.stage_status[WorkflowStage.SEARCH] = "in_progress"

        # 本次运行的输入指纹。它与"上一次运行记录下来的分阶段指纹"一起决定哪些
        # 阶段可以安全跳过（见 core/resume.py 的第一性原理）。
        fingerprint = RunFingerprint.build(
            topic=topic,
            sources=sources,
            max_results=max_results,
            data_path=data_path,
            reviewer_count=reviewer_count,
            figure_dpi=figure_dpi,
        )
        previous_steps = parse_recorded_steps(state.metadata.get(COMPLETED_STEPS_KEY))
        decision = plan_resume(
            artifact_store=self.artifact_store,
            project_name=project_name,
            fingerprint=fingerprint,
            previous_steps=previous_steps,
            enabled=resume,
        )
        self.state_manager.save(state)

        def progress(stage_name: str, status: str) -> None:
            nonlocal active_stage
            stage_map = {
                "search": WorkflowStage.SEARCH,
                "lit_review": WorkflowStage.LIT_REVIEW,
                "writing": WorkflowStage.WRITING,
            }
            next_stage = stage_map.get(stage_name, active_stage)
            if next_stage != active_stage:
                state.stage_status[next_stage] = "in_progress"
            active_stage = next_stage
            state.current_stage = active_stage
            state.stage_status[active_stage] = status
            # 安全性的关键：指纹必须在该阶段产物**已经落盘之后**才被记录。进度回调
            # 的 "completed" 标记恰好触发在产物写完之后（见 core/research_pipeline.py），
            # 因此这里记录指纹等价于“该阶段的产物确实与本次输入对应”。先写指纹，
            # 再 save，保证"输入变了但产物还没写出来"不会被误判为可复用。
            if status == "completed":
                recorded = state.metadata.setdefault(COMPLETED_STEPS_KEY, {})
                if isinstance(recorded, dict):
                    recorded[stage_name] = fingerprint.to_dict()
            self.state_manager.save(state)
            if on_progress:
                on_progress(stage_name, status)

        searcher = LiteratureSearcher.from_config(
            pubmed_email=os.getenv(
                "ARS_PUBMED_EMAIL",
                os.getenv("PUBMED_EMAIL", str(api_settings.get("pubmed_email", ""))),
            ),
            crossref_email=os.getenv(
                "ARS_CROSSREF_EMAIL",
                os.getenv("CROSSREF_EMAIL", str(api_settings.get("crossref_email", ""))),
            ),
            semantic_scholar_key=os.getenv(
                "SEMANTIC_SCHOLAR_API_KEY",
                str(api_settings.get("semantic_scholar_key", "")),
            ),
        )
        try:
            pipeline = ResearchPipeline(
                self.artifact_store,
                searcher=searcher,
                # 用量统计是**纯观测**包装：它只转发 complete/complete_json 并记账，
                # 不改变调用次数，也不改变返回内容。
                llm_client=UsageTrackingClient(
                    build_llm_client(
                        provider=selected_provider,
                        model=selected_model,
                        api_key=api_key,
                        base_url=selected_base_url,
                        timeout=llm_timeout,
                        workspace_dir=self.projects_dir / project_name,
                    ),
                    provider=str(selected_provider or ""),
                    model=str(selected_model or ""),
                ),
                progress=progress,
                reviewer_count=reviewer_count,
                figure_dpi=figure_dpi,
                resume=decision,
                should_continue=should_continue,
            )
            result = pipeline.run(
                ResearchPipelineConfig(
                    project_name=project_name,
                    topic=topic,
                    sources=sources,
                    max_results=max_results,
                    data_path=data_path,
                )
            )
        except ResearchPipelineCancelled:
            # 取消是用户**主动行为**，不是错误：状态记为 `"cancelled"` 而非
            # `"blocked"`，并把状态落盘，然后原样重新抛出（调用方需知道运行未完成）。
            # 用量的落盘由管线在自己的 finally 中完成，此处不重复。
            state.current_stage = active_stage
            state.stage_status[active_stage] = "cancelled"
            self.state_manager.save(state)
            raise
        except Exception:
            state.current_stage = active_stage
            state.stage_status[active_stage] = "blocked"
            self.state_manager.save(state)
            raise

        state.current_stage = WorkflowStage.EXPORT
        state.stage_status[WorkflowStage.EXPORT] = "ready_with_author_checks"
        self.state_manager.save(state)
        return result
