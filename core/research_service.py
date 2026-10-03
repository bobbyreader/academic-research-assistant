"""Project-level orchestration for the real external research pipeline."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from core.artifact_store import ArtifactStore
from core.config_loader import (
    get_bool,
    get_float,
    get_int,
    get_int_pair,
    get_str,
)
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
        settings: Mapping[str, Any] = runtime_settings
        # 配置警告必须**可见**：非阻断性问题（未知键、缺失配置文件等）不得被静默
        # 吞掉，否则用户会以为"配置生效了"。这里如实逐条记录到应用日志。
        for warning in validation.warnings:
            logging.getLogger(__name__).warning("配置警告：%s", warning)
        llm_settings = settings.get("llm", {})
        api_settings = settings.get("api_keys", {})
        if not isinstance(api_settings, dict):
            api_settings = {}
        # review.* / figures.* 一律经 config_loader 的取值接口读取（默认值只在
        # DEFAULTS 一处定义，不自造）：未配置时逐项回退到与改动前相同的默认值。
        reviewer_count = get_int(settings, "review.reviewer_count")
        figure_dpi = get_int(settings, "figures.default_dpi")
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
        # ------------------------------------------------------------------ #
        # 各节配置：一律经 config_loader 取值接口读取。默认值只在 DEFAULTS 一处
        # 定义（不自造），未配置时逐项回退到与改动前相同的默认值。这些值同时用于
        # (a) 参与续跑指纹，(b) 显式传给下游（这是既有模式：reviewer_count / figure_dpi）。
        # ------------------------------------------------------------------ #
        search_year_range = get_int_pair(settings, "search.year_range")
        figures_default_journal = get_str(settings, "figures.default_journal")
        figures_default_format = get_str(settings, "figures.default_format")
        figures_color_palette = get_str(settings, "figures.color_palette")
        figures_font_family = get_str(settings, "figures.font_family")
        figures_font_size_pt = get_int(settings, "figures.font_size_pt")
        writing_paper_type = get_str(settings, "writing.default_paper_type")
        writing_language = get_str(settings, "writing.default_language")
        writing_bilingual_abstract = get_bool(settings, "writing.bilingual_abstract")
        writing_style_guide = get_str(settings, "writing.style_guide")
        citation_style = get_str(settings, "citation.default_style")
        review_include_devil_advocate = get_bool(
            settings, "review.include_devil_advocate"
        )
        # 浮点键必须用 get_float：get_int 会把 0.6 抹成 0（"看起来配了、实际没生效"）。
        review_consensus_threshold = get_float(
            settings, "review.consensus_threshold"
        )
        review_score_scale = get_str(settings, "review.score_scale")
        export_default_format = get_str(settings, "export.default_format")
        export_pdf_engine = get_str(settings, "export.pdf_engine")
        export_pptx_template = get_str(settings, "export.pptx_template")
        export_include_speaker_notes = get_bool(settings, "export.include_speaker_notes")
        active_stage = WorkflowStage.SEARCH
        state.stage_status[WorkflowStage.BRAINSTORMING] = "completed"
        state.stage_status[WorkflowStage.SEARCH] = "in_progress"

        # 本次运行的输入指纹。它与"上一次运行记录下来的分阶段指纹"一起决定哪些
        # 阶段可以安全跳过（见 core/resume.py 的第一性原理）。
        #
        # 关键：**任何影响某阶段输出的配置都必须进指纹**。否则用户改了"撰写语言"
        # 或"引用样式"后重跑，系统会复用旧设置下产出的手稿并报告"复用了全部阶段"
        # ——那是一句谎，且用户看不出来。分组语义由 core/resume.py 负责，这里只需
        # 如实把值传进去。
        fingerprint = RunFingerprint.build(
            topic=topic,
            sources=sources,
            max_results=max_results,
            data_path=data_path,
            reviewer_count=reviewer_count,
            figure_dpi=figure_dpi,
            search_year_range=search_year_range,
            figures_default_journal=figures_default_journal,
            figures_default_format=figures_default_format,
            figures_color_palette=figures_color_palette,
            figures_font_family=figures_font_family,
            figures_font_size_pt=figures_font_size_pt,
            writing_paper_type=writing_paper_type,
            writing_language=writing_language,
            writing_bilingual_abstract=writing_bilingual_abstract,
            writing_style_guide=writing_style_guide,
            citation_style=citation_style,
            review_include_devil_advocate=review_include_devil_advocate,
            review_consensus_threshold=review_consensus_threshold,
            review_score_scale=review_score_scale,
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
                    # 显式传值给下游（既有模式）：字段名与 delivery-dev 冻结的一致。
                    writing_paper_type=writing_paper_type,
                    writing_language=writing_language,
                    writing_bilingual_abstract=writing_bilingual_abstract,
                    writing_style_guide=writing_style_guide,
                    citation_style=citation_style,
                    figures_default_journal=figures_default_journal,
                    figures_default_format=figures_default_format,
                    figures_color_palette=figures_color_palette,
                    figures_font_family=figures_font_family,
                    figures_font_size_pt=figures_font_size_pt,
                    review_include_devil_advocate=review_include_devil_advocate,
                    review_consensus_threshold=review_consensus_threshold,
                    review_score_scale=review_score_scale,
                    export_default_format=export_default_format,
                    export_pdf_engine=export_pdf_engine,
                    export_pptx_template=export_pptx_template,
                    export_include_speaker_notes=export_include_speaker_notes,
                    search_year_range=search_year_range,
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
