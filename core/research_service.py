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
        # 注意：``export.pdf_engine`` / ``export.pptx_template`` /
        # ``export.include_speaker_notes`` 只被**交付层**消费（``Orchestrator`` /
        # ``export_service``），不进入管线，因此这里-不-读取，也不传给
        # ``ResearchPipelineConfig``（后者刻意不持有这些字段）。留在这里只会让读者
        # 以为"管线用到了它"。``export.default_format`` 例外：它是管线字段，下面照常传。
        export_default_format = get_str(settings, "export.default_format")
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
            # 单调性守卫：状态一旦 `completed` / `cancelled` 就**不再被覆盖或降级**。
            # 否则异常路径（`except` 里把 `active_stage` 标成 "blocked"）会把一个**已经
            # 成功**的阶段改写成失败——状态文件对事实说谎，而它正是续跑判定的输入之一。
            if state.stage_status.get(next_stage) in {"completed", "cancelled"}:
                active_stage = next_stage
                state.current_stage = active_stage
                self.state_manager.save(state)
                if on_progress:
                    on_progress(stage_name, status)
                return
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

        def completed_steps() -> set[str]:
            raw = state.metadata.get(COMPLETED_STEPS_KEY)
            return set(raw) if isinstance(raw, dict) else set()

        #: 工作流主干顺序，用于定位"被中断时尚未进入的下一个阶段"。
        _STAGE_ORDER: tuple[WorkflowStage, ...] = (
            WorkflowStage.BRAINSTORMING,
            WorkflowStage.SEARCH,
            WorkflowStage.LIT_REVIEW,
            WorkflowStage.WRITING,
            WorkflowStage.EXPORT,
        )

        def _is_done(stage: WorkflowStage) -> bool:
            """该阶段是否**确实成功过**。

            依据是 ``COMPLETED_STEPS_KEY`` 的键集——它与续跑判定读的是**同一份事实**
            （该阶段产物已产出并落盘指纹），因此状态标志与续跑不会互相矛盾。
            """
            return (
                stage.value in completed_steps()
                or state.stage_status.get(stage) in {"completed", "cancelled"}
            )

        def mark_failed(stage: WorkflowStage, outcome: str) -> None:
            """把本次运行"没有成功"这件事如实记下；**已成功的阶段绝不改写**。

            ``outcome`` 为 ``"blocked"``（异常）或 ``"cancelled"``（用户取消）。

            两种情形分开处理，因为事实不同：

            * ``stage`` **尚未成功** → 它才是失败/被取消的阶段：标 ``outcome``，
              ``current_stage`` 指向它。（对应"在 search 阶段内失败"这类可归属情形。）
            * ``stage`` **已经成功** → 运行是在"该阶段已成功、还没进入下一阶段"的
              边界处停下的。此时**不得**把 ``stage`` 改写成失败（Run 2 的
              ``search: blocked`` 就是这么把一次真实成功说成失败的）；但**也不能**不
              留痕迹（否则 state 会显示"全部完成"，掩盖"这次其实没跑完"）。
              因此把 ``outcome`` 记到**下一个尚未成功的阶段**上——那正是我们没能进入
              的阶段——并让 ``current_stage`` 指向它。

            无论哪种情形，都额外写一个 run 级标记 ``metadata["last_run_outcome"]``，
            使状态文件永远能回答"上一次运行成功了吗、停在哪、为什么"。
            """
            if not _is_done(stage):
                state.stage_status[stage] = outcome
                state.current_stage = stage
                stopped_at = stage
            else:
                # 该阶段确实成功过：保持 completed（不降级）。
                state.stage_status[stage] = "completed"
                stopped_at = stage
                for candidate in _STAGE_ORDER:
                    if not _is_done(candidate):
                        state.stage_status[candidate] = outcome
                        state.current_stage = candidate
                        stopped_at = candidate
                        break
            state.metadata["last_run_outcome"] = {
                "outcome": outcome,
                "stopped_at": stopped_at.value,
                "completed_stages": sorted(completed_steps()),
            }
            self.state_manager.save(state)

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
                    search_year_range=search_year_range,
                )
            )
        except ResearchPipelineCancelled:
            # 取消是用户**主动行为**，不是错误：状态记为 `"cancelled"` 而非
            # `"blocked"`，并把状态落盘，然后原样重新抛出（调用方需知道运行未完成）。
            # 用量的落盘由管线在自己的 finally 中完成，此处不重复。
            # mark_failed 保证：只有真正停在、且尚未成功的阶段才被标为 cancelled；
            # 已成功的阶段不因取消而被回退（取消发生在阶段边界，不影响已完成阶段）。
            mark_failed(active_stage, "cancelled")
            raise
        except Exception:
            # 只有**真正失败的那个阶段**才标 blocked，且 `current_stage` 指向它；
            # 任何已成功（已完成并落盘）的阶段一律保持 completed，不被覆盖。
            mark_failed(active_stage, "blocked")
            raise

        state.current_stage = WorkflowStage.EXPORT
        state.stage_status[WorkflowStage.EXPORT] = "ready_with_author_checks"
        # run 级成功标记：与失败路径对称，使状态文件始终能回答"上一次运行成功了吗"。
        state.metadata["last_run_outcome"] = {
            "outcome": "completed",
            "stopped_at": WorkflowStage.EXPORT.value,
            "completed_stages": sorted(completed_steps()),
        }
        self.state_manager.save(state)
        return result
