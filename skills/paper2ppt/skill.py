"""Paper to PPT conversion skill.

Converts research papers, PDFs, preprints, or reading notes into
10-16 slide Chinese PPT presentations with speaker notes and QA reports.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.skill_bridge import BaseSkill, SkillOutput, SkillStatus

from .prompts import PROMPTS


class Paper2PPTSkill(BaseSkill):
    """Paper to PPT conversion module.

    Capabilities:
    - Extract research questions, key claims, core evidence, limitations,
      and reusable value from papers
    - Select and optimize key figures (including dense figure splitting)
    - Generate 10-16 slide Chinese PPT structure
    - Create editable .pptx content with speaker notes
    - Produce lightweight QA reports
    - Adapt for multiple scenarios (group meeting, literature review,
      thesis defense, conference)
    """

    name: str = "paper2ppt"
    version: str = "0.1.0"
    description: str = (
        "Converts academic papers into structured Chinese PPT presentations."
    )

    SUPPORTED_TASKS: List[str] = [
        "content_extraction",
        "slide_design",
        "speaker_notes",
        "qa_report",
        "scenarios",
    ]

    SUPPORTED_SCENARIOS: List[str] = [
        "group_meeting",
        "literature_review",
        "thesis_defense",
        "conference_presentation",
    ]

    def _validate_config(self) -> None:
        """Validate paper2ppt-specific configuration."""
        slide_count = self.config.get("target_slide_count", 12)
        if not isinstance(slide_count, int) or not 10 <= slide_count <= 16:
            raise ValueError(
                f"target_slide_count must be an integer between 10 and 16, "
                f"got {slide_count}."
            )

    def execute(self, input_data: Dict[str, Any]) -> SkillOutput:
        """Execute a paper2ppt task.

        Args:
            input_data: Dictionary with keys:
                - task (str): one of SUPPORTED_TASKS
                - subtask (str): specific subtask within the task
                - payload (dict): task-specific input data

        Returns:
            SkillOutput with status, data, errors, warnings, and
            author_checks.
        """
        output = self._create_output()

        try:
            task = input_data.get("task", "").strip()
            subtask = input_data.get("subtask", "").strip()
            payload = input_data.get("payload", {})

            if not task:
                output.add_error("Missing required field: task")
                return output

            if task not in self.SUPPORTED_TASKS:
                output.add_error(
                    f"Unknown task: {task}. Supported: {self.SUPPORTED_TASKS}"
                )
                return output

            handler = getattr(self, f"_handle_{task}", None)
            if handler is None:
                output.add_error(f"No handler implemented for task: {task}")
                return output

            return handler(subtask, payload, output)

        except Exception as exc:  # noqa: BLE001
            output.add_error(f"Paper2PPT skill failed: {exc}")
            return output

    # ------------------------------------------------------------------
    # Task handlers
    # ------------------------------------------------------------------
    def _handle_content_extraction(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle extraction of key content from papers."""
        valid_subtasks = [
            "research_question",
            "key_claims",
            "core_evidence",
            "limitations",
            "reusable_value",
        ]
        extraction_type = subtask or "research_question"
        if extraction_type not in valid_subtasks:
            output.add_error(
                f"Unknown extraction type: {extraction_type}. "
                f"Available: {valid_subtasks}"
            )
            return output

        prompt = PROMPTS["content_extraction"].get(extraction_type)
        if not prompt:
            output.add_error(f"No prompt template for extraction: {extraction_type}")
            return output

        content = payload.get("content", "")
        if not content:
            output.add_error("Missing required payload field: content")
            return output

        output.data = {
            "task": "content_extraction",
            "extraction_type": extraction_type,
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "extraction_result": {
                "type": extraction_type,
                "extracted_items": [],
                "summary": "Simulated extraction result...",
                "presentation_priority": "medium",
            },
        }
        output.add_author_check("Verify extracted content accuracy against source paper.")
        return output

    def _handle_slide_design(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle PPT slide structure and figure selection."""
        valid_subtasks = ["slide_structure", "figure_selection", "dense_figure_split"]
        design_task = subtask or "slide_structure"
        if design_task not in valid_subtasks:
            output.add_error(
                f"Unknown slide design task: {design_task}. "
                f"Available: {valid_subtasks}"
            )
            return output

        prompt = PROMPTS["slide_design"].get(design_task)
        if not prompt:
            output.add_error(f"No prompt template for design task: {design_task}")
            return output

        output.data = {
            "task": "slide_design",
            "design_task": design_task,
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "slide_design": {
                "slide_count": self.config.get("target_slide_count", 12),
                "slides": [],
                "figure_assignments": {},
                "layout_recommendations": [],
            },
        }
        output.add_author_check("Review slide structure and figure selections.")
        return output

    def _handle_speaker_notes(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle speaker notes generation."""
        valid_subtasks = ["comprehensive_notes", "q_a_preparation"]
        notes_task = subtask or "comprehensive_notes"
        if notes_task not in valid_subtasks:
            output.add_error(
                f"Unknown speaker notes task: {notes_task}. "
                f"Available: {valid_subtasks}"
            )
            return output

        prompt = PROMPTS["speaker_notes"].get(notes_task)
        if not prompt:
            output.add_error(f"No prompt template for notes task: {notes_task}")
            return output

        output.data = {
            "task": "speaker_notes",
            "notes_task": notes_task,
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "speaker_notes": {
                "slides_with_notes": [],
                "q_a_preparation": [],
                "timing_guide": {},
                "delivery_tips": [],
            },
        }
        output.add_author_check("Review and personalize speaker notes.")
        return output

    def _handle_qa_report(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle QA report generation."""
        valid_subtasks = ["content_accuracy", "presentation_quality", "accessibility_check"]
        qa_task = subtask or "content_accuracy"
        if qa_task not in valid_subtasks:
            output.add_error(
                f"Unknown QA task: {qa_task}. Available: {valid_subtasks}"
            )
            return output

        prompt = PROMPTS["qa_report"].get(qa_task)
        if not prompt:
            output.add_error(f"No prompt template for QA task: {qa_task}")
            return output

        output.data = {
            "task": "qa_report",
            "qa_task": qa_task,
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "qa_report": {
                "qa_type": qa_task,
                "passed_checks": [],
                "failed_checks": [],
                "warnings": [],
                "overall_score": 0.0,
                "recommendations": [],
            },
        }
        output.add_author_check("Address all failed QA checks before presentation.")
        return output

    def _handle_scenarios(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle scenario-specific presentation adaptation."""
        scenario = subtask or "group_meeting"
        if scenario not in self.SUPPORTED_SCENARIOS:
            output.add_error(
                f"Unknown scenario: {scenario}. "
                f"Available: {self.SUPPORTED_SCENARIOS}"
            )
            return output

        prompt = PROMPTS["scenarios"].get(scenario)
        if not prompt:
            output.add_error(f"No prompt template for scenario: {scenario}")
            return output

        output.data = {
            "task": "scenarios",
            "scenario": scenario,
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "scenario_adaptation": {
                "scenario": scenario,
                "adapted_slides": [],
                "emphasis_adjustments": [],
                "audience_considerations": [],
                "special_preparations": [],
            },
        }
        output.add_author_check(f"Verify {scenario} adaptation meets audience needs.")
        return output
