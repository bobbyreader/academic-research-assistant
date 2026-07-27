"""Nature paper writing skill.

Builds titles, abstracts, introductions, results narratives, discussions,
significance paragraphs, and first-submission packages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.skill_bridge import BaseSkill, SkillOutput, SkillStatus

from .prompts import PROMPTS


class WritingSkill(BaseSkill):
    """Nature paper writing module.

    Capabilities:
    - Title / abstract / introduction / results / discussion construction
    - Claim-evidence narrative organization
    - Chinese research notes to English manuscript paragraphs
    - Introduction logic-chain rebuilding
    - First-submission material package preparation
    - Reviewer recommendation and submission matrix
    - Pre-submission completeness checking
    """

    name: str = "writing"
    version: str = "0.1.0"
    description: str = (
        "Constructs Nature-style manuscript sections and submission packages."
    )

    SUPPORTED_TASKS: List[str] = [
        "title",
        "abstract",
        "introduction",
        "results",
        "discussion",
        "translation",
        "submission",
        "reviewer_recommendation",
        "completeness_check",
    ]

    def _validate_config(self) -> None:
        """Validate writing-specific configuration."""
        target_journal = self.config.get("target_journal", "")
        if target_journal and not isinstance(target_journal, str):
            raise ValueError("target_journal must be a string.")

    def execute(self, input_data: Dict[str, Any]) -> SkillOutput:
        """Execute a writing task.

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
            output.add_error(f"Writing skill failed: {exc}")
            return output

    # ------------------------------------------------------------------
    # Task handlers
    # ------------------------------------------------------------------
    def _handle_title(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle title generation/refinement."""
        prompt = PROMPTS["title"].get(subtask or "generate")
        if not prompt:
            output.add_error(f"Unknown title subtask: {subtask}")
            return output

        output.data = {
            "task": "title",
            "subtask": subtask or "generate",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "title_options": [
                {
                    "text": "Simulated Title Option 1",
                    "type": "declarative",
                    "character_count": 65,
                    "strengths": ["clear", "specific"],
                    "concerns": [],
                }
            ],
        }
        output.add_author_check("Select and refine the best title option.")
        return output

    def _handle_abstract(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle abstract construction."""
        prompt = PROMPTS["abstract"].get(subtask or "structured")
        if not prompt:
            output.add_error(f"Unknown abstract subtask: {subtask}")
            return output

        output.data = {
            "task": "abstract",
            "subtask": subtask or "structured",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "abstract": {
                "text": "Simulated abstract text...",
                "word_count": 150,
                "structure": "structured" if subtask != "unstructured" else "unstructured",
            },
        }
        output.add_author_check("Verify abstract accuracy against full manuscript.")
        return output

    def _handle_introduction(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle introduction construction."""
        prompt = PROMPTS["introduction"].get(subtask or "full")
        if not prompt:
            output.add_error(f"Unknown introduction subtask: {subtask}")
            return output

        output.data = {
            "task": "introduction",
            "subtask": subtask or "full",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "introduction": {
                "paragraphs": [],
                "logic_chain": ["background", "gap", "question", "contribution"],
                "word_count_estimate": 800,
            },
        }
        output.add_author_check("Verify logic chain flow and citation accuracy.")
        return output

    def _handle_results(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle results narrative construction."""
        prompt = PROMPTS["results"].get(subtask or "narrative")
        if not prompt:
            output.add_error(f"Unknown results subtask: {subtask}")
            return output

        output.data = {
            "task": "results",
            "subtask": subtask or "narrative",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "results_narrative": {
                "sections": [],
                "claim_evidence_map": {},
                "figure_references": [],
            },
        }
        output.add_author_check("Verify all claims are supported by data.")
        return output

    def _handle_discussion(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle discussion construction."""
        prompt = PROMPTS["discussion"].get(subtask or "full")
        if not prompt:
            output.add_error(f"Unknown discussion subtask: {subtask}")
            return output

        output.data = {
            "task": "discussion",
            "subtask": subtask or "full",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "discussion": {
                "paragraphs": [],
                "limitations_acknowledged": [],
                "implications": [],
            },
        }
        output.add_author_check("Verify appropriate hedging and no overclaiming.")
        return output

    def _handle_translation(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle Chinese-to-English translation."""
        prompt = PROMPTS["translation"].get(subtask or "paragraph")
        if not prompt:
            output.add_error(f"Unknown translation subtask: {subtask}")
            return output

        chinese_text = payload.get("chinese_text", "")
        if not chinese_text:
            output.add_error("Missing required payload field: chinese_text")
            return output

        output.data = {
            "task": "translation",
            "subtask": subtask or "paragraph",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "translation": {
                "source_text": chinese_text,
                "translated_text": "Simulated English translation...",
                "terminology_notes": {},
                "confidence_flags": [],
            },
        }
        output.add_author_check("Review translation for technical accuracy.")
        return output

    def _handle_submission(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle submission package preparation."""
        prompt = PROMPTS["submission"].get(subtask or "cover_letter")
        if not prompt:
            output.add_error(f"Unknown submission subtask: {subtask}")
            return output

        output.data = {
            "task": "submission",
            "subtask": subtask or "cover_letter",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "submission_materials": {
                "cover_letter": "Simulated cover letter...",
                "title_page": {},
                "highlights": [],
                "author_contributions": {},
                "data_availability": "",
            },
        }
        output.add_author_check("Verify all submission materials for accuracy.")
        return output

    def _handle_reviewer_recommendation(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle reviewer recommendation."""
        prompt = PROMPTS["reviewer_recommendation"].get(subtask or "suggest_reviewers")
        if not prompt:
            output.add_error(f"Unknown reviewer_recommendation subtask: {subtask}")
            return output

        output.data = {
            "task": "reviewer_recommendation",
            "subtask": subtask or "suggest_reviewers",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "reviewer_matrix": {
                "suggested_reviewers": [],
                "opposed_reviewers": [],
                "editor_suggestions": [],
            },
        }
        output.add_author_check("Verify reviewer suggestions for conflicts of interest.")
        return output

    def _handle_completeness_check(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle pre-submission completeness check."""
        prompt = PROMPTS["completeness_check"].get(subtask or "full_check")
        if not prompt:
            output.add_error(f"Unknown completeness_check subtask: {subtask}")
            return output

        output.data = {
            "task": "completeness_check",
            "subtask": subtask or "full_check",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "completeness_report": {
                "checklist": [],
                "missing_items": [],
                "format_issues": [],
                "recommendation": "ready_with_author_checks",
            },
        }
        output.add_warning("Completeness check is simulated. Verify all items manually.")
        return output
