"""Nature paper polishing and translation skill.

Provides Chinese-to-English academic translation, text condensation,
Nature-style section adjustment, AI-tell detection, and hourglass
structure optimization.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.skill_bridge import BaseSkill, SkillOutput, SkillStatus

from .prompts import PROMPTS


class PolishingSkill(BaseSkill):
    """Nature paper polishing and translation module.

    Capabilities:
    - Chinese-to-English academic paragraph translation
    - Text condensation and clarity enhancement
    - Nature / Nature Communications section style adjustment
    - Research paper vs methods paper writing emphasis
    - AI-tell detection and humanization
    - Hourglass structure and section moves optimization
    - Academic Phrasebank integration
    """

    name: str = "polishing"
    version: str = "0.1.0"
    description: str = (
        "Polishes and translates academic text to Nature publication standards."
    )

    SUPPORTED_TASKS: List[str] = [
        "translation",
        "polishing",
        "nature_style",
        "ai_detection",
        "hourglass",
        "phrasebank",
    ]

    def _validate_config(self) -> None:
        """Validate polishing-specific configuration."""
        target_journal = self.config.get("target_journal", "nature")
        if target_journal not in ("nature", "nature_communications", "other"):
            raise ValueError(
                f"Unsupported target_journal: {target_journal}. "
                "Use 'nature', 'nature_communications', or 'other'."
            )

    def execute(self, input_data: Dict[str, Any]) -> SkillOutput:
        """Execute a polishing task.

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
            output.add_error(f"Polishing skill failed: {exc}")
            return output

    # ------------------------------------------------------------------
    # Task handlers
    # ------------------------------------------------------------------
    def _handle_translation(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle Chinese-to-English academic translation."""
        prompt = PROMPTS["translation"].get(subtask or "academic_paragraph")
        if not prompt:
            output.add_error(f"Unknown translation subtask: {subtask}")
            return output

        chinese_text = payload.get("chinese_text", "")
        if not chinese_text:
            output.add_error("Missing required payload field: chinese_text")
            return output

        output.data = {
            "task": "translation",
            "subtask": subtask or "academic_paragraph",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "translation_result": {
                "source_text": chinese_text,
                "translated_text": "Simulated publication-quality English translation...",
                "alternative_phrasings": [],
                "terminology_notes": {},
                "confidence_flags": [],
            },
        }
        output.add_author_check(
            "Review translated text for technical accuracy and field-specific terminology."
        )
        return output

    def _handle_polishing(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle text polishing (condense, clarity, argument strengthen)."""
        prompt = PROMPTS["polishing"].get(subtask or "condense")
        if not prompt:
            output.add_error(f"Unknown polishing subtask: {subtask}")
            return output

        text = payload.get("text", "")
        if not text:
            output.add_error("Missing required payload field: text")
            return output

        output.data = {
            "task": "polishing",
            "subtask": subtask or "condense",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "polishing_result": {
                "original_text": text,
                "polished_text": "Simulated polished text...",
                "change_log": [],
                "readability_metrics": {"before": 0, "after": 0},
            },
        }
        output.add_author_check("Verify polished text preserves all key messages.")
        return output

    def _handle_nature_style(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle Nature journal style adaptation."""
        prompt = PROMPTS["nature_style"].get(subtask or "section_adjustment")
        if not prompt:
            output.add_error(f"Unknown nature_style subtask: {subtask}")
            return output

        text = payload.get("text", "")
        section_type = payload.get("section_type", "")
        if not text or not section_type:
            output.add_error(
                "Missing required payload fields: text and section_type"
            )
            return output

        output.data = {
            "task": "nature_style",
            "subtask": subtask or "section_adjustment",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "style_adjustment": {
                "original_text": text,
                "adjusted_text": "Simulated Nature-style adjusted text...",
                "section_type": section_type,
                "changes_made": [],
                "compliance_notes": [],
            },
        }
        output.add_author_check("Verify section-specific style compliance.")
        return output

    def _handle_ai_detection(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle AI-tell detection and humanization."""
        prompt = PROMPTS["ai_detection"].get(subtask or "ai_tells_check")
        if not prompt:
            output.add_error(f"Unknown ai_detection subtask: {subtask}")
            return output

        text = payload.get("text", "")
        if not text:
            output.add_error("Missing required payload field: text")
            return output

        output.data = {
            "task": "ai_detection",
            "subtask": subtask or "ai_tells_check",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "ai_detection_result": {
                "original_text": text,
                "flagged_passages": [],
                "ai_likelihood_score": 0.0,
                "humanized_text": "Simulated humanized text...",
                "recommendations": [],
            },
        }
        output.add_author_check("Review flagged passages and approve humanization changes.")
        return output

    def _handle_hourglass(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle hourglass structure and section moves optimization."""
        prompt = PROMPTS["hourglass"].get(subtask or "structure_check")
        if not prompt:
            output.add_error(f"Unknown hourglass subtask: {subtask}")
            return output

        output.data = {
            "task": "hourglass",
            "subtask": subtask or "structure_check",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "structure_analysis": {
                "hourglass_compliance": {},
                "section_moves": {},
                "flow_diagram": "",
                "restructuring_recommendations": [],
            },
        }
        output.add_author_check("Verify hourglass structure and section moves.")
        return output

    def _handle_phrasebank(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle Academic Phrasebank integration."""
        prompt = PROMPTS["phrasebank"].get(subtask or "phrase_suggestion")
        if not prompt:
            output.add_error(f"Unknown phrasebank subtask: {subtask}")
            return output

        output.data = {
            "task": "phrasebank",
            "subtask": subtask or "phrase_suggestion",
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "phrasebank_result": {
                "suggested_phrases": [],
                "usage_notes": {},
                "hedge_optimization": {},
            },
        }
        output.add_author_check("Select appropriate phrases for your writing context.")
        return output
