"""Nature simulated reviewer skill.

Generates three distinct reviewer reports, cross-review synthesis,
issue flagging, 12-axis technical checklist, and claim-evidence mapping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.skill_bridge import BaseSkill, SkillOutput, SkillStatus

from .prompts import PROMPTS


class ReviewerSkill(BaseSkill):
    """Nature simulated reviewer module.

    Capabilities:
    - Nature official dimension assessment (originality, importance,
      cross-disciplinary appeal, technical rigor, readability)
    - Three distinct reviewer reports with different emphases
    - Cross-review synthesis
    - Issue flagging (unsupported claims, technical flaws, evidence
      chain breaks, comprehension barriers)
    - Internal 12-axis technical checklist
    - Claim pointer and evidence location binding
    - Repetition checking across reviews (consensus requires >=2 reviewers)
    """

    name: str = "reviewer"
    version: str = "0.1.0"
    description: str = (
        "Simulates Nature peer review with multiple reviewer perspectives "
        "and comprehensive technical assessment."
    )

    SUPPORTED_TASKS: List[str] = [
        "nature_criteria",
        "reviewer_reports",
        "issue_flagging",
        "technical_checklist",
        "claim_pointers",
        "consensus",
    ]

    NATURE_DIMENSIONS: List[str] = [
        "originality",
        "scientific_importance",
        "cross_disciplinary_appeal",
        "technical_rigor",
        "readability",
    ]

    CHECKLIST_AXES: List[str] = [
        "axis_1_research_design",
        "axis_2_sample_size",
        "axis_3_measurement",
        "axis_4_statistical_methods",
        "axis_5_data_quality",
        "axis_6_reproducibility",
        "axis_7_ethics",
        "axis_8_reporting",
        "axis_9_figures_tables",
        "axis_10_references",
        "axis_11_interpretation",
        "axis_12_novelty",
    ]

    def _validate_config(self) -> None:
        """Validate reviewer-specific configuration."""
        strictness = self.config.get("strictness", "standard")
        if strictness not in ("lenient", "standard", "strict"):
            raise ValueError(
                f"Unsupported strictness: {strictness}. "
                "Use 'lenient', 'standard', or 'strict'."
            )

    def execute(self, input_data: Dict[str, Any]) -> SkillOutput:
        """Execute a reviewer task.

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
            output.add_error(f"Reviewer skill failed: {exc}")
            return output

    # ------------------------------------------------------------------
    # Task handlers
    # ------------------------------------------------------------------
    def _handle_nature_criteria(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle Nature official dimension assessment."""
        dimension = subtask or "originality"
        if dimension not in self.NATURE_DIMENSIONS:
            output.add_error(
                f"Unknown Nature dimension: {dimension}. "
                f"Available: {self.NATURE_DIMENSIONS}"
            )
            return output

        prompt = PROMPTS["nature_criteria"].get(dimension)
        if not prompt:
            output.add_error(f"No prompt template for dimension: {dimension}")
            return output

        manuscript = payload.get("manuscript", "")
        if not manuscript:
            output.add_error("Missing required payload field: manuscript")
            return output

        output.data = {
            "task": "nature_criteria",
            "dimension": dimension,
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "assessment": {
                "dimension": dimension,
                "rating": "moderate",
                "justification": "Simulated assessment...",
                "evidence": [],
                "recommendations": [],
            },
        }
        output.add_author_check(f"Review {dimension} assessment for accuracy.")
        return output

    def _handle_reviewer_reports(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle generation of individual or synthesized reviewer reports."""
        valid_subtasks = [
            "reviewer_1_methods",
            "reviewer_2_impact",
            "reviewer_3_clarity",
            "cross_review_synthesis",
        ]
        report_type = subtask or "reviewer_1_methods"
        if report_type not in valid_subtasks:
            output.add_error(
                f"Unknown reviewer report type: {report_type}. "
                f"Available: {valid_subtasks}"
            )
            return output

        prompt = PROMPTS["reviewer_reports"].get(report_type)
        if not prompt:
            output.add_error(f"No prompt template for report type: {report_type}")
            return output

        manuscript = payload.get("manuscript", "")
        if not manuscript and report_type != "cross_review_synthesis":
            output.add_error("Missing required payload field: manuscript")
            return output

        output.data = {
            "task": "reviewer_reports",
            "report_type": report_type,
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "review_report": {
                "reviewer_id": report_type,
                "summary": "Simulated review...",
                "major_strengths": [],
                "major_concerns": [],
                "minor_comments": [],
                "recommendation": "major_revision",
                "confidence": "medium",
            },
        }
        output.add_author_check("Review simulated report and replace with actual LLM output.")
        return output

    def _handle_issue_flagging(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle issue flagging (unsupported claims, technical flaws, etc.)."""
        valid_subtasks = [
            "unsupported_claims",
            "technical_flaws",
            "evidence_chain_breaks",
            "comprehension_barriers",
        ]
        issue_type = subtask or "unsupported_claims"
        if issue_type not in valid_subtasks:
            output.add_error(
                f"Unknown issue type: {issue_type}. Available: {valid_subtasks}"
            )
            return output

        prompt = PROMPTS["issue_flagging"].get(issue_type)
        if not prompt:
            output.add_error(f"No prompt template for issue type: {issue_type}")
            return output

        output.data = {
            "task": "issue_flagging",
            "issue_type": issue_type,
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "issue_report": {
                "issue_type": issue_type,
                "flagged_items": [],
                "severity_distribution": {"critical": 0, "major": 0, "minor": 0},
                "overall_risk": "low",
            },
        }
        output.add_author_check("Review flagged issues and prioritize corrections.")
        return output

    def _handle_technical_checklist(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle 12-axis technical checklist assessment."""
        axis = subtask or "axis_1_research_design"
        if axis not in self.CHECKLIST_AXES:
            output.add_error(
                f"Unknown checklist axis: {axis}. Available: {self.CHECKLIST_AXES}"
            )
            return output

        prompt = PROMPTS["technical_checklist"].get(axis)
        if not prompt:
            output.add_error(f"No prompt template for axis: {axis}")
            return output

        output.data = {
            "task": "technical_checklist",
            "axis": axis,
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "checklist_result": {
                "axis": axis,
                "rating": "pass",
                "justification": "Simulated checklist assessment...",
                "concerns": [],
                "recommendations": [],
            },
        }
        output.add_author_check(f"Verify {axis} assessment.")
        return output

    def _handle_claim_pointers(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle claim extraction and evidence mapping."""
        valid_subtasks = ["claim_extraction", "evidence_mapping", "verification_guide"]
        pointer_task = subtask or "claim_extraction"
        if pointer_task not in valid_subtasks:
            output.add_error(
                f"Unknown claim pointer task: {pointer_task}. "
                f"Available: {valid_subtasks}"
            )
            return output

        prompt = PROMPTS["claim_pointers"].get(pointer_task)
        if not prompt:
            output.add_error(f"No prompt template for task: {pointer_task}")
            return output

        output.data = {
            "task": "claim_pointers",
            "pointer_task": pointer_task,
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "claim_pointer_result": {
                "claims": [],
                "evidence_map": {},
                "verification_guide": [],
            },
        }
        output.add_author_check("Verify claim-evidence mappings for completeness.")
        return output

    def _handle_consensus(
        self, subtask: str, payload: Dict[str, Any], output: SkillOutput
    ) -> SkillOutput:
        """Handle reviewer consensus and repetition checking."""
        valid_subtasks = ["repetition_check", "consensus_synthesis"]
        consensus_task = subtask or "repetition_check"
        if consensus_task not in valid_subtasks:
            output.add_error(
                f"Unknown consensus task: {consensus_task}. "
                f"Available: {valid_subtasks}"
            )
            return output

        prompt = PROMPTS["consensus"].get(consensus_task)
        if not prompt:
            output.add_error(f"No prompt template for task: {consensus_task}")
            return output

        reviews = payload.get("reviews", [])
        if len(reviews) < 2:
            output.add_error(
                "Consensus checking requires at least 2 reviewer reports."
            )
            return output

        output.data = {
            "task": "consensus",
            "consensus_task": consensus_task,
            "prompt_used": prompt[:200] + "...",
            "payload_received": payload,
            "consensus_result": {
                "consensus_issues": [],
                "unique_issues": [],
                "contradictions": [],
                "overall_recommendation": "major_revision",
                "priority_revision_list": [],
            },
        }
        output.add_author_check(
            "Review consensus synthesis. Note: issues require >=2 reviewers to be listed as consensus."
        )
        return output
