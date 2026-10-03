"""相关性顾问级关口的对抗性测试。

所有测试都用假的 LLM 对象，绝不触网。重点覆盖四类失败模式：

* **绝不删文献**：判定与警告只做标记，传入的 ``papers`` 列表长度与内容必须不变；
* **反编造**：模型返回越界/未知 ``citation_id`` 必须被丢弃；
* **覆盖保证**：模型漏判某篇 → 该篇仍有一条 ``unclear`` verdict；
* **成本与失败语义**：非空恰好 1 次调用、空 0 次；模型抛异常 → ``ran=False``
  且 ``passed is False``，警告明确说明「未执行」。
"""

from __future__ import annotations

import json

import pytest

from core.relevance_verifier import (
    RELEVANCE_VALUES,
    RelevanceReport,
    RelevanceVerdict,
    verify_relevance,
)
from core.research_models import ArtifactDecodeError, PaperRecord

TOPIC = "Urban heat islands and human mortality"

P1_TITLE = "Urban heat and mortality"
P1_ABSTRACT = "Urban heat islands raise cardiovascular mortality in elderly residents."
P2_TITLE = "Green roofs"
P2_ABSTRACT = "Green roofs reduce indoor temperatures during summer heat waves."
P3_TITLE = "Quantum optics"
P3_ABSTRACT = "Cold atom interferometry for precision measurement of the fine constant."


class FakeLLM:
    """Deterministic stand-in for an LLM client (no network)."""

    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover
        raise AssertionError("complete() must not be used by the verifier")

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict:
        self.calls.append((system_prompt, user_prompt))
        if isinstance(self._response, Exception):
            raise self._response
        return self._response  # type: ignore[return-value]


def _papers() -> list[PaperRecord]:
    return [
        PaperRecord(title=P1_TITLE, abstract=P1_ABSTRACT),
        PaperRecord(title=P2_TITLE, abstract=P2_ABSTRACT),
        PaperRecord(title=P3_TITLE, abstract=P3_ABSTRACT),
    ]


def _verdict(citation_id: str, relevance: str, *, rationale: str = "judged") -> dict:
    return {
        "citation_id": citation_id,
        "relevance": relevance,
        "rationale": rationale,
    }


# --------------------------------------------------------------------------- #
# 1. 全部相关 → 通过
# --------------------------------------------------------------------------- #
def test_all_relevant_passes_without_warnings() -> None:
    llm = FakeLLM(
        {
            "verdicts": [
                _verdict("P1", "relevant"),
                _verdict("P2", "relevant"),
            ]
        }
    )

    report = verify_relevance(llm, TOPIC, _papers()[:2])

    assert report.passed is True
    assert report.warnings() == []
    assert len(report.verdicts) == 2


# --------------------------------------------------------------------------- #
# 2. 对抗性：明显不相关的文献被判 irrelevant → 记录，但检索结果未被修改
# --------------------------------------------------------------------------- #
def test_irrelevant_paper_is_recorded_and_papers_are_not_mutated() -> None:
    papers = _papers()
    before = [(paper.title, paper.abstract) for paper in papers]
    llm = FakeLLM(
        {
            "verdicts": [
                _verdict("P1", "relevant"),
                _verdict("P2", "relevant"),
                _verdict("P3", "irrelevant", rationale="量子光学与城市热岛无关"),
            ]
        }
    )

    report = verify_relevance(llm, TOPIC, papers)

    assert report.passed is False
    assert [item.citation_id for item in report.irrelevant] == ["P3"]
    assert any("明显不相关" in warning for warning in report.warnings())
    # 硬要求：绝不删除文献——传入的列表长度与内容必须原封不动。
    assert len(papers) == 3
    assert [(paper.title, paper.abstract) for paper in papers] == before


# --------------------------------------------------------------------------- #
# 3. 覆盖保证：模型漏判某篇 → 该篇仍有一条 unclear verdict
# --------------------------------------------------------------------------- #
def test_missing_verdict_is_filled_with_unclear() -> None:
    llm = FakeLLM({"verdicts": [_verdict("P1", "relevant")]})

    report = verify_relevance(llm, TOPIC, _papers()[:2])

    by_id = {item.citation_id: item for item in report.verdicts}
    assert set(by_id) == {"P1", "P2"}
    assert by_id["P2"].relevance == "unclear"
    assert "模型未对该文献给出判定" in by_id["P2"].rationale


# --------------------------------------------------------------------------- #
# 4. 反编造：模型返回越界标识 P99 → 丢弃，不进入 verdicts
# --------------------------------------------------------------------------- #
def test_out_of_range_id_is_dropped() -> None:
    llm = FakeLLM(
        {
            "verdicts": [
                _verdict("P1", "relevant"),
                _verdict("P99", "irrelevant"),  # 越界 → 丢弃
                _verdict("PX", "irrelevant"),  # 格式非法 → 丢弃
            ]
        }
    )

    report = verify_relevance(llm, TOPIC, _papers()[:2])

    ids = [item.citation_id for item in report.verdicts]
    assert ids == ["P1", "P2"]
    assert all(item.citation_id != "P99" for item in report.verdicts)
    assert report.passed is True  # 越界标识被丢弃，未污染判定


# --------------------------------------------------------------------------- #
# 5. 成本：3 篇 → 恰好 1 次调用；空列表 → 0 次
# --------------------------------------------------------------------------- #
def test_three_papers_use_exactly_one_call() -> None:
    llm = FakeLLM({"verdicts": []})

    verify_relevance(llm, TOPIC, _papers())

    assert len(llm.calls) == 1


def test_empty_papers_use_no_call_and_are_ran_true() -> None:
    llm = FakeLLM({"verdicts": []})

    report = verify_relevance(llm, TOPIC, [])

    assert llm.calls == []
    assert report.verdicts == []
    assert report.ran is True
    assert report.not_run_reason == ""
    # 确实执行且无事可做：应通过。
    assert report.passed is True


# --------------------------------------------------------------------------- #
# 6. 失败不抛：模型抛异常 → ran=False + reason，passed 为 False
# --------------------------------------------------------------------------- #
def test_model_failure_returns_not_run_without_raising() -> None:
    llm = FakeLLM(RuntimeError("upstream boom"))

    report = verify_relevance(llm, TOPIC, _papers())

    assert isinstance(report, RelevanceReport)
    assert report.ran is False
    assert report.not_run_reason
    assert report.passed is False


# --------------------------------------------------------------------------- #
# 7. ran=False 语义：警告明确说明「未执行」，不得被读作「检查通过」
# --------------------------------------------------------------------------- #
def test_not_run_warnings_say_not_executed() -> None:
    llm = FakeLLM(RuntimeError("boom"))

    report = verify_relevance(llm, TOPIC, _papers())

    joined = "".join(report.warnings())
    assert "未执行" in joined
    assert "不等于检查通过" in joined
    assert report.passed is False


# --------------------------------------------------------------------------- #
# 8. 噪声：超过半数被判不相关 → 概括说明，不逐条罗列
# --------------------------------------------------------------------------- #
def test_majority_flagged_warns_about_possible_noise() -> None:
    llm = FakeLLM(
        {
            "verdicts": [
                _verdict("P1", "irrelevant"),
                _verdict("P2", "irrelevant"),
                _verdict("P3", "relevant"),
            ]
        }
    )

    report = verify_relevance(llm, TOPIC, _papers())

    joined = "".join(report.warnings())
    assert "超过半数" in joined
    assert "主题" in joined or "判定过严" in joined


# --------------------------------------------------------------------------- #
# 9. 未知取值 → 保守回退为 unclear
# --------------------------------------------------------------------------- #
def test_unrecognised_relevance_falls_back_to_unclear() -> None:
    llm = FakeLLM({"verdicts": [_verdict("P1", "maybe-ish")]})

    report = verify_relevance(llm, TOPIC, _papers()[:1])

    assert report.verdicts[0].relevance == "unclear"


# --------------------------------------------------------------------------- #
# 10. 序列化往返 + 缺新键的旧产物兼容 + 非法 payload 抛错
# --------------------------------------------------------------------------- #
def test_roundtrip_to_dict_from_dict() -> None:
    llm = FakeLLM({"verdicts": [_verdict("P1", "irrelevant", rationale="无关")]})

    report = verify_relevance(llm, TOPIC, _papers()[:1])
    restored = RelevanceReport.from_dict(report.to_dict())

    assert restored.to_dict() == report.to_dict()
    assert restored.ran is True


def test_from_dict_without_new_keys_defaults_to_ran_true() -> None:
    # 模拟没有 ran / not_run_reason 的旧产物。
    payload = {
        "passed": True,
        "verdict_count": 1,
        "irrelevant_count": 0,
        "unclear_count": 0,
        "verdicts": [
            {
                "citation_id": "P1",
                "paper_title": P1_TITLE,
                "relevance": "relevant",
                "rationale": "相关",
            }
        ],
    }

    report = RelevanceReport.from_dict(payload)

    assert report.ran is True
    assert report.not_run_reason == ""
    assert report.passed is True


@pytest.mark.parametrize(
    "payload",
    [
        "not a dict",
        {"verdicts": "not a list"},
        {"verdicts": [{"citation_id": 1}]},
    ],
)
def test_from_dict_rejects_invalid_payload(payload: object) -> None:
    with pytest.raises(ArtifactDecodeError):
        RelevanceReport.from_dict(payload)


def test_artifact_decode_error_is_value_error() -> None:
    assert issubclass(ArtifactDecodeError, ValueError)


# --------------------------------------------------------------------------- #
# 11. Markdown 渲染
# --------------------------------------------------------------------------- #
def test_to_markdown_reports_marks_and_scope() -> None:
    llm = FakeLLM({"verdicts": [_verdict("P1", "irrelevant", rationale="无关")]})

    report = verify_relevance(llm, TOPIC, _papers()[:1])
    markdown = report.to_markdown()

    assert "相关性核验报告" in markdown
    assert "建议" in markdown
    assert "不会删除任何文献" in markdown
    assert "P1" in markdown
    assert P1_TITLE in markdown


def test_to_markdown_states_not_executed_when_not_run() -> None:
    llm = FakeLLM(RuntimeError("boom"))

    report = verify_relevance(llm, TOPIC, _papers())
    markdown = report.to_markdown()

    assert "未执行" in markdown
    assert "结论" in markdown


# --------------------------------------------------------------------------- #
# 契约健全性
# --------------------------------------------------------------------------- #
def test_relevance_values_are_exposed() -> None:
    assert RELEVANCE_VALUES == ("relevant", "unclear", "irrelevant")


def test_to_dict_exposes_stable_keys() -> None:
    llm = FakeLLM({"verdicts": [_verdict("P1", "relevant")]})

    report = verify_relevance(llm, TOPIC, _papers()[:1])
    payload = report.to_dict()

    for key in (
        "passed",
        "ran",
        "not_run_reason",
        "verdict_count",
        "irrelevant_count",
        "unclear_count",
        "verdicts",
    ):
        assert key in payload, f"missing key: {key}"

    assert set(payload["verdicts"][0]) == {
        "citation_id",
        "paper_title",
        "relevance",
        "rationale",
    }
    # 可 JSON 序列化（落盘前提）。
    json.dumps(payload, ensure_ascii=False)


def test_verdict_roundtrip() -> None:
    verdict = RelevanceVerdict(
        citation_id="P1",
        paper_title=P1_TITLE,
        relevance="relevant",
        rationale="相关",
    )

    assert RelevanceVerdict.from_dict(verdict.to_dict()) == verdict


# --------------------------------------------------------------------------- #
# 12. 确定性
# --------------------------------------------------------------------------- #
def test_identical_inputs_are_deterministic() -> None:
    def run() -> dict:
        llm = FakeLLM(
            {
                "verdicts": [
                    _verdict("P1", "relevant", rationale="a"),
                    _verdict("P2", "unclear", rationale="b"),
                    _verdict("P3", "irrelevant", rationale="c"),
                ]
            }
        )
        return verify_relevance(llm, TOPIC, _papers()).to_dict()

    assert run() == run()
