"""配置必须**产生影响**，而不仅是被读取。

为什么需要它（Phase 9 的真实事故）
----------------------------------
Phase 8 建立了 `tests/test_config_honesty.py`，验证"每个宣称可配置的键都真的被读取"。
Phase 9 的第一次真实端到端运行发现：**"被读取"不等于"生效"**。

`export.pdf_engine` 就是活证据——它被读取、被传递、被存进 `ResearchPipelineConfig`，
然后**再也没有人读它**：

    research_service.py:149   export_pdf_engine = get_str(settings, "export.pdf_engine")  # 读了
    research_service.py:280   export_pdf_engine=export_pdf_engine                         # 存了
    research_pipeline.py:107  export_pdf_engine: str = ""                                 # 定义了
    orchestrator.py:388       export_pdf(manuscript, output_path)                         # 真导出不传 engine

于是把它改成 `pandoc` 或 `reportlab` 对 CLI/Web 导出**毫无影响**，而
`test_config_honesty.py` 是绿的——它只检查"读取"，不检查"读取之后发生了什么"。

本文件补上这一半：**配置载体（dataclass）上的每个字段都必须被读取**。
"只赋值、从不读取"的字段是死配置——它让配置看起来生效，实际不生效。

一个字段**为什么**不该出现在配置载体上
--------------------------------------
如果某处代码不需要这个值，那这个值就不该被搬到那里。删掉字段、或让真正的消费者
去读取它，二者必居其一；**留着它是最坏的选择**——它会让后来读代码的人以为
"这里已经处理过了"。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.research_pipeline import ResearchPipelineConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILES = sorted(
    [*REPO_ROOT.glob("core/*.py"), REPO_ROOT / "orchestrator.py", REPO_ROOT / "web_app.py"]
)


def _dataclass_field_names(cls: type) -> list[str]:
    """取 dataclass 的字段名（按定义顺序）。"""
    return list(cls.__dataclass_fields__)


def _attribute_reads(field: str) -> list[str]:
    """返回所有 `X.<field>` 形式的属性访问位置（`文件:行`）。

    只看 `Attribute` 节点（即 `config.export_pdf_engine` 这种**读取**），
    不看 dataclass 的字段定义（那是 `AnnAssign`，不是属性访问）。
    """
    hits: list[str] = []
    for path in SOURCE_FILES:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == field:
                hits.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return hits


def test_every_pipeline_config_field_is_actually_read() -> None:
    """`ResearchPipelineConfig` 的每个字段都必须至少被读取一次。

    字段被赋值却从不被读取 = 死配置：它把配置值搬到一个用不到它的地方，
    让"配置已生效"看起来成立。`export.pdf_engine` 就是这样骗过了 Phase 8 的门禁。
    """
    fields = _dataclass_field_names(ResearchPipelineConfig)
    assert fields, "ResearchPipelineConfig 没有字段？请检查该测试是否已过期"

    never_read = sorted(field for field in fields if not _attribute_reads(field))

    assert not never_read, (
        "以下 ResearchPipelineConfig 字段被赋值、但**从未被任何代码读取**——"
        "它们是死配置，会让用户以为配置生效了："
        f"{never_read}\n"
        "正解是二选一：删掉字段，或让真正的消费者去读取它。"
    )


def test_the_check_can_actually_fail() -> None:
    """**证明这条检查能失败**——否则它是同义反复。

    `realapi-dev` 在 Phase 9 做过同样的事：把被验证的实现改成空操作，确认断言会失败。
    一条永远不会失败的测试，和没有测试是一样的。
    """
    assert _attribute_reads("__a_field_that_cannot_possibly_exist__") == []


@pytest.mark.parametrize("field", ["project_name", "topic"])
def test_known_live_fields_are_read(field: str) -> None:
    """抽样正向对照：确实被使用的字段必须能查到读取点。

    若这两个基础字段都查不到读取点，说明本文件的静态分析写错了（而不是产品有问题）
    ——**先怀疑自己的验证手段**。
    """
    assert _attribute_reads(field), f"{field} 应当被读取，但静态分析没找到——先检查本测试的取数逻辑"
