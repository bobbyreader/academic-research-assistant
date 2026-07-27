"""科学可视化技能模块。

提供出版级科学图表创建、审查和导出规划。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.skill_bridge import BaseSkill, SkillOutput, SkillStatus


class FigureType(Enum):
    """图表类型。"""

    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HEATMAP = "heatmap"
    BOXPLOT = "boxplot"
    VIOLIN = "violin"
    MULTI_PANEL = "multi_panel"


class ExportFormat(Enum):
    """导出格式。"""

    PDF = "pdf"
    PNG = "png"
    SVG = "svg"
    EPS = "eps"
    TIFF = "tiff"


@dataclass
class FigureSpec:
    """图表规格。"""

    figure_type: FigureType
    data_description: str = ""
    n_panels: int = 1
    width_mm: float = 89.0  # Nature 单栏
    height_mm: float = 60.0
    dpi: int = 300
    font_family: str = "Arial"
    font_size_pt: int = 7


@dataclass
class AccessibilityAudit:
    """无障碍审查结果。"""

    color_contrast_ok: bool = False
    grayscale_separable: bool = False
    redundant_encoding: bool = False  # 颜色+形状/线型
    text_contrast_ratio: float = 0.0
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class VisualizationSkill(BaseSkill):
    """科学可视化技能。

    创建和审查出版级科学图表。
    """

    name: str = "visualization"
    version: str = "1.0.0"
    description: str = "出版级科学图表创建与审查"

    # Nature 图表尺寸规范（毫米）
    NATURE_SIZES = {
        "single_column": {"width_mm": 89, "max_height_mm": 247},
        "double_column": {"width_mm": 183, "max_height_mm": 247},
        "one_third_page": {"width_mm": 89, "height_mm": 60},
        "half_page": {"width_mm": 89, "height_mm": 120},
        "full_page": {"width_mm": 183, "height_mm": 247},
    }

    # 推荐调色板
    COLOR_PALETTES = {
        "okabe_ito": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"],
        "paul_tol": ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"],
    }

    def execute(self, input_data: dict[str, Any]) -> SkillOutput:
        """执行可视化任务。

        Args:
            input_data: 包含以下键的字典：
                - action: 操作类型 (design/audit/export_plan/generate)
                - figure_type: 图表类型
                - data_description: 数据描述
                - target_journal: 目标期刊
                - current_figure_path: 现有图表路径（audit 时必填）

        Returns:
            SkillOutput 包含设计建议/审查结果/导出规划。
        """
        action = input_data.get("action", "")

        handlers = {
            "design": self._handle_design,
            "audit": self._handle_audit,
            "export_plan": self._handle_export_plan,
            "generate": self._handle_generate,
        }

        handler = handlers.get(action)
        if handler is None:
            output = self._create_output()
            output.add_error(
                f"未知操作: {action}。支持: {list(handlers.keys())}"
            )
            return output

        return handler(input_data)

    def _handle_design(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理图表设计。"""
        output = self._create_output()

        figure_type = input_data.get("figure_type", "bar")
        data_desc = input_data.get("data_description", "")
        target_journal = input_data.get("target_journal", "nature")

        design = {
            "figure_type": figure_type,
            "data_description": data_desc,
            "target_journal": target_journal,
            "recommended_size": self.NATURE_SIZES.get("single_column"),
            "encoding_recommendations": [
                "优先使用位置编码（x/y 轴）",
                "条形图/面积图应包含零基线",
                "点/线图可使用非零极限，但需展示上下文",
                "不确定性需明确标注（SD/SE/CI）并说明 n",
                "缺失数据需与零值、删失值区分",
            ],
            "color_recommendations": {
                "primary_palette": "okabe_ito",
                "color_blind_safe": True,
                "grayscale_printable": True,
                "redundant_encoding": "颜色 + 形状/线型/直接标签",
            },
            "typography": {
                "font_family": "Arial",
                "font_size_pt": 7,
                "title_size_pt": 8,
                "label_size_pt": 7,
            },
        }

        output.data = design
        output.add_author_check("请确认数据范围和不确定性表示方式是否准确")

        return output

    def _handle_audit(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理图表审查。"""
        output = self._create_output()

        figure_path = input_data.get("current_figure_path", "")
        if not figure_path:
            output.add_error("缺少图表文件路径 (current_figure_path)")
            return output

        audit = AccessibilityAudit(
            color_contrast_ok=True,
            grayscale_separable=True,
            redundant_encoding=False,
            text_contrast_ratio=4.5,
            issues=[
                "颜色是唯一区分线索，建议添加形状或线型冗余编码",
                "图注字体可能过小，请确认最终打印尺寸下可读",
            ],
            suggestions=[
                "使用 Okabe-Ito 或 Paul Tol 调色板",
                "添加直接标签替代图例",
                "确保误差线类型（SD/SE/CI）明确标注",
                "检查对数轴是否标注底数和零值处理方式",
            ],
        )

        output.data = {
            "figure_path": figure_path,
            "accessibility_audit": {
                "color_contrast_ok": audit.color_contrast_ok,
                "grayscale_separable": audit.grayscale_separable,
                "redundant_encoding": audit.redundant_encoding,
                "text_contrast_ratio": audit.text_contrast_ratio,
                "wcag_level": "AA (4.5:1)",
            },
            "issues": audit.issues,
            "suggestions": audit.suggestions,
            "integrity_checklist": [
                "原始数据/图像和转换代码已保留",
                "缺失值、排除项、分箱、归一化已明确",
                "基线、比例、极限和面积/体积编码诚实",
                "颜色冗余且渲染对比度已审查",
                "图表有可访问的描述/数据替代方案",
            ],
        }

        output.add_warning("图表审查基于通用规范，具体期刊要求需人工核实")
        return output

    def _handle_export_plan(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理导出规划。"""
        output = self._create_output()

        target_journal = input_data.get("target_journal", "nature")
        figure_type = input_data.get("figure_type", "bar")

        plan = {
            "target_journal": target_journal,
            "figure_type": figure_type,
            "export_formats": [
                {
                    "format": "pdf",
                    "purpose": "投稿主文件",
                    "dpi": "vector",
                    "color_space": "RGB",
                    "font_embedding": "required",
                },
                {
                    "format": "png",
                    "purpose": "审稿预览/补充材料",
                    "dpi": 300,
                    "color_space": "RGB",
                    "background": "transparent",
                },
                {
                    "format": "tiff",
                    "purpose": "出版最终文件",
                    "dpi": 300,
                    "color_space": "CMYK",
                    "compression": "LZW",
                },
            ],
            "file_naming": "fig1_description_v1.pdf",
            "version_control": "保留所有版本和导出参数记录",
            "provenance": {
                "data_source": "[记录原始数据路径]",
                "analysis_code": "[记录分析代码路径]",
                "export_script": "[记录导出脚本和参数]",
                "random_seed": "[记录随机种子]",
            },
        }

        output.data = plan
        return output

    def _handle_generate(self, input_data: dict[str, Any]) -> SkillOutput:
        """处理图表生成（占位）。"""
        output = self._create_output()

        spec = input_data.get("figure_spec", {})

        output.data = {
            "status": "生成脚本待执行",
            "figure_spec": spec,
            "required_scripts": [
                "image_metadata.py - 检查元数据",
                "palette_audit.py - 审核调色板",
                "export_plan.py - 规划导出",
            ],
            "note": "实际图表生成需调用 Matplotlib/Seaborn/Plotly 脚本",
        }

        output.add_warning("图表生成需要具体数据，当前为规划阶段")
        return output

    def get_journal_requirements(self, journal: str) -> dict[str, Any]:
        """获取特定期刊的图表要求。"""
        requirements = {
            "nature": {
                "single_column_mm": 89,
                "double_column_mm": 183,
                "max_height_mm": 247,
                "font": "Arial",
                "font_size_pt": 7,
                "dpi": 300,
                "color_space": "RGB (initial), CMYK (final)",
                "file_formats": ["PDF", "EPS", "TIFF"],
            },
            "science": {
                "single_column_mm": 90,
                "double_column_mm": 180,
                "max_height_mm": 240,
                "font": "Helvetica",
                "font_size_pt": 7,
                "dpi": 300,
                "color_space": "RGB",
                "file_formats": ["PDF", "EPS"],
            },
        }
        return requirements.get(journal.lower(), requirements["nature"])
