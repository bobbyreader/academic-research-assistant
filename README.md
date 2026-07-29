# 🔬 Academic Research Assistant / 学术研究助手

> [English](#-english) | [中文](#-中文)

---

## 🇬🇧 English

A comprehensive AI-powered workflow system for scientific research — from hypothesis generation to paper publication and presentation.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/downloads/)
[![Stage: Alpha](https://img.shields.io/badge/Stage-Alpha-orange.svg)]()

---

### 🎯 What Is This?

**Academic Research Assistant** is a modular, end-to-end research workflow system that orchestrates 10 specialized AI skills across the entire research lifecycle — from brainstorming ideas and searching literature, through statistical analysis and visualization, all the way to manuscript writing, peer review simulation, and presentation generation.

Think of it as a **research cockpit**: instead of juggling disconnected tools, you have one intelligent system where every module passes data seamlessly to the next, following battle-tested workflows inspired by Nature journal standards, PRISMA guidelines, and academic best practices.

---

### ⚡ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize a new research project
python orchestrator.py init my_paper --mode hybrid

# Run the workflow
python orchestrator.py run my_paper

# Check progress
python orchestrator.py status my_paper

# Export deliverables
python orchestrator.py export my_paper --format md
```

**That's it.** In under 5 commands, you've run an 11-stage AI-assisted research pipeline.

---

### 🗺️ The Research Lifecycle

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  💡 Ideate          📚 Investigate        📊 Analyze        ✍️ Write     │
│  ─────────         ─────────────        ─────────        ─────────      │
│  Brainstorming  →  Academic Search   →  Statistics   →  Writing        │
│                  →  Literature Rev.   →  Visualization→  Polishing      │
│                  →  ARS Deep Research                      ↓            │
│                                                     🔍 Review          │
│                                                     Reviewer            │
│                                                     ARS Pipeline        │
│                                                     ARS Reviewer        │
│                                                          ↓               │
│                                              📢 Communicate              │
│                                              Paper2PPT                  │
└──────────────────────────────────────────────────────────────────────────┘
```

#### Three Workflow Modes

| Mode | Best For | Stages |
|------|----------|--------|
| **🔵 Lightweight** | You have data, need a paper fast | 7 stages |
| **🟡 Heavyweight** | Full research project from scratch | 7 stages |
| **🟢 Hybrid** *(recommended)* | Maximum flexibility & quality | 11 stages |

---

### 🧩 The 10 Skills

| # | Skill | What It Does |
|---|-------|-------------|
| 1 | **Scientific Brainstorming** | 5-phase collaborative ideation — cross-disciplinary analogies, hypothesis flipping, constraint removal, SCAMPER, TRIZ |
| 2 | **Academic Search** | Multi-source literature retrieval (CrossRef, PubMed, arXiv, Scopus, ScienceDirect) with citation formatting (APA, Nature, IEEE, Vancouver) and self-citation audit |
| 3 | **Literature Review** | Systematic review with PRISMA flow diagrams, quality assessment tools (Cochrane, Newcastle-Ottawa, AMSTAR 2), and citation verification |
| 4 | **Nature Statistics** | Statistical method audit — detects pseudoreplication, nested data, multiple comparisons, significance misuse; generates reviewer responses |
| 5 | **Scientific Visualization** | Publication-ready figures with WCAG 2.2 accessibility review, uncertainty visualization, journal-specific export planning |
| 6 | **Nature Writing** | Nature-style manuscript drafting — claim-evidence narratives, Chinese-to-English translation, cover letter, highlights, submission checklist |
| 7 | **Nature Polishing** | Sentence-level editing — translation refinement, AI-taste detection, Nature/Nature Communications paradigm alignment |
| 8 | **Nature Reviewer** | Pre-submission mock peer review — 3 independent reviewer reports, cross-review synthesis, 12-axis technical checklist, claim-pointer traceability |
| 9 | **Paper2PPT** | Paper to 10-16 slide Chinese presentation with speaker notes and key figure extraction |
| 10 | **ARS Integration** | Full Academic Research Suite wrapper — 8 Deep Research modes, 11 Paper modes, 6 Reviewer modes, 10-stage pipeline with mandatory academic integrity checkpoints |

---

### 🏗️ Architecture

```
academic-research-assistant/
├── orchestrator.py          # CLI entry point (init/run/status/list/export)
├── config/
│   ├── workflow.yaml        # Workflow mode definitions
│   └── settings.yaml        # API keys, output paths, citation formats
├── core/
│   ├── skill_bridge.py      # BaseSkill + SkillOutput (universal interface)
│   ├── state_manager.py     # Project state + checkpoint persistence
│   └── artifact_store.py    # Versioned artifact management (v1/v2/v3...)
├── skills/                  # 10 modular skill implementations
│   ├── brainstorming/       # Conversational ideation engine
│   ├── academic_search/     # Multi-source search + citation engine
│   ├── literature_review/   # Systematic review executor
│   ├── statistics/          # Statistical audit engine
│   ├── visualization/       # Publication figure builder
│   ├── writing/             # Nature manuscript generator
│   ├── polishing/           # Sentence-level refinement
│   ├── reviewer/            # Mock peer review simulator
│   ├── paper2ppt/           # Presentation generator
│   └── ars_integration/    # ARS pipeline orchestrator
├── workflows/               # YAML workflow definitions
├── scripts/                  # CLI tools (citation verifier, PDF generator, PPTX exporter)
├── templates/                # Document templates (review, cover letter, response)
└── examples/                 # Quick start demos
```

---

### 🔑 Key Features

- **🔗 Seamless Data Flow** — Each skill passes structured outputs to the next. No manual copy-paste between tools.
- **📈 Versioned Artifacts** — Every output is versioned (v1, v2, v3...). Roll back anytime.
- **⏸️ Checkpoint & Resume** — Interrupt a workflow and pick up exactly where you left off.
- **🎯 Status Tracking** — Every stage has a clear status: `ready` / `ready_with_author_checks` / `blocked`. You always know what's done and what needs attention.
- **📋 Submission Checklist** — Built-in pre-submission integrity checks before you hit "Submit".
- **🌐 Multi-Source Search** — Query CrossRef, PubMed, arXiv, Scopus, and ScienceDirect simultaneously.
- **✅ Citation Verification** — Every DOI is verified against CrossRef. No fake citations.
- **🔒 Academic Integrity Gates** — ARS Pipeline includes mandatory integrity checkpoints (Stage 2.5 & 4.5) that cannot be bypassed.
- **🎨 Publication-Ready Output** — Figures, manuscripts, and presentations follow Nature journal standards.

---

### 👥 Who Is This For?

- **PhD students** navigating the literature → writing → submission gauntlet
- **Researchers** needing systematic review support with PRISMA compliance
- **Lab groups** wanting consistent, high-quality manuscript standards
- **Postdocs** preparing for journal submission under Nature or Nature Communications
- **Academic writers** seeking AI assistance that respects academic integrity

---

### 📋 Requirements

- Python 3.11+
- API keys for external services (PubMed, Semantic Scholar, CrossRef — free tiers available)
- Pandoc + xelatex (for PDF export)
- python-pptx (for PPTX generation)

---

### 🤝 Contributing

Contributions are welcome! Please read our contribution guidelines and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

### 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

### 🙏 Acknowledgments

This system is inspired by and integrates concepts from:

- [davila7/claude-code-templates](https://github.com/davila7/claude-code-templates) — Scientific Brainstorming & Literature Review
- [Yuan1z0825/nature-skills](https://github.com/Yuan1z0825/nature-skills) — Nature Academic Search, Writing, Polishing, Reviewer, Paper2PPT
- [Imbad0202/academic-research-skills](https://github.com/Imbad0202/academic-research-skills) — Academic Research Suite (ARS)
- [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) — Scientific Visualization

---

*Built with ❤️ for researchers who deserve better tools.*

---

## 🇨🇳 中文

一个全面的 AI 驱动科研工作流系统——从假设生成到论文发表和演示汇报。

---

### 🎯 这是什么？

**Academic Research Assistant** 是一个模块化、端到端的研究工作流系统，能够编排 **10 个专业 AI 技能**，覆盖整个研究生命周期——从头脑风暴、文献检索，到统计分析、图表绘制，再到手稿撰写、同行评审模拟和演示文稿生成。

把它想象成一个**研究驾驶舱**：你不再需要在各种割裂的工具之间来回切换，而是拥有一个智能系统，每个模块都能将数据无缝传递给下一个，遵循经过实战检验的工作流——灵感来自 Nature 期刊标准、PRISMA 指南和学术最佳实践。

---

### ⚡ 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 初始化一个新研究项目
python orchestrator.py init my_paper --mode hybrid

# 运行工作流
python orchestrator.py run my_paper

# 查看进度
python orchestrator.py status my_paper

# 导出产出物
python orchestrator.py export my_paper --format md
```

**就这么简单。** 不到 5 条命令，你已经运行了一个 11 阶段的 AI 辅助研究流水线。

---

### 🗺️ 研究生命周期

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  💡 构思           📚 调研             📊 分析          ✍️ 写作        │
│  ──────          ─────────            ──────         ──────           │
│  头脑风暴   →   学术检索      →      统计审计  →    论文撰写           │
│             →   文献综述      →      科学可视化 →   润色翻译           │
│             →   ARS 深度研究                     ↓                    │
│                                                  🔍 评审             │
│                                                  模拟审稿             │
│                                                  ARS 流水线          │
│                                                  ARS 审稿人          │
│                                                       ↓               │
│                                               📢 传播                │
│                                               论文转 PPT             │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 三种工作流模式

| 模式 | 适用场景 | 阶段数 |
|------|---------|--------|
| **🔵 轻量级** | 已有数据，需要快速成文 | 7 个阶段 |
| **🟡 重量级** | 从零开始完整研究项目 | 7 个阶段 |
| **🟢 混合型**（推荐）| 最高灵活性与质量 | 11 个阶段 |

---

### 🧩 10 大核心技能

| # | 技能 | 功能描述 |
|---|-------|-------------|
| 1 | **Scientific Brainstorming** | 5 阶段协作构思——跨学科类比、假设翻转、约束移除、SCAMPER、TRIZ |
| 2 | **Academic Search** | 多源文献检索（CrossRef、PubMed、arXiv、Scopus、ScienceDirect），支持引用格式（APA、Nature、IEEE、Vancouver）和他引审计 |
| 3 | **Literature Review** | 系统性综述，含 PRISMA 流程图、质量评估工具（Cochrane、Newcastle-Ottawa、AMSTAR 2）和引用验证 |
| 4 | **Nature Statistics** | 统计方法审计——检测伪重复、嵌套数据、多重比较、显著性滥用；生成审稿人回复 |
| 5 | **Scientific Visualization** | 出版级图表，含 WCAG 2.2 无障碍审查、不确定性可视化、期刊特定导出规划 |
| 6 | **Nature Writing** | Nature 风格手稿撰写——claim-evidence 叙事、中英翻译、投稿信、亮点、提交清单 |
| 7 | **Nature Polishing** | 句子级润色——翻译优化、AI 味检测、Nature/Nature Communications 范式对齐 |
| 8 | **Nature Reviewer** | 投稿前模拟同行评审——3 份独立审稿报告、交叉综合、12 轴技术清单、claim-pointer 可追溯性 |
| 9 | **Paper2PPT** | 论文转 10-16 页中文演示文稿，含演讲备注和关键图表提取 |
| 10 | **ARS Integration** | 完整学术研究套件封装——8 种深度研究模式、11 种论文模式、6 种审稿模式、10 阶段流水线（含强制学术诚信检查点） |

---

### 🏗️ 系统架构

```
academic-research-assistant/
├── orchestrator.py          # CLI 入口（init/run/status/list/export）
├── config/
│   ├── workflow.yaml        # 工作流模式定义
│   └── settings.yaml        # API 密钥、输出路径、引用格式
├── core/
│   ├── skill_bridge.py      # BaseSkill + SkillOutput（通用接口）
│   ├── state_manager.py     # 项目状态 + 检查点持久化
│   └── artifact_store.py    # 版本化产出物管理（v1/v2/v3...）
├── skills/                  # 10 个模块化技能实现
│   ├── brainstorming/       # 对话式构思引擎
│   ├── academic_search/     # 多源搜索 + 引用引擎
│   ├── literature_review/   # 系统性综述执行器
│   ├── statistics/          # 统计审计引擎
│   ├── visualization/       # 出版图表构建器
│   ├── writing/             # Nature 手稿生成器
│   ├── polishing/           # 句子级优化
│   ├── reviewer/            # 模拟同行评审
│   ├── paper2ppt/           # 演示文稿生成器
│   └── ars_integration/    # ARS 流水线编排器
├── workflows/               # YAML 工作流定义
├── scripts/                  # CLI 工具（引用验证器、PDF 生成器、PPTX 导出器）
├── templates/                # 文档模板（综述、投稿信、审稿回复）
└── examples/                 # 快速开始演示
```

---

### 🔑 核心特性

- **🔗 无缝数据流** — 每个技能将结构化输出传递给下一个，无需手动复制粘贴
- **📈 版本化产出物** — 每个输出都有版本号（v1、v2、v3...），随时可回滚
- **⏸️ 检查点与恢复** — 中断工作流后，可从上次离开的地方精确恢复
- **🎯 状态追踪** — 每个阶段都有清晰状态：`ready` / `ready_with_author_checks` / `blocked`，随时知道进展
- **📋 提交清单** — 内置投稿前诚信检查，在点击"提交"前确保万无一失
- **🌐 多源搜索** — 同时查询 CrossRef、PubMed、arXiv、Scopus 和 ScienceDirect
- **✅ 引用验证** — 每个 DOI 都通过 CrossRef 验证，杜绝虚假引用
- **🔒 学术诚信关口** — ARS 流水线包含强制诚信检查点（第 2.5 和 4.5 阶段），无法绕过
- **🎨 出版级输出** — 图表、手稿和演示文稿均遵循 Nature 期刊标准

---

### 👥 适用人群

- **博士生**——需要应对文献→写作→投稿的完整挑战
- **研究人员**——需要符合 PRISMA 规范的系统性综述支持
- **实验室团队**——希望保持一贯的高质量手稿标准
- **博士后**——准备向 Nature 或 Nature Communications 投稿
- **学术写作者**——寻求尊重学术诚信的 AI 辅助

---

### 📋 环境要求

- Python 3.11+
- 外部服务 API 密钥（PubMed、Semantic Scholar、CrossRef——提供免费额度）
- Pandoc + xelatex（用于 PDF 导出）
- python-pptx（用于 PPTX 生成）

---

### 🤝 贡献指南

欢迎贡献！请阅读贡献指南并提交 Pull Request。对于重大更改，请先开 Issue 讨论您希望做出的改变。

---

### 📄 开源协议

本项目采用 MIT 协议开源——详见 [LICENSE](LICENSE) 文件。

---

### 🙏 致谢

本系统灵感来源于并整合了以下项目的理念：

- [davila7/claude-code-templates](https://github.com/davila7/claude-code-templates) — Scientific Brainstorming & Literature Review
- [Yuan1z0825/nature-skills](https://github.com/Yuan1z0825/nature-skills) — Nature Academic Search, Writing, Polishing, Reviewer, Paper2PPT
- [Imbad0202/academic-research-skills](https://github.com/Imbad0202/academic-research-skills) — Academic Research Suite (ARS)
- [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) — Scientific Visualization

---

*为值得更好工具的研究者，用心打造。❤️*
