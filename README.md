# 🔬 Academic Research Assistant / 学术研究助手

> [English](#-english) | [中文](#-中文)

---

## 🇬🇧 English

An AI-assisted literature research pipeline: give it a topic (and optionally a CSV dataset) and it retrieves real papers, synthesises the evidence, drafts a manuscript, **verifies every citation**, and exports the result.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/downloads/)
[![Stage: Alpha](https://img.shields.io/badge/Stage-Alpha-orange.svg)]()

### 🎯 What it is (and is not)

It is a **research cockpit** that turns one topic into a traceable, exportable draft.

It is **not** a paper generator. Without experimental data the output is explicitly a
**literature synthesis / research proposal** — the system will not fabricate results,
sample sizes, or significance tests.

### ⚡ Quick start (CLI)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# Sign in to the locally installed Codex CLI once (default engine, no API key needed)
codex login

# Topic -> search, analyse, draft, verify, export
python3 orchestrator.py research my_paper \
  --topic "Climate adaptation and urban heat resilience" \
  --sources crossref,pubmed,semantic_scholar,arxiv \
  --max-results 10 \
  --export md

python3 orchestrator.py status my_paper
python3 orchestrator.py export my_paper --format pdf
```

### 🖥️ Web UI

On macOS, double-click `启动研究助手.command` in the project root. Or run it manually:

```bash
source .venv/bin/activate
python3 web_app.py            # http://127.0.0.1:5050
```

Enter a topic, pick sources and export formats, optionally upload a CSV, and follow the
four-stage progress view. The server binds to your own machine by default.

> Do **not** open `web/templates/index.html` directly — it is a server-rendered template.

### 🗺️ The pipeline (4 stages)

```
topic (+ optional CSV)
   │
   ├─ 1. 真实检索 (search)     Crossref / PubMed / Semantic Scholar / arXiv, deduplicated
   ├─ 2. 证据分析 (analysis)   inferential statistics + figures, then LLM synthesis
   ├─ 3. 研究写作 (writing)    LLM drafts the manuscript
   │        ├─ 🔒 引用验证     citation integrity gate      ← see below
   │        ├─ 📊 统计追溯     statistics traceability gate ← see below
   │        └─ 🧪 模拟评审     advisory simulated peer review
   └─ 4. 整理下载 (export)     Markdown / PDF / PPTX
```

Every stage writes **versioned artifacts** (`v1`, `v2`, …) plus a `.meta` file, so any
run can be rolled back.

### 🔒 Citation integrity gate

This is the feature that separates the tool from a chat wrapper. Before a manuscript is
persisted, `core/citation_verifier.py` checks it:

| Check | Behaviour |
|-------|-----------|
| **Marker integrity** (deterministic, always on) | Every `[Pn]` marker in the draft must map to a record that was actually retrieved. An unknown marker means the model invented a citation → the run is **blocked** and no manuscript is written. |
| **DOI resolvability** (network, advisory) | Every reference carrying a DOI is checked against Crossref. Unresolved DOIs downgrade the run to *author checks required* (a warning) instead of blocking, so transient network failures are never confused with fabrication. |
| **No-DOI records** | Preprints without a DOI (e.g. arXiv) are recorded as *not checked* rather than treated as failures. |

The result is written to `artifacts/writing/citation_verification.{json,md}`.

### 📊 Data analysis and statistics traceability

When you pass `--data data.csv`, everything is derived from that dataset and the
model is never allowed to invent a number:

| Step | What happens |
|------|--------------|
| **Inferential statistics** | `core/statistics_engine.py` auto-detects numeric/categorical columns and runs only tests the structure supports: Welch t-test (2 groups), one-way ANOVA (3+), Pearson/Spearman correlation. It always reports effect size + n, applies Holm-Bonferroni correction, checks normality, and refuses to run on constants or n<3. |
| **Publication figures** | `core/figure_builder.py` renders colour-blind-safe (Okabe-Ito) figures at 300 dpi: per-column distribution histograms, group comparisons (box + individual points + median + per-group n), and a correlation scatter with fit line. Output is byte-for-byte deterministic. |
| **Traceability gate** | `core/statistics_verifier.py` checks every p-value the model wrote against the computed ones, matching at the precision actually written (a fixed tolerance would wrongly accept `p = 0.0001` for a computed `0.001`). Unmatched claims are flagged for author review. |
| **System-injected sections** | The computed statistics table and the figure list are appended to the manuscript by the *system*, not the model — so the numbers in the deliverable are traceable by construction. |

Statistics and figures always use the **same resolved grouping**, so a figure can
never illustrate a different grouping than the test it accompanies.

### 🧪 Simulated peer review

`core/peer_reviewer.py` produces an **advisory** pre-submission review from the
artifacts the pipeline has already built:

- `review.reviewer_count` in `config/settings.yaml` selects how many roles from
  `(methodology, statistics, novelty)` act as reviewers — **default `1`** (each
  reviewer costs one extra LLM call); set `0` to disable the stage;
- **evidence-bound**: every concern must quote the manuscript, and any concern
  whose evidence is empty is dropped before a reader sees it;
- **cross-checked**: untraceable citations and untraceable statistics are injected
  as concerns by the system, so the review cannot silently pass a manuscript whose
  own gates already failed;
- the headline count is the number of **distinct** concerns rather than the sum
  over reviewers, and the decision comes from a documented mechanical rule
  (worst-case wins, ties by majority).

A review is a language-model opinion, not evidence, so it **never blocks** the
pipeline — it is surfaced as a warning. Blocking stays reserved for the two
deterministic gates. Results are written to `artifacts/review/review_reports.{json,md}`.

### 🧰 CLI commands

| Command | Purpose |
|---------|---------|
| `research <name> --topic "..."` | Run the full pipeline (auto-creates the project) |
| `init <name> [--mode …]` | Create an empty project |
| `status <name>` | Show stage status and artifacts |
| `list` | List projects |
| `export <name> --format md\|pdf\|pptx` | Export artifacts |

`--mode lightweight|heavyweight|hybrid` is currently recorded as project metadata only;
the pipeline does not branch on it yet.

### 🔑 LLM providers

| Provider | How to enable |
|----------|---------------|
| **Codex CLI** (default) | `codex login` |
| **Gemini** | `export ARS_LLM_PROVIDER=gemini` + `GEMINI_API_KEY` |
| **OpenAI-compatible** | `export ARS_LLM_PROVIDER=openai_compatible` + `ARS_LLM_API_KEY`, `ARS_LLM_MODEL`, `ARS_LLM_BASE_URL` |

See [.env.example](.env.example). Never commit real keys.

### 🏗️ Architecture

```
academic-research-assistant/
├── orchestrator.py          # CLI entry point (init / research / status / list / export)
├── web_app.py               # Local browser UI entry point
├── web/                     # Page, styles, interactions
├── config/
│   └── settings.yaml        # Provider, API keys, citation style, export defaults
├── core/
│   ├── citation_verifier.py    # 🔒 Citation integrity gate
│   ├── statistics_engine.py    # Defensible inferential statistics
│   ├── figure_builder.py       # Publication-ready figures
│   ├── statistics_verifier.py  # 📊 Statistics traceability gate
│   ├── peer_reviewer.py        # 🧪 Advisory simulated peer review
│   ├── research_pipeline.py    # Search -> data -> draft -> verify -> persist
│   ├── research_service.py  # Binds project state + clients to the pipeline
│   ├── external_clients.py  # Crossref / PubMed / Semantic Scholar / arXiv adapters
│   ├── llm_client.py        # Codex CLI / Gemini / OpenAI-compatible clients
│   ├── data_analyzer.py     # Descriptive statistics for optional CSV data
│   ├── http_client.py       # Dependency-free HTTP transport with retries
│   ├── artifact_store.py    # Versioned artifact storage (v1/v2/v3…)
│   ├── state_manager.py     # Project state + checkpoints
│   ├── export_service.py    # Markdown -> PDF / PPTX
│   └── config_loader.py     # settings.yaml loader
├── scripts/                 # export_pptx.py, generate_pdf.py, verify_citations.py
├── templates/               # Document templates
└── tests/                   # pytest suite
```

### 📋 Requirements

- Python 3.11+
- `pytest` for the test suite (`python3 -m pytest -q`)
- PDF export: Pandoc + XeLaTeX (falls back to WeasyPrint, which needs system Pango)
- PPTX export: bundled `python-pptx` exporter
- A configured LLM engine (see above)

### 🚧 Roadmap

This project deliberately keeps **one** execution spine (`core/research_pipeline.py`).
Capabilities are added only when they can be wired into that spine and proven by tests.
Full phase detail and the audit log live in [ROADMAP.md](ROADMAP.md).

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Remove the stub skill layer, dead configs and the fake `run` path; align docs | ✅ done |
| 1 | Citation integrity gate (marker traceability + DOI verification) | ✅ done |
| 2 | Inferential statistics, publication figures, statistics traceability gate | ✅ done |
| 3 | Simulated peer review; presentation outline built from the real artifacts | ✅ done |
| 4 | CI (pytest/ruff/mypy), packaging, config validation | ✅ done |

### 🧪 Development

```bash
python3 -m pip install -e ".[dev]"
ruff check .   # static checks (ruff version pinned via [tool.ruff].required-version)
mypy           # type checks
pytest         # test suite
```

CI runs all three on Python 3.11 and 3.12 (see `.github/workflows/ci.yml`).
`config/settings.yaml` is validated before a run starts and fails fast, naming the
offending key instead of crashing halfway through.

### 🤝 Contributing

Issues and pull requests are welcome. For major changes, please open an issue first.

### 📄 License

MIT — see [LICENSE](LICENSE).

### 🙏 Acknowledgments

- [davila7/claude-code-templates](https://github.com/davila7/claude-code-templates)
- [Yuan1z0825/nature-skills](https://github.com/Yuan1z0825/nature-skills)
- [Imbad0202/academic-research-skills](https://github.com/Imbad0202/academic-research-skills)
- [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills)

---

## 🇨🇳 中文

一个 AI 辅助的文献研究工作流：输入一个研究主题（可选上传 CSV 数据），系统会真实检索文献、
综合证据、撰写手稿、**校验每一条引用**，并导出结果。

### 🎯 它是什么（以及不是什么）

它是一台**研究驾驶舱**，把"一个主题"变成一份可追溯、可导出的草稿。

它**不是**论文生成器。在没有实验数据时，输出会被明确标注为**文献综合/研究计划**——
系统不会伪造实验结果、样本量或显著性检验。

### ⚡ 快速开始（命令行）

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# 首次使用登录本机 Codex CLI（默认引擎，无需 API 密钥）
codex login

# 主题 -> 检索、分析、写作、校验、导出
python3 orchestrator.py research my_paper \
  --topic "城市热环境中的气候适应与韧性" \
  --sources crossref,pubmed,semantic_scholar,arxiv \
  --max-results 10 \
  --export md

python3 orchestrator.py status my_paper
python3 orchestrator.py export my_paper --format pdf
```

### 🖥️ 网页界面

macOS 上双击项目根目录的 `启动研究助手.command` 即可；也可以手动运行：

```bash
source .venv/bin/activate
python3 web_app.py            # http://127.0.0.1:5050
```

输入研究主题、选择文献来源与导出格式（可上传 CSV），页面会展示"真实检索 → 证据分析 →
研究写作 → 整理下载"四阶段进度。服务默认只监听本机。

> 请不要直接打开 `web/templates/index.html`，它是需要本地服务渲染的模板。

### 🗺️ 工作流（4 个阶段）

```
主题（可选 CSV 数据）
   │
   ├─ 1. 真实检索     Crossref / PubMed / Semantic Scholar / arXiv，按 DOI 去重
   ├─ 2. 证据分析     推断统计 + 图表，再由 LLM 归纳
   ├─ 3. 研究写作     LLM 撰写手稿
   │        ├─ 🔒 引用验证     引用可信性关口   ← 见下
   │        ├─ 📊 统计追溯     统计可追溯性关口 ← 见下
   │        └─ 🧪 模拟评审     顾问级模拟同行评审
   └─ 4. 整理下载     Markdown / PDF / PPTX
```

每个阶段都会写入**版本化产出物**（`v1`、`v2`……）及 `.meta` 元数据文件，随时可回滚。

### 🔒 引用可信性关口

这是本工具区别于"聊天套壳"的关键能力。在手稿被保存之前，`core/citation_verifier.py`
会执行以下校验：

| 校验项 | 行为 |
|--------|------|
| **引用标识完整性**（确定性，始终执行） | 正文中的每个 `[Pn]` 都必须对应一条真实检索到的文献。出现无法追溯的标识，说明模型编造了引用 → **阻断本次运行**，不保存手稿。 |
| **DOI 可解析性**（联网，建议性） | 带 DOI 的参考文献会与 Crossref 核对。未解析的 DOI 只会把结果降级为"需作者确认"（警告），不会阻断，避免把网络抖动误判为造假。 |
| **无 DOI 文献** | 没有 DOI 的预印本（如 arXiv）记为"未做解析校验"，不计为失败。 |

校验结果写入 `artifacts/writing/citation_verification.{json,md}`。

### 📊 数据分析与统计可追溯性

传入 `--data data.csv` 后，一切结论都从这份数据推导，绝不允许模型编造数字：

| 环节 | 行为 |
|------|------|
| **推断统计** | `core/statistics_engine.py` 自动识别数值/分类列，只运行数据结构支持的检验：Welch t 检验（2 组）、单因素 ANOVA（3 组以上）、Pearson/Spearman 相关。始终报告效应量与 n，执行 Holm-Bonferroni 校正，检查正态性；对常量列或 n<3 拒绝运行。 |
| **出版级图表** | `core/figure_builder.py` 以 300 dpi 输出色盲友好（Okabe-Ito）图件：逐列分布直方图、分组比较图（箱体 + 个体散点 + 中位数 + 每组 n）、带拟合线的相关散点图。输出具备字节级确定性。 |
| **可追溯性关口** | `core/statistics_verifier.py` 校验模型写出的每个 p 值，按模型**实际书写的小数位数**与计算结果匹配（固定容差会把 `p = 0.0001` 误判为匹配计算值 `0.001`）。无法追溯的陈述会被标记待作者核对。 |
| **系统注入章节** | 统计表与图表清单由**系统**（而非模型）附加到手稿，因此交付物中的数字天然可追溯。 |

统计与图表始终使用**同一个已解析的分组列**，图表不可能与被说明的检验使用不同分组。

### 🧪 模拟同行评审

`core/peer_reviewer.py` 基于管线已产出的产物生成**顾问级**预提交评审：

- `config/settings.yaml` 的 `review.reviewer_count` 决定从
  `(methodology, statistics, novelty)` 中启用几位审稿人——**默认 `1`**
  （每多一位就多一次 LLM 调用）；设为 `0` 即关闭该阶段；
- **证据约束**：每条意见必须引用稿件原文，证据为空的意见在到达读者之前就被丢弃；
- **交叉核对**：无法追溯的引用与统计陈述由系统**确定性注入**为意见，
  因此评审不可能放过一个自身关口已失败的稿件；
- 头条数字是**去重后的独立问题数**（而非各审稿人意见之和），决定由**文档化的机械规则**推导
  （最坏情况优先、平局取多数）。

评审是语言模型的**意见而非证据**，因此**永不阻断**管线——只以警告形式呈现。
阻断始终保留给两个确定性关口。结果写入 `artifacts/review/review_reports.{json,md}`。

### 🧰 命令行命令

| 命令 | 作用 |
|------|------|
| `research <名称> --topic "..."` | 执行完整工作流（自动创建项目） |
| `init <名称> [--mode …]` | 创建空项目 |
| `status <名称>` | 查看阶段状态与产出物 |
| `list` | 列出所有项目 |
| `export <名称> --format md\|pdf\|pptx` | 导出产出物 |

`--mode lightweight|heavyweight|hybrid` 目前仅作为项目元数据记录，工作流尚未按模式分支。

### 🔑 内容生成引擎

| 引擎 | 启用方式 |
|------|---------|
| **本机 Codex CLI**（默认） | `codex login` |
| **Gemini** | `export ARS_LLM_PROVIDER=gemini` + `GEMINI_API_KEY` |
| **OpenAI 兼容接口** | `export ARS_LLM_PROVIDER=openai_compatible` + `ARS_LLM_API_KEY`、`ARS_LLM_MODEL`、`ARS_LLM_BASE_URL` |

参考 [.env.example](.env.example)，请勿提交真实密钥。

### 🏗️ 系统架构

```
academic-research-assistant/
├── orchestrator.py          # CLI 入口（init / research / status / list / export）
├── web_app.py               # 本地网页工作台入口
├── web/                     # 页面、样式与交互
├── config/
│   └── settings.yaml        # 引擎、API 密钥、引用格式、导出默认值
├── core/
│   ├── citation_verifier.py    # 🔒 引用可信性关口
│   ├── statistics_engine.py    # 可辩护的推断统计
│   ├── figure_builder.py       # 出版级图表
│   ├── statistics_verifier.py  # 📊 统计可追溯性关口
│   ├── peer_reviewer.py        # 🧪 顾问级模拟同行评审
│   ├── research_pipeline.py    # 检索 -> 数据 -> 写作 -> 校验 -> 持久化
│   ├── research_service.py  # 串联项目状态、外部客户端与管线
│   ├── external_clients.py  # Crossref / PubMed / Semantic Scholar / arXiv 适配器
│   ├── llm_client.py        # Codex CLI / Gemini / OpenAI 兼容客户端
│   ├── data_analyzer.py     # 可选 CSV 数据的描述性统计
│   ├── http_client.py       # 无第三方依赖、带重试的 HTTP 传输层
│   ├── artifact_store.py    # 版本化产出物存储（v1/v2/v3…）
│   ├── state_manager.py     # 项目状态与检查点
│   ├── export_service.py    # Markdown -> PDF / PPTX
│   └── config_loader.py     # settings.yaml 加载器
├── scripts/                 # export_pptx.py、generate_pdf.py、verify_citations.py
├── templates/               # 文档模板
└── tests/                   # pytest 测试
```

### 📋 环境要求

- Python 3.11+
- 运行测试：`python3 -m pytest -q`
- PDF 导出：Pandoc + XeLaTeX（缺失时回退到 WeasyPrint，需要系统 Pango 库）
- PPTX 导出：项目内置的 `python-pptx` 导出器
- 已配置的内容生成引擎（见上）

### 🚧 路线图

本项目刻意只保留**一条**执行主干（`core/research_pipeline.py`）。只有当某项能力能够接入
主干并有测试证明时，才会被加入。完整阶段说明与审计日志见 [ROADMAP.md](ROADMAP.md)。

| 阶段 | 范围 | 状态 |
|------|------|------|
| 0 | 删除桩技能层、死配置与假 `run` 路径；文档对齐现实 | ✅ 已完成 |
| 1 | 引用可信性关口（引用可追溯 + DOI 校验） | ✅ 已完成 |
| 2 | 推断统计、出版级图表、统计可追溯性关口 | ✅ 已完成 |
| 3 | 模拟同行评审；基于真实产物生成演示大纲 | ✅ 已完成 |
| 4 | CI（pytest/ruff/mypy）、打包、配置校验 | ✅ 已完成 |

### 🧪 开发

```bash
python3 -m pip install -e ".[dev]"
ruff check .   # 静态检查（版本由 [tool.ruff].required-version 锁定）
mypy           # 类型检查
pytest         # 测试
```

CI 在 Python 3.11 / 3.12 上执行以上三项（见 `.github/workflows/ci.yml`）。
`config/settings.yaml` 在运行开始前校验，出错即**快速失败并点名出错的键**，
而不是跑到一半才崩溃。

### 🤝 贡献指南

欢迎提交 Issue 和 Pull Request；重大改动请先开 Issue 讨论。

### 📄 开源协议

MIT——详见 [LICENSE](LICENSE)。

### 🙏 致谢

- [davila7/claude-code-templates](https://github.com/davila7/claude-code-templates)
- [Yuan1z0825/nature-skills](https://github.com/Yuan1z0825/nature-skills)
- [Imbad0202/academic-research-skills](https://github.com/Imbad0202/academic-research-skills)
- [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills)
