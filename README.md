# 🔬 Academic Research Assistant

> A comprehensive AI-powered workflow system for scientific research — from hypothesis generation to paper publication and presentation.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/downloads/)
[![Stage: Alpha](https://img.shields.io/badge/Stage-Alpha-orange.svg)]()

---

## 🎯 What Is This?

**Academic Research Assistant** is a modular, end-to-end research workflow system that orchestrates 10 specialized AI skills across the entire research lifecycle — from brainstorming ideas and searching literature, through statistical analysis and visualization, all the way to manuscript writing, peer review simulation, and presentation generation.

Think of it as a **research cockpit**: instead of juggling disconnected tools, you have one intelligent system where every module passes data seamlessly to the next, following battle-tested workflows inspired by Nature journal standards, PRISMA guidelines, and academic best practices.

---

## ⚡ Quick Start

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

## 🗺️ The Research Lifecycle

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

### Three Workflow Modes

| Mode | Best For | Stages |
|------|----------|--------|
| **🔵 Lightweight** | You have data, need a paper fast | 7 stages |
| **🟡 Heavyweight** | Full research project from scratch | 7 stages |
| **🟢 Hybrid** *(recommended)* | Maximum flexibility & quality | 11 stages |

---

## 🧩 The 10 Skills

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

## 🏗️ Architecture

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

## 🔑 Key Features

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

## 👥 Who Is This For?

- **PhD students** navigating the literature → writing → submission gauntlet
- **Researchers** needing systematic review support with PRISMA compliance
- **Lab groups** wanting consistent, high-quality manuscript standards
- **Postdocs** preparing for journal submission under Nature or Nature Communications
- **Academic writers** seeking AI assistance that respects academic integrity

---

## 📋 Requirements

- Python 3.11+
- API keys for external services (PubMed, Semantic Scholar, CrossRef — free tiers available)
- Pandoc + xelatex (for PDF export)
- python-pptx (for PPTX generation)

---

## 🤝 Contributing

Contributions are welcome! Please read our contribution guidelines and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This system is inspired by and integrates concepts from:

- [davila7/claude-code-templates](https://github.com/davila7/claude-code-templates) — Scientific Brainstorming & Literature Review
- [Yuan1z0825/nature-skills](https://github.com/Yuan1z0825/nature-skills) — Nature Academic Search, Writing, Polishing, Reviewer, Paper2PPT
- [Imbad0202/academic-research-skills](https://github.com/Imbad0202/academic-research-skills) — Academic Research Suite (ARS)
- [K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills) — Scientific Visualization

---

*Built with ❤️ for researchers who deserve better tools.*
