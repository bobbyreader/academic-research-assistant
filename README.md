# Nature Skills 科研工作流系统

一套完整的学术研究辅助工作流系统，覆盖从研究构思到论文发表再到成果汇报的全生命周期。

## 系统架构

```
Nature_Skills/
├── orchestrator.py          # 主调度器（CLI 入口）
├── config/
│   ├── workflow.yaml        # 工作流配置
│   └── settings.yaml        # 全局设置
├── core/
│   ├── __init__.py
│   ├── skill_bridge.py      # Skill 基类和数据桥接
│   ├── state_manager.py     # 项目状态管理
│   └── artifact_store.py    # 产出物版本管理
├── skills/                  # 10 个 Skill 模块
│   ├── brainstorming/       # 科学头脑风暴
│   ├── academic_search/     # 学术检索
│   ├── literature_review/   # 文献综述
│   ├── ars_integration/     # ARS 集成
│   ├── statistics/          # 统计报告
│   ├── visualization/       # 科学可视化
│   ├── writing/             # 论文撰写
│   ├── polishing/           # 润色翻译
│   ├── reviewer/            # 模拟审稿
│   └── paper2ppt/           # 论文转 PPT
├── workflows/               # 工作流模板
│   ├── lightweight.yaml     # 轻量级
│   ├── heavyweight.yaml     # 重量级
│   └── hybrid.yaml          # 混合型（推荐）
├── scripts/                 # 工具脚本
├── templates/               # 模板文件
└── output/                  # 输出目录
```

## 10 大 Skill 功能

| Skill | 功能 | 适用阶段 |
|-------|------|---------|
| **Brainstorming** | 研究构思、假设生成、跨学科探索 | 阶段 0：构思 |
| **Academic Search** | 多源检索、引用格式、他引审计 | 阶段 1：文献 |
| **Literature Review** | 系统性综述、PRISMA、质量评估 | 阶段 1：文献 |
| **ARS Integration** | 深度研究、论文撰写、多视角评审 | 全流程 |
| **Statistics** | 统计审查、重复类型区分、审稿回应 | 阶段 2：数据 |
| **Visualization** | 出版级图表、无障碍审查、导出规划 | 阶段 2：数据 |
| **Writing** | 手稿起草、claim-evidence 叙事、投稿包 | 阶段 3：撰写 |
| **Polishing** | 润色翻译、AI 味检查、风格调整 | 阶段 3：撰写 |
| **Reviewer** | 模拟审稿、三报告交叉验证、12 轴清单 | 阶段 4：审查 |
| **Paper2PPT** | 论文转中文 PPT、speaker notes | 阶段 5：传播 |

## 三种工作流模式

### 1. 轻量级（Lightweight）
> 适合已有实验数据、需要快速撰写 Nature 风格论文

```
academic_search → statistics → visualization → writing → polishing → reviewer → paper2ppt
```

### 2. 重量级（Heavyweight）
> 适合从零开始做完整研究项目，以 ARS 为核心调度器

```
brainstorming → ARS Deep Research → ARS Academic Paper → ARS Integrity → ARS Reviewer → ARS Revision → paper2ppt
```

### 3. 混合型（Hybrid，推荐）
> 取各 Skill 优势组合，最大化效率

```
brainstorming → academic_search → literature_review → ARS Deep Research → statistics → visualization → ARS Academic Paper → polishing → reviewer → ARS Integrity Final → paper2ppt
```

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 初始化项目

```bash
python orchestrator.py init my_research --mode hybrid
```

### 运行工作流

```bash
python orchestrator.py run my_research --workflow hybrid
```

### 查看状态

```bash
python orchestrator.py status my_research
```

### 导出产出物

```bash
python orchestrator.py export my_research --format md
python orchestrator.py export my_research --format pdf
python orchestrator.py export my_research --format pptx
```

## 使用示例

### 示例 1：科学头脑风暴

```python
from skills.brainstorming.skill import BrainstormingSkill

skill = BrainstormingSkill()
result = skill.execute({
    "research_topic": "AI 对高等教育质量保障的影响",
    "current_stage": "understand_context",
})
print(result.data["prompts"])
```

### 示例 2：学术检索

```python
from skills.academic_search.skill import AcademicSearchSkill

skill = AcademicSearchSkill()
result = skill.execute({
    "action": "search",
    "query": "artificial intelligence higher education quality assurance",
    "sources": ["crossref", "pubmed", "semantic_scholar"],
    "max_results": 20,
})
print(result.data["results"])
```

### 示例 3：统计审查

```python
from skills.statistics.skill import StatisticsSkill

skill = StatisticsSkill()
result = skill.execute({
    "action": "audit",
    "statistical_text": "n=5, t-test, p<0.05",
    "figure_legends": "Figure 1: Bar chart showing...",
})
print(result.data["issues"])
```

### 示例 4：论文撰写

```python
from skills.writing.skill import WritingSkill

skill = WritingSkill()
result = skill.execute({
    "action": "draft",
    "section": "abstract",
    "claims": [
        {"text": "AI 显著提升教育质量保障效率", "figures": ["fig1"], "citations": ["ref1"]},
    ],
    "figures": [{"id": "fig1", "caption": "效率对比图"}],
})
print(result.data["draft_text"])
```

## 关键特性

- **人机协作**：AI 处理繁琐工作，人类专注思考与判断
- **多层质量保障**：引用验证、主张审计、学术诚信闸门
- **透明可追溯**：Material Passport 记录全流程
- **反 AI 局限性**：针对框架锁定、谄媚倾向、意图检测错误专门优化
- **模块化设计**：各 Skill 独立可用，也可组合成完整 Pipeline

## 学术诚信保障

系统内置多重学术诚信防线：

1. **引用查验 Gate**：确定性引用存在性查验（Semantic Scholar/OpenAlex/Crossref/arXiv resolver）
2. **L3 Claim-Faithfulness**：三层引用 anchor + opt-in 审计
3. **时序验证层**：5 种时序失效模式检测
4. **跨模型验证**：可选第二 AI 模型独立审查
5. **不可跳过阶段**：Stage 2.5（审稿前）+ Stage 4.5（最终）强制诚信验证

## 常见问题

### Q: 如何选择工作流模式？
- **轻量级**：已有数据，快速成文
- **重量级**：从零开始，完整研究
- **混合型**：灵活组合，推荐大多数场景

### Q: 如何配置 API 密钥？
编辑 `config/settings.yaml`，填入对应的 API 密钥。注意：不要将真实密钥提交到版本控制。

### Q: 学术诚信验证可以跳过吗？
不可以。Stage 2.5 和 Stage 4.5 是强制阶段，这是系统的核心安全设计。

### Q: 支持哪些引用格式？
Nature、APA 7.0、IEEE、Vancouver、Chicago，可导出 `.ris`/`.bib`/`.nbib`/`.enw`。

## 许可证

CC-BY-NC 4.0（署名-非商业性使用）

## 参考

- [claude-code-templates](https://github.com/davila7/claude-code-templates)
- [nature-skills](https://github.com/Yuan1z0825/nature-skills)
- [academic-research-skills](https://github.com/Imbad0202/academic-research-skills)
- [scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills)
