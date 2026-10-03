# 进度记录 / Roadmap

> 本文档是项目的**唯一进度记录**：阶段状态、审计结论、已知分歧。
> 面向使用者的功能说明见 [README.md](README.md)。

---

## 一、项目定位

一个 AI 辅助的文献研究工作流：输入研究主题（可选 CSV 数据），真实检索文献、综合证据、
撰写手稿、**校验每一条引用与每一个统计数字**、产出演示大纲并导出。

**它不伪造结果**：没有实验数据时，输出被明确标注为"文献综合/研究计划"。

---

## 二、设计原则（决定一切取舍）

第一性原理推导出的核心命题：**用户要的是一份"可信"的手稿——每个论断都能追溯到真实证据。**

由此得出唯一主干：

```
主题(+数据) → ① 检索证据 → ② 综合 → ③ 成稿 → ④ 可信性校验 → ⑤ 交付
```

并得出两条**判断准则**，用于裁决所有"该不该阻断"的争论：

| 证据类型 | 处理方式 | 理由 |
|---|---|---|
| **确定性可验证**（引用标识是否真实存在、统计陈述能否追溯到计算结果） | **硬阻断** | 没有合理解释空间；放过即等于造假 |
| **判断/意见**（LLM 的审稿意见、DOI 网络解析失败） | **只告警，不阻断** | 意见不是证据；网络抖动不等于造假 |

**架构约束**：只保留**一条**执行主干 `core/research_pipeline.py`。
任何能力只有能接入主干、并有测试证明时才被加入。

---

## 三、阶段状态

| 阶段 | 范围 | 状态 |
|---|---|---|
| **Phase 0** | 收敛与清账：删除桩层与死代码、移除假执行路径、文档对齐现实 | ✅ 完成 |
| **Phase 1** | 引用可信性关口（标识完整性硬阻断 + DOI 校验顾问级） | ✅ 完成 |
| **Phase 2** | 推断统计引擎、出版级图表、统计可追溯性关口、管线集成 | ✅ 完成 |
| **Phase 3** | 顾问级模拟同行评审、基于真实产物的演示大纲 | ✅ 完成 |
| **Phase 4** | CI（pytest/ruff/mypy）、打包（pyproject）、配置校验 | ✅ 完成 |
| **Phase 5** | PDF 交付保障、配置诚实化、死枚举清理、端到端无断点审计 | ✅ 完成 |
| **Phase 5.1** | 论断—证据语义追溯（核心承诺的最后一块） | ✅ 完成 |

---

## 四、审计日志

每个阶段完成后，**必须独立复算/实证**验证，不采信开发者自评。以下是审计发现并修复的缺陷。

### Phase 0
| 缺陷 | 处理 |
|---|---|
| `skills/`（31 文件）返回模拟数据、未被任何流程调用 | 删除 |
| `core/base_skill.py` 引用不存在的 API（`update_stage`/`stage_records`/`save`） | 删除 |
| `core/skill_bridge.py` 删除 skills 后成为纯死代码 | 删除 |
| `config/workflow.yaml` + `workflows/` 无任何代码读取，且与实现不符 | 删除 |
| `orchestrator.py` 的 `run` 命令只标记阶段完成、不做实际工作 | 删除 |

### Phase 1
| 缺陷 | 处理 |
|---|---|
| README 承诺"每个 DOI 都经 CrossRef 验证"，实际只把检索结果拼成参考文献 | 实现真正的关口 |
| 固定容差 `0.005` 使 `p = 0.0001` 误配计算值 `0.001`（假阴性） | 改为按**书写精度**匹配 |

### Phase 2
| 缺陷 | 性质 |
|---|---|
| `_load_csv` 全局数值化摧毁分类列 → 分组检验永远失效 | 宣称可用实则失效 |
| 不传分组列时静默不生成分组图**且不告警** | 静默降级 |
| 同一列产出两张相同图表 | 冗余（规格自相矛盾） |
| `title.split(" 的")` 用展示字符串反解列名 | 代码坏味 |
| **Spearman 的 CI 用 Fisher-z 推导**（该变换假设双变量正态） | 方法论错误 |
| ANOVA 未做事后比较却未声明 | 可被过度解读 |
| 正态性被拒后不升级提示 | 提示不足 |

### Phase 3
| 缺陷 | 性质 |
|---|---|
| 评审综合意见把 5 个真实问题虚报成 15 个（注入意见按审稿人重复计数） | 诚实性 |
| docstring 声称"意见必须引用原文"，但注入意见用关口指针 | 文档与行为矛盾 |
| `_has_tests` 对 int 输入返回错误 | 潜在集成 bug |

### Phase 4
| 缺陷 | 性质 |
|---|---|
| 我显式按"规则族"配置 ruff，把违规从 53 项放大到 **740 项** | 配置失误：ruff 默认是**精选规则**而非整族，按族选择会引入 E501/PLR0915/PLR2004/RUF001 等噪声 |
| `pyproject.toml` 的 `addopts="-q"` 与命令行 `-q` 叠加为 `-qq`，**吞掉测试汇总行** | 自伤配置：验证被静默削弱 |
| `llm_client` 把 `str \| None` 传给要求 `str` 的参数（8 处） | 真实类型缺陷 |
| `web_app` 把 `list[str]` 赋给 `str` 类型的字典值 | 真实类型谎言 |
| `orchestrator` 的 `**options: object` 无法与目标签名匹配 | 类型不精确 |
| 声明 Python 3.11 后暴露 `UP017`（应使用 `datetime.UTC`） | 声明真实最低版本后正确浮现的现代化项 |

### Phase 5
| 缺陷 | 性质 |
|---|---|
| PDF 兜底路径 WeasyPrint **依赖系统 Pango**，本机不可用 → 用户根本拿不到 PDF | 兜底即第二个故障点 |
| `settings.yaml` 共 9 个 section，**代码只读 3 个**（`llm`/`api_keys`/`review`） | 配置文件成了最后一个说谎的地方 |
| `WorkflowStage` 9 个成员**只用 5 个** | 声明但未使用 |
| **删除枚举成员会让 3 个真实项目崩溃**：容错只覆盖了 `stage_status`，未覆盖 `current_stage` → `current_stage=None` → CLI `status`/`list` 崩溃 + Web `/api/projects` 500 | **真实断点**（本机 `projects/` 中的真实数据触发） |
| `llm.timeout_seconds` 对 `openai_compatible`/`gemini` 生效，对 **`codex_cli`（默认 provider）无效** | 读取 ≠ 生效 |
| 我的端到端测试台最初**绕过了 `ResearchService`**，导致项目状态从未被更新 | 测试台错误：状态归 Service 所有，端到端测试必须走用户真实链路 |

### Phase 5.1
| 缺口 / 缺陷 | 性质 |
|---|---|
| 引用关口只证明 `[P1]` **存在**，不证明 P1 **支持**该论断——模型可写"X 使 Y 上升 30% [P1]"而 P1 讲的是别的东西，**所有关口全部通过** | **核心承诺的最后一块缺口** |
| 我的端到端测试台最初绕过了 `ResearchService`，导致项目状态从未更新 | 测试台错误（已在 Phase 5 修正） |

审计验证的 10 项：空引用强制降级、模型失败不外泄、未知标识/无摘要不问模型、覆盖保证、
反编造、最坏判定、定死键、确定性、空输入零调用、跨模块注入。

### 一次误报的澄清
两位成员先后报告"测试顺序相关抖动"。判定性检查结果：**未安装随机化插件**（顺序固定）、
可疑测试**连跑 10 次全过**、**文件顺序反转通过**、**5 次全量运行全过**。
结论：**不是产品缺陷，是并行开发的竞态**（成员跑测试时其他 agent 正在编辑同一文件）。
教训：**审计必须先复现，再决定是否修改**。

---

## 五、已解决的分歧：`d7db5db`

远端曾有一个方向与本项目**相反**的提交 `d7db5db`：

| 项目 | `d7db5db` | 本项目（方案 A） |
|---|---|---|
| 技能层 | 把 Skill 接进调度器（`_execute_stage()` 真正调用各 Skill） | 整体删除 |
| 状态模型 | 新增 `stage_records` / `update_stage()` | 未采用 |
| `core/base_skill.py` | 删除 | 删除（结论一致） |

**处理方式（已完成）**：

1. `d7db5db` 保留为分支 `legacy-skills`，并推送到远端同名分支——**先在远端保存**，
   再改动 main，确保任何情况下该提交都不会失去引用。
2. 用 `git merge -s ours origin/main` 让 `main` 成为 `origin/main` 的**超集**，
   于是推送是普通快进（`d7db5db..958a52b`）——**无需强推，不重写历史**。
3. 该提交至今仍是 `main` 的祖先，`git log` 可追溯。

**为何不采用强推**：强推 main 会重写历史；且若 `legacy-skills` 尚未推送到远端，
被覆盖的提交会在远端失去引用（只能依赖本地 reflog 找回）。
"先保存、再超集、后推送"的顺序把风险降到零。

---

## 六、质量门禁

三道门禁，全部在 CI（`.github/workflows/ci.yml`）中强制执行：

| 门禁 | 命令 | 现状 |
|---|---|---|
| 静态检查 | `ruff check .` | 全部通过（ruff 版本锁定 `==0.16.0`，保证可复现） |
| 类型检查 | `mypy` | 20 个源文件 0 问题 |
| 测试 | `pytest` | 152 通过 / 0 跳过（PDF 走纯 Python 兜底后不再需要跳过） |

### 端到端无断点审计

`tests/test_end_to_end.py` 走**用户真实链路**
（`Orchestrator → ResearchService → ResearchPipeline → 三重校验 → 评审 → 导出 → CLI → Web`），
只对外部世界（文献 API / LLM / DOI 解析）打桩，并在**每个交接处**断言数据未断裂：

| 交接 | 断言 |
|---|---|
| 检索 → 分析 | `literature.json` 非空且含真实 DOI |
| 数据集 → 统计 | 产出推断检验（Welch t + Cohen's d），非仅描述统计 |
| 数据集 → 图表 | 每个登记的图表**文件真实落盘**且非空 |
| 写作 → 引用关口 | `passed == True` 且无未知标识 |
| 写作 → 统计关口 | 关口**确实看到了**计算出的 p 值 |
| 关口 → 手稿 | 统计表、图表清单、参考文献均出现在手稿中 |
| 写作 → 评审 | 评审报告存在且有结论 |
| 写作 → 演示大纲 | 大纲含真实统计与图表 |
| 全部 → 导出 | MD 非空、PDF 以 `%PDF` 开头、PPTX 为合法 zip |
| 状态 | `current_stage == export`，`export == ready_with_author_checks` |
| CLI / Web | `status`/`list`/`/api/projects`/下载端点全部可用 |
| 旧项目 | Phase-5 之前的 `state.json`（含已删阶段名）仍可驱动全部界面 |

配置校验：`core/config_validation.py` 把 `config/settings.yaml` 视为不可信输入，
在 `ResearchService` 产生**任何副作用之前**快速失败，并点名出错的键。

**已知限制**：`pyproject.toml` 只做元数据 + 工具配置 + 依赖声明；应用按源码目录运行
（`scripts/`、`web/`、`templates/`、`config/` 相对仓库根解析），wheel 分发尚未打包这些运行时资产。

---

## 七、计划中的配置项

`config/settings.yaml` 只保留**当前有代码读取**的键——未被读取的键会误导用户，
让他们以为自己能配置某件事。以下键已从配置文件中移除，未来的实现方向记录在此，
而不是留在配置里。

| 原键 | 现状 | 重新接回所需的实现 |
|---|---|---|
| `paths.*`（projects_dir / templates_dir / output_dir / scripts_dir） | 未读取；目录由仓库根硬编码解析 | 让 `ArtifactStore`、模板与脚本加载改从该节读取路径，并提供默认值 |
| `search.*`（default_databases / max_results_per_source / default_year_range / deduplication_fields / real_sources） | 未读取；检索参数由调用方 / 环境变量决定 | 在 `ResearchService`/`LiteratureSearcher` 中接入检索配置，并把 `real_sources` 校验接回 `config_validation` |
| `citation.*`（default_style / supported_styles / export_formats） | 未读取；引用格式固定 | 实现可选引用样式渲染器，并把样式导出格式接入导出层 |
| `writing.*`（default_paper_type / default_language / bilingual_abstract / style_guide） | 未读取；写作提示词为固定常量 | 让 `ResearchPipeline` 的写作提示词按这些键参数化 |
| `integrity.*`（mandatory_stages / citation_verification / cross_model_check / temporal_validation） | 未读取；可信性关口按确定性规则硬编码 | 把关口开关接入管线，并让 `mandatory_stages` 真正决定阻断点 |
| `export.*`（default_format / pdf_engine / pptx_template / include_speaker_notes） | 未读取；导出参数由 `export_service` 的调用方决定 | 让 `export_service` 从该节读取默认格式与引擎（注意：由 `pdf-dev` 负责该文件） |
| `logging.*`（level / format / file） | 未读取；日志在入口按库默认配置 | 在应用入口用该节初始化 `logging` |
| `llm.model` / `llm.base_url` | **保留**（由 `ResearchService` 读取） | — |
| `api_keys.scopus_key` / `api_keys.elsevier_key` | 未读取；无对应检索实现 | 实现 Scopus / Elsevier 检索源后接回 `LiteratureSearcher.from_config` |
| `figures.default_journal` / `default_format` / `color_palette` / `font_family` / `font_size_pt` | 未读取；图表样式为固定常量 | 让 `figure_builder.build_figures` 接受这些样式参数并由配置驱动 |
| `review.include_devil_advocate` / `consensus_threshold` / `score_scale` | 未读取；评审规则在 `peer_reviewer` 中固定 | 让 `peer_reviewer` 接受这些参数并在配置校验中保留 |
