# Raven 三大机制源码笔记（移植依据）

> 来源：`EverMind-AI/Raven`（Apache-2.0）与 `EverMind-AI/EverOS`（Apache-2.0）主分支源码阅读。
> 本仓库 **不依赖** 这两个包；下面记录的是我们重新实现时参照的设计，以及针对 ARC-AGI-3 的改动。
> 路径均相对于各自仓库根目录。

---

## ① DAG 调度

**关键文件**
- `raven/agent/subagent/dag_graph.py` — 数据模型、静态校验、Kahn 拓扑排序
- `raven/agent/subagent/dag_runner.py` — 调度器 `run_dag`
- `raven/agent/subagent/dag_tool.py` — Host 可调用的 `run_subagent_dag` 工具及 schema
- `raven/agent/subagent/dag_render.py` — 上游输出如何进入下游 prompt
- `raven/agent/subagent/dag_verdict.py` — 节点完成后的 LLM 裁判
- `raven/agent/subagent/dag_adjudication.py`, `dag_control_tools.py` — suspend / continue / abandon / replan
- `raven/memory_engine/skills/subagent-dag-orchestration/SKILL.md` — 写图的规则

**要点**
1. **没有独立 planner**：Host LLM 在一次 `run_subagent_dag` 工具调用里写出整张图。
   节点 `DagNodeSpec{id, subagent, node_summary, prompt_template, depends_on, inputs, instance, skills, mcps}`，
   pydantic `extra="forbid"`；图 `SubAgentDagSpec{task_summary, nodes, confirm}`。边只以 `depends_on` 表达。
2. **派发前静态校验**（在 Host 自己的回合内完成）：id 唯一（大小写折叠，且在整个会话内唯一）；
   依赖可解析（本图节点或更早已完成节点）；`{{x.output}}` 必须引用已声明依赖（default-deny）；
   每个声明的 input 必须被占位符引用；Kahn 排序检测环：`len(order) != len(nodes) → "graph contains a cycle"`。
3. **调度器**：不按拓扑序走，而是每轮重算 ready 集：
   `pending ∧ 所有依赖 completed ∧ 依赖已被裁判定案`。ready 节点按 `instance` 分组——**同一 instance 串行**
   （有状态会话），其余各自成组；每组一个 asyncio task；共享 `Semaphore`（默认 8）。
   用 `asyncio.wait(FIRST_COMPLETED)` 等待，外加 cancel / replanned（硬）与 settled / continued（软）信号：
   任一节点完成即结束本轮，未完成的 task 带入下一轮，下游立即启动。
4. **输出传递**：每个节点的回复写 `nodes/<id>.out.md`；`render_prompt` 单次左到右替换
   `{{id.output}}`（内容，包成不可信数据）或 `{{id.output_path}}`（只给路径），替换后的文本不再二次扫描。
5. **失败**：后端异常 → `failed`；`_cascade_failures` 把依赖失败/跳过的 pending 节点置 `skipped`；
   不自动重试崩溃，只通过 continuation（最多 2 次）或裁判 `follow_up`。
6. **裁判**：每个节点结束后一次 `report_verdict{outcome, category, what_is_missing, evidence}` 调用，
   180s 超时，**出错即放行（fail-open）**。未完成 → 节点挂起为 `exception`，
   Host 通过 `resolve_dag_node` 选择：
   - `continue` + 补充信息：节点回到 pending（无状态节点收到 "Your previous attempt returned: … Now: …"）；
   - `abandon`：节点失败，下游级联跳过；
   - `replan` + 新节点：完整重新校验，停止本次运行、提交新图（新图可引用已完成节点 id，不重复声明）。
7. **SKILL.md 规则摘录**："Iteration is graphs in series, not a cycle in one graph"；
   "Replanning because a node is hard is how a graph loops without progressing"；
   节点 id 全会话唯一；大内容用 `_path` 形式传递。

**ARC 化改动（`arc_harness/dag/`）**
- 环境是有状态资源 → 所有触碰环境的节点共享 `instance="env:<game>"`，自动串行；只有"思考"节点并行。
- 一局游戏 = 多张图串联（每个"回合"一张图），而不是一张带环的图。
- 跨游戏并发：所有游戏的 DAG 共享一个 LLM 信号量，喂饱 vLLM 的连续批处理。
- 裁判默认用确定性判据（帧是否变化、关卡是否 +1、是否 GAME_OVER），LLM 裁判只在确定性判据无法判定时调用。

---

## ② 自动沉淀 skill

**关键发现**：Raven 自身的"本地抽取流水线"（`skillForge.extraction`）只存在于配置里，未接线；
真正的 case / skill 抽取发生在 **EverOS 服务** 里（`everos/memory/strategies/*`），prompt 在
`everalgo-*` PyPI 包中。Raven 负责：每轮把消息送进去、检索、注入。

**流水线（EverOS）**
1. 触发：`raven/agent/loop/turn_path.py` 每轮结束 `_dispatch_backend_store`；`flush_every_turns=1`。
   **没有成功与否的门控**。
2. 边界：`AgentBoundaryDetector`（一次 LLM 调用）把消息切成 memcell（任务段）。
3. Case：`AgentCaseExtractor` → `AgentCase{task_intent, approach, key_insight, quality_score∈[0,1]}`（第三人称）。
4. 聚类：`quality_score < 0.2` 丢弃；`task_intent` 做 embedding，`cluster_by_llm` 决定并入已有簇或新建。
5. Skill：`AgentSkillExtractor(target_case, 同簇已有 skill ≤10, 支持 case ≤9)` → 新增或改写 `SKILL.md`；
   这一步即去重/合并。**未实现退役**。E2E 测试需要约 8 次同类会话才能稳定产出一个 skill。

**存储**：`SKILL.md` + YAML frontmatter。
- Raven 本地：`name, description, metadata{raven:{always, inject, requires}}`。
- EverOS：`type: agent_skill, name, description, confidence, maturity_score, source_case_ids, cluster_id`。

**检索与注入**（`raven/memory_engine/skill_forge/`）
- 各源（local BM25、EverOS recall、Hub）各取 `k×2`，RRF 融合 `Σ w_i/(10+rank)`（权重 local 0.96 / everos 0.9 / hub 0.85），按 name 去重。
- `QueryRewriter`（是否需要检索 + 去噪改写）→ `LLMGateFilter`：
  "AT MOST 2 … selecting an irrelevant or unexecutable skill is strictly worse than selecting none"，
  且必须"仅用现有工具就能执行"。
- Push 模式注入完整正文（`### Skill: name`）；Pull 模式（默认）只给菜单，模型用 `find_skill/read_skill` 自取。

**上下文压缩**（`raven/agent/window/compaction.py`）：先把旧工具输出替换为占位符（保留最近 3 个），
再把对话头部摘要为"交接简报"（目标/验收标准、已确认事实、已完成、测试结果原文、剩余计划），保留尾部 25% 原文。

**Raven 的短板 → 我们的改动（`arc_harness/memory/`）**
| Raven/EverOS | 本仓库 |
|---|---|
| LLM 边界检测 | 游戏天然边界：关卡通过 / GAME_OVER / 游戏结束 |
| LLM 自评 quality_score | 客观质量：是否过关 × 动作效率（状态图最短路 / 实际步数） |
| embedding 聚类 | 观测特征（可用动作集、对象种类、点击/方向类）相似度 + LLM 合并决策；embedding 可选 |
| 使用反馈是 no-op | 记录 skill 被注入后关卡的表现，更新 `confidence`，低于阈值退役 |
| 依赖 EverOS 服务 + 云端 LLM/embedding | 纯本地 Markdown + JSON，LLM 指向本地 vLLM |

---

## ③ 自进化

**离线 evolver（`evolver/`）** —— 贪心爬山 + MAP-Elites 精英库，不是 beam/MCTS。
1. **冷启动**：vanilla 在 train 上 K=3，得每任务稳定性桶（`STABLE_PASS / BORDERLINE_2_3 / BORDERLINE_1_3 / STABLE_FAIL`）。
2. **诊断**（`nodes/taxonomy.py`, `nodes/diagnose.py`）：失败轨迹多标签 `{why, where, dominant, reasoning, fix_hint}`
   （dominant 权重 1.0，其余 0.5），合并进 `failure_map.json`。新 benchmark 用 map-reduce 归纳 5–9 个 `W*_` 类。
3. **选 WHY + 设计**：每轮 2 个 WHY × 每 WHY 3 个候选 + 1 个重组。候选由 bash-editor agent 在父提交的
   git worktree 里生成（动作 `read_trajectory | bash | write_file | done`，≤22 轮，4 次只读后强制编辑），
   禁止硬编码任务 id；WHERE 由实际改动的文件推导。
4. **零成本剪枝**：编译、import smoke、结构性改动必须带 beacon（AST 比较，纯 prompt 改动免 beacon）。
5. **筛选**：锚点子集 K=1，只淘汰"比 vanilla 低 ≥1.5σ"者（或 Fisher 单侧显著更差）。
6. **确认**：全 train K=3。
7. **三道门**：Gate-f（infra 失败重跑 ≤2 次，仍失败记 0 分留在分母）；Gate-b（只在 beacon 触发的任务上算配对统计）；
   Gate2（配对 z：`z = mean(d)/(stdev(d)/√n)`，z≥2 只是标签）。晋升条件：候选均值 > 父节点均值（严格）。
8. **终止**：连续 10 轮未超过 vanilla 或 20 轮。**密封测试集**：`assert_no_test_leak`，评分写入循环读不到的目录，
   结束时按 train 分数选交付版本，报告 retention。
9. **护栏**：可编辑路径白名单；不可变内核（evolver、评分器、tests）；`config_fingerprint` 不一致拒绝续跑。
10. **模型角色**：driver（诊断/选 WHY）、design（编辑器）、verdict（可选）；可全部是同一个本地模型
    （作者在 qwen3.6-27B 上跑通）。关键是 `nodes/semantic.py::SemanticNode`：每次调用做 schema 校验，
    解析失败把错误回灌重试。

**在线"热进化"（`raven/contracts/harness.py`, `raven/agent/subagent/charter.py`）**
- 四个策略模块 `HarnessModules(memory, planning, capability, action)` 不被替换；每次派发由 LLM 生成一份
  `Charter`（systemPrompt / stopWhen / 可见工具子集 / 声明式 checks / 可选受 AST 门控的小函数），
  通过 ContextVar 在下一轮生效。没有评测、没有门控。

**ARC 化改动（`evolve/` 与 `arc_harness/charter.py`）**
- 分数是连续的 RHAE（加上是否过关），配对统计用每游戏 RHAE 差值（配对 z + Wilcoxon 符号秩）。
- 25 个公开游戏 → 17 train / 8 sealed test（按 tags 分层）。
- ARC WHY 种子：`W1_action_loop, W2_mechanic_undiscovered, W3_wrong_hypothesis_persisted,
  W4_inefficient_path, W5_repeated_death, W6_click_target_miss, W7_budget_exhausted, W8_perception_error`。
- 白名单：`prompts/`, `arc_harness/roles/`, `arc_harness/charter.py`, `arc_harness/explore/config.py`, `skills/`。
- 在线 Charter：开局与每关结束时生成，内容为声明式 action checks（如"同一无效动作重复 ≥3 次禁止"）
  和早停阈值；第一版不允许生成代码。
