# arc26 MCP 与 REPL 工具参考

智能体是**不做修改的 MiniMax Code**（`mcode acp`，Goal 模式）。每局游戏启动三个进程：

| 进程 | 文件 | 运行身份 | 作用 |
|---|---|---|---|
| 游戏守护进程 | `arc_mcp/server.py --daemon` | root | 持有游戏引擎和账本（步数与官方 scorecard 一致），unix socket |
| REPL kernel | `arc_mcp/kernel.py` | root，内存上限 6GB | 常驻 Python 命名空间；没有引擎，只能通过守护进程动作；审计钩子禁止读游戏文件、禁止子进程 |
| MCP 前端 | `arc_mcp/server.py`（stdio） | 沙箱用户 `arcagent` | 把下面三个工具暴露给 MiniMax Code |

kernel 挂掉（内存溢出、硬崩溃）时，`arc_runner/batch.py` 会自动重启它。DAG 从文件重新加载，游戏状态在守护进程里，只丢失模型自己定义的变量。模型会收到明确提示。

## MCP 工具（3 个）

| 工具 | 参数 | 返回 |
|---|---|---|
| `arc_python` | `code`、`purpose`（一句话：这个格子回答什么问题）、`expect`（预期结果）必填；`check`（判定对错的 Python 表达式）、`revises`（修正哪个意外节点）、`parents` 可选 | 状态行 + DAG 节点头（预期 → RIGHT/WRONG）+ 实时对错日志 + 感知块（执行了动作时附 4 倍放大图） |
| `arc_note` | `section` ∈ rules / goal / levels / plan，`text`，`mode` = replace / append | 局内笔记；每段最多 1500 字；上下文压缩或重启后原样交还 |
| `arc_dag` | `last` | 探索 DAG：每个格子一行（id、父节点、关卡、步数、目的，`!` 表示意外），以及还没修正的意外 |

## REPL 命名空间

### 状态变量（每次动作后刷新；都是副本，改了也不影响游戏）
- `grid`：当前帧，`np.ndarray (64, 64)`，`grid[y, x]`，颜色 0–15。被模型改写后，格子结束时自动恢复。
- `prev_grid`：上一个动作之前的帧。`frames`：上一个动作的全部动画帧。
- `state`：`'NOT_FINISHED' | 'GAME_OVER' | 'WIN'`；`level`：已完成关卡数；`win_levels`：总关卡数。
- `available`：当前可用的动作 id；`actions_used`、`level_actions_used`：与 scorecard 一致的步数（RESET 也计 1 步）。
- `history`：每个动作一条记录，字段有 `n, action, x, y, level, state, changed, note, moves, small, target, effect, death`。
- `lattice_cells`：最近一次 `cells()` 得到的格子数组。

### 函数
| 函数 | 作用 |
|---|---|
| `act(a, x=None, y=None, force=False)` | 执行一个动作：0 RESET（只在 GAME_OVER 后有效）、1–4 方向（每个游戏含义不同）、5 交互、6 点击 (x, y)、7 撤销。返回 `{changed, state, level, level_up, game_over, note, frames}`。如果某个动作在完全相同的画面下曾导致死亡，且这个组合从没安全执行过，就拒绝执行、不耗步数；`force=True` 强制执行 |
| `reset()` | 等于 `act(0)` |
| `show(g=None, y0, y1, x0, x1)` | 某个区域的十六进制文本，带行列标号 |
| `changes(a=None, b=None)` | 两帧之间变化的像素 `[(y, x, old, new)]` |
| `objects(g=None)` | 4 连通同色块 `{color, size, bbox, center}` |
| `regions(a=None, b=None)` | 把变化的像素聚成区域，显示前后颜色 |
| `anim()` | 上一个动作各动画帧之间的差异 |
| `cells(step=None, ox=None, oy=None)` | 把画面压成格子地图：从棋盘上反复出现的方块尺寸或动作位移的最大公约数推断格子大小和起点，裁掉背景，用十六进制显示 |
| `action_stats(level=None)` | 每个动作的效果统计（ACTION6 按被点中的物体分开）：次数、无效果次数、移动向量、改变的区域（已排除步数条） |
| `deaths(level=None)` | 每次死亡的报告：死亡那一步动了什么、步数条状态、是否和之前某次死法相同 |
| `look()` | 在本格子的回复里附上当前帧图像 |
| `journal`、`dag()`、`node(i)`、`rerun(i)` | 实时对错日志、DAG、第 i 个格子的记录、重跑第 i 个格子的代码 |
| `np` | numpy |

### 感知块（执行了动作的格子会自动附上）
1. 当前帧的 4 倍放大 PNG（每个像素放大成 4×4），以图片形式发给模型。
2. 物体列表：按颜色和尺寸分组，罕见的排前面。
3. 上一个动作改变的区域；整个格子改变的区域（多个区域 = 有副作用）。
4. 移动的物体和位移向量。
5. 计量条：连续几步都在同一行（或列）小范围变化的条，报告剩余格数和"大约还能走几步"。
6. 格子地图：每关第一次推断出格子时显示一次。
7. 动作效果表：RESET 之后显示本关已知效果；过关后显示上一关的效果。
8. 动画差异。

## MiniMax Code 插件（`plugin/arc26`，原生 `.minimax-plugin` 格式）
| 钩子 | 作用 |
|---|---|
| `SessionStart` | 把状态、笔记、对错日志、最近的 DAG 节点作为 additionalContext 注入，压缩或重启后原样恢复 |
| `PreToolUse` | 禁用 bash、read、grep、glob、web、ls 等工具，只允许通过 arc 工具玩；write/edit 只能写 `.minimax/skills/` |
| `PostToolUse` | 把非 REPL 的工具调用（todowrite 计划、笔记、goal 等）记进 `dag_events.jsonl`，在 DAG 里可见 |

规划用 MiniMax Code 自带的 todowrite 和 Goal 模式（`beta.threadGoal`），不自己实现。

## 运行器 `arc_runner/batch.py`
主要参数：`--games`（逗号分隔或 all）、`--k`（每个游戏跑几局）、`--conc`、`--max-game-seconds`、`--hours`、`--base-url`、`--model`、`--reasoning`（开思考）、`--output-limit`、`--agent-user arcagent`（沙箱用户）、`--env-dir`、`--gateway`（比赛模式）。

托管 API 走 `arc_runner/llm_proxy.py`：
- `--inject` 合并到请求体，例如 `chat_template_kwargs.enable_thinking`；
- `--drop max_tokens,max_completion_tokens` 去掉输出上限。

技能只在同一次运行的不同游戏之间共享，按来源过滤：同一个游戏再玩时，读不到自己上次留下的技能。

## 评估
- `python -m arc_eval.summarize <run_dir>`：每个游戏的关卡数、每关步数/人类步数、RHAE（与引擎公式一致），以及功能使用统计。
- 游戏清单与切分：见 `games/manifest.json` 和 `docs/GAMES.md`。
