# arc26：ARC Prize 2026（ARC-AGI-3）手写 Agent Loop

**每一步动作就是一轮，每一轮都从全新的上下文开始。** 模型看不到之前的对话，它能看到的只有下面这些：

1. **图（graph）：** 所有观察、规则、假设、目标、计划、问题，以及每个动作和它核对过的结果。节点之间用有类型的边做符号连接，例如 supports、refutes、causes、leads_to、tests。**每一轮开始前都要完整读一遍图，每一轮都必须新增至少一个节点。**
2. **交接文档：** 上一轮写给这一轮的说明，包括当前目标、接下来的计划、还不确定的地方。
3. **预测核对：** 上一轮对这一步的预测，已经和实际结果自动比对过（RIGHT / WRONG）。
4. **画面 diff：** 这个游戏每步画面变化很小，所以观测以变化为中心：
   - 哪些格子变了，聚成区域；
   - 哪些物体移动了、位移多少；
   - 哪些变化属于 HUD（步数条），单独报告；
   - 动画帧。

   另外附上当前帧的物体列表、十六进制网格和 4 倍放大图。

每一轮模型输出一个 JSON：`graph_update`（新节点、边、状态变更）、`action`、`prediction`（可自动核对：棋盘是否变化、哪些物体怎么移动、是否过关、是否死亡）、`handoff`。

**模拟器与搜索（在上面这套底层之上）：**
- **写模拟器：** 模型可以在回复里附一段 Python，写成游戏的模拟器：`step(grid, action, x, y)` 给出下一帧，`goal(grid)` 判断是否过关，`key`、`clicks` 可选。新版本会替换旧版本。
- **自动核对：** 每个真实动作都用 `step()` 核对，报告哪些格子预测错了（HUD 另算）。新版本到达时，先在本关最近 20 个动作上回放，报告每个动作错了几格。
- **搜索路线：** 回复里 `"plan": true` 时，循环在模拟器里用 BFS 搜到 `goal()` 的最短路线，这一步不花真实步数。
- **执行路线：** 找到的路线逐步执行，真实画面和预测不一致就立即停下，把差异交回模型修改代码。这一轮记为一个路线动作节点，图照常更新。
- **沙箱：** 模型的代码在子进程里运行，有时间限制，崩溃或死循环只会被报告，不会中断游戏。

| 文件 | 作用 |
|---|---|
| `arc_agent/loop.py` | 单局循环：读取、调 LLM、校验、更新图、执行动作、算 diff、核对预测 |
| `arc_agent/graph.py` | 图的存储、更新、序列化 |
| `arc_agent/sim.py` | 模型写的模拟器：子进程沙箱、逐步核对、回放、BFS 搜路线 |
| `arc_agent/vision.py` | 帧 diff、变化区域、物体位移、HUD 识别、物体列表、4 倍图 |
| `arc_agent/prompt.py` | 系统说明、每轮输入、JSON 解析与校验、预测核对 |
| `arc_agent/llm.py` | OpenAI 兼容客户端：开思考、不设输出上限、单次调用限时后重试、不超过本局截止时间 |
| `arc_agent/run.py` / `report.py` | 批量运行（每局一个进程、比赛模式计分卡）/ 成绩汇总 |
| `arc_agent/session.py` | 比赛计步规则（与官方 scorecard 一致） |
| `arc_eval/` | 游戏集与切分、RHAE |
| `games/manifest.json`、`scripts/` | 全部游戏的清单与校验、下载脚本 |
| `docs/` | [游戏清单](docs/GAMES.md)、[实验记录](docs/RESULTS.md) |

旧的 MiniMax Code 加 MCP 方案保存在历史提交 `6ea847c`。

## 运行

```bash
scripts/fetch_games.sh                                   # 官方 25 个 + 社区 249 个游戏，按 sha256 校验
python -m arc_agent.run --out-dir runs/a1 --games ls20,vc33,tu93 --conc 3 --game-seconds 2700 \
    --base-url https://<host>/v1 --model <model> --api-key-file .secrets/dots_api_key --call-seconds 300
python -m arc_agent.report runs/a1
```

密钥只放在被 git 忽略的 `.secrets/` 和 `.kaggle/` 里。
