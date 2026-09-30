# 交接文档（2026-09-30）

这份文档写给接手本地开发的人：目标、现状、代码结构、怎么在本地跑起来、Kaggle 上在跑什么、下一步做什么。实验数据见 [RESULTS.md](RESULTS.md)，游戏清单见 [GAMES.md](GAMES.md)。

## 1. 目标与现状

- **比赛：** Kaggle ARC Prize 2026 ARC-AGI-3。
  - 排行榜（09-27）：第一名 Tufa Labs 27.29，第二名 20.53，第 20 名约 7.3。
  - 我们还没提交过。
- **分数（RHAE）：**
  - 每关得分 min(115, 100·(人类步数/AI步数)²)，没过的关为 0。
  - 按关卡序号加权（第 k 关权重 k），每个游戏取加权平均，再对所有游戏取平均。
  - **只算动作步数，不算时间。**
  - 例：tu93 共 9 关，权重合计 45。按人类效率只过第 1 关，全局才 2.2 分。
- **当前阶段目标：** 在 3 个调参游戏上用 Dots 刷到平均 30 分以上，再跑完整的 25 个。调参游戏按人类步数挑选：简单 cd82（171 步）、中等 tu93（462 步）、最难 wa30（1843 步）。
- **现状：** Dots 在调参游戏上平均 1–3 分。
  - 过了的关，步数基本和人类持平。
  - 问题是每局只过 0–1 关：每步 20–45 秒，还有很多步花在试探上。
- **比赛的硬约束：** 隐藏集约 110 个游戏，重跑约 9 小时，只能用比赛 GPU（RTX PRO 6000，本地 Qwen3.8-27B）。
  - Dots 每局 4 小时的跑法只用来调方法，不能直接提交。
  - 本地 GPU 版目前每步约 150 秒，得分 0.44。

## 2. 架构（手写 Agent Loop）

入口是 `arc_agent/loop.py` 的 `play()`。**每一步动作就是一次 LLM 调用。**

### 2.1 底层（不变）

- **图（`graph.py`）：**
  - 节点类型：observation / rule / hypothesis / goal / plan / question；动作、结果和模拟器节点由 loop 自己记录。
  - 边类型：supports / refutes / causes / part_of / leads_to / about / tests / revises。
  - **每轮回答必须至少新增一个节点，否则打回重答。**
  - `render()` 先列知识再列历史；旧的、预测正确的动作会折叠。
- **交接文档：** 每轮回答里的 `handoff`，写给下一轮。
- **预测：**
  - 每轮都要预测：棋盘是否变化、哪个颜色的物体怎么移动、是否过关、是否死亡。
  - 执行后由 `prompt.check_prediction` 自动核对，结果是 RIGHT 或 WRONG。
- **画面 diff（`vision.py`）：** 变化区域、物体位移、按形状边缘判断的移动、HUD 识别、SIDE EFFECT 标记、格子地图（lattice）、动画帧。
- **上下文：**
  - `--context rolling`：对话持续进行，每轮只追加新内容。
  - **压缩：** prompt 超过 `--compact-tokens` 时压缩；每到新的一关、每次 GAME_OVER 后也压缩。压缩就是开一个新段，只带完整的图、交接文档和当前模拟器。
  - **缓存：** 旧轮次从不修改，Dots 前缀缓存命中约 86%。
  - `fresh` 模式每步开一个新对话。
- **LLM（`llm.py`）：**
  - 流式输出，思考开，不设 token 上限。
  - 单次调用限时（默认 180 秒）。超时后带着最后 12k 字符的思考续写。
  - 空回复会重试。
- **回答格式：**
  - 一个 JSON：`graph_update`、`action`、`prediction`、`handoff`、`think_next`、`need_grid`、`plan`。
  - 可以在 JSON 后面附一个 ```python 模拟器代码块；写在 JSON 的 `"python"` 字段里也能识别。

### 2.2 模拟器与搜索（`sim.py`，最新加入，结构性强制）

- **写法：** 模型写 Python。
  - 必须定义 `step(grid, action, x, y) -> grid`（预测下一帧）和 `goal(grid) -> bool`（判断是否过关）。
  - 可选 `key(grid)`：状态去重用；`clicks(grid)`：点击候选位置。
- **loop 像 graph 一样强制执行，不靠提示词劝说：**
  - **必须有：** 一关的第 3 个动作起还没有模拟器，回答就打回重答（一次修复机会）。
  - **错了必须修：** 模拟器上一步预测错了、这轮又没给新版本，同样打回。
  - **不许变差：** 新版本在本关最近 20 个动作上回放；崩溃，或者比当前版本错的格子更多，就拒收。
  - **修复失败：** 用第一次的回答照常走这一步，问题报告给下一轮。
- **进入图：**
  - 每个模拟器版本是一个 `simulator` 节点，新版本 revises 旧版本。
  - 每个动作的结果节点 supports 或 refutes 当前模拟器。
- **自动搜索：**
  - 模拟器连续 3 步预测完全正确后，每轮调用模型前，自动在模拟器里 BFS 搜到 `goal()` 的最短路线，放进输入。
  - 模型回答 `"plan": true` 就执行这条路线。
  - 执行中真实画面和 `step()` 不一致就立刻停下；遇到新关或 GAME_OVER 也停。
- **沙箱：** 模型的代码在子进程里跑，有时间限制，崩溃或死循环只报告，不会中断游戏。
- **参数：** 在 `loop.py` 顶部：`SIM_FROM`、`TRUST`、`TRANSITIONS`、`PLAN_MAX`、`SEARCH_SECONDS`、`AUTO_SEARCH_SECONDS`。

### 2.3 文件

| 文件 | 作用 |
|---|---|
| `arc_agent/loop.py` | 单局循环：组织输入、调 LLM、校验（graph 和模拟器）、执行动作或路线、diff、核对预测、写图 |
| `arc_agent/sim.py` | 模拟器沙箱：`check`（一步核对）、`replay`（回放）、`search`（BFS） |
| `arc_agent/prompt.py` | 系统说明、每轮消息、JSON 和代码解析、校验、预测核对 |
| `arc_agent/graph.py` | 图的存储、更新、渲染 |
| `arc_agent/vision.py` | 感知：diff、物体、HUD、格子地图、PNG |
| `arc_agent/llm.py` | OpenAI 兼容流式客户端 |
| `arc_agent/run.py` | 批量运行（每局一个进程）；接 `--gateway` 时走比赛模式 |
| `arc_agent/report.py` | 成绩汇总：关数、RHAE、预测正确率、每步耗时、token、缓存命中率 |
| `arc_agent/session.py` | 与官方 scorecard 一致的计步 |
| `arc_eval/` | 游戏集登记（官方 25 个 + 社区 249 个）、RHAE 计算、随机基线 |
| `kaggle/build_notebook.py`、`kaggle/config.json` | 生成 Kaggle notebook：submit / dev（GPU）/ api / loop |
| `scripts/` | `fetch_games.sh`（下载并校验全部游戏）、`games_manifest.py`、`dots_loop.sh`（本地调参循环） |

每局的输出在 `<out-dir>/games/<game>_k<n>/`：
- `steps.jsonl`：每步记录，包括 usage、预测核对结果、执行的动作 `played`、`sim_checks`、`sim_problems`；
- `reasoning.jsonl`：模型的思考和回复；失败的回复也会记下；
- `graph.json`；
- `result.json`。

## 3. 本地运行

```bash
git clone https://github.com/Timothy-kira/arc26 && cd arc26
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# 密钥（都在 .gitignore 里，不要提交）
mkdir -p .kaggle .secrets
#   .kaggle/access_token  ← Kaggle token（需先在网页上接受比赛规则）
#   .secrets/dots_api_key ← Dots API key

make data                      # 比赛数据：25 个公开游戏 + 引擎 wheel，放到 data/
pip install data/arc_agi_3_wheels/arc_agi-*.whl data/arc_agi_3_wheels/arcengine-*.whl
bash scripts/fetch_games.sh    # 可选：社区游戏 + 清单校验

# 跑调参游戏（Dots，滚动上下文），每局 25 分钟
python -m arc_agent.run --out-dir runs/x1 --games cd82,tu93,wa30 --conc 3 \
  --base-url https://note3-prev-api.askdiandian.com/v1 --model dots3-note-prev \
  --api-key-file .secrets/dots_api_key --game-seconds 1500 \
  --context rolling --compact-tokens 60000
python -m arc_agent.report runs/x1   # 汇总
```

其他常用参数：
- `--k`：每个游戏跑几局；
- `--think-policy always|model|exception`；
- `--call-seconds`；
- `--no-image`。

`scripts/dots_loop.sh` 会一轮接一轮地跑调参游戏；在 `runs/dots3/` 下建一个 `STOP` 文件就停。

## 4. Kaggle

- **账号：** xishengfeng。
- **notebook：** 用 `python kaggle/build_notebook.py --variant loop|api|dev|submit` 生成到 `build/<variant>/`，再用 `kaggle kernels push -p build/<variant>` 推送。
- **密钥：** 放在私有数据集 `xishengfeng/arc26-secrets` 里。
- **`arc26-agent-loop`（CPU，Dots 调参循环）：**
  - 每轮 `git clone` `main` 分支的最新代码（分支名在 `kaggle/config.json` 的 `loop.branch`）。notebook 在构建时就定下了拉哪个分支，所以改了分支名要重新构建并推送 notebook 才生效。09-30 在跑的两版是改成 main 之前推的，仍然拉 `claude/arc-prize-local-setup-u7haly`。
  - 读同一文件里 `loop` 段的设置：每局 4 小时，每个游戏 2 局。
  - 在日志里打印每轮的报告。平均分达到 30 以后，自动把 25 个公开游戏各跑一遍。
- **正在运行（09-30）：**
  - v1：第 3 轮，旧代码 31b4bad，约 09:00 结束。
  - v2：缓存修复后的代码，第 0 轮约 10:05 结束。
  - 之后两版的新一轮都会拉到最新代码，包括模拟器。
- **其他 notebook：**
  - `arc26-agent-api`：迭代集；
  - `arc26-agent-val-api`：验证集；
  - `arc26-agent-dev`：比赛 GPU、Qwen3.8、vLLM。
- **看日志：**
  - 运行中：`KaggleApi().kernels_logs_stream(...)`；
  - 跑完后：`kaggle kernels output <kernel> -p <dir>`。

## 5. 已知问题与下一步

1. **模拟器刚接进去，需要长局数据。**
   - 本地 s5 测试（cd82）里，模型从第 3 步起写了模拟器，前几步就改到了 v3，回放在 3 个动作上已经完全正确。
   - 还不知道能不能到"连续 3 步正确 → 自动搜路线 → 走路线过关"这一步。
   - 要看：`sim_checks` 趋势、有没有路线被执行、过关数的变化。
2. **cd82 理解差：** 这是个"选色、涂色、盖章"类游戏，旧代码的预测几乎全错。模拟器的核对报告应该能帮上忙。
3. **速度：** 难的一步要想 1–2.5 分钟。`think_next` / `--think-policy` 的取舍还没调好。
4. **比赛时间预算：** 最终要压进约 9 小时、110 个游戏、本地 27B 模型。
   - 模拟器加搜索的方向本来就能减少 LLM 调用。
   - 本地 GPU 的前缀缓存只能在 align 模式下按 1568 token 一整块命中，目前命中率约 11%。
5. **没有单元测试目录：** 模拟器的各条路径（打回、拒收、入图、搜索、执行路线）目前用假的 LLM 离线验证过，写法见提交 d03ed52 的说明。可以补成 `tests/`。

## 6. 注意

- **密钥：** Kaggle token 和 Dots key 只放在被 gitignore 的文件里，不要提交，也不要打印。
- **注意 Kaggle 上的其他 notebook：** 账号里还有和本项目无关的 notebook，别碰。
- **旧方案：** 旧的 MiniMax Code + MCP 方案已从仓库删除，需要时看历史提交 `6ea847c`。
