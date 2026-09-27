# arc26：ARC Prize 2026（ARC-AGI-3），MiniMax Code 加实时 Python REPL

智能体是**不做修改的 MiniMax Code**（ACP 模式加 Goal 模式）。每局游戏都当作一个交互式编程题来解：游戏状态放在一个常驻 Python REPL 的变量里，模型写代码、运行、执行动作、观察结果。每个格子都记成探索 DAG 里的一个节点。

- **比赛提交：** Qwen3.8-27B 在 Kaggle RTX PRO 6000 上用 vLLM 离线运行。
- **日常迭代：** 用托管 API（Dots）。

| 目录 | 内容 |
|---|---|
| `arc_mcp/` | 游戏守护进程和账本（`server.py`、`session.py`、`ledger.py`）、REPL kernel（`kernel.py`）、感知（`percept.py`：4 倍图、物体、变化区域、位移、格子地图、计量条） |
| `plugin/arc26/` | MiniMax Code 原生插件：SessionStart 注入状态和笔记，PreToolUse 只允许用 arc 工具，PostToolUse 把工具调用记进 DAG |
| `arc_runner/` | `batch.py`：每局一个 MiniMax Code 会话，负责沙箱、kernel 守护、技能隔离。`acp_driver.py`：Goal 模式驱动。`llm_proxy.py`：托管 API 代理，控制思考开关、去掉输出上限 |
| `arc_eval/` | 游戏集与切分（`datasets.py`）、RHAE 计算、`summarize.py`（成绩和功能使用统计） |
| `kaggle/` | `build_notebook.py`：submit（比赛提交）、dev（GPU 自测）、api（托管 API）三种 notebook |
| `games/manifest.json` | 我们运行的全部游戏：切分、人类基线、sha256 |
| `scripts/` | `fetch_games.sh` 下载并校验游戏；`games_manifest.py` 生成或校验清单 |
| `docs/` | [MCP 与 REPL 工具参考](docs/MCP_TOOLS.md)、[游戏清单](docs/GAMES.md)、[实验记录](docs/RESULTS.md) |

## 快速开始

```bash
# 1. 游戏：官方 25 个来自 Kaggle 比赛数据（需要 .kaggle/access_token），社区 249 个（已排除官方游戏副本）来自 arc-interactive
scripts/fetch_games.sh

# 2. 用托管 API 跑 3 个迭代游戏：本地代理加沙箱用户 arcagent
python arc_runner/llm_proxy.py --upstream https://<host>/v1 --api-key-file .secrets/api_key --port 8012 \
    --inject '{"chat_template_kwargs": {"enable_thinking": true}}' --drop max_tokens,max_completion_tokens &
python arc_runner/batch.py --out-dir runs/demo --games ls20,vc33,tu93 --conc 2 --max-game-seconds 2700 \
    --base-url http://127.0.0.1:8012/v1 --model <model> --reasoning --context 131072 --output-limit 65536 \
    --agent-user arcagent --node <node22> --mcode <mcode cli.js>
python -m arc_eval.summarize runs/demo

# 3. Kaggle
python kaggle/build_notebook.py --variant api     # 或 dev / submit；然后 kaggle kernels push -p build/<variant>
```

密钥只放在被 git 忽略的 `.secrets/` 和 `.kaggle/` 里，不要提交。
