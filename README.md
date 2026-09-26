# arc26 — ARC Prize 2026 (ARC-AGI-3) with MiniMax Code + Qwen3.8-27B

The agent is **MiniMax Code** (`mcode exec`, unmodified; github.com/Timothy-kira/minimax-code) driving
Qwen3.8-27B served offline by vLLM on the Kaggle RTX PRO 6000. The game is exposed to it as an MCP server.

- `arc_mcp/server.py` — the ARC-AGI-3 game as a dependency-free stdio MCP server (`arc_observe`, `arc_act`);
  plays public games offline or joins the competition gateway's scorecard.
- `arc_runner/batch.py` — one headless `mcode exec` per game (own data dir, `AGENTS.md` instructions,
  a shared skills directory the agent writes `SKILL.md` files into for later games, `--continue` while unfinished).
- `kaggle/build_notebook.py` — the Kaggle notebook (submit / dev variants); MiniMax Code + Node 22 come from the
  `xishengfeng/mcode-offline` dataset, vLLM from `driessmit1/arc3-vllm-h100-wheelhouse-v3`.

```bash
make data                                           # competition data into data/
python arc_runner/batch.py --out-dir runs/dev --games ls20 --base-url http://127.0.0.1:8000/v1
python kaggle/build_notebook.py --variant dev       # then: kaggle kernels push -p build/dev --accelerator NvidiaRtxPro6000
```
