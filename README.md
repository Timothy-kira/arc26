# arc26 — ARC Prize 2026 (ARC-AGI-3) with Qwen3.8-27B

An offline agent for the Kaggle `arc-prize-2026-arc-agi-3` competition. Qwen3.8-27B (served by vLLM on the RTX 6000) is driven by a lean harness that re-implements three mechanisms studied in EverMind's Raven — **DAG scheduling**, **automatic skill sedimentation** and **self-evolution** — locally, with no cloud memory or skill hub. See [`docs/raven_mechanisms.md`](docs/raven_mechanisms.md) for the source-level notes.

## How it plays
```
explorer burst (no LLM, ms/step)  ──event──►  reasoning round = one DAG
  state graph over HUD-masked frame hashes      perceive ─► {objects, diff, image analysts} ─► plan ─► act(env, serialized)
  BFS to the nearest untried (state, action)    judge: no progress ─► replan once (successor graph)
```
* **Events**: level start, GAME_OVER, no new state for a while, burst exhausted.
* **Memory**: per-game notebook (the compaction brief); a *case* per level with objective quality (win × efficiency);
  cases cluster by observable features; skills (`SKILL.md`) are added/updated from cases, retrieved by BM25 + RRF + an LLM gate, and their confidence is updated from outcomes (retired when they keep failing).
* **Charter**: per-level action priorities / bans / stop threshold, refined by every plan.
* **Budget**: ~110 hidden games in ~9 h — games run concurrently and share one LLM semaphore; per-game deadlines adapt to the time left.
* **Fallback**: if vLLM is not healthy the same run degrades to the pure explorer.

## Self-evolution (`evolve/`)
Offline greedy hill-climbing on the 25 public games (17 train / 8 sealed test): diagnose failures into WHY × WHERE,
design candidates with a bounded file editor over whitelisted paths (`prompts/`, `arc_harness/roles/`, `charter.py`,
`explore/config.py`, `skills/`), prune for free, screen on an anchor subset, confirm at K=3, paired gate, strict promotion,
patience stop, sealed test opened once. Candidates are overlays (no git needed), state is resumable across Kaggle sessions.

## Workflow
```bash
make setup          # venv, competition data, offline engine wheels
make test           # unit + smoke tests (CPU)
make baseline       # pure explorer on the 25 public games
make mock           # full pipeline with a scripted LLM
make push-dev       # Kaggle RTX 6000: Qwen on all public games  -> make pull-dev
make push-evolve    # Kaggle RTX 6000: one evolver session       -> make pull-evolve, apply evolved.patch
make push-submit    # submission kernel, then "Submit to Competition" on the kernel page
```
Model and vLLM wheelhouse come from public Kaggle datasets configured in `kaggle/config.json`.
