# arc26 — ARC Prize 2026 (ARC-AGI-3) with Qwen3.8-27B on EverMind Raven

The agent is **EverMind Raven itself** (fork: `timothy-kira/Raven`, branch `arc3`), driving Qwen3.8-27B served
offline by vLLM on the Kaggle RTX PRO 6000. The only ARC-specific code is Raven's own benchmark-integration
pattern (`benchmarks/arc3/`, mirroring `benchmarks/appworld/`): game tools, a runner, and the evolver bench bundle.
Memory/skill sedimentation (EverOS plugin, run locally) and self-evolution (`evolver/`) are Raven's.

This repo keeps only the Kaggle side: competition data download, offline packaging and the notebook builder
(being ported from the previous hand-written harness to Raven), plus `docs/raven_mechanisms.md`.

```bash
make data    # download competition data (wheels, 25 public games) into data/
```
