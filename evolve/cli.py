"""``python -m evolve {run,status,unseal,export}``."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path

from arc_harness.llm.client import LLMConfig, OpenAICompatClient

from .bench import SubprocessEvaluator
from .loop import Evolver
from .sealed import unseal
from .state import RunSpec, RunState
from .workspace import overlay_diff


def split_games(games: list[str], n_test: int, seed: int = 26) -> tuple[list[str], list[str]]:
    rnd = random.Random(seed)
    g = sorted(games)
    rnd.shuffle(g)
    return sorted(g[n_test:]), sorted(g[:n_test])


def cmd_init(args: argparse.Namespace) -> None:
    from arc_harness.env.arcade import ArcEnv

    games = [g.split("-")[0] for g in ArcEnv("offline").list_games()]
    train, test = split_games(games, args.n_test)
    spec = RunSpec(work_dir=args.work_dir, repo_dir=args.repo, train_games=train, test_games=test)
    if args.smoke:
        spec.train_games, spec.test_games = train[:2], test[:1]
        spec.k_confirm, spec.anchor_size, spec.max_rounds, spec.candidates_per_why, spec.max_why_per_round = 1, 2, 1, 1, 1
        spec.run_args = ["--max-actions", "200", "--hours", "0.1", "--concurrency", "2"]
    Path(args.config).write_text(json.dumps(spec.__dict__, indent=2))
    print(f"wrote {args.config}: train={spec.train_games} test={spec.test_games}")


def cmd_run(args: argparse.Namespace) -> None:
    spec = RunSpec.load(Path(args.config))
    llm = OpenAICompatClient(LLMConfig())
    ev = Evolver(spec, SubprocessEvaluator(spec), llm, spec.train_games + spec.test_games)
    best = asyncio.run(ev.run())
    print(json.dumps({"parent": best.id, "train_mean": best.mean(spec.train_games), "lineage": [n.id for n in ev.lineage()]}, indent=2))


def cmd_status(args: argparse.Namespace) -> None:
    spec = RunSpec.load(Path(args.config))
    st = RunState(spec)
    for n in sorted(st.nodes.values(), key=lambda n: n.created):
        print(f"{n.id} parent={n.parent_id} {n.status:9s} why={n.why:28s} train={n.mean(spec.train_games) if n.scores else float('nan'):.3f} {n.reason[:80]}")
    print(json.dumps({k: v for k, v in st.meta.items() if k != "history"}, indent=1))


def cmd_unseal(args: argparse.Namespace) -> None:
    spec = RunSpec.load(Path(args.config))
    print(json.dumps(unseal(RunState(spec), SubprocessEvaluator(spec)), indent=2))


def cmd_export(args: argparse.Namespace) -> None:
    spec = RunSpec.load(Path(args.config))
    st = RunState(spec)
    node = st.nodes[args.node] if args.node else st.parent
    assert node is not None
    patch = overlay_diff(Path(spec.repo_dir), node.overlay)
    Path(args.out).write_text(patch)
    print(f"wrote {args.out} ({len(patch)} bytes) for {node.id}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="evolve")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("--config", default="evolve_run.json")
    p.add_argument("--work-dir", default="runs/evolve")
    p.add_argument("--repo", default=".")
    p.add_argument("--n-test", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    for name in ("run", "status", "unseal", "export"):
        q = sub.add_parser(name)
        q.add_argument("--config", default="evolve_run.json")
        if name == "export":
            q.add_argument("--node", default="")
            q.add_argument("--out", default="evolved.patch")
    args = ap.parse_args()
    {"init": cmd_init, "run": cmd_run, "status": cmd_status, "unseal": cmd_unseal, "export": cmd_export}[args.cmd](args)


if __name__ == "__main__":
    main()
