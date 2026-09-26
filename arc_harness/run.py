"""Entry point: play every game concurrently with one shared LLM and memory.

Local:   python -m arc_harness.run --mode offline --games ls20,ft09
Kaggle:  python -m arc_harness.run --mode competition   (gateway at http://gateway:8001)
If the LLM endpoint is unhealthy the run degrades to the pure explorer instead of failing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .agent import AgentConfig, GameAgent
from .budget import GlobalBudget
from .dag.runner import InstanceLocks
from .env.arcade import ArcEnv
from .llm.client import LLMConfig, OpenAICompatClient
from .llm.mock import MockLLM, schema_default_handler
from .memory.hub import MemoryHub
from .scoring import summarize_scorecard

logger = logging.getLogger("arc26")


async def run(args: argparse.Namespace) -> dict:
    out = Path(args.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    env = ArcEnv(args.mode, args.env_dir)
    games = env.list_games()
    if args.games != "all":
        prefixes = args.games.split(",")
        games = [g for g in games if any(g.startswith(p) for p in prefixes)]
    env.open(tags=["arc26", args.tag])

    llm = None
    if args.llm == "mock":
        llm = MockLLM(schema_default_handler)
    elif args.llm == "vllm":
        cfg = LLMConfig()
        if args.llm_base_url:
            cfg.base_url = args.llm_base_url
        client = OpenAICompatClient(cfg)
        deadline = time.monotonic() + args.llm_wait
        while not await client.healthy():
            if time.monotonic() > deadline:
                logger.error("LLM endpoint %s unhealthy; falling back to the pure explorer", cfg.base_url)
                client = None
                break
            await asyncio.sleep(5)
        llm = client

    memory = MemoryHub(out / "memory", llm, [Path(p) for p in args.skills_dir.split(",") if p])
    total = args.hours * 3600
    budget = GlobalBudget(total, len(games), args.concurrency, reserve_seconds=min(args.reserve, 0.1 * total))
    agent_cfg = AgentConfig(max_actions=args.max_actions)
    if args.no_image:
        agent_cfg.hypothesizers = tuple(h for h in agent_cfg.hypothesizers if h != "image")
    locks = InstanceLocks()
    sem = asyncio.Semaphore(args.concurrency)
    reports = []

    async def one(gid: str) -> None:
        async with sem:
            try:
                session = await asyncio.to_thread(env.make, gid, args.seed)
            except Exception as exc:
                logger.error("make %s failed: %s", gid, exc)
                return
            agent = GameAgent(session, llm, memory, budget.for_game(), agent_cfg, out / "games" / gid, locks)
            rep = await agent.play()
            budget.done()
            reports.append(asdict(rep))
            logger.info(
                "%s done: levels %d/%d actions %d rounds %d %.0fs %s",
                gid, rep.levels_completed, rep.win_levels, rep.actions, rep.llm_rounds, rep.seconds, rep.error,
            )

    t0 = time.time()
    await asyncio.gather(*(one(g) for g in games))
    card = env.scorecard()
    summary = summarize_scorecard(card)
    env.close()
    summary["reports"] = reports
    summary["wall_seconds"] = time.time() - t0
    summary["llm"] = llm.stats.as_dict() if llm is not None else None
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if hasattr(llm, "aclose"):
        await llm.aclose()  # type: ignore[union-attr]
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="arc26 ARC-AGI-3 agent")
    ap.add_argument("--mode", default="auto", choices=["auto", "offline", "competition"])
    ap.add_argument("--env-dir", default=None)
    ap.add_argument("--games", default="all")
    ap.add_argument("--llm", default="vllm", choices=["vllm", "mock", "none"])
    ap.add_argument("--llm-base-url", default=None)
    ap.add_argument("--llm-wait", type=float, default=float(os.getenv("ARC26_LLM_WAIT", "60")))
    ap.add_argument("--concurrency", type=int, default=int(os.getenv("ARC26_GAME_CONCURRENCY", "12")))
    ap.add_argument("--hours", type=float, default=float(os.getenv("ARC26_HOURS", "8.5")))
    ap.add_argument("--reserve", type=float, default=900.0)
    ap.add_argument("--max-actions", type=int, default=2500)
    ap.add_argument("--skills-dir", default=str(Path(__file__).resolve().parents[1] / "skills"))
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    return ap


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("arc_agi", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    summary = asyncio.run(run(args))
    print(json.dumps(summary["totals"], indent=2))


if __name__ == "__main__":
    main()
