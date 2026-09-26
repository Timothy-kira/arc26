"""Play many ARC-AGI-3 games with MiniMax Code (``mcode exec``) driving a local model.

Per attempt a workspace is prepared with:
  .mcp.json        the ARC game MCP server (``arc_mcp/server.py``) for this game
  AGENTS.md        how to play (MiniMax Code injects it into the system prompt)
  .minimax/skills  -> the run's shared skills directory (cross-game learning: the agent is
                      asked to write what it learned as SKILL.md files that later games load)
and ``mcode exec`` runs there headless (``--permission full``), continuing the same session
while the game is unfinished and time remains. Each attempt has its own MINIMAX_DATA_DIR so
concurrent processes never share a session database.

Offline mode plays the public games; gateway mode (Kaggle rerun) opens the single scorecard
the gateway allows, hands its id to every game, and closes it at the end.

Standard library only (runs on Kaggle's python); the MCP server needs arc_agi + numpy + pillow.

Usage::

    python arc_runner/batch.py --out-dir runs/dev --env-dir data/environment_files \
        --base-url http://127.0.0.1:8000/v1 --model qwen3.8-27b --games all --conc 12
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "arc_mcp" / "server.py"
PRELOAD = ROOT / "arc_runner" / "no_fetch_timeouts.cjs"

AGENTS_MD = """# Playing an ARC-AGI-3 game

You are playing an unknown turn-based puzzle game on a 64x64 grid of 16 colours (0-f), through the
`arc` MCP tools. Nobody tells you the rules or the goal: discover them by acting and watching what
changes, then win every level.

- `arc_observe`: the current frame as hex rows (row number first; x = column, y = row; "NN-MM" = identical rows). Free, but
  `arc_act` already shows the new cells of small changes, so observe only when you need the whole frame.
- `arc_act`: send up to 20 actions: 0=RESET (restart the level; needed after GAME_OVER), 1=up, 2=down,
  3=left, 4=right, 5=interact, 6=click at (x, y), 7=undo. Only the listed available actions do anything.

Scoring: each level scores (human_actions / your_actions)^2, so every wasted action hurts.
There is no time limit you need to track and no other action budget than `actions_used/N` in the
status line: the runner stops you when time is up. Never stop on your own while the game is unfinished.
`levels_completed` rising means you won a level. GAME_OVER means the attempt failed: RESET.

Work like a scientist: hypothesise the goal and the mechanics, run the cheapest action that tests a
hypothesis, and never repeat an action that was useless in the same situation. Keep a short notes file
(`notes.md`) of confirmed rules so you do not lose them when the conversation is compacted.

Skills from earlier games are in `.minimax/skills/` - read the ones whose description matches what you
see before you start. When the game ends (won, or told to stop), write or update ONE skill there:
`.minimax/skills/<short-kebab-name>/SKILL.md` with frontmatter `name` and `description` (the observable
cues: available actions, layout, colours) and a short numbered procedure that would solve that kind of
game faster. Never mention game ids. Do not use the shell or other files for anything else.
"""

FIRST_PROMPT = "Play the ARC game. Start with arc_observe. Keep playing until the game is won or the tools tell you to stop."
NEXT_PROMPT = "Continue playing the same game from where you are (check notes.md and arc_observe). Stop only when the game is won or the tools tell you to stop."


def write_mcode_config(data_dir: Path, base_url: str, model: str, context: int, output: int, reasoning: bool = False,
                       api_key: str = "EMPTY") -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = f"""custom_provider:
  vllm:
    api: openai-completions
    options: {{ apiKey: "{api_key}", baseURL: "{base_url}", timeout: 900000 }}
    models:
      {model}:
        tool_call: true
        reasoning: {str(reasoning).lower()}
        limit: {{ context: {context}, output: {output} }}
        compat: {{ thinkingFormat: qwen-chat-template, supportsDeveloperRole: false, maxTokensField: max_tokens }}
defaultModel: custom_provider:vllm/{model}
permissionMode: bypassPermissions
telemetry: {{ enabled: false, metrics: false, diagnostics: false }}
agents:
  default:
    features: {{ webSearch: false, mavis: false }}
"""
    p = data_dir / "config.yaml"
    p.write_text(cfg)
    os.chmod(p, 0o600)


def open_scorecard(args: argparse.Namespace) -> tuple[Any, Optional[str], list[str]]:
    """Game ids, plus (gateway mode) the one scorecard every game joins."""
    import logging

    from arc_agi import Arcade, OperationMode

    quiet = logging.getLogger("arc.engine")
    if args.gateway:
        arc = Arcade(arc_api_key="test-key-123", arc_base_url=args.gateway.rstrip("/"),
                     operation_mode=OperationMode.ONLINE, logger=quiet)
        card = arc.open_scorecard(tags=["arc26-mcode"])
    else:
        arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=args.env_dir, logger=quiet)
        card = None
    return arc, card, sorted(e.game_id for e in arc.get_environments())


def read_json(p: Path) -> dict[str, Any]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def runtime_errors(data_dir: Path, n: int = 4) -> str:
    """The last ERROR/WARN lines of MiniMax Code's runtime log (why a model call failed)."""
    lines: list[str] = []
    for log in sorted((data_dir / "v2" / "observability" / "logs").glob("runtime-*.log")):
        try:
            lines += [l for l in log.read_text(errors="replace").splitlines() if "ERROR" in l or "WARN" in l]
        except OSError:
            pass
    return " | ".join(l[:300] for l in lines[-n:])


def run_attempt(game: str, k: int, args: argparse.Namespace, card: Optional[str], time_limit: float) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    final = out_dir / f"{game}_k{k}.json"
    if final.exists() and read_json(final):
        return read_json(final)
    ws = out_dir / "ws" / f"{game}_k{k}"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / ".minimax").mkdir(exist_ok=True)
    skills_link = ws / ".minimax" / "skills"
    if not skills_link.exists():
        skills_link.symlink_to(Path(args.skills_dir).resolve(), target_is_directory=True)
    (ws / "AGENTS.md").write_text(AGENTS_MD)
    result = ws / "result.json"
    env = {
        "ARC_GAME": game,
        "ARC_ENV_DIR": str(Path(args.env_dir).resolve()),
        "ARC_RESULT": str(result),
        "ARC_MAX_ACTIONS": str(args.max_actions),
    }
    if args.gateway:
        env.update(ARC_GATEWAY=args.gateway, ARC_CARD_ID=card or "")
    # One game daemon per attempt: every mcode exec round (and any MCP server restart) plays the
    # same game instead of starting a new one. Unix socket paths must stay short.
    sock = Path("/tmp") / f"arc26-{os.getpid()}-{game[:12]}-{k}.sock"
    sock.unlink(missing_ok=True)
    daemon = subprocess.Popen([args.server_python, str(SERVER), "--daemon", str(sock)], env={**os.environ, **env},
                              stdout=subprocess.DEVNULL, stderr=(ws / "daemon.stderr").open("w"))
    for _ in range(600):
        if sock.exists() or daemon.poll() is not None:
            break
        time.sleep(0.1)
    if not sock.exists():
        daemon.kill()
        res = {"game": game, "k": k, "rounds": 0, "infra_error": "game daemon failed: " + (ws / "daemon.stderr").read_text()[-500:]}
        final.write_text(json.dumps(res, indent=1))
        return res
    (ws / ".mcp.json").write_text(json.dumps(
        {"mcpServers": {"arc": {"command": args.server_python, "args": [str(SERVER)], "env": {**env, "ARC_SOCKET": str(sock)},
                                "timeout": 120000}}}, indent=1))
    data_dir = ws / ".mcode-data"
    write_mcode_config(data_dir, args.base_url, args.model, args.context, args.output_limit, args.reasoning, args.api_key)
    # The model server is local: no proxy (its dispatcher has 300 s timeouts), and no fetch timeouts at all.
    penv = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
    penv.update(MINIMAX_DATA_DIR=str(data_dir), MCODE_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
                NODE_OPTIONS=(penv.get("NODE_OPTIONS", "") + f" --require {PRELOAD}").strip())
    deadline = time.monotonic() + time_limit
    rounds, idle, log = 0, 0, []
    while rounds < args.max_rounds:
        left = deadline - time.monotonic()
        if left < 60:
            break
        state = read_json(result)
        if state.get("state") == "WIN" or int(state.get("actions") or 0) >= args.max_actions:
            break
        cmd = [args.node, args.mcode, "exec", FIRST_PROMPT if rounds == 0 else NEXT_PROMPT, "--cwd", str(ws),
               "--permission", "full", "--max-steps", str(args.max_steps), "--timeout", f"{int(left)}s",
               "--output-format", "json"]
        if rounds > 0:
            cmd.append("--continue")
        before = int(state.get("actions") or 0)
        with (ws / f"exec_{rounds}.stderr").open("w") as err:
            try:
                r = subprocess.run(cmd, cwd=ws, env=penv, stdout=subprocess.PIPE, stderr=err, text=True, timeout=left + 120)
                out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
                log.append({"round": rounds, "exit": r.returncode, "result": out[-2000:]})
                tail = (ws / f"exec_{rounds}.stderr").read_text()[-400:].replace("\n", " | ") if r.returncode else ""
                if r.returncode:
                    tail += " || " + runtime_errors(data_dir)
                acted = int(read_json(result).get("actions") or 0) - before
                sys.stderr.write(f"[exec] {game} round {rounds} exit={r.returncode} actions+={acted} {out[-500:]} {tail}\n")
            except subprocess.TimeoutExpired:
                log.append({"round": rounds, "exit": "timeout"})
        rounds += 1
        idle = idle + 1 if int(read_json(result).get("actions") or 0) == before else 0
        if idle >= 2 and rounds > 2:
            break  # two rounds in a row without an action: the agent stopped acting
    daemon.terminate()
    try:
        daemon.wait(10)
    except subprocess.TimeoutExpired:
        daemon.kill()
    sock.unlink(missing_ok=True)
    res = {"game": game, "k": k, "rounds": rounds, **read_json(result), "exec": log}
    res["infra_error"] = None if result.exists() else "no game progress recorded"
    final.write_text(json.dumps(res, indent=1))
    return res


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="arc26-batch")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--games", default="all")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--conc", type=int, default=8)
    p.add_argument("--env-dir", default=str(ROOT / "data" / "environment_files"))
    p.add_argument("--gateway", default=None)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key-file", default="", help="file holding the API key of a hosted endpoint")
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--context", type=int, default=262144)
    p.add_argument("--output-limit", type=int, default=8192)
    p.add_argument("--reasoning", action="store_true",
                   help="let the model think before each step (off: long reasoning exhausted the output budget)")
    p.add_argument("--hours", type=float, default=8.0)
    p.add_argument("--max-game-seconds", type=float, default=3600.0)
    p.add_argument("--max-actions", type=int, default=100000)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--max-rounds", type=int, default=100)
    p.add_argument("--skills-dir", default=None, help="shared skills directory (default <out-dir>/skills)")
    p.add_argument("--node", default=shutil.which("node") or "node")
    p.add_argument("--mcode", default=str(ROOT.parent / "minimax-code" / "dist" / "cli.js"))
    p.add_argument("--server-python", default=sys.executable)
    args = p.parse_args(argv)
    args.api_key = Path(args.api_key_file).read_text().strip() if args.api_key_file else "EMPTY"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.skills_dir = args.skills_dir or str(out_dir / "skills")
    Path(args.skills_dir).mkdir(parents=True, exist_ok=True)
    arc, card, all_games = open_scorecard(args)
    wanted = all_games if args.games == "all" else args.games.split(",")
    games = [g for g in all_games if any(g == w or g.startswith(w) for w in wanted)]
    jobs = [(g, k) for k in range(args.k) for g in games]

    t_end = time.monotonic() + args.hours * 3600
    lock = threading.Lock()
    left = [len(jobs)]
    results: list[dict[str, Any]] = []

    def worker(job: tuple[str, int]) -> None:
        with lock:
            waves = max(1.0, left[0] / args.conc)
            share = max(120.0, min(args.max_game_seconds, (t_end - time.monotonic()) / waves))
            left[0] -= 1
        try:
            r = run_attempt(job[0], job[1], args, card, share)
        except Exception as exc:  # one broken game must not stop the batch
            r = {"game": job[0], "k": job[1], "infra_error": f"{type(exc).__name__}: {exc}"}
        with lock:
            results.append(r)
        sys.stderr.write(f"[batch] {job[0]} k{job[1]} levels={r.get('levels_completed')}/{r.get('win_levels')} "
                         f"actions={r.get('actions')} score={r.get('score')} rounds={r.get('rounds')} infra={r.get('infra_error')}\n")

    try:
        with ThreadPoolExecutor(max_workers=args.conc) as pool:
            list(pool.map(worker, jobs))
    finally:
        if args.gateway and card:
            for _ in range(3):  # closing finalizes the submission on the gateway
                try:
                    arc.close_scorecard(card)
                    break
                except Exception as exc:
                    sys.stderr.write(f"close_scorecard failed: {exc}\n")
    ok = [r for r in results if not r.get("infra_error")]
    summary = {
        "games": len(games),
        "attempts": len(results),
        "infra": len(results) - len(ok),
        "levels_completed": sum(int(r.get("levels_completed") or 0) for r in ok),
        "levels_total": sum(int(r.get("win_levels") or 0) for r in ok),
        "mean_score": (sum(float(r.get("score") or 0) for r in ok) / len(ok)) if ok else 0.0,
        "actions": sum(int(r.get("actions") or 0) for r in ok),
        "skills": sorted(p.parent.name for p in Path(args.skills_dir).glob("*/SKILL.md")),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    sys.stderr.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
