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
PLUGIN = ROOT / "plugin" / "arc26"

AGENTS_MD = """# Playing an ARC-AGI-3 game as an interactive programming problem

You play an unknown turn-based puzzle game (64x64 grid, 16 colours). Nobody tells you the rules or
the goal. The game lives in a Python REPL that you drive with the `arc_python` tool: the state is in
variables (`grid`, `prev`, `frames`, `state`, `level`, `available`, `history`, ...) and `act(a, x, y)`
plays one action. Score per level = (human_actions / your_actions)^2: every action counts, thinking and
code that does not act are free.

Method (a scientist with a programmable lab):
1. Look first, for free: after every cell that plays actions you get a PERCEPTION block (a 4x image of
   the frame, the objects grouped by colour and size, the diff between animation frames); `look()`
   attaches an image without acting, `objects()` / `anim()` / `show()` give the raw data.
2. Ask one question per cell and spend as few actions as that question needs (e.g. "what does
   ACTION1 do?" = one act() and a look at changes()). Give every cell a `purpose`, `parents` = the
   cells it builds on, `expect` = what you think will happen and, when you can, `check` = a Python
   expression that decides it (e.g. "outcome['levels'] > 0", "grid[y, x] == 12"). Every cell is judged
   RIGHT or WRONG in real time and kept in this game's JOURNAL (shown in every reply; `journal` and
   `dag()` in the REPL). Learn from it: never repeat a WRONG idea unchanged; fix it in a cell with
   `revises` = that node's id, and build on what was RIGHT.
3. Turn what you learn into code: helpers that find the avatar, list objects, simulate a move, run BFS
   to a target. Keep them in the REPL and reuse them; later levels usually share the rules and only
   change the layout, so a working solver from level 1 often solves level 2 with few actions.
4. Plan with the `todowrite` tool: keep a short task list for the current level (what to test, what
   to build, what to play), exactly one item in progress, and update it as results come in. Compute
   paths or click sequences in code first, then play them.
5. After GAME_OVER call reset() (action 0) and change what killed you.
6. Novelty first: every distinct kind of object on screen is a candidate interaction (a key, a switch,
   a door, a paint pot). Before concluding how to win, touch or click each kind once, cheapest first,
   and record what it did. Never write "unreachable", "static" or "useless" about an object you have
   not actually tried to reach or click.
7. Do not re-test what the notebook already marks as confirmed: build on it. Re-probing known rules
   wastes actions that count against the level.
8. Keep the notebook up to date with `arc_note` (rules, goal, levels, plan): the server hands it back
   verbatim after every restart or context compaction, so it is your reliable memory.

The REPL cannot read files or start processes; do not use the shell or other tools to look at game
files. Skills in `.minimax/skills/` come from OTHER games played earlier in this run: read the ones whose
description matches what you see. When the game ends (won, or told to stop), write or update ONE skill
there: `.minimax/skills/<short-kebab-name>/SKILL.md` with frontmatter `name` and `description` (the
observable cues) and a short procedure plus reusable helper code. Never mention game ids.
You work in goal mode: the session keeps going until the game is won, so never end a turn with a plan
in words only: every turn ends with a tool call (think, then call arc_python). Every reply shows the budget (actions this level, total,
time left): spend actions only when a cell has a clear question or a computed plan.
"""

GOAL_OBJECTIVE = ("Win every level of the ARC game in this workspace, using as few game actions as possible. Follow "
                  "AGENTS.md. At the start of every level write a short todowrite plan; play only through the arc_python "
                  "REPL and give every cell an `expect`; when a result contradicts your expectation, fix it in a cell with "
                  "`revises`; after every completed level record the rule that won it with arc_note. The goal is complete "
                  "only when the tools say the game is won.")
FIRST_PROMPT = "Play the ARC game. Start by looking at the state in the REPL with arc_python (looking costs no actions). Keep playing until the game is won or the tools tell you to stop."


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
        modalities: {{ input: [text, image], output: [text] }}
        reasoning: {str(reasoning).lower()}
        limit: {{ context: {context}, output: {output} }}
        compat: {{ thinkingFormat: qwen-chat-template, supportsDeveloperRole: false, maxTokensField: max_tokens }}
defaultModel: custom_provider:vllm/{model}
permissionMode: bypassPermissions
beta: {{ threadGoal: true }}
goal:
  verification: none
  breaker: {{ repeatedReplyLimit: 3 }}
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


def sandbox(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """Command prefix and env to run the agent CLI as ``--agent-user`` (no access to game files)."""
    if not args.agent_user:
        return [], {}
    import pwd

    pw = pwd.getpwnam(args.agent_user)
    return ([shutil.which("setpriv") or "setpriv", f"--reuid={pw.pw_uid}", f"--regid={pw.pw_gid}", "--clear-groups", "--"],
            {"HOME": pw.pw_dir, "USER": args.agent_user})


def hand_over(path: Path, args: argparse.Namespace) -> None:
    """Give the agent user ownership of a tree (its workspace, the shared skills)."""
    if not args.agent_user:
        return
    import pwd

    pw = pwd.getpwnam(args.agent_user)
    for root, dirs, files in os.walk(path):
        for n in [root] + [os.path.join(root, d) for d in dirs] + [os.path.join(root, f) for f in files]:
            try:
                os.lchown(n, pw.pw_uid, pw.pw_gid)
            except OSError:
                pass


def game_key(game: str) -> str:
    """Versions of one game share a key (``ls20-9607627b`` -> ``ls20``)."""
    return game.split("-")[0]


def prepare_skills(ws: Path, shared: Path, key: str) -> None:
    """The workspace sees the run's shared skills except those that came from this game (a game must
    never read what it wrote itself: that would leak its own solution into its evaluation)."""
    view = ws / ".minimax" / "skills"
    if view.is_symlink():
        view.unlink()
    view.mkdir(parents=True, exist_ok=True)
    for d in sorted(shared.glob("*/SKILL.md")):
        origins = (d.parent / ".origin").read_text().split() if (d.parent / ".origin").exists() else []
        link = view / d.parent.name
        if key not in origins and not link.exists():
            link.symlink_to(d.parent, target_is_directory=True)


def merge_skills(ws: Path, shared: Path, key: str, t_start: float) -> None:
    """New skills written in the workspace move into the shared library; edited shared skills record
    this game as an origin too, so later attempts at this game will not see them."""
    view = ws / ".minimax" / "skills"
    for d in sorted(view.iterdir()) if view.exists() else []:
        if d.is_symlink():
            target = d.resolve()
            if (target / "SKILL.md").exists() and (target / "SKILL.md").stat().st_mtime > t_start:
                o = target / ".origin"
                origins = set(o.read_text().split()) if o.exists() else set()
                o.write_text(" ".join(sorted(origins | {key})))
            continue
        if not (d / "SKILL.md").exists():
            continue
        dest, n = shared / d.name, 1
        while dest.exists():
            n += 1
            dest = shared / f"{d.name}-{n}"
        shutil.move(str(d), dest)
        (dest / ".origin").write_text(key)


def next_prompt(ws: Path, idle: int) -> str:
    """Continuation prompt with the notebook, status and recent DAG copied verbatim from disk, so
    nothing the agent recorded depends on how the conversation was compacted."""
    led = read_json(ws / "ledger.json")
    book = led.get("notebook") or {}
    notes = "\n".join(f"[{k}]\n{v}" for k, v in book.items() if v) or "(empty)"
    try:
        nodes = json.loads((ws / "dag.json").read_text())[-10:]
    except (OSError, ValueError):
        nodes = []
    dag = "\n".join(f"[{n['id']}]{'!' if n.get('flag') else ''} <- {n['parents']} {n['actions']}a {n['purpose'][:100]}"
                    + (f" | expected: {n['expect'][:80]}" if n.get("expect") else "") for n in nodes) or "(no cells yet)"
    try:
        jr = json.loads((ws / "dag_journal.json").read_text())
    except (OSError, ValueError):
        jr = []
    wrong = [j for j in jr if j.get("ok") is False][-5:]
    right = [j for j in jr if j.get("ok") is True][-4:]
    journal = "\n".join([f"WRONG n{j['node']} L{j['level']}: {j['expect'] or j['purpose']} -> {j['why']}" for j in wrong]
                        + [f"RIGHT n{j['node']} L{j['level']}: {j['expect'] or j['purpose']}" for j in right]) or "(empty)"
    dag += "\nJournal (what was right and wrong so far):\n" + journal
    nudge = "Your last turn ended without playing. Do not narrate: call arc_python now.\n" if idle else ""
    return (f"{nudge}Continue playing the same game from where you are. The REPL still holds your variables and "
            f"helper functions.\nStatus: {led.get('status', '?')}\nYour notebook (verbatim):\n{notes}\n"
            f"Your last REPL cells (arc_dag for more; node(i)['code'] to read one):\n{dag}\n"
            "Stop only when the game is won or the tools tell you to stop.")


def run_attempt(game: str, k: int, args: argparse.Namespace, card: Optional[str], time_limit: float) -> dict[str, Any]:
    out_dir = Path(args.out_dir).resolve()
    final = out_dir / f"{game}_k{k}.json"
    if final.exists() and read_json(final):
        return read_json(final)
    ws = out_dir / "ws" / f"{game}_k{k}"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / ".minimax").mkdir(exist_ok=True)
    t_attempt = time.time()
    prepare_skills(ws, Path(args.skills_dir).resolve(), game_key(game))
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
    env["ARC_DEADLINE"] = str(time.time() + time_limit)  # shown as "time left" in every tool reply
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
    # The REPL kernel: its own process, talks to the daemon, never sees the engine or game files.
    ksock = Path("/tmp") / f"arc26-{os.getpid()}-{game[:12]}-{k}.k.sock"
    ksock.unlink(missing_ok=True)
    blocked = [str(Path(args.env_dir).resolve()), str(ROOT / "data")]
    if Path("/kaggle/input").exists():  # competition files (game sources) and secrets on Kaggle
        blocked.append("/kaggle/input")
    kenv = {**os.environ, "ARC_DAG": str(ws / "dag.json"), "ARC_BLOCK_PATHS": os.pathsep.join(blocked),
            "PYTHONPATH": str(ROOT)}
    kernel = subprocess.Popen([args.server_python, "-m", "arc_mcp.kernel", str(ksock), str(sock)], env=kenv, cwd=str(ws),
                              stdout=subprocess.DEVNULL, stderr=(ws / "kernel.stderr").open("w"))
    for _ in range(300):
        if ksock.exists() or kernel.poll() is not None:
            break
        time.sleep(0.1)
    (ws / ".mcp.json").write_text(json.dumps(
        {"mcpServers": {"arc": {"command": args.server_python, "args": [str(SERVER)],
                                "env": {"ARC_SOCKET": str(sock), "ARC_KERNEL": str(ksock)}, "timeout": 300000}}}, indent=1))
    for p in (sock, ksock):
        if p.exists():
            os.chmod(p, 0o666)  # the agent CLI may run as another user
    data_dir = ws / ".mcode-data"
    write_mcode_config(data_dir, args.base_url, args.model, args.context, args.output_limit, args.reasoning, args.api_key)
    # The arc26 MiniMax Code plugin: hooks restore state after start/compaction, guard tools, log the
    # trajectory into the DAG and keep the session from stopping while the game is unfinished.
    plugin_dst = data_dir / "plugins" / "arc26"
    if not plugin_dst.exists():
        shutil.copytree(PLUGIN, plugin_dst)
    # The model server is local: no proxy (its dispatcher has 300 s timeouts), and no fetch timeouts at all.
    penv = {k: v for k, v in os.environ.items() if k.lower() not in ("http_proxy", "https_proxy", "all_proxy")}
    penv.update(MINIMAX_DATA_DIR=str(data_dir), MCODE_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1",
                ARC_DEADLINE=env["ARC_DEADLINE"],
                NODE_OPTIONS=(penv.get("NODE_OPTIONS", "") + f" --require {PRELOAD}").strip())
    prefix, uenv = sandbox(args)
    penv.update(uenv)
    hand_over(ws, args)
    en = subprocess.run(prefix + [args.node, args.mcode, "plugin", "enable", "arc26"], cwd=ws, env=penv,
                        capture_output=True, text=True, timeout=120)
    if en.returncode:
        sys.stderr.write(f"[plugin] {game}: enable failed: {(en.stderr or en.stdout)[-300:]}\n")
    deadline = time.monotonic() + time_limit
    rounds, idle, log = 0, 0, []
    if args.driver == "acp":
        from acp_driver import run_goal

        mcp = [{"name": "arc", "command": args.server_python, "args": [str(SERVER)],
                "env": [{"name": "ARC_SOCKET", "value": str(sock)}, {"name": "ARC_KERNEL", "value": str(ksock)}]}]

        def done() -> bool:
            st = read_json(result)
            return st.get("state") == "WIN" or int(st.get("actions") or 0) >= args.max_actions

        # One goal session after another: a session ends when its goal ends (complete / blocked /
        # breaker); the REPL, notebook and DAG persist, so the next session picks up from them.
        while rounds < args.max_rounds and deadline - time.monotonic() > 60 and not done():
            before = int(read_json(result).get("actions") or 0)
            # state, notebook, journal and DAG reach the model through the plugin's SessionStart hook
            objective = GOAL_OBJECTIVE
            r = run_goal(args.node, args.mcode, ws, penv, mcp, objective, deadline, done, ws / f"acp_{rounds}.log",
                         prefix=prefix)
            acted = int(read_json(result).get("actions") or 0) - before
            log.append({"round": rounds, **r, "actions": acted})
            sys.stderr.write(f"[goal] {game} session {rounds} actions+={acted} stop={r.get('stop')} "
                             f"tokens={r.get('tokens_used')} err={r.get('error')}\n")
            rounds += 1
            idle = idle + 1 if acted == 0 else 0
            if idle >= args.max_idle:
                break
    while args.driver == "exec" and rounds < args.max_rounds:
        left = deadline - time.monotonic()
        if left < 60:
            break
        state = read_json(result)
        if state.get("state") == "WIN" or int(state.get("actions") or 0) >= args.max_actions:
            break
        cmd = [args.node, args.mcode, "exec", FIRST_PROMPT if rounds == 0 else next_prompt(ws, idle), "--cwd", str(ws),
               "--permission", "full", "--max-steps", str(args.max_steps), "--timeout", f"{int(left)}s",
               "--output-format", "json"]
        if rounds > 0:
            cmd.append("--continue")
        before = int(state.get("actions") or 0)
        with (ws / f"exec_{rounds}.stderr").open("w") as err:
            try:
                r = subprocess.run(prefix + cmd, cwd=ws, env=penv, stdout=subprocess.PIPE, stderr=err, text=True, timeout=left + 120)
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
        if idle >= args.max_idle:
            break  # the agent keeps stopping without acting
    merge_skills(ws, Path(args.skills_dir).resolve(), game_key(game), t_attempt)
    for proc in (kernel, daemon):
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    sock.unlink(missing_ok=True)
    ksock.unlink(missing_ok=True)
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
    p.add_argument("--driver", default="acp", choices=["acp", "exec"],
                   help="acp: MiniMax Code goal mode over ACP (auto-continuation); exec: one mcode exec per round")
    p.add_argument("--agent-user", default="", help="run the agent CLI as this unprivileged user (data/ is made root-only)")
    p.add_argument("--max-idle", type=int, default=6, help="give a game up after this many rounds in a row without an action")
    p.add_argument("--skills-dir", default=None, help="shared skills directory (default <out-dir>/skills)")
    p.add_argument("--node", default=shutil.which("node") or "node")
    p.add_argument("--mcode", default=str(ROOT.parent / "minimax-code" / "dist" / "cli.js"))
    p.add_argument("--server-python", default=sys.executable)
    args = p.parse_args(argv)
    args.api_key = Path(args.api_key_file).read_text().strip() if args.api_key_file else "EMPTY"

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.skills_dir = args.skills_dir or str(out_dir / "skills")
    Path(args.skills_dir).mkdir(parents=True, exist_ok=True)
    if args.agent_user:  # the agent may not read game files; the root-run game daemon still can
        for d in {ROOT / "data", Path(args.env_dir).resolve()}:
            if d.exists():
                os.chmod(d, 0o700)
        hand_over(Path(args.skills_dir), args)
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
