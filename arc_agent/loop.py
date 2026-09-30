"""The agent loop for one game: one LLM call per action.

    step:  read (graph + handoff + checked prediction + diff + frame)  ->  LLM  ->  validate
           ->  apply graph_update  ->  play the action  ->  diff  ->  check the prediction
           ->  record action and outcome nodes  ->  next step

Two context modes. ``fresh``: every step is a new conversation holding the whole graph and the
handoff. ``rolling``: the conversation continues, each turn adding only what is new (the check, the
diff, the graph nodes added, the frame), and it is compacted into a fresh segment (whole graph +
handoff) once the prompt passes ``compact_tokens``, at every new level and after GAME_OVER. In both
modes every answer must add graph nodes. The graph and the step log are saved after every step.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from arc_agent import prompt, sim, vision
from arc_agent.graph import Graph
from arc_agent.llm import LLM
from arc_agent.session import Session

GRID_EVERY = 8  # rolling mode: at most one full grid per this many turns of a segment
TRANSITIONS = 20  # recent real actions a new simulator is replayed on
SIM_AFTER = 5  # actions on a level after which a missing simulator is asked for every turn
PLAN_MAX = 40  # longest route played from one search
SEARCH_SECONDS = 40.0
ACTION_NAMES = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5", 6: "ACTION6", 7: "UNDO"}


def route_text(route: list[tuple]) -> str:
    return "ROUTE " + ", ".join(ACTION_NAMES.get(a, str(a)) + (f"({x},{y})" if a == 6 else "") for a, x, y in route)


def replay_text(code: str, version: int, transitions: list[dict], mask: np.ndarray) -> str:
    """A new simulator replayed on the level's recent real actions."""
    if not transitions:
        return f"SIMULATOR v{version} saved (no actions on this level yet to replay it on)"
    res = sim.run(code, "replay", 15, transitions=[dict(t) for t in transitions], mask=mask)
    if "error" in res:
        return f"SIMULATOR v{version} saved; replaying it failed: " + sim.error_text(res)
    w = res["wrong"]
    exact = sum(1 for n in w if n == 0)
    return (f"SIMULATOR v{version} saved; replayed on this level's last {len(w)} actions it predicts {exact} "
            f"exactly; board cells wrong per action, oldest first: {w}")


def plan_route(code: str, grid: np.ndarray, available: list[int], t_end: float,
               fallback: list[tuple]) -> tuple[list[tuple], str]:
    """Search the simulator for a route to goal(); the model's own action when there is none."""
    if not code:
        return fallback, "PLAN: there is no simulator yet (write step() and goal()); your action was played"
    secs = min(SEARCH_SECONDS, t_end - time.time() - 30)
    if secs < 5:
        return fallback, "PLAN: no time left to search; your action was played"
    res = sim.run(code, "search", secs, grid=grid, actions=[a for a in available if a != 0])
    if "error" in res:
        return fallback, "PLAN: " + sim.error_text(res) + "; your action was played"
    if res["found"] and res["path"]:
        path = [tuple(p) for p in res["path"]]
        note = f"PLAN: the search found a route of {len(path)} actions ({res['nodes']} simulated actions)"
        return path[:PLAN_MAX], note + (f"; its first {PLAN_MAX} are played" if len(path) > PLAN_MAX else "")
    if res["found"]:
        return fallback, "PLAN: goal() already holds for the current frame, yet the level is not cleared: goal() is wrong; your action was played"
    return fallback, f"PLAN: {res['reason']}; your action was played"


def make_session(game: str, env_dir: str, gateway: Optional[str], card: Optional[str]) -> tuple[Session, str, Any]:
    from arc_agi import Arcade, OperationMode

    quiet = logging.getLogger("arc.engine")
    if gateway:
        arc = Arcade(arc_api_key="test-key-123", arc_base_url=gateway.rstrip("/"),
                     operation_mode=OperationMode.ONLINE, logger=quiet)
    else:
        arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir, logger=quiet)
        card = arc.open_scorecard(tags=["arc26-agent"])
    envs = {e.game_id: e for e in arc.get_environments()}
    gid = next((g for g in envs if g == game or g.startswith(game)), game)
    return Session(arc.make(gid, scorecard_id=card)), gid, envs.get(gid)


def play(game: str, out_dir: Path, llm: LLM, env_dir: str = "", gateway: Optional[str] = None,
         card: Optional[str] = None, seconds: float = 2700.0, max_actions: int = 2000,
         policy: str = "model", image: bool = True, context: str = "fresh",
         compact_tokens: int = 30000) -> dict[str, Any]:
    """Play one game. ``policy`` decides which steps run with thinking:
      always     every step
      model      the model marks routine next steps (``think_next`` false): those run without thinking
      exception  no thinking unless it is needed: the first 3 steps, a failed prediction, a new level,
                 GAME_OVER, or the model asks for it (``think_next`` true)
    Under both adaptive policies a failed prediction, a finished level or GAME_OVER always think."""
    out_dir.mkdir(parents=True, exist_ok=True)
    s, gid, info = make_session(game, env_dir, gateway, card)
    obs = s.start()
    graph = Graph()
    hud = vision.HudTracker()
    t0 = time.time()
    t_end = t0 + seconds
    handoff, verdict, diff, problems = "", "", "", []
    prev_grid: Optional[np.ndarray] = None
    move_steps: list[int] = []  # displacements seen on this level (lattice cue)
    think_next = True
    last_pos: Optional[tuple[int, int]] = None
    log = (out_dir / "steps.jsonl").open("a")
    thoughts = (out_dir / "reasoning.jsonl").open("a")  # the model's reasoning per step, for analysis
    step = 0
    tokens = 0
    convo: list[dict] = []  # rolling mode: the current segment after the system prompt
    compact = True  # the next step opens a fresh segment
    added_ids: list[int] = []  # graph nodes added since the model's last turn
    need_grid = False
    grid_turn = 0  # the turn of this segment that last carried the full grid
    segments = 0
    sim_code, sim_version = "", 0  # the model's simulator (arc_agent.sim)
    transitions: list[dict] = []  # this level's recent real actions, for replaying a new simulator
    sim_wrong = 0  # board cells the simulator got wrong on the last action (-1: it failed)
    while time.time() < t_end and s.actions < max_actions:
        tp = time.time()
        grid = np.asarray(obs.grid)
        state = obs.state.name
        if state == "WIN":
            break
        view = {"level": obs.levels_completed, "win_levels": s.win_levels or obs.win_levels, "state": state,
                "level_actions": s.level_so_far, "actions": s.actions, "available": obs.available_actions,
                "step": step, "time_left": t_end - time.time()}
        lat = vision.infer_lattice(grid, move_steps)
        lattice = ""
        if lat:
            st, ox, oy = lat
            if ox is None:
                ox, oy = (last_pos[0] % st, last_pos[1] % st) if last_pos else (0, 0)
            lattice = vision.lattice_text(grid, st, ox, oy)
        thinks = policy == "always" or think_next
        img = vision.png(grid) if image else None
        asks = list(problems)
        if not sim_code and s.level_so_far >= SIM_AFTER and state != "GAME_OVER":
            asks.append(f"NO SIMULATOR YET after {s.level_so_far} actions on this level: write step() (and goal()) in a "
                        "```python block this step, from what the graph says the actions do. It may be rough; the "
                        "reports on it tell you what to fix.")
        elif sim_code and sim_wrong:
            asks.append("YOUR SIMULATOR was wrong on the last action (see SIMULATOR above): send a corrected ```python "
                        "block this step, or record in the graph why the difference does not matter.")
        if context == "fresh" or compact:
            # a routine step reads the cell map and the objects; the full hex grid (~4k tokens) only when
            # the model thinks or there is no cell map
            grid_text = vision.hex_grid(grid) if thinks or not lattice or context != "fresh" else \
                "(left out on routine steps: use the cell map and objects)"
            user = prompt.user_message(view, graph.render(), handoff, verdict, diff, vision.objects_text(grid),
                                       grid_text, img, asks, lattice, sim_code)
            convo = []
            segments += 1
            compact = False
            grid_turn = 0
        else:
            # a full grid stays in the segment until the next compaction, so a request for one is honoured
            # only when none was sent in the last GRID_EVERY turns; otherwise the changed rows go
            turn = len(convo) // 2
            full = need_grid and turn - grid_turn >= GRID_EVERY
            if full:
                grid_turn = turn
            grid_text = vision.hex_grid(grid) if full else None
            rows = vision.changed_rows(prev_grid, grid) if prev_grid is not None and not full else ""
            if need_grid and not full:
                rows = (f"(need_grid: a full grid was sent {turn - grid_turn} turns ago; it and the changed rows "
                        f"since then give the whole frame)\n") + rows
            # earlier turns are never edited (the server's prefix cache only hits a byte-identical prefix),
            # so a step turn carries no image: the frame is in the changed rows / grid, and the image is
            # in the turn that opened the segment
            user = prompt.step_message(view, graph.lines(added_ids) or "(none)", verdict, diff,
                                       vision.objects_text(grid), lattice, grid_text, None, asks, rows)
        msgs = [{"role": "system", "content": prompt.SYSTEM}] + convo + [user]
        answer, errs, rep = None, [], None
        tl = time.time()
        for _ in range(2):  # one repair round when the answer is unusable
            try:
                rep = llm.chat(msgs, t_end, None if thinks else False)
            except RuntimeError as exc:
                errs = [str(exc)]
                break
            tokens += int((rep.usage or {}).get("total_tokens") or 0)
            answer = prompt.parse(rep.content)
            errs = prompt.validate(answer, obs.available_actions, state)
            if not errs:
                break
            msgs = msgs + [{"role": "assistant", "content": rep.content},
                           {"role": "user", "content": "Your answer is unusable: " + "; ".join(errs) +
                            ". Reply again with the complete JSON object."}]
        te = time.time()
        if context == "rolling":
            # the segment keeps this turn: the user message and the answer itself, as compact JSON (not
            # the reasoning, nor any prose written before the JSON)
            kept = json.dumps(answer, ensure_ascii=False, separators=(",", ":")) if answer is not None else \
                (rep.content[-2000:] if rep else "")
            code = prompt.parse_code(rep.content) if rep and answer is not None else None  # a block (a JSON field is in kept)
            if code:
                kept += "\n```python\n" + code + "\n```"
            convo += [user, {"role": "assistant", "content": kept}]
            used = int((rep.usage or {}).get("prompt_tokens") or 0) if rep else compact_tokens + 1
            if used > compact_tokens:
                compact = True
        rec: dict[str, Any] = {"step": step, "segment": segments, "t": round(time.time(), 1), "level": obs.levels_completed,
                               "llm_s": round(rep.seconds, 1) if rep else None, "attempts": rep.attempts if rep else 0,
                               "usage": rep.usage if rep else None,
                               "llm_failures": list(rep.failures) if rep else None, "thinking": policy == "always" or think_next,
                               "reasoning_chars": len(rep.reasoning) if rep else None}
        if errs:
            rec["error"] = errs
            if rep is not None:
                thoughts.write(json.dumps({"step": step, "error": errs, "reasoning": rep.reasoning[-8000:],
                                           "content": rep.content[-6000:]}) + "\n")
                thoughts.flush()
            log.write(json.dumps(rec) + "\n")
            log.flush()
            problems = errs
            step += 1
            if rep is None and time.time() < t_end:  # the endpoint is down: wait instead of spinning
                time.sleep(min(20, max(0.0, t_end - time.time())))
            continue
        assert answer is not None
        new_ids, problems = graph.apply(answer.get("graph_update") or {}, obs.levels_completed, step)
        need_grid = answer.get("need_grid") is True
        sim_lines: list[str] = []
        code = prompt.parse_code(rep.content, answer) if rep else None
        if code and code != sim_code:
            sim_code, sim_version = code, sim_version + 1
            sim_lines.append(replay_text(sim_code, sim_version, transitions, hud.mask()))
        a = answer["action"]
        aid = int(a["id"])
        x, y = (int(a["x"]), int(a["y"])) if aid == 6 else (None, None)
        todo: list[tuple] = [(aid, x, y)]
        if answer.get("plan") is True:
            todo, note = plan_route(sim_code, grid, obs.available_actions, t_end, todo)
            sim_lines.append(note)
        pred = answer.get("prediction") or {}
        before = obs
        prev_grid = grid
        played: list[tuple] = []
        level_up = False
        for i, (aid, x, y) in enumerate(todo):
            g0 = np.asarray(obs.grid)
            lv = obs.levels_completed
            obs = s.step(aid, x, y)
            after = np.asarray(obs.grid)
            played.append((aid, x, y))
            up = obs.levels_completed > lv
            wrong: Optional[int] = None
            if up:
                hud.reset()
                move_steps, last_pos, transitions, sim_wrong = [], None, [], 0
            elif aid != 0:
                hud.update(g0, after)
                transitions = (transitions + [{"grid": g0, "action": aid, "x": x, "y": y, "after": after}])[-TRANSITIONS:]
                if sim_code:
                    res = sim.run(sim_code, "check", 8, grid=g0, action=aid, x=x, y=y, after=after, mask=hud.mask())
                    wrong = -1 if "error" in res else res["wrong"]
                    sim_wrong = wrong
                    if len(todo) == 1 or wrong != 0 or i == len(todo) - 1:
                        sim_lines.append((f"[route action {i + 1}] " if len(todo) > 1 else "") + sim.check_text(res))
            level_up = level_up or up
            if up or obs.state.name in ("GAME_OVER", "WIN") or time.time() > t_end:
                break
            if len(todo) > 1 and wrong != 0 and i < len(todo) - 1:
                sim_lines.append(f"ROUTE stopped after {i + 1} of {len(todo)} actions: the real frame differed from step()")
                break
        if len(todo) > 1:
            sim_lines.append(f"ROUTE played {len(played)} of {len(todo)} actions: " + route_text(played))
        tx = time.time()
        after = np.asarray(obs.grid)
        diff, facts = vision.diff_text(prev_grid, after, None if level_up else hud.mask())
        anim = vision.anim_text([np.asarray(f) for f in obs.frames], prev_grid) if len(played) == 1 else ""
        if anim:
            diff += "\n" + anim
        facts.update(level_up=level_up, game_over=obs.state.name == "GAME_OVER")
        if not level_up and facts["moves"] and len(played) == 1:
            m0 = facts["moves"][0]
            move_steps += [abs(v) for v in (m0["dx"], m0["dy"]) if v]
            same = [m for m in facts["moves"] if (m["dx"], m["dy"]) == (m0["dx"], m0["dy"])]  # parts of one mover
            last_pos = (min(m["x"] for m in same), min(m["y"] for m in same))
        ok, verdict = prompt.check_prediction(pred, facts)
        if sim_lines:
            verdict += "\n" + "\n".join(sim_lines)
        needed = ok is not True or level_up or facts["game_over"]
        if policy == "exception":
            think_next = needed or step < 2 or answer.get("think_next") is True
        else:
            think_next = needed or answer.get("think_next") is not False
        if level_up:
            verdict += f"\nLEVEL {before.levels_completed + 1} COMPLETED in {s.level_actions[-1]} actions: the frame is now level {obs.levels_completed + 1}."
            diff = "(new level: the whole frame changed)"
        if facts["game_over"]:
            verdict += "\nGAME_OVER: the level must be restarted with action 0 (RESET)."
        act_text = route_text(played) if len(played) > 1 else \
            ACTION_NAMES.get(aid, str(aid)) + (f" at ({x},{y})" if aid == 6 else "")
        act_id = graph.add_node("action", f"{act_text}; predicted: {pred.get('text', '')}", before.levels_completed, step)
        out_id = graph.add_node("outcome", verdict.splitlines()[0][:300], before.levels_completed, step,
                                "confirmed" if ok else "refuted" if ok is False else "open")
        graph.add_edge(act_id, out_id, "leads_to")
        for nid in new_ids:
            if graph.nodes[nid].type in ("hypothesis", "rule", "plan"):
                graph.add_edge(act_id, nid, "tests")
        handoff = str(answer.get("handoff", ""))
        added_ids = new_ids + [act_id, out_id]
        if level_up or facts["game_over"]:
            compact = True  # a new level or a restart opens a fresh segment
        graph.save(out_dir / "graph.json")
        rec["timing"] = {"prep": round(tl - tp, 2), "llm": round(te - tl, 2), "env": round(tx - te, 2),
                         "post": round(time.time() - tx, 2)}
        rec.update(played=played, sim_version=sim_version, action=aid, x=x, y=y, prediction=pred, verdict=verdict, ok=ok, facts=facts,
                   new_nodes=len(new_ids), handoff=handoff, actions=s.actions, state=obs.state.name,
                   levels=obs.levels_completed)
        log.write(json.dumps(rec, default=str) + "\n")
        log.flush()
        if rep is not None:
            thoughts.write(json.dumps({"step": step, "reasoning": rep.reasoning[-20000:], "content": rep.content[-6000:]}) + "\n")
            thoughts.flush()
        step += 1
    log.close()
    thoughts.close()
    baseline = list(getattr(info, "baseline_actions", None) or [])
    result = {"game": gid, "levels": obs.levels_completed, "win_levels": s.win_levels, "actions": s.actions,
              "level_actions": s.level_actions, "baseline": baseline, "state": obs.state.name, "steps": step,
              "nodes": len(graph.nodes), "tokens": tokens, "seconds": round(time.time() - t0), "context": context,
              "segments": segments}
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    return result
