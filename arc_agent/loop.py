"""The agent loop for one game: one LLM call per action, a fresh context every step.

    step:  read (graph + handoff + checked prediction + diff + frame)  ->  LLM  ->  validate
           ->  apply graph_update  ->  play the action  ->  diff  ->  check the prediction
           ->  record action and outcome nodes  ->  next step

Everything the model knows between steps is in the graph and the handoff; both are saved after
every step (graph.json, steps.jsonl) so a run can be inspected or resumed.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from arc_agent import prompt, vision
from arc_agent.graph import Graph
from arc_agent.llm import LLM
from arc_agent.session import Session

ACTION_NAMES = {0: "RESET", 1: "ACTION1", 2: "ACTION2", 3: "ACTION3", 4: "ACTION4", 5: "ACTION5", 6: "ACTION6", 7: "UNDO"}


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
         card: Optional[str] = None, seconds: float = 2700.0, max_actions: int = 2000) -> dict[str, Any]:
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
    last_pos: Optional[tuple[int, int]] = None
    log = (out_dir / "steps.jsonl").open("a")
    step = 0
    tokens = 0
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
        msgs = [{"role": "system", "content": prompt.SYSTEM},
                prompt.user_message(view, graph.render(), handoff, verdict, diff, vision.objects_text(grid),
                                    vision.hex_grid(grid), vision.png(grid), problems, lattice)]
        answer, errs, rep = None, [], None
        tl = time.time()
        for _ in range(2):  # one repair round when the answer is unusable
            try:
                rep = llm.chat(msgs, t_end)
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
        rec: dict[str, Any] = {"step": step, "t": round(time.time(), 1), "level": obs.levels_completed,
                               "llm_s": round(rep.seconds, 1) if rep else None, "attempts": rep.attempts if rep else 0,
                               "usage": rep.usage if rep else None,
                               "llm_failures": list(rep.failures) if rep else None,
                               "reasoning_chars": len(rep.reasoning) if rep else None}
        if errs:
            rec["error"] = errs
            log.write(json.dumps(rec) + "\n")
            log.flush()
            problems = errs
            step += 1
            if rep is None and time.time() < t_end:  # the endpoint is down: wait instead of spinning
                time.sleep(min(20, max(0.0, t_end - time.time())))
            continue
        assert answer is not None
        new_ids, problems = graph.apply(answer.get("graph_update") or {}, obs.levels_completed, step)
        a = answer["action"]
        aid = int(a["id"])
        x, y = (int(a["x"]), int(a["y"])) if aid == 6 else (None, None)
        pred = answer.get("prediction") or {}
        before = obs
        prev_grid = grid
        obs = s.step(aid, x, y)
        tx = time.time()
        after = np.asarray(obs.grid)
        level_up = obs.levels_completed > before.levels_completed
        if level_up:
            hud.reset()
            move_steps, last_pos = [], None
        else:
            hud.update(prev_grid, after)
        diff, facts = vision.diff_text(prev_grid, after, None if level_up else hud.mask())
        anim = vision.anim_text([np.asarray(f) for f in obs.frames], prev_grid)
        if anim:
            diff += "\n" + anim
        facts.update(level_up=level_up, game_over=obs.state.name == "GAME_OVER")
        if not level_up and facts["moves"]:
            m0 = facts["moves"][0]
            move_steps += [abs(v) for v in (m0["dx"], m0["dy"]) if v]
            same = [m for m in facts["moves"] if (m["dx"], m["dy"]) == (m0["dx"], m0["dy"])]  # parts of one mover
            last_pos = (min(m["x"] for m in same), min(m["y"] for m in same))
        ok, verdict = prompt.check_prediction(pred, facts)
        if level_up:
            verdict += f"\nLEVEL {before.levels_completed + 1} COMPLETED in {s.level_actions[-1]} actions: the frame is now level {obs.levels_completed + 1}."
            diff = "(new level: the whole frame changed)"
        if facts["game_over"]:
            verdict += "\nGAME_OVER: the level must be restarted with action 0 (RESET)."
        act_text = ACTION_NAMES.get(aid, str(aid)) + (f" at ({x},{y})" if aid == 6 else "")
        act_id = graph.add_node("action", f"{act_text}; predicted: {pred.get('text', '')}", before.levels_completed, step)
        out_id = graph.add_node("outcome", verdict.splitlines()[0][:300], before.levels_completed, step,
                                "confirmed" if ok else "refuted" if ok is False else "open")
        graph.add_edge(act_id, out_id, "leads_to")
        for nid in new_ids:
            if graph.nodes[nid].type in ("hypothesis", "rule", "plan"):
                graph.add_edge(act_id, nid, "tests")
        handoff = str(answer.get("handoff", ""))
        graph.save(out_dir / "graph.json")
        rec["timing"] = {"prep": round(tl - tp, 2), "llm": round(te - tl, 2), "env": round(tx - te, 2),
                         "post": round(time.time() - tx, 2)}
        rec.update(action=aid, x=x, y=y, prediction=pred, verdict=verdict, ok=ok, facts=facts,
                   new_nodes=len(new_ids), handoff=handoff, actions=s.actions, state=obs.state.name,
                   levels=obs.levels_completed)
        log.write(json.dumps(rec, default=str) + "\n")
        log.flush()
        step += 1
    log.close()
    baseline = list(getattr(info, "baseline_actions", None) or [])
    result = {"game": gid, "levels": obs.levels_completed, "win_levels": s.win_levels, "actions": s.actions,
              "level_actions": s.level_actions, "baseline": baseline, "state": obs.state.name, "steps": step,
              "nodes": len(graph.nodes), "tokens": tokens, "seconds": round(time.time() - t0)}
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    return result
