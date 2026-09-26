"""The evolution loop: greedy hill-climbing with one incumbent parent.

cold start (vanilla, train, K) ->
  per round: diagnose parent's failures -> choose WHYs -> design candidates ->
  free pruning -> anchor screen (K=1, cull clear losers) -> confirm top (train, K) ->
  paired gate -> promote the best candidate if its train mean strictly beats the parent.
Stops after ``patience`` rounds without promotion or ``max_rounds``.
Every step persists to ``work_dir`` so a new Kaggle session can resume.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from arc_harness.llm.client import ChatModel

from .bench import Evaluator
from .design import design_candidate
from .diagnose import choose_whys, diagnose, why_brief
from .state import RunSpec, RunState, TreeNode
from .stats import paired, per_game_mean, screen_cull
from .workspace import check_candidate, materialize

logger = logging.getLogger(__name__)


def select_anchor(vanilla: TreeNode, games: list[str], size: int) -> list[str]:
    """Evenly spaced over the vanilla score ranking: sentinels, borderline and hard games."""
    ranked = sorted(games, key=lambda g: sum(vanilla.scores.get(g, [0])) / max(len(vanilla.scores.get(g, [0])), 1))
    if len(ranked) <= size:
        return ranked
    step = (len(ranked) - 1) / (size - 1)
    return [ranked[round(i * step)] for i in range(size)]


class Evolver:
    def __init__(self, spec: RunSpec, evaluator: Evaluator, llm: ChatModel, all_game_ids: Optional[list[str]] = None) -> None:
        self.spec = spec
        self.state = RunState(spec)
        self.eval = evaluator
        self.llm = llm
        self.forbidden = sorted(set(all_game_ids or []) | set(spec.train_games) | set(spec.test_games))

    def _log(self, **event) -> None:
        self.state.meta["history"].append(event)
        self.state.save()
        logger.info("evolve: %s", event)

    def cold_start(self) -> TreeNode:
        st = self.state
        if st.meta.get("vanilla"):
            return st.nodes[st.meta["vanilla"]]
        v = TreeNode(TreeNode.new_id(), None, status="promoted", summary="vanilla")
        v.scores = self.eval.eval(v, self.spec.train_games, self.spec.k_confirm, "train")
        st.save_node(v)
        st.meta.update(vanilla=v.id, parent=v.id)
        self._log(event="cold_start", node=v.id, train_mean=v.mean(self.spec.train_games))
        return v

    async def _design(self, parent: TreeNode, why: str, brief: str, distinct: list[str]) -> Optional[TreeNode]:
        node = TreeNode(TreeNode.new_id(), parent.id, why=why)
        ws = Path(self.spec.work_dir).resolve() / "ws" / node.id
        materialize(Path(self.spec.repo_dir).resolve(), parent.overlay, ws)
        tried = [
            f"- {n.summary} -> {n.status} ({n.reason})"
            for n in self.state.nodes.values()
            if n.why == why and n.id != node.id
        ][-6:]
        changes, summary = await design_candidate(
            self.llm, ws, self.spec.whitelist, brief, "\n".join(tried), distinct, self.spec.editor_turns
        )
        node.overlay = {**parent.overlay, **changes}
        node.summary = summary[:400]
        if not changes:
            node.status, node.reason = "pruned", "no edit"
        else:
            err = check_candidate(ws, changes, self.forbidden)
            if err:
                node.status, node.reason = "pruned", err
        self.state.save_node(node)
        return node if node.status == "new" else None

    async def round(self) -> bool:
        st, spec = self.state, self.spec
        parent = st.parent
        vanilla = st.nodes[st.meta["vanilla"]]
        assert parent is not None
        st.meta["round"] += 1
        r = st.meta["round"]
        fmap_path = Path(spec.work_dir) / "failure_map.json"
        diagnosed = st.meta.setdefault("diagnosed", [])
        if parent.id not in diagnosed:
            means = per_game_mean(parent.scores, spec.train_games)
            worst = sorted(spec.train_games, key=lambda g: means[g])[: max(4, len(spec.train_games) // 2)]
            traj = self.eval.trajectories(parent, worst, "train")
            fmap = await diagnose(self.llm, traj, fmap_path)
            diagnosed.append(parent.id)
        else:
            fmap = json.loads(fmap_path.read_text()) if fmap_path.exists() else {"why_distribution": {}, "cells": {}}
        whys = choose_whys(fmap, st.meta["attempts"], spec.max_why_per_round)
        if not whys:
            whys = ["W2_mechanic_undiscovered"]
        cands: list[TreeNode] = []
        for why in whys:
            brief = why_brief(fmap, why)
            distinct: list[str] = []
            for _ in range(spec.candidates_per_why):
                node = await self._design(parent, why, brief, distinct)
                if node is not None:
                    cands.append(node)
                    distinct.append(node.summary[:120])
        anchor = select_anchor(vanilla, spec.train_games, spec.anchor_size)
        survivors: list[tuple[float, TreeNode]] = []
        for node in cands:
            s = self.eval.eval(node, anchor, 1, "screen")
            node.screen = {g: v[0] for g, v in s.items()}
            culled, why_txt = screen_cull(node.screen, parent.scores, anchor, spec.screen_sigma)
            if culled:
                node.status, node.reason = "culled", why_txt
            else:
                survivors.append((sum(node.screen.values()) / len(anchor), node))
            st.save_node(node)
        survivors.sort(key=lambda x: -x[0])
        best: Optional[TreeNode] = None
        for _, node in survivors[: spec.confirm_top]:
            node.scores = self.eval.eval(node, spec.train_games, spec.k_confirm, "train")
            res = paired(node.scores, parent.scores, spec.train_games)
            passed = res.cand_mean > res.ctrl_mean
            node.status = "confirmed" if passed else "rejected"
            node.reason = (
                f"train {res.cand_mean:.3f} vs parent {res.ctrl_mean:.3f}, z={res.z:.2f}, "
                f"wilcoxon p={res.wilcoxon_p:.3f}{' (2sigma)' if res.credited_2sigma else ''}"
            )
            st.save_node(node)
            if passed and (best is None or node.mean(spec.train_games) > best.mean(spec.train_games)):
                best = node
        for node in cands:
            if node is not best:
                st.meta["attempts"][node.why] = st.meta["attempts"].get(node.why, 0) + 1
        if best is not None:
            best.status = "promoted"
            st.save_node(best)
            st.meta["parent"] = best.id
            st.meta["no_improve"] = 0
        else:
            st.meta["no_improve"] += 1
        beats_vanilla = best is not None and best.mean(spec.train_games) > vanilla.mean(spec.train_games)
        self._log(
            event="round", round=r, whys=whys, candidates=[n.id for n in cands],
            promoted=best.id if best else None, beats_vanilla=beats_vanilla,
            parent_mean=st.parent.mean(spec.train_games) if st.parent else None,
        )
        return best is not None

    async def run(self) -> TreeNode:
        self.cold_start()
        while self.state.meta["round"] < self.spec.max_rounds and self.state.meta["no_improve"] < self.spec.patience:
            await self.round()
        parent = self.state.parent
        assert parent is not None
        return parent

    def lineage(self) -> list[TreeNode]:
        out = []
        n = self.state.parent
        while n is not None:
            out.append(n)
            n = self.state.nodes.get(n.parent_id) if n.parent_id else None
        return out[::-1]

    def export(self) -> dict:
        return {nid: asdict(n) for nid, n in self.state.nodes.items()}
