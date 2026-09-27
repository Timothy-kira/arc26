"""Write or check games/manifest.json: every game we run, its split, tags, human per-level
baselines (official) or category and input type (community), and the sha256 of its files, so a
fresh checkout can fetch the same games and verify them. Also renders docs/GAMES.md.

    python scripts/games_manifest.py            # rewrite the manifest and docs/GAMES.md from data/
    python scripts/games_manifest.py --check    # verify data/ against the manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arc_eval.datasets import games, official_val_ids  # noqa: E402

MANIFEST = ROOT / "games" / "manifest.json"
DOC = ROOT / "docs" / "GAMES.md"
ITERATION = ["ls20", "vc33", "tu93"]  # the three games the harness is iterated on
COMMUNITY_REPO = "https://github.com/theredbluepill/arc-interactive"


# arc-interactive's own GAMES.md has ~110 fine categories; they are grouped into families by their
# first word (a category like "Graph / Plumbing" belongs to the "Graph" family).
FAMILY = {
    "Tutorial": "教程 / 移动基础", "Movement": "移动与协调", "Coordination": "移动与协调", "Multi-Agent": "移动与协调",
    "Escort": "移动与协调", "Puzzle": "机关谜题（钥匙、开关、传送、冰面）", "Pattern": "图案 / 拼贴 / 对称",
    "Tiling": "图案 / 拼贴 / 对称", "Spatial": "图案 / 拼贴 / 对称", "Environmental": "推箱 / 环境操作",
    "Manipulation": "推箱 / 环境操作", "Push": "推箱 / 环境操作", "Logistics": "推箱 / 环境操作",
    "Survival": "生存 / 时机 / 危险", "Timing": "生存 / 时机 / 危险", "Hazard": "生存 / 时机 / 危险",
    "Dynamic": "生存 / 时机 / 危险", "Stealth": "记忆 / 隐藏信息 / 探索", "Memory": "记忆 / 隐藏信息 / 探索",
    "Exploration": "记忆 / 隐藏信息 / 探索", "Simulation": "模拟 / 场", "Field": "模拟 / 场", "Growth": "模拟 / 场",
    "Coverage": "路径 / 覆盖 / 资源", "Path": "路径 / 覆盖 / 资源", "Territory": "路径 / 覆盖 / 资源",
    "Resource": "路径 / 覆盖 / 资源", "Collection": "收集", "Graph": "图 / 电路", "Circuit": "图 / 电路",
    "Logic": "逻辑 / 推理 / 排序", "Cognitive": "逻辑 / 推理 / 排序", "Permutation": "逻辑 / 推理 / 排序",
    "Ordering": "逻辑 / 推理 / 排序", "Sequencing": "逻辑 / 推理 / 排序", "Precision": "拓扑 / 几何",
    "Topology": "拓扑 / 几何", "Geometry": "拓扑 / 几何",
}


def digest(d: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.is_file()}


def plain(md: str, limit: int = 160) -> str:
    t = re.sub(r"\*\*|`|!\[[^\]]*\]\([^)]*\)", "", md).replace("|", "/").strip()
    return t if len(t) <= limit else t[:limit - 1].rstrip() + "…"


def input_type(actions: str) -> str:
    click = bool(re.search(r"\b6\b", actions))
    moves = bool(re.search(r"1-4|\b[1-5]:", actions)) and "No-op" not in actions
    return "keyboard_click" if click and moves else "click" if click else "keyboard"


def community_rows(cdir: Path) -> dict[str, list[str]]:
    rows = {}
    for line in (cdir / "GAMES.md").read_text().splitlines():
        if line.startswith("| ") and not line.startswith(("| Game", "|---")):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if len(c) >= 7:
                rows[c[0]] = c
    return rows


def build() -> dict:
    val = official_val_ids()
    official = []
    for g in games("official"):
        short = g.game_id.split("-")[0]
        split = "official_val" if g.game_id in val else "official_dev"
        official.append({"game_id": g.game_id, "short": short, "split": split, "iteration": short in ITERATION,
                         "tags": list(g.tags), "human_baseline": list(g.baseline or []),
                         "path": str(g.env_dir.relative_to(ROOT / "data")), "sha256": digest(g.env_dir)})
    cdir = ROOT / "data" / "community"
    commit = subprocess.run(["git", "-C", str(cdir), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows = community_rows(cdir)
    community = []
    for g in games("community"):
        short = g.game_id.split("-")[0]
        d = next((cdir / "environment_files" / short).iterdir())
        r = rows.get(short, [short, "Uncategorised", "", "", "", "", ""])
        cat = r[1]
        community.append({"game_id": g.game_id, "short": short, "family": FAMILY.get(re.split(r"[ /]", cat)[0], "其他"),
                          "category": cat, "input": input_type(r[6]), "grid": r[2], "levels": r[3],
                          "description": plain(r[4]), "tags": list(g.tags),
                          "path": str(d.relative_to(ROOT / "data")), "sha256": digest(d)})
    return {"official": {"source": "kaggle competitions download -c arc-prize-2026-arc-agi-3 (make data)",
                         "games": official},
            "community": {"source": COMMUNITY_REPO, "commit": commit or None,
                          "excluded": "copies of official games (ft09, ls20, vc33)", "games": community},
            "iteration_games": ITERATION}


def render(m: dict) -> str:
    from collections import Counter

    L = ["# 我们运行的游戏", "",
         "游戏源码不放进本仓库：官方游戏是 Kaggle 比赛数据，社区游戏属于 arc-interactive（MIT）。"
         "用 `scripts/fetch_games.sh` 下载，并按 `games/manifest.json` 里的 sha256 校验。"
         "本页由 `python scripts/games_manifest.py` 生成。", "",
         "| 集合 | 数量 | 用途 |", "|---|---|---|",
         f"| official_dev | {sum(g['split'] == 'official_dev' for g in m['official']['games'])} | 开发；其中 ls20、vc33、tu93 是迭代用的 3 个游戏 |",
         f"| official_val | {sum(g['split'] == 'official_val' for g in m['official']['games'])} | 验证：只评估，不看、不调参 |",
         f"| community | {len(m['community']['games'])} | 开发集（防过拟合、覆盖更多机制）；没有人类基线，只看过关数和步数 |", "",
         "## 官方 25 个", "", "| 游戏 | 切分 | 迭代 | 类型 | 关卡数 | 人类每关步数 |", "|---|---|---|---|---|---|"]
    for g in m["official"]["games"]:
        L.append(f"| `{g['game_id']}` | {g['split']} | {'✓' if g['iteration'] else ''} | {', '.join(g['tags']) or '-'} | "
                 f"{len(g['human_baseline'])} | {' '.join(map(str, g['human_baseline']))} |")
    c = m["community"]
    fam = Counter(g["family"] for g in c["games"])
    inp = Counter(g["input"] for g in c["games"])
    L += ["", f"## 社区 {len(c['games'])} 个（arc-interactive）", "",
          f"- **来源：** {c['source']}，固定在提交 `{c['commit']}`。",
          "- **已排除：** 原仓库里混入的官方游戏副本（ft09、ls20、vc33），避免开发集和官方集重叠。",
          f"- **输入方式：** keyboard {inp['keyboard']}、click {inp['click']}、keyboard_click {inp['keyboard_click']}。",
          "- **分类：** 按原仓库人工标注的类别，归并成下面的大类；每个游戏的原始类别也一并列出。", "",
          "| 大类 | 数量 |", "|---|---|"]
    L += [f"| {f} | {n} |" for f, n in fam.most_common()]
    for f, _ in fam.most_common():
        L += ["", f"### {f}", "", "| 游戏 | 原始类别 | 输入 | 网格 | 关卡 | 说明 |", "|---|---|---|---|---|---|"]
        for g in sorted((g for g in c["games"] if g["family"] == f), key=lambda g: (g["category"], g["short"])):
            L.append(f"| `{g['short']}` | {g['category']} | {g['input']} | {g['grid']} | {g['levels']} | {g['description']} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not a.check:
        MANIFEST.parent.mkdir(exist_ok=True)
        m = build()
        MANIFEST.write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n")
        DOC.write_text(render(m))
        print(f"wrote {MANIFEST} and {DOC}")
        return 0
    want = json.loads(MANIFEST.read_text())
    bad = 0
    for kind in ("official", "community"):
        n = 0
        for g in want[kind]["games"]:
            d = ROOT / "data" / g["path"]
            got = digest(d) if d.exists() else {}
            if got != g["sha256"]:
                n += 1
                print(f"MISMATCH {kind} {g['game_id']}: {'missing' if not got else 'files differ'}")
        print(f"{len(want[kind]['games']) - n}/{len(want[kind]['games'])} {kind} games verified")
        bad += n
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
