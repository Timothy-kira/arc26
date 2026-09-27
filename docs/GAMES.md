# 我们运行的游戏

游戏源码不放进本仓库：官方游戏是 Kaggle 比赛数据，社区游戏属于 arc-interactive（MIT）。用 `scripts/fetch_games.sh` 下载，并按 `games/manifest.json` 里的 sha256 校验。本页由 `python scripts/games_manifest.py` 生成。

| 集合 | 数量 | 用途 |
|---|---|---|
| official_dev | 20 | 开发；其中 ls20、vc33、tu93 是迭代用的 3 个游戏 |
| official_val | 5 | 验证：只评估，不看、不调参 |
| community | 249 | 开发集（防过拟合、覆盖更多机制）；没有人类基线，只看过关数和步数 |

## 官方 25 个

| 游戏 | 切分 | 迭代 | 类型 | 关卡数 | 人类每关步数 |
|---|---|---|---|---|---|
| `ar25-0c556536` | official_dev |  | keyboard_click | 8 | 32 50 75 37 89 159 233 73 |
| `bp35-0a0ad940` | official_dev |  | keyboard_click | 9 | 21 48 44 38 33 87 86 131 163 |
| `cd82-fb555c5d` | official_dev |  | keyboard_click | 6 | 55 8 41 21 23 23 |
| `cn04-2fe56bfb` | official_dev |  | keyboard_click | 6 | 29 54 85 300 208 113 |
| `dc22-fdcac232` | official_dev |  | keyboard_click | 6 | 59 102 67 98 324 578 |
| `ft09-0d8bbf25` | official_dev |  | - | 6 | 43 12 23 28 65 37 |
| `g50t-5849a774` | official_val |  | keyboard | 7 | 78 175 179 230 96 54 67 |
| `ka59-38d34dbb` | official_dev |  | keyboard_click | 7 | 28 109 51 51 33 132 326 |
| `lf52-271a04aa` | official_dev |  | click | 10 | 32 81 60 71 205 148 244 109 164 225 |
| `lp85-305b61c3` | official_dev |  | click | 8 | 17 38 31 16 41 60 26 159 |
| `ls20-9607627b` | official_dev | ✓ | keyboard | 7 | 22 123 73 84 96 192 186 |
| `m0r0-492f87ba` | official_dev |  | keyboard_click | 6 | 30 111 203 26 500 237 |
| `r11l-495a7899` | official_dev |  | click | 6 | 22 33 51 26 52 49 |
| `re86-8af5384d` | official_dev |  | keyboard_click | 8 | 26 42 86 108 189 139 424 241 |
| `s5i5-18d95033` | official_val |  | click | 8 | 20 89 106 54 162 38 86 83 |
| `sb26-7fbdac44` | official_val |  | keyboard_click | 8 | 18 28 18 19 31 23 58 18 |
| `sc25-635fd71a` | official_dev |  | keyboard_click | 6 | 36 6 32 83 143 50 |
| `sk48-d8078629` | official_val |  | keyboard_click | 8 | 61 177 101 103 230 181 125 92 |
| `sp80-589a99af` | official_dev |  | keyboard_click | 6 | 39 58 25 148 96 152 |
| `su15-1944f8ab` | official_val |  | click | 9 | 22 42 26 115 36 31 8 40 41 |
| `tn36-ef4dde99` | official_dev |  | click | 7 | 32 72 26 40 30 55 62 |
| `tr87-cd924810` | official_dev |  | keyboard | 6 | 54 58 40 45 71 146 |
| `tu93-0768757b` | official_dev | ✓ | keyboard_click | 9 | 19 16 34 42 123 80 14 23 111 |
| `vc33-5430563c` | official_dev | ✓ | click | 7 | 7 18 44 61 131 34 152 |
| `wa30-ee6fef47` | official_dev |  | keyboard | 9 | 71 119 183 98 368 68 79 442 415 |

## 社区 249 个（arc-interactive）

- **来源：** https://github.com/theredbluepill/arc-interactive，固定在提交 `b6cbf21a36f029882b72c702bbbfd45455ce330d`。
- **已排除：** 原仓库里混入的官方游戏副本（ft09、ls20、vc33），避免开发集和官方集重叠。
- **输入方式：** keyboard 140、click 70、keyboard_click 39。
- **分类：** 按原仓库人工标注的类别，归并成下面的大类；每个游戏的原始类别也一并列出。

| 大类 | 数量 |
|---|---|
| 机关谜题（钥匙、开关、传送、冰面） | 40 |
| 生存 / 时机 / 危险 | 29 |
| 图案 / 拼贴 / 对称 | 29 |
| 推箱 / 环境操作 | 24 |
| 逻辑 / 推理 / 排序 | 23 |
| 模拟 / 场 | 20 |
| 图 / 电路 | 18 |
| 路径 / 覆盖 / 资源 | 17 |
| 记忆 / 隐藏信息 / 探索 | 16 |
| 移动与协调 | 13 |
| 拓扑 / 几何 | 10 |
| 收集 | 6 |
| 教程 / 移动基础 | 4 |

### 机关谜题（钥匙、开关、传送、冰面）

| 游戏 | 原始类别 | 输入 | 网格 | 关卡 | 说明 |
|---|---|---|---|---|---|
| `tc01` | Puzzle / Fields | keyboard | 16×16 | 5 | Conveyor layer: after each resolved move, you are pushed one more step by the arrow at your destination cell (arrows in level data). |
| `kb01` | Puzzle / Key | keyboard | 10×10 | 5 | Key leash: yellow key must stay within Manhattan R of you or lose — positional rule, not fg01 sight fog. |
| `pw01` | Puzzle / Logic | keyboard | 10×10 | 5 | Two yellow pads + crates (pw01): each pad needs a crate on it, then touch goal — not a waypoint order puzzle (tw01 / cq01) or weight sum (wp01). |
| `kn01` | Puzzle / Movement | keyboard | 16×16 | 5 | Knight’s courier: ACTION1–4 use L-shaped knight hops from the active bank; ACTION5 toggles between two banks (eight directions total). |
| `bl01` | Puzzle / Phase | keyboard | 10×10 | 5 | Phase-through (one tile): ACTION5 arms next cardinal move to walk through one wall cell — not rc01 warp to start. |
| `dp01` | Puzzle / Planes | keyboard | 10×14 | 5 | Spatial A/B planes (not dv01 timeline): ACTION5 swaps plane_a vs plane_b wall layers on the same board; HUD yellow top cue = overlay planes. |
| `dl01` | Puzzle / Planning | keyboard | 12×12 | 5 | Delay line: moves queue (max 3); each step runs the oldest pending move then enqueues the current direction; ACTION5 clears the queue. |
| `em01` | Puzzle / Planning | keyboard | 10×10 | 5 | Echo move: every second step repeats your previous cardinal delta. |
| `dr01` | Puzzle / Rush | keyboard | 10×10 | 5 | Drill (wall + rush): ACTION5 destroys one orth-adjacent grid wall; strict step budget — not mx01 melt budget style or dg01 mud dig. |
| `rc01` | Puzzle / Teleport | keyboard | 10×10 | 5 | Recall to spawn: ACTION5 jumps you to level start (once) — teleport, not bl01 wall phasing. |
| `dv01` | Puzzle / Timeline | keyboard | 10×10 | 5 | Timeline walls (not dp01 planes): ACTION5 swaps which wall set is solid (t0 / t1); HUD red top cue = branch timeline. |
| `or01` | Puzzle / Timing | keyboard | 10×10 | 5 | Orbit keys: keys rotate clockwise around a pillar each step; pick up when orth-adjacent, then reach the goal. |
| `co01` | Puzzle Mechanics | keyboard | 10×10 | 5 | Color gate: recolor pads set active hue; only matching doors open. |
| `dt01` | Puzzle Mechanics | keyboard | 8-10 | 5 | Detour: cyan waypoint arms only when you enter it from the required side (waypoint_enter_from: n/ e/ s/ w); then the yellow goal counts. |
| `ex01` | Puzzle Mechanics | keyboard | 8-10 | 5 | Exit hold: stand on green exit pad and repeat ACTION5 hold_frames times to clear. |
| `ex02` | Puzzle Mechanics | keyboard | 8-10 | 5 | Sliding exit hold: ACTION5 on pad increments hold; moving only decays hold by 1 (not full reset). |
| `ex03` | Puzzle Mechanics | keyboard | 8×8 | 5 | Moving exit hold: like ex02, but the green pad slides on a timer (pad_period, pad_delta). |
| `fs01` | Puzzle Mechanics | keyboard | 8-10 | 5 | Floor switches. Step on every yellow pressure plate (any order) to open the gray door, then reach the green goal. |
| `fs02` | Puzzle Mechanics | keyboard | 8-10 | 5 | Floor switches OR: orange plates — stepping on any one opens the door (redundant branches, not a count). |
| `fs03` | Puzzle Mechanics | keyboard | 8-10 | 5 | Floor switches k-of-n: step on k distinct yellow plates (any order); k = required_plates in level data (corner HUD shows k ticks). Extras optional. |
| `gr01` | Puzzle Mechanics | keyboard | 8-10 | 6 | Gravity: after each move, one auto-step down (if clear). Lose if the goal is unreachable (BFS with same rules), confirmed on two consecutive steps. Dense gray… |
| `ic01` | Puzzle Mechanics | keyboard | 8-10 | 5 | Ice slide: one movement action = slide in a straight line until the next cell would be OOB, a wall, or a red hazard (you stop before entering those). Yellow go… |
| `ic02` | Puzzle Mechanics | keyboard | 8-10 | 5 | Torus ice slide: wraps at grid edges until a wall/hazard stops you. |
| `ic03` | Puzzle Mechanics | keyboard | 8-10 | 5 | Capped slide: each move travels at most slide_cap cells (level.data). |
| `mo01` | Puzzle Mechanics | keyboard | 8-10 | 5 | Momentum: need ≥2 steps in a row before changing direction; early turn = lose. |
| `mx01` | Puzzle Mechanics | keyboard | 10×10 | 5 | Melt (wall erosion): ACTION5 removes one adjacent wall tile up to a melt budget; open a path to exit — walls only, not dg01 mud. |
| `nw01` | Puzzle Mechanics | keyboard | 8-10 | 5 | Arrow tiles: stepping onto one queues its direction; your next move uses that vector instead of your key (then it clears). Orange HUD = forced move queued. |
| `nw02` | Puzzle Mechanics | keyboard | 8-12 | 5 | Vector arrows: arrow tiles add to a pending (dx,dy); next move executes the sum clamped to one step. |
| `nw03` | Puzzle Mechanics | keyboard | 8-12 | 5 | Sticky vector arrows: pending impulse is only consumed after a successful move (blocked moves keep the queue). |
| `rf01` | Puzzle Mechanics | keyboard | 8-10 | 5 | Mirror half-plane: on x >= mid, left/right inputs are swapped. |
| `tp01` | Puzzle Mechanics | keyboard | 8-10 | 5 | Symmetric teleporters: either end of a pair warps to the other (HUD: ↔ + pair ticks). Reach the yellow goal. |
| `tp02` | Puzzle Mechanics | keyboard | 8-10 | 5 | One-way teleporters (directed_pairs): only the source tile warps; orange HUD arrows mark sources + exit direction. |
| `tp03` | Puzzle Mechanics | keyboard | 8-10 | 5 | Single-use links: each pair vanishes after one warp (HUD: remaining pair ticks + burn flash on landing). |
| `ul01` | Puzzle Mechanics | keyboard | 8×8 | 5 | Pick up the key to unlock the door and advance. |
| `ul02` | Puzzle Mechanics | keyboard | 8-12 | 5 | Two-key unlock: key A before door A and key B before door B; wrong door order loses. |
| `ul03` | Puzzle Mechanics | keyboard | 10×10 | 5 | Master key: both silver keys reveal gold; gold opens both doors. |
| `wk01` | Puzzle Mechanics | keyboard | 8-10 | 5 | Weak floor: brown tiles collapse to holes after you leave; holes are lethal. HUD turns red on hole death. |
| `zq01` | Puzzle Mechanics | keyboard | 8-10 | 5 | Zone timer: blinking red hazard cells toggle on a fixed period (period, hazard_cells). |
| `zq02` | Puzzle Mechanics | keyboard | 8-10 | 5 | Dual-phase hazards: two independent blinking hazard sets with different period / phase offset in data. |
| `zq03` | Puzzle Mechanics | keyboard | 8-10 | 5 | Synced hazard masks: one safe beat per cycle where both fields are off (mask_a / mask_b). |

### 生存 / 时机 / 危险

| 游戏 | 原始类别 | 输入 | 网格 | 关卡 | 说明 |
|---|---|---|---|---|---|
| `rr01` | Dynamic / Physics | keyboard_click | 12×12 | 7 | Ramp rollout: ball keeps a heading; ACTION5 steps one cell (ramps rotate CW first); ACTION6 toggles a ramp on a free cell. Reach the green exit. |
| `wb01` | Dynamic / Walls | keyboard | 16×16 | 5 | Wall belt: marked wall segments on a row shift east each step (wrap); crushed if a wall lands on you. |
| `hz01` | Hazard / Growth | keyboard | 10×10 | 5 | Hazard growth: red hazards spread orthogonally every M steps. |
| `gl01` | Hazard / Path | keyboard | 10×10 | 5 | Revisit rule: 2 free, 3rd loses — same cell third visit loses (vs vp01 one-and-done). |
| `tr01` | Hazard / Path | keyboard | 10×10 | 5 | Revisit rule: time-decay — after first visit, cell expires in ttl world steps; standing there when expired loses (not simple visit-count like gl01). |
| `vp01` | Hazard / Path | keyboard | 10×10 | 5 | Revisit rule: one visit only — second entry to any cell loses (path hazard; not cq01/tw01 ordering or pw01 plates). |
| `fb01` | Hazard / Push | keyboard | 12×12 | 5 | Fuse bombs: push bombs with countdown; blast removes adjacent weak walls and loses if you are adjacent. |
| `cf01` | Hazard / Time | keyboard | 10×10 | 5 | Revisit rule: cooldown — re-enter a cell before F world steps since leave loses (cooldown gate, not tr01 TTL decay). |
| `tg01` | Survival | keyboard | 12×12 | 5 | Tag evasion: chaser moves every other step; survive T steps or reach a safe zone. |
| `rh01` | Survival / Hazard | keyboard | 10×10 | 5 | Advancing band: lethal horizontal band shifts one row every N moves — sweeps down grid, not lf01 stationary blink row. |
| `vi01` | Survival / Hazard | keyboard | 12×12 | 5 | Infection: plague spreads every K steps; cyan vaccine then green exit. |
| `tm01` | Survival / Resource | keyboard | 10×10 | 5 | Twin decay meters (not bt01): oxygen + heat each tick down; cyan / magenta pads refill respective bars; survive T steps. HUD shows two bar rows + time-left row. |
| `fw01` | Survival / Simulation | keyboard_click | 24×24 | 5 | Wildfire: fire spreads on a timer; ACTION6 splashes water (3×3); standing on fire (walk or spread) loses; reach the green exit. |
| `hd01` | Survival / Timing | keyboard | 16×16 | 5 | Heat front: a heat band advances south every N steps; ACTION5 on a magenta station charges temporary immunity; reach the goal. |
| `sg01` | Survival / Timing | keyboard | 8×8 | 5 | Signal lock: sweeping cursor; ACTION5 in the green window scores; miss shrinks the window. |
| `sv01` | Survival / Timing | keyboard | 8-24 | 5 | Survival game. Manage hunger and warmth. Green food restores hunger; orange warm zones stop warmth loss. Survive 60 frames to advance. |
| `sv02` | Survival / Timing | keyboard | 8-24 | 5 | Shelter survival: warmth decay pauses only inside magenta shelter zones; hunger rules unchanged; survive 60 steps per level. |
| `sv03` | Survival / Timing | keyboard | 8-24 | 5 | Dual shelters: yellow pauses hunger decay, magenta pauses warmth; alternate shelter types on entry. |
| `wm01` | Survival / Timing | click | 32×32 | 5 | Whack-a-Mole! Click moles before they escape. Meet checkpoint requirements or lose. |
| `wm02` | Survival / Timing | click | 32×32 | 5 | Lane moles: moles spawn in rotating lane columns; wrong-lane click while a mole is up costs a life. |
| `wm03` | Survival / Timing | click | 32×32 | 5 | Decoy moles: gray decoys in wrong lanes; whack decoy = lose life. |
| `lf01` | Timing | keyboard | 10×10 | 5 | Blinking row hazard: one fixed row toggles lethal/safe every P steps — on/off stripe, not rh01 advancing band. |
| `pj01` | Timing / Deflect | keyboard_click | 14×14 | 5 | Bolt bounce: bolt steps each frame; ACTION6 toggles / vs \\ mirrors; sink the bolt without getting hit. |
| `pj04` | Timing / Deflect | keyboard_click | 14×14 | 7 | Fixed shooter: ACTION5 bolt; ACTION6 places mirrors (minimal avatar). |
| `ml04` | Timing / Laser | keyboard_click | 10×10 | 7 | Stepped bolt: ACTION5 advances laser one cell per press; ACTION6 cycles fixed mirror tiles / → \\ → empty (no placement inventory). Optional emit_dir in level… |
| `wm04` | Timing / Reflex | click | 32×32 | 7 | Whack: wrong click shortens the timer. |
| `rb01` | Timing / Rhythm | keyboard | 11×11 | 5 | Global movement beat: all cardinal moves only register every B world steps — not tf01 where only orange crossing tiles are phase-gated. |
| `sg04` | Timing / Rhythm | click | 12×12 | 7 | Dual commit: two-arm timing like sg01. |
| `tf01` | Timing / Traffic | keyboard | 14×14 | 5 | Local crossing phase: orange marked cells may be entered only on green phase of a short cycle — other floor always walkable (unlike rb01 global beat). |

### 图案 / 拼贴 / 对称

| 游戏 | 原始类别 | 输入 | 网格 | 关卡 | 说明 |
|---|---|---|---|---|---|
| `jw01` | Pattern | keyboard_click | 10×10 | 5 | Jigsaw swap: ACTION6 on magenta trigger swaps two authored rectangles; push boxes to the goal. |
| `eg01` | Pattern / Click | click | 6×6 | 7 | Hamming target: ACTION6 cycles cell color; win when distance to target ≤ D. |
| `ng04` | Pattern / Layer | click | 12×12 | 7 | Multi-layer cycle path edit (ng01 family). |
| `sf04` | Pattern / Stencil | keyboard_click | 16×16 | 7 | Rotate stencil: ACTION5 spins L-tromino mask; ACTION6 paints. |
| `gp01` | Pattern Puzzles | click | 8×8 | 5 | Per-cell paint to match hints: ACTION6 flips only the clicked cell yellow/off to match gray template — no neighbor kernel (lo01 / lo05). |
| `gp02` | Pattern Puzzles | click | 8×8 | 5 | Grid paint erase: floor starts fully painted; ACTION6 erases yellow; leave paint only on gray hint cells. |
| `gp03` | Pattern Puzzles | click | 8×8 | 5 | Three-state grid paint: cycle cell colors with ACTION6; match per-cell goal palette in data. |
| `gp04` | Pattern Puzzles | click | 8×8 | 7 | Blob cap: paint like gp01 but also keep ≤ K orthogonal yellow components. |
| `lo01` | Pattern Puzzles | click | 3×3–5×5 | 5 | Orthogonal Lights Out: ACTION6 toggles self + 4 orth neighbors; clear all on — not gp01 hint paint or lo05 knight kernel. |
| `lo02` | Pattern Puzzles | click | 4×4–6×6 | 5 | Torus Lights Out: ACTION6 toggles a cell and orthogonal neighbors with edge wrap; walls block toggles on their cells. |
| `lo03` | Pattern Puzzles | click | 4×4–6×6 | 5 | Diagonal torus Lights Out: ACTION6 toggles cell + 8 neighbors with wrap (lo02 + diagonals). |
| `lo04` | Pattern Puzzles | click | 5–6 | 7 | Diagonal Lights Out: ACTION6 toggles cell + four diagonals only. |
| `lo05` | Pattern Puzzles | click | 8×8 | 7 | Knight kernel: ACTION6 toggles self + all chess-knight L-step neighbors — not lo01 orth neighborhood or gp01 single-cell paint. |
| `pt01` | Pattern Puzzles | click | 64×64 | 5 | Pattern rotation puzzle. Click tiles to rotate them 90° clockwise and match the target pattern. |
| `pt02` | Pattern Puzzles | click | 64×64 | 5 | Row/column rotate: ACTION6 rotates a full row or column of 3×3 tiles (nearest axis wins). |
| `pt03` | Pattern Puzzles | click | 64×64 | 5 | Band lock: odd levels rotate rows only, even levels columns only (difficulty). |
| `pt04` | Pattern Puzzles | click | 8×8 | 7 | Tile swap: two ACTION6 picks swap 1×1 colors; match the target grid. |
| `pt05` | Pattern Puzzles | click | 8×8 | 7 | Slide row: click yellow header (x=0) to cycle that row’s colors left; match key. |
| `qr01` | Pattern Puzzles | click | 8×8 | 5 | Quad twist: ACTION6 rotates the 2×2 block of tiles anchored at the click clockwise; match the target pattern. |
| `qr04` | Pattern Puzzles | click | 8×8 | 7 | 3×3 twist: ACTION6 rotates a 3×3 color block clockwise (qr01 kernel). |
| `sf01` | Pattern Puzzles | keyboard | 64×64 | 5 | Stencil paint: move a 3×3 stencil with ACTION1–4; ACTION5 paints all non-wall cells under it; match gray hints. |
| `sy01` | Pattern Puzzles | click | 11×11 | 5 | Mirror Maker. Mirror the pattern from the left side onto the right side. Create a perfect reflection! |
| `sy02` | Pattern Puzzles | click | 11×11 | 5 | Staggered mirror: mirror targets use half-row offset (mirror_stagger in data). |
| `sy03` | Pattern Puzzles | click | 11×11 | 5 | Vertical mirror: template above divider y=5, build mirrored copy below (sy02 geometry rotated). |
| `sy04` | Pattern Puzzles | click | 11×11 | 7 | Diagonal mirror: template on i<j; build matching dots on x>y across y=x. |
| `wr01` | Spatial / Rotation | keyboard | 12×12 | 5 | World rotation: entire level rotates 90° CW every N steps; ACTION5 braces to skip the next rotation (budget). |
| `dm01` | Tiling | click | 8×8 | 5 | Domino cover: ACTION6 toggles dominoes on valid pairs; cover all marked cells once. |
| `dm04` | Tiling | click | 8×8 | 7 | L-tromino cover: three ACTION6 picks forming an L in a 2×2 cover marked cells. |
| `cu01` | Tiling / Click | keyboard_click | 10×10 | 7 | Cover yellow: ACTION5 domino vs L-tool; ACTION6 places. |

### 推箱 / 环境操作

| 游戏 | 原始类别 | 输入 | 网格 | 关卡 | 说明 |
|---|---|---|---|---|---|
| `dg01` | Environmental | keyboard | 10×10 | 5 | Dig mud only: ACTION5 clears one orth-adjacent brown mud cell (budget); does not break dr01/mx01 stone walls. |
| `mb01` | Environmental | keyboard | 10×10 | 5 | Magnet crates: after each move, metal crates slide one cell toward you when free. |
| `op01` | Environmental | keyboard | 8-10 | 5 | One-push crate: each crate may be pushed at most once; second push loses. |
| `rn01` | Environmental / Graph | keyboard_click | 12×12 | 5 | Rope anchors: ACTION6 two orth-adjacent anchors over water places a walkable rope on the middle cell. |
| `ci01` | Environmental Manipulation | keyboard | 8–12 | 5 | Crate ice: player moves normally; pushed crates slide until wall, crate, or mud. |
| `cr01` | Environmental Manipulation | keyboard_click | 16×16 | 5 | Creek crossing: limited ACTION6 planks on river; planks break when you leave; reach the goal. |
| `pb01` | Environmental Manipulation | keyboard | 8-10 | 5 | Single-crate sokoban: one box + one pad + normal walking; not multi-crate sk01 complexity. Step limit exceeded = lose. |
| `pb02` | Environmental Manipulation | keyboard | 8-10 | 5 | Two crates, two yellow goals; push both blocks onto pads (sk01-style). |
| `pb03` | Environmental Manipulation | keyboard | 8-10 | 5 | Decoy orange pad — pushing a crate onto it loses; real goals stay yellow. |
| `rz01` | Environmental Manipulation | keyboard | 12×12 | 5 | Rush grid: push 1×2 cars along their axis like sokoban; clear the exit car. |
| `sk01` | Environmental Manipulation | keyboard | 8-12 | 5 | Classic multi-crate sokoban: several crates/pads; green = placed; harder topology than pb01 one-box intro. Step limit exceeded = lose. |
| `sk02` | Environmental Manipulation | keyboard | 8-12 | 5 | Sliding crate sokoban: after a successful push, if the cell beyond the crate is empty, the crate slides one more step. |
| `sk03` | Environmental Manipulation | keyboard | 8-12 | 5 | Sticky mud sokoban: sliding crate chain stops on mud floor (mud tag); mud is walkable. |
| `tb01` | Environmental Manipulation | keyboard_click | 24×24 | 5 | Bridge Builder. Multi-island routes (waypoints + optional reef clusters); bridge open water (ACTION6), walk island-to-island to the green goal. Later levels ad… |
| `tb02` | Environmental Manipulation | keyboard_click | 24×24 | 5 | Bridge decay: like bridge builder, but a bridge sprite is removed when you leave that water cell. |
| `tb03` | Environmental Manipulation | keyboard_click | 24×24 | 5 | Reef growth: like bridge decay, plus random rock spawns on water every M steps (reef_every). |
| `wl01` | Environmental Manipulation | keyboard_click | 32×32 | 5 | Wall craft: maroon chasms are deadly—bridge them with ACTION6 (place/remove, budget); red hazards are separate one-touch deaths. |
| `as01` | Logistics | keyboard | 12×12 | 5 | Assembly fetch: ACTION5 pickup/drop tagged parts; deliver to matching workstations in order from level.data. |
| `dd01` | Logistics | keyboard | 48×48 | 5 | Routed delivery: colored parcels → matching customers (same tile color); each order has an SLA deadline (late = lose) plus a global step budget. ACTION5 picks… |
| `wp01` | Logistics | keyboard | 10×10 | 5 | Weight plate (wp01): one tile; sum of orth-adjacent weights (you=1, crate=2) = W unlocks — not pw01 dual yellow pads. |
| `sk04` | Manipulation | click | 10×10 | 7 | Fixed avatar, click-pull: ACTION6 on an orth-adjacent crate pulls it one step toward you; ACTION1–4 no-op — not pb04’s remote ACTION5 winch. |
| `sw01` | Manipulation | keyboard | 10×10 | 5 | Swap positions: ACTION5 swaps you with the orth-adjacent magenta partner — no terrain removal (dg01 / dr01). |
| `tk01` | Manipulation | keyboard | 10×10 | 5 | Telekinetic tug: ACTION5 pulls the nearest crate one greedy Manhattan hop toward you; else push sokoban-style. |
| `pb04` | Push / Minimal | keyboard | 10×10 | 7 | No player movement: cyan winch tile; ACTION5 pulls the nearest crate one step along the level’s (dx,dy) toward the winch; cover all yellow pads with crates — n… |

### 逻辑 / 推理 / 排序

| 游戏 | 原始类别 | 输入 | 网格 | 关卡 | 说明 |
|---|---|---|---|---|---|
| `rs02` | Cognitive Flexibility | keyboard | 8-16 | 5 | Dual safe: collect targets matching either color of the active pair (dual_pairs); after all pairs have been safe, any remaining target is allowed. |
| `rs03` | Cognitive Flexibility | keyboard | 8-16 | 5 | Forbidden stripe: HUD shows a color that is never collectible; clear all other targets. |
| `rs01` | Cognitive Flexibility / Rule Switching | keyboard | 8-16 | 5 | Rule Switcher. Collect colored targets that match the signpost color at top. Wrong color = lose. After all colors cycle through as safe, collect remaining targ… |
| `ph04` | Logic / Algebra | keyboard | 8×1 row | 7 | Mod row step: ACTION5 applies neighbor-sum mod R on a cyclic row. |
| `sr01` | Logic / Algebra | keyboard | 12×12 | 5 | Aura XOR: RGB zones toggle aura bits; wash clears; door opens when aura matches the target triple. |
| `hn01` | Logic / Classic | keyboard | 10×12 | 5 | Tower of Hanoi: ACTION1–3 pick peg; ACTION5 pick/drop top disk; stack all on the right peg. |
| `hn04` | Logic / Classic | keyboard | 10×12 | 7 | 4 peg Hanoi: ACTION1–4 select peg; ACTION5 pick/drop. |
| `cw01` | Logic / Click | click | 12×12 | 7 | 2-SAT literals: ACTION6 toggles a variable; satisfy all clauses. |
| `cz01` | Logic / Click | keyboard_click | 8×8 | 7 | Row XOR pick: ACTION1–4 select row 0–3; ACTION6 flips all bits in that row; match target. |
| `ms04` | Logic / Deduction | click | 8×8 | 7 | Edge-adjacent clues: ACTION6 flags mines. |
| `ng01` | Logic / Deduction | keyboard_click | 8×8 | 5 | Nonogram lite: ACTION1–4 move the cursor; ACTION6 (click) cycles empty / filled / mark on the clicked cell (cursor jumps to click); filled cells must match the… |
| `lk01` | Logic / Lock | keyboard | 10×10 | 5 | Lock tumblers: while the door exists ACTION1–3 cycle digits; when matched the door vanishes; then 1-4 move to goal. |
| `rs04` | Logic / Phase | keyboard_click | 10×10 | 7 | XOR-safe collect: phase bits pick which target color may clear. |
| `qt01` | Logic / Split | keyboard | 10×10 | 5 | Quantum split: magenta splitter duplicates offset; cyan observer collapses; both ghosts cannot hit walls. |
| `vt01` | Logic / Weights | keyboard | 12×12 | 5 | Vote plates: sum of crate weights on the plate mod M opens the door; reach the goal. |
| `tw01` | Ordering | keyboard | 10×10 | 5 | Two-stop order: orange waypoint before green goal validates — only two landmarks, fixed order (not cq01 many ring markers). |
| `lp01` | Ordering / Lexical | keyboard | 9×11 | 5 | Letter path: visit letter pads in the authored word order; wrong letter loses. |
| `sl01` | Permutation / Puzzle | keyboard | 3×3 | 5 | Slide puzzle: swap the hole with adjacent tiles until the board matches the goal; step limit. |
| `sq02` | Sequencing | click | 12×12 | 5 | Shrinking queue: like sequencing, but only the current expected color block is eligible/visible until unlocked; wrong order resets progress. |
| `sq03` | Sequencing | click | 12×12 | 5 | Dual queue: two color queues; ACTION6 on either head; wrong tap resets both. |
| `sq01` | Sequencing / Ordering | click | 12×12 | 5 | Sequencing. Click colored blocks in the correct order. Follow the sequence shown at the top! |
| `sq04` | Sequencing / Ordering | click | 12×12 | 7 | LIFO sequencing: clear blocks in stack order (last in HUD first), sq01-style clicks. |
| `sq05` | Sequencing / Ordering | click | 12×12 | 7 | Double-tap lock: FIFO order; each block needs two consecutive correct taps to clear. |

### 模拟 / 场

| 游戏 | 原始类别 | 输入 | 网格 | 关卡 | 说明 |
|---|---|---|---|---|---|
| `ph01` | Field / Math | click | 24×24 | 5 | Phase interference: phases 0–3; ACTION6 increments a cell; ACTION5 applies (self + Σ orth neighbors) mod 4 on non-walls; match marked targets. |
| `ph02` | Field / Math | click | 24×24 | 5 | Phase multiply: ACTION5 applies multiply-style update with orthogonal neighbors mod N (mod_n in data). |
| `ph03` | Field / Math | click | 24×24 | 5 | XOR neighbor step (ph02 variant): ACTION5 XORs cell with orth neighbors mod N. |
| `df01` | Field / Simulation | keyboard_click | 16×16 | 5 | Heat diffusion: hot/cold sources; temperature diffuses each step; goal needs band hit; ACTION6 vents 3×3. |
| `df04` | Field / Simulation | keyboard_click | 10×10 | 7 | Diffuse only: ACTION5 heat step; ACTION6 3×3 vent; probe in band. |
| `ih01` | Field / Simulation | keyboard_click | 10×10 | 7 | Heaters: ACTION5 global chill; ACTION6 toggles heater pads. |
| `ox01` | Field / XOR | keyboard | 10×10 | 5 | XOR hazards: red vs magenta hazard layers alternate collidable each step. |
| `gc01` | Growth / Routing | keyboard | 14×14 | 5 | Coral creep: coral spreads to empty neighbors every M steps (blocking, non-lethal); reach the goal. |
| `ab01` | Simulation | click | 10×10 | 7 | Win: stable + (total grains mod P) == R — modular target, not sp01 fixed sum or sp04 sinks. |
| `av01` | Simulation | keyboard | 10×10 | 5 | Avalanche: rocks fall after each move; crushed loses. |
| `ll01` | Simulation | click | 32×32 | 6 | Generations lock: Conway Life; ACTION6 toggles cells (budget); ACTION5 advances one generation; after exactly N steps the live set must equal the target. Level… |
| `ll04` | Simulation | keyboard_click | 12×12 | 7 | Torus Life: Conway on a torus; ACTION6 toggles; ACTION5 steps. |
| `sb01` | Simulation | keyboard | 10×10 | 5 | Sand fall: sand piles fall like rocks; crushed loses. |
| `sp01` | Simulation | click | 12×12 | 5 | Win: stable + Σ grains = target_sum (classic sandpile). ACTION6 adds a grain; ≥4 topples to neighbors; no sinks (sp04) / no mod goal (ab01). |
| `sp04` | Simulation | click | 12×12 | 7 | Win: stable + same target-sum rule as sp01, but grains that topple into cyan sink cells are removed (mass loss; not ab01 mod P/R). |
| `fe02` | Simulation / Abstract | keyboard | 8×8 | 7 | Vote + ratify: ACTION1–4 tally; ACTION5 picks the leading rule (tie → higher index) and applies one (a,b,c) tick; need ratifications / level; keep a,b,c in 1..… |
| `fi01` | Simulation / Hazard | keyboard | 14×14 | 5 | Firebreak: fire spreads each step; ACTION5 places a blue break tile blocking spread. |
| `tc04` | Simulation / Routing | keyboard_click | 10×10 | 7 | Conveyor: ACTION5 moves packages on arrows; ACTION6 rotates arrow. |
| `zm04` | Simulation / Spread | keyboard_click | 12×12 | 7 | Strains: ACTION5 switch+spread; ACTION6 blocks a cell. |
| `at01` | Simulation / Steering | keyboard | 16×16 | 5 | Ant trail: ACTION5 drops pheromone; ant follows strongest adjacent scent toward the exit hole. |

### 图 / 电路

| 游戏 | 原始类别 | 输入 | 网格 | 关卡 | 说明 |
|---|---|---|---|---|---|
| `ck04` | Circuit / Logic | click | 14×14 | 7 | Directed wire: each tile’s arrow rotates with ACTION6; reach out. |
| `kv01` | Circuit / Logic | keyboard | 8×8 | 5 | Voltage ladder: cycle three series resistors (1/2/4 Ω); ACTION5 checks probe voltages vs targets. |
| `kv04` | Circuit / Logic | keyboard | 8×8 | 7 | Two ladders: ACTION1–3 / ACTION4 cycle branches; ACTION5 verify. |
| `cs01` | Graph / Click | click | 8×8 | 7 | Vertex cover: toggle cyan vertices so every edge touches a selected vertex within budget. |
| `ct01` | Graph / Click | click | 8×8 | 7 | Independent cover: no adjacent selections; every gray mark must be selected or orth-adjacent to one. |
| `cv01` | Graph / Click | click | 8×8 | 7 | List coloring: ACTION6 cycles allowed colors; neighbors differ. |
| `cx01` | Graph / Click | click | 10×10 | 7 | Disconnect cut: toggle gates until S and T split. |
| `ck02` | Graph / Logic | keyboard_click | 24×24 | 5 | Circuit junction: ACTION6 toggles wire; ACTION5 tests path; on junction cells, no right turn relative to incoming wire direction. |
| `ck03` | Graph / Logic | click | 24×24 | 5 | Checkpoint wire (ck02 variant): cyan checkpoint must lie on successful test path. |
| `rp01` | Graph / Logic | click | 32×32 | 5 | Relay pulse: ACTION6 toggles relays; ACTION5 fires from the source; orthogonal relay chain must light every lamp (adjacent to a visited relay). |
| `rp02` | Graph / Logic | click | 32×32 | 5 | Pulse depth: relays + amplifiers reset hop budget (max_pulse_depth); light all lamps. |
| `rp03` | Graph / Logic | click | 32×32 | 5 | Splitter relay (rp02 variant): T-relays fork pulses to both arms. |
| `pd01` | Graph / Plumbing | click | 10×10 | 5 | Pipe drop: ACTION6 cycles empty → H → V pipe; connect cyan source to yellow sink. |
| `pd04` | Graph / Plumbing | click | 12×12 | 7 | Plus junction: toggle full + open/closed; connect flow. |
| `pu01` | Graph / Plumbing | click | 16×16 | 5 | Pipe twist: ACTION6 toggles horizontal vs vertical pipe on a cell; connect cyan source to yellow sink with orthogonal flow. |
| `pu04` | Graph / Plumbing | click | 16×16 | 7 | Tee pipes: ACTION6 cycles H / V / T; connect source→sink. |
| `bp01` | Graph / Power | keyboard | 14×14 | 5 | Tower power: stand on each light-blue tower and use ACTION5 to power it (sprite turns green); power all towers, then reach the yellow goal. |
| `rp04` | Graph / Relay | keyboard_click | 12×12 | 7 | Relay pulse: ACTION6 toggles relays; ACTION5 floods from source. |

### 路径 / 覆盖 / 资源

| 游戏 | 原始类别 | 输入 | 网格 | 关卡 | 说明 |
|---|---|---|---|---|---|
| `bd01` | Coverage / Path | keyboard | 5-8 | 5 | Revisit rule: forbidden — any cell entered twice loses (stricter than visit-all va01). Red marks cells you left; HUD flashes red on revisit fail. |
| `hm01` | Coverage / Path | keyboard | 3-6 | 5 | Revisit rule: forbidden — Hamiltonian: every open cell exactly once (same strictness as bd01 but win = full grid, not goal-only). |
| `va01` | Coverage / Path | keyboard | 4-8 | 5 | Revisit rule: allowed — cover every walkable floor at least once; revisits OK. Goal: full coverage, not Hamiltonian. |
| `va02` | Coverage / Path | keyboard | 4-8 | 5 | Visit every non-hazard floor cell; red hazard cells never need coverage. Visited cells stay green (trail). Three blocked moves (OOB / wall / hazard) on a level… |
| `va03` | Coverage / Path | keyboard | 4-8 | 5 | Visit yellow waypoints in order with a fixed move budget (shortest solve length). OOB/wall don’t use moves. Waste the budget or mistime the final step = lose. |
| `ep01` | Path / Budget | keyboard | 12×12 | 5 | Pain budget: entering a cell costs its visit count; exceed the global budget to lose. |
| `fl04` | Path / Click | keyboard_click | 12×12 | 7 | Capped path: connect A→B with click path length ≤ max_len. |
| `lw04` | Path / Click | keyboard_click | 12×12 | 7 | Corner budget: connect A→B with length and turn caps. |
| `in01` | Path / Hazard | keyboard | 10×10 | 5 | Ink trail: leaving a cell drops blocking ink. |
| `fl01` | Path / Numberlink | keyboard_click | 12×12 | 5 | Numberlink: connect numbered endpoints with paths of exact per-pair length; no overlap (pairs, length in data). ACTION6 extends/cuts at the clicked grid cell. |
| `lw01` | Path / Topology | click | 24–32 | 5 | Line weave: connect colored starts to matching ends with orthogonal paths; colors cannot share cells. |
| `lw02` | Path / Topology | click | 24–32 | 5 | Shared corridor weave: like lw01 but paths may share cells; perpendicular entry to another color’s visited cell is forbidden. |
| `lw03` | Path / Topology | click | 24–32 | 5 | Shared edges OK (lw02 variant): relaxed perpendicular corridor rule; segment edge ownership. |
| `bt01` | Resource / Movement | keyboard | 10×10 | 5 | Single charge bar (not tm01): one charge integer drops every move; cyan pads refill; 0 charge loses after the step unless you reach the goal that same step (go… |
| `au01` | Resource / Path | keyboard | 10×10 | 5 | Bonus steps: tight move budget; cyan bonus pads add steps when entered. |
| `zm01` | Territory | keyboard_click | 16×16 | 5 | Flood duel: two colors expand from seeds; ACTION5 switches the active color; ACTION6 claims a floor cell orthogonally adjacent to your region (cover a target f… |
| `cq01` | Territory / Coverage | keyboard | 10×10 | 5 | Many ordered rings: hit every orange ring marker (set), then goal — multi-stop route (not tw01’s single orange-before-green). |

### 记忆 / 隐藏信息 / 探索

| 游戏 | 原始类别 | 输入 | 网格 | 关卡 | 说明 |
|---|---|---|---|---|---|
| `bn01` | Exploration | keyboard_click | 64×64 | 5 | Beacon sweep: hidden targets; ACTION5 drops a beacon (Chebyshev radius); ghosts show only under light; ACTION6 flags a cell (wrong cell = lose). |
| `bn02` | Exploration | keyboard_click | 64×64 | 5 | Manhattan beacon: like bn01 but reveal uses L1 (Manhattan) radius; wrong ACTION6 (not on a hidden ghost) costs a step instead of instant game over. |
| `bn03` | Exploration | keyboard_click | 64×64 | 5 | Shrinking beacon (bn02 variant): each ACTION5 beacon reduces max light radius by 1. |
| `mm04` | Memory | keyboard_click | 64×64 | 7 | One peek / level: ACTION5 briefly reveals unmatched tiles. |
| `mm05` | Memory | click | 64×64 | 7 | Sticky matches: an edge-aligned matched pair blocks flips on cells that sit orthogonally next to both tiles (diagonal pairs do not). |
| `fg01` | Memory / Exploration | keyboard | 10×10 | 5 | Fog (visibility): Chebyshev radius r hides far tiles in render only — no kb01 key-distance leash rule. |
| `mm01` | Memory / Hidden State | click | 64×64 | 7 | Memory Match. Level 1 has 2 pairs, then +1 pair per level up to 8 pairs. Flip pairs of hidden tiles to find matching colors. Match all pairs to win. Time runs… |
| `mm02` | Memory / Hidden State | click | 64×64 | 5 | Memory triples: flip three tiles; clear when all three match color. |
| `mm03` | Memory / Hidden State | click | 64×64 | 5 | Memory quads (mm02 variant): flip four tiles; clear when all four match. |
| `ms01` | Memory / Hidden State | keyboard | 8-16 | 5 | Blind Sapper. Navigate a hidden minefield using deduction. Safe tiles reveal adjacent mine counts. Step on a mine = lose. Reach the goal to win. Tests working… |
| `ms02` | Memory / Hidden State | keyboard_click | 8-16 | 5 | Flag sapper: ACTION6 plants flags on hidden mines; wrong flag = lose; reach the goal. |
| `ms03` | Memory / Hidden State | keyboard_click | 8-12 | 5 | Chord sapper: clues count mines in Chebyshev radius 2; ACTION6 flags (display_to_grid). |
| `st01` | Stealth | keyboard | 16×16 | 5 | Sentry sweep: cone-vision guards; spotted = lose; ACTION5 whistles to nudge a guard forward one cell. |
| `hs01` | Stealth / Chase | keyboard | 10×10 | 5 | Hostile chase: hunter moves one toward you every two player moves (slower than es01 NPC); reach goal without capture. |
| `bn04` | Stealth / Reveal | keyboard_click | 16×16 | 7 | Line/column flash: ACTION5 axis reveal (bn01 family). |
| `sc01` | Stealth / Trail | keyboard | 14×14 | 5 | Scent + cone: guard sees eastward; high scent under LOS loses; ACTION5 masks scent briefly. |

### 移动与协调

| 游戏 | 原始类别 | 输入 | 网格 | 关卡 | 说明 |
|---|---|---|---|---|---|
| `mc01` | Coordination | keyboard | 16×16 | 5 | Tandem: two players take the same Δ each step; ACTION5 swaps which avatar is “lead” for collision resolution; both must reach their goals. |
| `fc01` | Coordination / Chain | keyboard | 14×10 | 5 | Follow chain: three tails copy prior segment positions; you and tails must reach the exit neighborhood together. |
| `ec01` | Coordination / Mirror | keyboard | 12×12 | 5 | Echo ghost: a mirror copy moves with reflected horizontal delta; wall collision for the echo blocks the whole move. |
| `es01` | Escort | keyboard | 10×10 | 5 | Co-op escort: friendly NPC steps one toward you every player move; both must reach colored goals; bump = lose — not hs01 hunter cadence. |
| `wa01` | Movement | keyboard | 12×12 | 5 | Warp line: crossing a warp band shifts +2 on its axis when clear. |
| `wg01` | Movement / Chaos | keyboard | 10×10 | 5 | Wind gust: every K steps you are pushed one cell east when free. |
| `bi01` | Movement / Chess | keyboard | 10×10 | 5 | Bishop courier: ACTION1–4 slide diagonally (NW/NE/SW/SE) until a wall blocks. |
| `rk01` | Movement / Chess | keyboard | 10×10 | 5 | Rook courier: ACTION1–4 slide orthogonally until a wall blocks. |
| `cy01` | Movement / Field | keyboard | 10×10 | 5 | Cyclic conveyor: on conveyor row, after each step slide west while free. |
| `eb01` | Movement / Field | keyboard | 10×10 | 5 | Escalator row: on the marked row, after each move you auto-slide east while free. |
| `pm01` | Movement / Meta | keyboard | 10×10 | 5 | Prime steps: cardinal moves only apply on prime step indices; composites no-op. |
| `ju01` | Movement / Planning | keyboard | 10×10 | 5 | Jump tile: landing on a jump floor moves two cells in the same direction when the skip cell is clear. |
| `ob01` | Multi-Agent | keyboard | 16×16 | 5 | Odd one out: three bodies; ACTION5 cycles which avatar ACTION1–4 moves; each reaches its pad. |

### 拓扑 / 几何

| 游戏 | 原始类别 | 输入 | 网格 | 关卡 | 说明 |
|---|---|---|---|---|---|
| `ml01` | Geometry | click | 24×24 | 5 | Mirror laser (single goal): global ACTION6 clicks place/cycle mirrors (purple = /, light magenta = \\) on any valid floor cell; ACTION5 fires a full ray; 1–4 n… |
| `ml02` | Geometry | keyboard_click | 24×24 | 5 | Dual-receptor laser: like ml01 optics but all yellow receptors in one shot; blue technician moves (1–4); ACTION6 only on cells orthogonally adjacent to technic… |
| `ml03` | Geometry | keyboard_click | 24×24 | 5 | Fragile mirrors (ml02 rules): same move + adjacent mirror play as ml02, but every mirror the beam reflects from is removed after that shot (failed shots eat op… |
| `ff01` | Precision / Topology | click | 64×64 | 5 | Flood fill: click inside closed regions to paint them yellow. Five levels mix rectangles, donut/ring, and C-bays with ramping shape count. Sq01-style click rip… |
| `ff02` | Precision / Topology | click | 64×64 | 5 | Flood unpaint: interiors start filled; ACTION6 erases a clicked enclosure; gray hints must end empty. ACTION1–4 no-op. |
| `ff03` | Precision / Topology | click | 64×64 | 5 | Limited erases (ff02 variant): capped enclosure erasers per level in data. |
| `ff04` | Precision / Topology | click | 8×8 | 7 | Gradient budget flood: grow paint from seed; total Manhattan cost to seed ≤ budget; match gray hints. |
| `pk01` | Topology / Packing | keyboard_click | 8–12 | 5 | Polyomino pack: ACTION5 toggles domino vs straight-tromino mode; ACTION6 clicks place pieces on marked cells. |
| `pk02` | Topology / Packing | click | 10×10 | 7 | Ribbon edges: two-click adjacent vertices to claim a marked unit edge; cover all. |
| `dn01` | Topology / Torus | keyboard | 16×16 | 5 | Donut wind: torus wrap east increments winding; need winding ≥ 2 on the goal to clear. |

### 收集

| 游戏 | 原始类别 | 输入 | 网格 | 关卡 | 说明 |
|---|---|---|---|---|---|
| `tt01` | Collection | keyboard | 8-24 | 3 | Collection game. Navigate grid to collect yellow targets while avoiding red hazards (static collidable cells). |
| `tt02` | Collection | keyboard | 16-24 | 3 | Patrol hazards: collect yellow targets while red patrol hazards step along authored tracks (patrols) every player step. |
| `tt03` | Collection | keyboard | 16-24 | 3 | Collector spawns: patrol collection plus new yellow targets every K steps until cap (spawn_every, target_cap). |
| `sn01` | Collection / Classic | keyboard | 12×12 | 5 | Snake collect: eat all yellow food; grow; walls and self-collision lose. |
| `rv01` | Collection / Hazard | keyboard | 8–16 | 5 | Rotating sparks: collect targets while red hazards step together in a wind direction that cycles N→E→S→W. |
| `nu01` | Collection / Order | keyboard | 10×10 | 5 | Number fuse: collect numbered tokens in descending order; wrong pickup loses. |

### 教程 / 移动基础

| 游戏 | 原始类别 | 输入 | 网格 | 关卡 | 说明 |
|---|---|---|---|---|---|
| `ez01` | Tutorial / Movement Basics | keyboard | 8×8 | 5 | Go UP to reach the target. |
| `ez02` | Tutorial / Movement Basics | keyboard | 8×8 | 5 | Go LEFT to reach the target. |
| `ez03` | Tutorial / Movement Basics | keyboard | 8×8 | 5 | Go RIGHT to reach the target. |
| `ez04` | Tutorial / Movement Basics | keyboard | 8×8 | 5 | Go DOWN to reach the target. |
