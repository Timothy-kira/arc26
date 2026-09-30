#!/bin/bash
# Keep playing the three tuning games (easy cd82, medium tu93, hardest wa30 by human steps) with the
# hosted Dots model, round after round, every round comparing the configurations listed in
# runs/dots3/arms (one "name|flags" per line). Results: runs/dots3/<round>/<arm>/, summary in
# runs/dots3/log.txt. Safe to start again after a container restart: it continues numbering.
cd "$(dirname "$0")/.."
D=runs/dots3; mkdir -p $D
[ -f $D/arms ] || printf 'fresh|--context fresh\nrolling|--context rolling\n' > $D/arms
GAMES=${GAMES:-cd82,tu93,wa30}
while [ ! -f $D/STOP ]; do
  R=$(date -u +%m%d-%H%M); mkdir -p $D/$R
  echo "== round $R $(git rev-parse --short HEAD) games $GAMES" >> $D/log.txt
  while IFS='|' read -r name flags; do
    [ -z "$name" ] && continue
    .venv/bin/python -m arc_agent.run --out-dir $D/$R/$name --games $GAMES --conc 3 \
      --base-url https://note3-prev-api.askdiandian.com/v1 --model dots3-note-prev \
      --api-key-file .secrets/dots_api_key --game-seconds ${GAME_SECONDS:-1800} $flags > $D/$R/$name.log 2>&1 &
  done < $D/arms
  wait
  while IFS='|' read -r name flags; do
    [ -z "$name" ] && continue
    echo "-- $name ($flags)" >> $D/log.txt
    .venv/bin/python -m arc_agent.report $D/$R/$name | tail -5 >> $D/log.txt
  done < $D/arms
done
