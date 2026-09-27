#!/usr/bin/env bash
# Fetch every game the harness runs and verify it against games/manifest.json.
#   official:  the 25 public ARC-AGI-3 games (Kaggle competition data; needs a Kaggle token in
#              .kaggle/access_token and the competition rules accepted; not redistributed here)
#   community: arc-interactive (MIT; 249 games once its copies of official games are left out) at the pinned commit
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d data/environment_files ]; then
  make data
fi
COMMIT=$(python3 -c "import json; print(json.load(open('games/manifest.json'))['community']['commit'])")
if [ ! -d data/community/.git ]; then
  git clone -q https://github.com/theredbluepill/arc-interactive data/community
fi
git -C data/community fetch -q origin "$COMMIT" 2>/dev/null || true
git -C data/community checkout -q "$COMMIT"
python3 scripts/games_manifest.py --check
