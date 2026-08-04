#!/usr/bin/env bash
# Export block centroids for every city so a GEE-capable machine can sample
# slope and land cover for all of them, not just the two that happened to be
# done first. Run on the server; the resulting centroid files are pulled
# locally, sampled through scripts/fetch_terrain_resistance.py, and the
# resistance files sent back.
set -uo pipefail

PY=/workspace/venvs/xichang-agent/bin/python
ROOT=/workspace/xichang-agentic-evacuation
CLAUDE=$ROOT/claude
OUT=$CLAUDE/results_resistance

cd "$CLAUDE" || exit 1
mkdir -p "$OUT"

# slug|bbox(W S E N)  -- identical bboxes to build_all_cities.sh, so the
# centroids line up with the blocks the simulation actually uses.
CITIES=(
  "chengdu|103.95 30.55 104.20 30.78"
  "naples|14.14 40.79 14.35 40.92"
  "kathmandu|85.20 27.57 85.57 27.82"
  "mandalay|95.95 21.85 96.25 22.15"
  "taipei|121.4636 24.9659 121.6255 25.1503"
  "wellington|174.68 -41.36 174.90 -41.18"
  "l_aquila|13.28 42.30 13.50 42.43"
  "noto|136.70 37.25 137.36 37.55"
  "los_angeles|-118.67 33.70 -118.15 34.34"
)

for entry in "${CITIES[@]}"; do
  IFS='|' read -r slug bbox <<< "$entry"
  roads="$ROOT/data/external/${slug}_roads.geojson"
  [ -f "$roads" ] || { echo "SKIP $slug (no roads)"; continue; }
  out="$OUT/${slug}_block_centroids.json"
  [ -f "$out" ] && { echo "have $slug"; continue; }
  echo "=== $slug ==="
  $PY scripts/export_block_centroids.py --roads "$roads" --bbox $bbox --output "$out" || echo "FAILED $slug"
done

echo
ls -lh "$OUT"/*_block_centroids.json | awk '{print $5, $9}'
