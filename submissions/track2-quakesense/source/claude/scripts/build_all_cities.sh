#!/usr/bin/env bash
# Build block-scale evacuation data for every city that has the required
# inputs, into the portal's data directory so all nine appear in the demo.
#
# Chengdu takes the census/land-use-calibrated population grid; every other
# city takes its raw WorldPop raster, because no equivalent census pipeline
# exists for them -- the same "one pipeline, per-city data quality declared
# honestly" principle the manuscript describes. Terrain resistance is passed
# where a resistance file has been fetched (see fetch_terrain_resistance.py);
# cities without one fall back to the flat-plane assumption and say so in
# their provenance.
#
#   bash scripts/build_all_cities.sh
set -uo pipefail

PY=/workspace/venvs/xichang-agent/bin/python
ROOT=/workspace/xichang-agentic-evacuation
CLAUDE=$ROOT/claude
PORTAL=$CLAUDE/portal/data
WORLDPOP=/workspace/persistence/worldpop

cd "$CLAUDE" || exit 1
mkdir -p "$PORTAL"

# slug|label|bbox(W S E N)|epicentre(lon lat)|magnitude|worldpop raster
CITIES=(
  "naples|Naples|14.14 40.79 14.35 40.92|14.25 40.84|6.5|ita_ppp_2020_1km_Aggregated_UNadj.tif"
  "kathmandu|Kathmandu|85.20 27.57 85.57 27.82|85.35 27.70|7.0|npl_ppp_2020_1km_Aggregated_UNadj.tif"
  "mandalay|Mandalay|95.95 21.85 96.25 22.15|96.10 22.00|7.0|mmr_ppp_2020_1km_Aggregated_UNadj.tif"
  "taipei|Taipei|121.4636 24.9659 121.6255 25.1503|121.54 25.05|6.8|twn_ppp_2020_1km_Aggregated_UNadj.tif"
  "wellington|Wellington|174.68 -41.36 174.90 -41.18|174.78 -41.28|7.2|nzl_ppp_2020_1km_Aggregated_UNadj.tif"
  "l_aquila|L Aquila|13.28 42.30 13.50 42.43|13.38 42.36|6.3|ita_ppp_2020_1km_Aggregated_UNadj.tif"
  "noto|Noto|136.70 37.25 137.36 37.55|137.00 37.40|7.6|jpn_ppp_2020_1km_Aggregated_UNadj.tif"
  "los_angeles|Los Angeles|-118.67 33.70 -118.15 34.34|-118.40 34.05|7.0|usa_ppp_2020_1km_Aggregated_UNadj.tif"
)

ok=(); failed=(); skipped=()

for entry in "${CITIES[@]}"; do
  IFS='|' read -r slug label bbox epi mag raster <<< "$entry"
  roads="$ROOT/data/external/${slug}_roads.geojson"
  shelters="$ROOT/data/multi_city/${slug}/shelters.geojson"
  pop="$WORLDPOP/$raster"
  resist="$CLAUDE/results_naples/${slug}_block_resistance.json"
  [ -f "$resist" ] || resist="$CLAUDE/results_${slug}/${slug}_block_resistance.json"

  if [ ! -f "$roads" ] || [ ! -f "$shelters" ]; then
    echo "SKIP $slug (missing roads or shelters)"; skipped+=("$slug"); continue
  fi
  if [ ! -f "$pop" ]; then
    echo "SKIP $slug (missing WorldPop raster $raster)"; skipped+=("$slug"); continue
  fi

  extra=()
  if [ -f "$resist" ]; then extra+=(--resistance "$resist"); fi

  echo "=== $slug ==="
  if $PY scripts/build_city_block_evacuation.py \
        --city "$label" \
        --roads "$roads" \
        --population-raster "$pop" \
        --shelters "$shelters" \
        --bbox $bbox \
        --epicenter $epi \
        --magnitude "$mag" \
        --duration-minutes 180 \
        --rendered-agents 15000 \
        \
        "${extra[@]}" \
        --output-dir "$PORTAL/$slug"; then
    ok+=("$slug")
  else
    echo "FAILED $slug"; failed+=("$slug")
  fi
done

echo
echo "=========================================="
echo "built:   ${ok[*]:-none}"
echo "failed:  ${failed[*]:-none}"
echo "skipped: ${skipped[*]:-none}"
echo "=========================================="
