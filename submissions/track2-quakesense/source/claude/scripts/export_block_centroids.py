#!/usr/bin/env python3
"""Export road-enclosed block centroids as a compact JSON, so a machine with
Earth Engine access can sample slope and land cover for them.

This project's cloud GPU server blocks outbound traffic to googleapis.com
(see scripts/test_gee_auth.py), so the terrain-resistance surface cannot be
fetched where the pipeline runs. Splitting it in two -- centroids out here,
resistance values back in -- keeps block extraction in exactly one place
(geo.chengdu_blocks, the same code the simulation itself uses) instead of
reimplementing it on the machine that happens to have network access.

    python scripts/export_block_centroids.py \\
        --roads ../data/external/naples_roads.geojson \\
        --bbox 14.14 40.79 14.35 40.92 \\
        --output results_naples/naples_block_centroids.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geo.chengdu_blocks import blocks_from_geojson  # noqa: E402
from scripts.build_city_block_evacuation import clip_blocks  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roads", type=Path, required=True)
    p.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    p.add_argument("--drop-highways", nargs="*", default=["motorway", "motorway_link"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    blocks, _, _ = blocks_from_geojson(args.roads, drop_highways=args.drop_highways)
    blocks = clip_blocks(blocks, tuple(args.bbox))
    if not blocks:
        raise SystemExit("No blocks in --bbox")

    payload = {
        "bbox_west_south_east_north": list(args.bbox),
        "source_roads": str(args.roads),
        "count": len(blocks),
        "blocks": [
            {"block_id": b.block_id, "lon": round(float(b.lon), 6), "lat": round(float(b.lat), 6)}
            for b in blocks
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {len(blocks):,} block centroids -> {args.output}")


if __name__ == "__main__":
    main()
