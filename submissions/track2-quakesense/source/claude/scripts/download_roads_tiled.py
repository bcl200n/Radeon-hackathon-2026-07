#!/usr/bin/env python3
"""Download an OSM road network for a bbox too large for one Overpass query.

A single query over Chengdu's full municipal extent (2.0 deg x 1.35 deg,
roughly 200 x 150 km) times out on every public Overpass endpoint. Splitting
it into tiles that are each about the size of a query already known to
succeed, then concatenating the features, gets the same result. Ways that
straddle a tile boundary are returned by both tiles, so they are
de-duplicated by OSM id on merge -- without that the block extractor would
see doubled geometry along every seam.

    python scripts/download_roads_tiled.py \\
        --bbox 102.90 30.10 104.90 31.45 --tiles 4 3 \\
        --output ../data/external/chengdu_admin_roads.geojson
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def fetch_tile(west, south, east, north, retries=3):
    query = f"""
[out:json][timeout:180];
(
  way["highway"]({south},{west},{north},{east});
);
out geom;
""".strip()
    for attempt in range(retries):
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                r = requests.post(endpoint, data={"data": query},
                                  headers={"User-Agent": "SeismicEvacuationResearchDemo/0.1"},
                                  timeout=240)
                r.raise_for_status()
                return r.json().get("elements", [])
            except Exception as exc:
                print(f"      {endpoint.split('/')[2]}: {exc}")
        time.sleep(5 * (attempt + 1))
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox", type=float, nargs=4, required=True, metavar=("W", "S", "E", "N"))
    p.add_argument("--tiles", type=int, nargs=2, default=[4, 3], metavar=("NX", "NY"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    w, s, e, n = args.bbox
    nx, ny = args.tiles
    dx, dy = (e - w) / nx, (n - s) / ny

    seen: set[int] = set()
    features: list[dict] = []
    failed: list[str] = []

    for iy in range(ny):
        for ix in range(nx):
            tw, ts = w + ix * dx, s + iy * dy
            te, tn = tw + dx, ts + dy
            label = f"{ix},{iy}"
            print(f"  tile {label} [{tw:.3f},{ts:.3f},{te:.3f},{tn:.3f}] ...", flush=True)
            els = fetch_tile(tw, ts, te, tn)
            if els is None:
                failed.append(label)
                print(f"    FAILED")
                continue
            added = 0
            for el in els:
                if el.get("type") != "way" or "geometry" not in el:
                    continue
                # Ways crossing a seam come back from both tiles; keep one.
                if el["id"] in seen:
                    continue
                seen.add(el["id"])
                coords = [[pt["lon"], pt["lat"]] for pt in el["geometry"]]
                if len(coords) < 2:
                    continue
                tags = el.get("tags", {})
                features.append({
                    "type": "Feature",
                    "properties": {"highway": tags.get("highway"), "osm_id": el["id"],
                                   "name": tags.get("name", ""), "source": "OpenStreetMap"},
                    "geometry": {"type": "LineString", "coordinates": coords},
                })
                added += 1
            print(f"    +{added:,} new (running total {len(features):,})")
            time.sleep(2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"type": "FeatureCollection", "features": features},
                                      ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(features):,} unique ways -> {args.output}")
    if failed:
        print(f"WARNING: {len(failed)} tile(s) failed and are missing from the output: {failed}")


if __name__ == "__main__":
    main()
