"""Candidate shelter sites from Chengdu POI, and what they would buy.

The official register has 1,252 sites. The POI file carries roughly four
thousand spaces that could physically serve as emergency shelter -- parks,
squares, school grounds, sports halls, exhibition centres. This asks the
cheapest useful question about them, the one that needs no capacity assumption
at all:

    if every one of these were opened, how far would 15-minute reach improve?

Reach is pure geometry, so the answer is defensible without deciding how many
people a school playground holds. If it barely moves, the second step -- siting
optimisation with assumed capacities -- is not worth running.

    python -m scripts.poi_candidates --poi 成都市POI数据.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

import numpy as np

CACHE = Path("/data/quakesense/cache")
WALK_MPS = 1.34

#: Subcategories that are open space or have open ground attached. Parking is
#: excluded on purpose: a car park is nominally open but is not a place a city
#: shelters people, and including it would inflate the count fourfold.
KEEP = {
    "公园": "park", "广场": "square", "小学": "school", "中学": "school",
    "综合体育馆": "sports", "会展展馆": "exhibition", "展览馆": "exhibition",
    "文化馆": "culture", "体育场馆": "sports", "运动场馆": "sports",
}


def load_poi(path: Path):
    lon, lat, kind, district = [], [], [], []
    with io.open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) < 8:
                continue
            k = KEEP.get(row[2])
            if not k:
                continue
            try:
                x, y = float(row[3]), float(row[4])
            except ValueError:
                continue
            # Chengdu sits near 104E 30.7N; anything else is a parsing error.
            if not (102.5 < x < 105.0 and 30.0 < y < 31.6):
                continue
            lon.append(x); lat.append(y); kind.append(k); district.append(row[7])
    return (np.array(lon), np.array(lat), kind, district)


def nearest_block(plon, plat, blon, blat, chunk=256):
    """Snap each candidate to a block, chunked so the cross product never
    materialises in full."""
    out = np.empty(len(plon), dtype=np.int64)
    for i in range(0, len(plon), chunk):
        dx = blon[None, :] - plon[i:i + chunk, None]
        dy = blat[None, :] - plat[i:i + chunk, None]
        out[i:i + chunk] = np.argmin(dx * dx + dy * dy, axis=1)
    return out


def reach_curve(dist_to_nearest, pop, marks=(5, 10, 15, 20, 30)):
    ok = np.isfinite(dist_to_nearest) & (dist_to_nearest < 3e38)
    out = {}
    for m in marks:
        r = WALK_MPS * 60 * m
        out[m] = float(pop[ok & (dist_to_nearest <= r)].sum() / pop.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poi", required=True)
    ap.add_argument("--cache", default=str(CACHE / "chengdu_city.npz"))
    ap.add_argument("--route", default=str(CACHE / "route_city.npz"))
    a = ap.parse_args()

    z = np.load(a.cache)
    blon, blat, pop = z["lon"], z["lat"], z["pop"]
    r = np.load(a.route)
    dist = r["dist"]                     # (n_official_shelters, n_blocks)

    plon, plat, kind, district = load_poi(Path(a.poi))
    import collections
    print(f"candidate sites kept: {len(plon):,}")
    for k, v in collections.Counter(kind).most_common():
        print(f"  {k:<12}{v:>7,}")
    print(f"official register:    {dist.shape[0]:,}")

    # Reach today, from the official register only.
    base = dist.min(axis=0)
    now = reach_curve(base, pop)

    # Reach if every candidate were opened. Candidates are snapped to blocks,
    # so "distance to a candidate" is the network distance between blocks --
    # which the routing table already holds only for official shelters. Use
    # straight-line between block centroids as a lower bound instead, and say
    # so: it flatters the candidates, which makes a weak result conclusive.
    cb = nearest_block(plon, plat, blon, blat)
    kx = 111_320.0 * math.cos(math.radians(float(blat.mean())))
    bx = (blon - blon.mean()) * kx
    by = (blat - blat.mean()) * 110_540.0
    cand_x, cand_y = bx[cb], by[cb]

    best = np.full(len(bx), np.inf)
    for i in range(0, len(cand_x), 256):
        dx = bx[:, None] - cand_x[None, i:i + 256]
        dy = by[:, None] - cand_y[None, i:i + 256]
        best = np.minimum(best, np.sqrt(dx * dx + dy * dy).min(axis=1))
    # Combined: whichever is nearer, an official shelter or a candidate.
    both = np.minimum(base, best)
    added = reach_curve(both, pop)
    cand_only = reach_curve(best, pop)

    print(f"\n{'walk':>6}{'official only':>16}{'candidates only':>18}"
          f"{'both':>10}{'gain':>10}")
    print("-" * 60)
    for m in (5, 10, 15, 20, 30):
        g = added[m] - now[m]
        print(f"{m:>4} m{now[m]:>15.1%}{cand_only[m]:>18.1%}"
              f"{added[m]:>10.1%}{g:>+10.1%}")

    print("\n  Candidate distances are straight-line between block centroids,")
    print("  an optimistic bound; official reach uses true network distance.")
    print("  A small gain under a generous assumption is therefore a real")
    print("  negative result, while a large one needs the network check.")

    out = CACHE / "poi_candidates.json"
    out.write_text(json.dumps({
        "n_candidates": int(len(plon)),
        "by_kind": dict(collections.Counter(kind)),
        "reach_official": now, "reach_candidates_only": cand_only,
        "reach_both": added,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(CACHE / "poi_candidate_blocks.npz", block=cb,
             lon=plon, lat=plat, kind=np.array(kind))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
