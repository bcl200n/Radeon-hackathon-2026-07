"""Clip the block layer and shelters to Chengdu's real administrative area.

The block layer was cut from a bounding box, so it carries 16,220 blocks and
3,513,188 people that belong to Deyang, Meishan and Ziyang rather than Chengdu.
Reporting those as Chengdu residents inflated the population to 22,381,554
against a census figure of about 20.94 M, and the two errors -- outsiders
included, insiders under-allocated -- had been hiding each other.

Clipping changes the block indices, so neighbours are remapped and the routing
tables have to be rebuilt; there is no way to reuse the old cache. Shelters
outside the boundary go too: a shelter in another prefecture is not a place
Chengdu's emergency plan can send anyone.

    python -m scripts.clip_to_districts \
        --cache /data/quakesense/cache/chengdu_bridged.npz \
        --owner /data/quakesense/cache/district_owner.npz \
        --boundary chengdu_districts.geojson \
        --out /data/quakesense/cache/chengdu_city.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.assign_districts import _rings, points_in_ring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--owner", required=True)
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = np.load(a.cache)
    ow = np.load(a.owner)
    owner = ow["owner"]
    keep = owner >= 0
    n_old = len(owner)
    n_new = int(keep.sum())
    print(f"blocks {n_old:,} -> {n_new:,}  "
          f"population {z['pop'].sum():,.0f} -> {z['pop'][keep].sum():,.0f}")

    # Old index -> new index, with -1 for everything dropped, so a neighbour
    # that fell outside the city becomes a missing link rather than silently
    # pointing at whichever block inherited its number.
    remap = np.full(n_old, -1, dtype=np.int32)
    remap[keep] = np.arange(n_new, dtype=np.int32)

    neigh = z["neigh"]
    nb = np.where(neigh >= 0, remap[np.clip(neigh, 0, None)], -1)
    nb = nb[keep]
    # Compact each row so the valid neighbours sit at the front; the step loop
    # reads a fixed-width table and treats -1 as padding.
    out_nb = np.full_like(nb, -1)
    for i in range(nb.shape[0]):
        v = nb[i][nb[i] >= 0]
        out_nb[i, :len(v)] = v
    deg = (out_nb >= 0).sum(axis=1).astype(np.int32)
    print(f"mean degree {z['deg'].mean():.2f} -> {deg.mean():.2f}  "
          f"isolated {int((deg == 0).sum()):,}")

    # Shelters: keep the ones inside the boundary.
    gj = json.loads(Path(a.boundary).read_text(encoding="utf-8"))
    feats = gj["features"]
    s_lon, s_lat, s_cap = z["s_lon"], z["s_lat"], z["s_cap"]
    inside = np.zeros(len(s_lon), dtype=bool)
    for f in feats:
        for ring in _rings(f.get("geometry") or {}, False):
            lo, hi = ring.min(axis=0), ring.max(axis=0)
            box = ((s_lon >= lo[0]) & (s_lon <= hi[0])
                   & (s_lat >= lo[1]) & (s_lat <= hi[1]) & ~inside)
            cand = np.flatnonzero(box)
            if cand.size:
                inside[cand[points_in_ring(s_lon[cand], s_lat[cand], ring)]] = True
    print(f"shelters {len(s_lon):,} -> {int(inside.sum()):,}  "
          f"capacity {s_cap.sum():,.0f} -> {s_cap[inside].sum():,.0f}")

    pop = z["pop"][keep]
    cap = s_cap[inside].sum()
    print(f"capacity share of population "
          f"{z['s_cap'].sum() / z['pop'].sum() * 100:.2f}% -> "
          f"{cap / pop.sum() * 100:.2f}%")

    np.savez(a.out, lon=z["lon"][keep], lat=z["lat"][keep], pop=pop,
             area=z["area"][keep], neigh=out_nb, deg=deg,
             s_lon=s_lon[inside], s_lat=s_lat[inside], s_cap=s_cap[inside],
             district=owner[keep].astype(np.int32),
             adcode=ow["adcode"])
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
