"""Attach Chengdu's real district boundaries to the block layer.

The model has been reporting "administrative boundaries are constructed" as a
limitation while an official 20-district boundary file sat unused. It is a
limitation only below the district level: the file carries level="district"
with childrenNum=0, so subdistricts and communities really are absent, but the
districts themselves are real and the 20 district-level LLM leaders can sit on
them instead of on Hilbert cells.

The file comes from an AMap-style source (adcode / acroutes / childrenNum),
which usually means GCJ-02, while the block centroids come from OSM in WGS84 --
about 500 m apart in Chengdu, enough to misassign every block near a boundary.
Rather than assume, this tries both and keeps whichever leaves fewer blocks
outside the city.

    python -m scripts.assign_districts --boundary 成都市.geojson \
        --cache /data/quakesense/cache/chengdu_bridged.npz
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PI = math.pi
A = 6378245.0
EE = 0.00669342162296594323


def _tf_lat(x, y):
    v = (-100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y
         + 0.2 * np.sqrt(np.abs(x)))
    v += (20.0 * np.sin(6.0 * x * PI) + 20.0 * np.sin(2.0 * x * PI)) * 2.0 / 3.0
    v += (20.0 * np.sin(y * PI) + 40.0 * np.sin(y / 3.0 * PI)) * 2.0 / 3.0
    v += (160.0 * np.sin(y / 12.0 * PI) + 320 * np.sin(y * PI / 30.0)) * 2.0 / 3.0
    return v


def _tf_lon(x, y):
    v = (300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y
         + 0.1 * np.sqrt(np.abs(x)))
    v += (20.0 * np.sin(6.0 * x * PI) + 20.0 * np.sin(2.0 * x * PI)) * 2.0 / 3.0
    v += (20.0 * np.sin(x * PI) + 40.0 * np.sin(x / 3.0 * PI)) * 2.0 / 3.0
    v += (150.0 * np.sin(x / 12.0 * PI) + 300.0 * np.sin(x / 30.0 * PI)) * 2.0 / 3.0
    return v


def gcj02_to_wgs84(lon, lat):
    """Vectorised inverse of the GCJ-02 obfuscation, to first order."""
    dlat = _tf_lat(lon - 105.0, lat - 35.0)
    dlon = _tf_lon(lon - 105.0, lat - 35.0)
    rad = lat / 180.0 * PI
    magic = 1 - EE * np.sin(rad) ** 2
    sqrt_magic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((A * (1 - EE)) / (magic * sqrt_magic) * PI)
    dlon = (dlon * 180.0) / (A / sqrt_magic * np.cos(rad) * PI)
    return lon - dlon, lat - dlat


def _rings(geom, convert):
    """Exterior rings only. A district's holes are other districts, and a
    block sitting in one will be claimed by that district's own ring."""
    t = geom.get("type")
    polys = ([geom["coordinates"]] if t == "Polygon"
             else geom["coordinates"] if t == "MultiPolygon" else [])
    for poly in polys:
        if not poly:
            continue
        r = np.asarray(poly[0], dtype=np.float64)
        if convert:
            lo, la = gcj02_to_wgs84(r[:, 0], r[:, 1])
            r = np.stack([lo, la], axis=1)
        yield r


def points_in_ring(px, py, ring):
    """Ray casting, vectorised over points."""
    x1, y1 = ring[:-1, 0], ring[:-1, 1]
    x2, y2 = ring[1:, 0], ring[1:, 1]
    inside = np.zeros(px.shape, dtype=bool)
    for i in range(len(x1)):
        cond = ((y1[i] > py) != (y2[i] > py))
        if not cond.any():
            continue
        xint = (x2[i] - x1[i]) * (py - y1[i]) / (y2[i] - y1[i] + 1e-300) + x1[i]
        inside ^= cond & (px < xint)
    return inside


def assign(features, lon, lat, convert):
    owner = np.full(len(lon), -1, dtype=np.int32)
    for j, f in enumerate(features):
        for ring in _rings(f.get("geometry") or {}, convert):
            lo, hi = ring.min(axis=0), ring.max(axis=0)
            box = ((lon >= lo[0]) & (lon <= hi[0])
                   & (lat >= lo[1]) & (lat <= hi[1]) & (owner < 0))
            cand = np.flatnonzero(box)
            if not cand.size:
                continue
            hit = points_in_ring(lon[cand], lat[cand], ring)
            owner[cand[hit]] = j
    return owner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    gj = json.loads(Path(a.boundary).read_text(encoding="utf-8"))
    feats = gj["features"]
    names = [f["properties"]["name"] for f in feats]
    codes = [f["properties"]["adcode"] for f in feats]
    print(f"boundary: {len(feats)} features, "
          f"level={feats[0]['properties'].get('level')}, "
          f"childrenNum={feats[0]['properties'].get('childrenNum')}")

    z = np.load(a.cache)
    lon, lat, pop = z["lon"], z["lat"], z["pop"]
    print(f"blocks: {len(lon):,}  population {pop.sum():,.0f}")

    # Decide the datum by measurement, not by assumption.
    best = None
    for convert in (False, True):
        owner = assign(feats, lon, lat, convert)
        outside = int((owner < 0).sum())
        tag = "GCJ-02 -> WGS84" if convert else "as supplied"
        print(f"  {tag:<18} unassigned blocks {outside:>7,} "
              f"({outside / len(lon) * 100:5.2f}%)  "
              f"population outside {pop[owner < 0].sum():>12,.0f}")
        if best is None or outside < best[0]:
            best = (outside, owner, convert)
    outside, owner, convert = best
    print(f"\nchosen: {'converted from GCJ-02' if convert else 'as supplied'}")

    print(f"\n{'adcode':>8} {'district':<14}{'blocks':>9}{'population':>14}"
          f"{'share':>8}")
    print("-" * 56)
    tot = pop.sum()
    rows = []
    for j, (c, n) in enumerate(zip(codes, names)):
        m = owner == j
        p = pop[m].sum()
        rows.append({"adcode": int(c), "name": n, "blocks": int(m.sum()),
                     "population": float(p)})
        print(f"{c:>8} {n:<14}{int(m.sum()):>9,}{p:>14,.0f}{p / tot * 100:>7.2f}%")
    if outside:
        print(f"{'--':>8} {'(outside)':<14}{outside:>9,}"
              f"{pop[owner < 0].sum():>14,.0f}"
              f"{pop[owner < 0].sum() / tot * 100:>7.2f}%")

    out = Path(a.out) if a.out else Path(a.cache).with_name("district_owner.npz")
    np.savez(out, owner=owner, adcode=np.array(codes, dtype=np.int64),
             converted=np.array([convert]))
    Path(out).with_suffix(".json").write_text(
        json.dumps({"converted_from_gcj02": bool(convert),
                    "unassigned_blocks": outside, "districts": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out} and {Path(out).with_suffix('.json')}")


if __name__ == "__main__":
    main()
