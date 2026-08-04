"""Compress a city's road network down to a drawable basemap layer.

Chengdu's OSM extract is 141,089 ways and 1.58 M vertices. At the replay map's
scale one screen pixel is roughly 300 m, so almost all of that detail is
invisible: what makes the map readable is the skeleton -- ring roads,
expressways, the radial trunks -- not every service alley.

Three reductions, in order:
  1. keep only the classes that carry the city's shape;
  2. drop vertices closer together than the screen can resolve;
  3. quantise to a 16-bit lattice over the bounding box (~3 m) and delta-encode,
     so a coordinate costs 2 bytes per axis instead of 8.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import zlib
from pathlib import Path

import numpy as np

#: Three grades, drawn at three widths. Collapsing them into one "major road"
#: class flattens a ring road into the same stroke as a district street, which
#: is what made the first version read as an undifferentiated mesh.
#: Heaviest -- expressways and the ring-road skeleton.
TIER0 = {"motorway", "trunk", "motorway_link", "trunk_link"}
#: Medium -- the radial arterials carrying traffic between districts.
TIER1 = {"primary", "primary_link"}
#: Lightest -- sub-arterials, enough to give each district its grain.
TIER2 = {"secondary", "secondary_link"}


def _simplify(pts: np.ndarray, tol_m: float, kx: float, ky: float) -> np.ndarray:
    """Radial-distance filter: keep a vertex only once it is `tol` from the
    last kept one. Cheaper than Douglas-Peucker and indistinguishable at a
    tolerance far below one pixel."""
    if len(pts) < 3:
        return pts
    keep = [0]
    lx, ly = pts[0]
    for i in range(1, len(pts) - 1):
        dx = (pts[i, 0] - lx) * kx
        dy = (pts[i, 1] - ly) * ky
        if dx * dx + dy * dy >= tol_m * tol_m:
            keep.append(i)
            lx, ly = pts[i]
    keep.append(len(pts) - 1)
    return pts[keep]


def extract(path: Path, bbox, tol_m: float, log=print):
    lo_lon, lo_lat, hi_lon, hi_lat = bbox
    kx = 111_320.0 * math.cos(math.radians((lo_lat + hi_lat) / 2))
    ky = 110_540.0
    span_lon = max(hi_lon - lo_lon, 1e-9)
    span_lat = max(hi_lat - lo_lat, 1e-9)

    with open(path, "rb") as fh:
        gj = json.load(fh)

    tiers = {0: [], 1: [], 2: []}
    n_in = n_pts_in = n_pts_out = 0
    for f in gj.get("features", []):
        hw = (f.get("properties") or {}).get("highway")
        tier = (0 if hw in TIER0 else
                1 if hw in TIER1 else
                2 if hw in TIER2 else None)
        if tier is None:
            continue
        g = f.get("geometry") or {}
        lines = ([g["coordinates"]] if g.get("type") == "LineString"
                 else g.get("coordinates", []) if g.get("type") == "MultiLineString"
                 else [])
        for ln in lines:
            if len(ln) < 2:
                continue
            a = np.asarray(ln, dtype=np.float64)[:, :2]
            n_in += 1
            n_pts_in += len(a)
            a = _simplify(a, tol_m, kx, ky)
            if len(a) < 2:
                continue
            n_pts_out += len(a)
            tiers[tier].append(a)

    def encode(lines):
        """uint16 absolute start, int16 deltas, per line prefixed by length."""
        buf = []
        for a in lines:
            u = np.clip((a[:, 0] - lo_lon) / span_lon * 65535, 0, 65535).astype(np.int32)
            v = np.clip((a[:, 1] - lo_lat) / span_lat * 65535, 0, 65535).astype(np.int32)
            du = np.diff(u)
            dv = np.diff(v)
            # A delta beyond int16 means the segment jumped more than half the
            # bbox; that is a data artefact, so cut the line rather than wrap.
            if len(du) and (np.abs(du).max() > 32767 or np.abs(dv).max() > 32767):
                continue
            buf.append(np.array([len(a)], dtype=np.int16))
            buf.append(np.array([u[0] - 32768, v[0] - 32768], dtype=np.int16))
            if len(du):
                buf.append(np.stack([du, dv], axis=1).astype(np.int16).reshape(-1))
        if not buf:
            return "", 0
        raw = np.concatenate(buf).tobytes()
        return base64.b64encode(zlib.compress(raw, 9)).decode(), len(buf) // 3

    out = {}
    for t in (0, 1, 2):
        enc, _ = encode(tiers[t])
        out[str(t)] = enc
    kb = sum(len(v) for v in out.values()) / 1024
    log(f"roads: {n_in:,} ways kept ("
        + ", ".join(f"tier{t} {len(tiers[t]):,}" for t in (0, 1, 2))
        + f"), {n_pts_in:,} -> {n_pts_out:,} vertices "
          f"(tol {tol_m:.0f} m), encoded {kb:.0f} KB base64")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roads", required=True)
    ap.add_argument("--payload", required=True,
                    help="Existing payload.json to merge the road layer into.")
    ap.add_argument("--tol-m", type=float, default=220.0)
    a = ap.parse_args()

    p = Path(a.payload)
    payload = json.loads(p.read_text())
    payload["roads"] = extract(Path(a.roads), payload["bbox"], a.tol_m)
    p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"payload now {p.stat().st_size/1024**2:.2f} MB")


if __name__ == "__main__":
    main()
