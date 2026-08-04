"""Turn a run's per-block frames into a self-contained replay map.

The map has to answer three questions that the summary numbers cannot:
where the crowd actually is at minute t, where shelters fill and stall, and
how the "that one is full" signal spreads outward from a rejection. The third
is the whole point -- belief is a field, not an agent property, so it appears
nowhere in the event stream and can only be seen here.

82,766 blocks x 90 frames of float is 119 MB, which no browser should be asked
to parse. Blocks are binned to a ~1.5 km lattice and quantised to uint8, which
is lossless at the resolution a screen can show and lands under 3 MB.
"""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path

import numpy as np


def build(run: Path, out: Path, cell_km: float, max_frames: int, title: str):
    z = np.load(run / "frames.npz")
    lon, lat, pop = z["lon"], z["lat"], z["pop"]
    t_min = z["t_min"]
    n_frames = len(t_min)
    take = np.linspace(0, n_frames - 1, min(max_frames, n_frames)).astype(int)

    lo_lon, hi_lon = float(lon.min()), float(lon.max())
    lo_lat, hi_lat = float(lat.min()), float(lat.max())
    kx = 111.320 * np.cos(np.radians((lo_lat + hi_lat) / 2))
    ncol = max(int((hi_lon - lo_lon) * kx / cell_km) + 1, 2)
    nrow = max(int((hi_lat - lo_lat) * 110.54 / cell_km) + 1, 2)

    col = np.clip(((lon - lo_lon) / (hi_lon - lo_lon) * (ncol - 1)).astype(np.int32),
                  0, ncol - 1)
    row = np.clip(((lat - lo_lat) / (hi_lat - lo_lat) * (nrow - 1)).astype(np.int32),
                  0, nrow - 1)
    cell = row * ncol + col
    ncell = nrow * ncol

    base_pop = np.bincount(cell, weights=pop, minlength=ncell)
    occupied = base_pop > 0
    idx = np.flatnonzero(occupied)

    # Basemap. External tile servers are unreachable under the artifact CSP, so
    # the basemap is derived from the data instead of fetched: resident
    # population per cell draws the city's actual footprint -- built-up areas,
    # the ring roads, the gaps where the rivers and farmland are. It is a more
    # honest backdrop than a generic street tile anyway, because every pixel of
    # it is an input to the simulation running on top.
    base = np.sqrt(base_pop[idx] / max(base_pop.max(), 1e-9))
    base_u8 = np.clip(base * 255.0, 0, 255).astype(np.uint8)

    def q(vals, scale):
        """Bin to the lattice and quantise to a byte on a square-root ramp.

        Population per cell spans four orders of magnitude; a linear ramp would
        show the city centre and nothing else.
        """
        binned = np.bincount(cell, weights=vals.astype(np.float64), minlength=ncell)
        v = np.sqrt(np.maximum(binned[idx], 0.0) / max(scale, 1e-9))
        return np.clip(v * 255.0, 0, 255).astype(np.uint8)

    peak_transit = max(float(np.bincount(cell, weights=z["transit"][t].astype(np.float64),
                                         minlength=ncell).max())
                       for t in take)
    peak_shelt = max(float(np.bincount(cell, weights=z["sheltered"][t].astype(np.float64),
                                       minlength=ncell).max())
                     for t in take)
    peak_gave = max(float(np.bincount(cell, weights=z["gaveup"][t].astype(np.float64),
                                      minlength=ncell).max())
                    for t in take)

    layers = {"transit": [], "sheltered": [], "gaveup": [], "belief": [],
              "known": []}
    for t in take:
        layers["transit"].append(q(z["transit"][t], peak_transit))
        layers["sheltered"].append(q(z["sheltered"][t], peak_shelt))
        layers["gaveup"].append(q(z["gaveup"][t], peak_gave))
        # Belief is a mean, not a sum: it is what a person standing there
        # thinks, so adding two blocks' beliefs would be meaningless.
        b = z["belief"][t]
        s = np.bincount(cell, weights=b.astype(np.float64), minlength=ncell)
        c = np.bincount(cell, minlength=ncell).astype(np.float64)
        m = np.divide(s, np.maximum(c, 1.0))[idx]
        layers["belief"].append(np.clip(m / 600.0 * 255.0, 0, 255).astype(np.uint8))
        # Mean shelters known per person here, over the K=16 candidate slots.
        # Same mean-not-sum reasoning as belief.
        if "known" in z:
            k = z["known"][t]
            ks = np.bincount(cell, weights=k.astype(np.float64), minlength=ncell)
            km = np.divide(ks, np.maximum(c, 1.0))[idx]
            layers["known"].append(np.clip(km / 16.0 * 255.0, 0, 255).astype(np.uint8))
        else:
            layers["known"].append(np.zeros(len(idx), dtype=np.uint8))

    def pack(name):
        raw = np.stack(layers[name]).tobytes()
        return base64.b64encode(zlib.compress(raw, 6)).decode()

    payload = {
        "title": title,
        "nrow": nrow, "ncol": ncol,
        "bbox": [lo_lon, lo_lat, hi_lon, hi_lat],
        "cells": base64.b64encode(zlib.compress(idx.astype(np.int32).tobytes(), 6)).decode(),
        "basemap": base64.b64encode(zlib.compress(base_u8.tobytes(), 6)).decode(),
        "pop_peak": float(base_pop.max()),
        "n_cells": int(len(idx)),
        "t_min": [round(float(t_min[t]), 1) for t in take],
        "peaks": {"transit": peak_transit, "sheltered": peak_shelt, "gaveup": peak_gave},
        # True per-frame totals. Reconstructing these from the quantised bytes
        # is not just imprecise, it is wrong: the ramp is square-root, so the
        # mean of the bytes is not the byte of the mean.
        "totals": {k: [int(z[k][t].sum()) for t in take]
                   for k in ("transit", "sheltered", "gaveup")},
        "layers": {k: pack(k) for k in layers},
        "shelters": {
            "lon": [round(float(v), 5) for v in z["s_lon"]],
            "lat": [round(float(v), 5) for v in z["s_lat"]],
            "cap": [int(v) for v in z["s_cap"]],
        },
    }
    # Study-area outline, decimated: at 1.5 km cells nobody can see a vertex
    # finer than that, and the full ring is 40k points.
    ring = []
    for cand in (run.parent.parent / "boundary.geojson",
                 Path("/workspace/xichang-agentic-evacuation/data_validation/"
                      "chengdu_study_boundary.geojson")):
        if cand.exists():
            gj = json.loads(cand.read_text())
            feats = gj.get("features") or [gj]
            for f in feats:
                g = f.get("geometry") or f
                polys = ([g["coordinates"]] if g["type"] == "Polygon"
                         else g.get("coordinates", []))
                for poly in polys:
                    if not poly:
                        continue
                    a = np.asarray(poly[0], dtype=np.float64)
                    step = max(1, len(a) // 900)
                    ring.append([[round(float(x), 4), round(float(y), 4)]
                                 for x, y in a[::step]])
            break
    payload["outline"] = ring

    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}
    payload["summary"] = summary.get("totals", {})
    payload["perf"] = {k: summary.get(k) for k in
                       ("gpu_only_ms_per_step", "peak_vram_gib", "steps", "n_agents",
                        "n_blocks", "n_shelters")}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    mb = out.stat().st_size / 1024 ** 2
    print(f"grid {nrow} x {ncol}, {len(idx):,} occupied cells, "
          f"{len(take)} frames -> {out} ({mb:.2f} MB)")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/data/quakesense/runs/viz")
    ap.add_argument("--out", default="/data/quakesense/runs/viz/webgis_payload.json")
    ap.add_argument("--cell-km", type=float, default=1.5)
    ap.add_argument("--max-frames", type=int, default=46)
    ap.add_argument("--title", default="Chengdu")
    a = ap.parse_args()
    build(Path(a.run), Path(a.out), a.cell_km, a.max_frames, a.title)


if __name__ == "__main__":
    main()
