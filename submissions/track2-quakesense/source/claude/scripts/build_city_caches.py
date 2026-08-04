"""Lane B: build every other city's block cache on the CPU, in parallel.

Parsing a GeoJSON, hashing shared polygon edges into an adjacency, and bridging
the disconnected pieces is pure CPU work -- 128 cores sitting idle while the
GPU runs Chengdu's LLM arms. Doing it now means each city's simulation is then
only a routing build plus a run.

Los Angeles is 153 MB and gets its own worker; the small ones share.

    python -m scripts.build_city_caches --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import orjson as _fj

    def _load(p):
        return _fj.loads(Path(p).read_bytes())
except ImportError:
    def _load(p):
        with open(p, "rb") as fh:
            return json.load(fh)

from simulator.block_graph import build_adjacency, bridge_components, _components

BASE = Path("/workspace/xichang-agentic-evacuation/claude/portal/data")
OUT = Path("/data/quakesense/cache")


def build(city: str, max_degree: int = 14):
    t0 = time.perf_counter()
    d = BASE / city
    geo = _load(d / f"{city}_blocks.geojson")
    feats = geo["features"]
    props = [f["properties"] for f in feats]
    lon = np.fromiter((p["lon"] for p in props), dtype=np.float64, count=len(props))
    lat = np.fromiter((p["lat"] for p in props), dtype=np.float64, count=len(props))
    pop = np.fromiter((p["population"] for p in props), dtype=np.float64,
                      count=len(props))
    area = np.fromiter(
        (float(p.get("area_m2") or p.get("area") or 40_000.0) for p in props),
        dtype=np.float64, count=len(props))

    neigh, deg = build_adjacency(feats, max_degree=max_degree)
    n_before = len(np.unique(_components(neigh, len(lon))))

    kx = 111_320.0 * math.cos(math.radians(float(lat.mean())))
    bx = (lon - lon.mean()) * kx
    by = (lat - lat.mean()) * 110_540.0
    # Bridging is iterative: one pass can leave pieces behind when the degree
    # cap truncates a repair, so keep going until the graph is one piece.
    for _ in range(6):
        lab = _components(neigh, len(lon))
        if len(np.unique(lab)) == 1:
            break
        neigh, _n = bridge_components(neigh, bx, by, pop, max_degree=max_degree,
                                      log=lambda *a: None)
    n_after = len(np.unique(_components(neigh, len(lon))))
    deg = (neigh >= 0).sum(axis=1).astype(np.int32)

    sim = _load(d / f"{city}_block_evacuation.json")
    sh = sim["shelters"]
    s_lon = np.fromiter((s["lon"] for s in sh), dtype=np.float64, count=len(sh))
    s_lat = np.fromiter((s["lat"] for s in sh), dtype=np.float64, count=len(sh))
    s_cap = np.fromiter((float(s.get("capacity", 0.0)) for s in sh),
                        dtype=np.float64, count=len(sh))

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{city}_city.npz"
    np.savez(out, lon=lon, lat=lat, pop=pop, area=area, neigh=neigh, deg=deg,
             s_lon=s_lon, s_lat=s_lat, s_cap=s_cap)
    return {"city": city, "blocks": len(lon), "population": float(pop.sum()),
            "shelters": len(sh), "capacity": float(s_cap.sum()),
            "cap_share": float(s_cap.sum() / max(pop.sum(), 1)),
            "components_before": int(n_before), "components_after": int(n_after),
            "mean_degree": float(deg.mean()), "seconds": round(time.perf_counter() - t0, 1)}


def _safe(city):
    try:
        return build(city)
    except Exception:                                        # noqa: BLE001
        return {"city": city, "error": traceback.format_exc(limit=3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cities", nargs="*", default=[
        "los_angeles", "taipei", "kathmandu", "naples", "mandalay",
        "wellington", "l_aquila", "noto"])
    a = ap.parse_args()
    print(f"building {len(a.cities)} city caches on {a.workers} workers\n")
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_safe, c): c for c in a.cities}
        for f in as_completed(futs):
            r = f.result()
            if "error" in r:
                print(f"  {r['city']:<13} FAILED\n{r['error']}")
            else:
                print(f"  {r['city']:<13} {r['blocks']:>8,} blocks  "
                      f"pop {r['population']:>12,.0f}  "
                      f"shelters {r['shelters']:>5,}  "
                      f"cap {r['cap_share']*100:>6.2f}%  "
                      f"comp {r['components_before']:>5,}->{r['components_after']}  "
                      f"{r['seconds']:>6.1f}s")
            rows.append(r)
    Path("/data/quakesense/cache/city_build_report.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    ok = [r for r in rows if "error" not in r]
    print(f"\n{len(ok)}/{len(rows)} cities cached")


if __name__ == "__main__":
    main()
