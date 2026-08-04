"""Separate the two ways a shelter system fails, district by district.

The 15-minute life circle is a standard built for daily services -- clinics,
schools, markets. Demand for those is staggered across a day and capacity means
throughput per visit. An emergency shelter inverts both: demand arrives at once
and capacity is a hard stock. So a system can pass an accessibility standard and
still fail completely, and the two failures need different remedies:

* reach-limited  -> build closer, or open more sites
* stock-limited  -> build bigger, or the same sites hold more

Chengdu's aggregate numbers already suggest which one binds -- 35.8 % of
residents are within a 15-minute walk of some shelter, but total capacity holds
13.45 % -- yet the aggregate hides which districts have which problem. A
planning department cannot act on a city-wide average.

    python -m scripts.two_constraints
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CACHE = Path("/data/quakesense/cache")
WALK_MPS = 1.34
MINUTES = 15


def main():
    z = np.load(CACHE / "chengdu_city.npz")
    r = np.load(CACHE / "route_city.npz")
    owner = np.load(CACHE / "district_owner_city.npz")

    pop = z["pop"]
    cap = z["s_cap"]
    dist = r["dist"]                       # (n_shelters, n_blocks) network metres
    own = owner["owner"]                   # block -> district index
    names = json.loads((CACHE / "district_owner.json").read_text(
        encoding="utf-8"))["districts"]

    reach_m = WALK_MPS * 60 * MINUTES
    nearest = dist.min(axis=0)
    within = np.isfinite(nearest) & (nearest <= reach_m)

    # A shelter belongs to the district containing the block it sits in, so its
    # capacity counts towards that district's stock.
    s_block = owner["shelter_blocks"] if "shelter_blocks" in owner else None
    if s_block is None:
        s_block = np.argmin(dist, axis=1)
    s_dist = own[s_block]

    rows = []
    for j, d in enumerate(names):
        m = own == j
        if not m.any():
            continue
        p = pop[m].sum()
        if p <= 0:
            continue
        reach = pop[m & within].sum() / p
        stock = cap[s_dist == j].sum() / p
        rows.append({"name": d["name"], "adcode": d["adcode"], "pop": p,
                     "reach": reach, "stock": stock})

    rows.sort(key=lambda r_: r_["reach"])
    print(f"{'district':<12}{'population':>12}{'reach@15min':>13}"
          f"{'capacity':>11}{'binding constraint':>22}")
    print("-" * 71)
    for r_ in rows:
        # Which constraint bites first: the share who could physically get
        # there, or the share the buildings can hold.
        if r_["stock"] < r_["reach"]:
            verdict = f"stock ({r_['stock'] / max(r_['reach'], 1e-9):.2f}x tighter)"
        else:
            verdict = f"reach ({r_['reach'] / max(r_['stock'], 1e-9):.2f}x tighter)"
        print(f"{r_['name']:<12}{r_['pop']:>12,.0f}{r_['reach']:>12.1%}"
              f"{r_['stock']:>11.1%}{verdict:>22}")

    tot_p = sum(r_["pop"] for r_ in rows)
    tot_reach = sum(r_["pop"] * r_["reach"] for r_ in rows) / tot_p
    tot_stock = sum(r_["pop"] * r_["stock"] for r_ in rows) / tot_p
    print("-" * 71)
    print(f"{'CITY':<12}{tot_p:>12,.0f}{tot_reach:>12.1%}{tot_stock:>11.1%}")
    print()
    n_stock = sum(1 for r_ in rows if r_["stock"] < r_["reach"])
    print(f"  {n_stock} of {len(rows)} districts are stock-limited, "
          f"{len(rows) - n_stock} reach-limited.")
    print("  Stock-limited means building closer changes nothing: the people "
          "who can already\n  get there outnumber the places waiting for them.")

    out = CACHE / "two_constraints.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2,
                              default=float), encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
