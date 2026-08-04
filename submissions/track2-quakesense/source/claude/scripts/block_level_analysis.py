#!/usr/bin/env python3
"""Block-level determinants of evacuation outcome, across the nine-city set.

Why block level and not city level: nine cities is nine observations, which
cannot support a mediation model with several mediators -- any significance
star printed on an n=9 fit would be fiction. The blocks are the real sample:
tens of thousands of them, each with its own terrain, population, egress
geometry and simulated outcome. So the model is fitted per city over its own
blocks, and the cities are then compared as separate fits rather than pooled
into a single nine-point regression.

Emits a tidy table plus per-city fitted coefficients with standard errors and
P values, for the figure to draw. Uses ordinary least squares with
heteroskedasticity-robust (HC1) standard errors, since block-level residual
variance is obviously not constant across a city.

    python scripts/block_level_analysis.py --cities chengdu_full naples ... \\
        --output results_analysis/block_level_fits.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PORTAL = Path(__file__).resolve().parents[1] / "portal" / "data"

#: Predictors of whether a block's residents get out. Chosen because each is a
#: physically distinct mechanism the simulator actually implements, not because
#: they maximise fit: terrain friction, how far help is, how wide the way out
#: is, how many people must use it, and how hard the ground shook.
PREDICTORS = [
    ("log_shelter_distance", "Distance to shelter (log m)"),
    ("resistance", "Terrain walking resistance"),
    ("log_egress_width", "Egress width (log m)"),
    ("log_population", "Block population (log)"),
    ("mmi", "Shaking intensity (MMI)"),
]


def ols_hc1(X: np.ndarray, y: np.ndarray):
    """OLS with HC1 robust standard errors. Returns (beta, se, t, p, r2, n)."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    # HC1: finite-sample-corrected White estimator
    S = (X * (resid ** 2)[:, None]).T @ X
    cov = XtX_inv @ S @ XtX_inv * (n / max(n - k, 1))
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    # Normal approximation is fine at these sample sizes (n in the thousands).
    p = 2.0 * 0.5 * np.erfc(np.abs(t) / math.sqrt(2.0))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, se, t, p, r2, n


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def load_blocks(city_dir: Path) -> list[dict] | None:
    for f in sorted(city_dir.glob("*_block_evacuation.json")):
        payload = json.loads(f.read_text(encoding="utf-8"))
        blocks = payload.get("blocks")
        if blocks:
            return blocks
    return None


def build_design(blocks: list[dict]):
    """Tidy the raw block records into a design matrix, dropping only rows that
    are genuinely unusable (missing terrain, or an infinite shelter distance
    which means no shelter is reachable at all -- those become the outcome's
    1s, not predictors)."""
    rows = []
    for b in blocks:
        dist = b.get("shelter_distance_m")
        res = b.get("resistance")
        pop = b.get("population") or 0.0
        egw = b.get("egress_width_m") or 0.0
        if pop <= 0 or egw <= 0:
            continue
        # A run that predates the terrain surface has no resistance field at
        # all; that is not missing data, it is the flat-plane assumption,
        # which is exactly resistance 1.0. Recording it as such keeps those
        # cities in the sample instead of silently dropping every row.
        if res is None:
            res = 1.0
        # Unreachable blocks have no finite distance; give them the city's
        # max finite distance so they stay in the sample as the extreme case
        # rather than being silently dropped (dropping them would remove
        # exactly the outcome we are modelling).
        rows.append({
            "shelter_distance_m": dist,
            "resistance": float(res),
            "egress_width_m": float(egw),
            "population": float(pop),
            "mmi": float(b.get("mmi") or 0.0),
            "unserved": 1.0 if b.get("unserved") else 0.0,
        })
    if not rows:
        return None, None, 0
    finite = [r["shelter_distance_m"] for r in rows if r["shelter_distance_m"] is not None]
    fallback = max(finite) if finite else 6000.0
    for r in rows:
        if r["shelter_distance_m"] is None:
            r["shelter_distance_m"] = fallback

    X = np.column_stack([
        np.ones(len(rows)),
        np.log10([max(r["shelter_distance_m"], 1.0) for r in rows]),
        [r["resistance"] for r in rows],
        np.log10([max(r["egress_width_m"], 0.1) for r in rows]),
        np.log10([max(r["population"], 0.1) for r in rows]),
        [r["mmi"] for r in rows],
    ])
    y = np.array([r["unserved"] for r in rows])
    return X, y, len(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cities", nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    report = {
        "model": "OLS, HC1 robust SE. Outcome: block is unserved (1) vs served (0).",
        "why_block_level": (
            "Nine cities cannot support a multi-mediator model; the blocks are the "
            "sample. Each city is fitted separately over its own blocks and the fits "
            "are compared, rather than pooling nine city-level points."
        ),
        "predictors": [{"key": k, "label": lbl} for k, lbl in PREDICTORS],
        "cities": {},
    }

    for city in args.cities:
        cdir = PORTAL / city
        if not cdir.exists():
            print(f"  {city:14s} SKIP (no directory)")
            continue
        blocks = load_blocks(cdir)
        if not blocks:
            print(f"  {city:14s} SKIP (run used --no-block-detail; no per-block records)")
            continue
        X, y, n = build_design(blocks)
        if X is None or y.sum() == 0 or y.sum() == len(y):
            print(f"  {city:14s} SKIP (no outcome variation: {int(y.sum()) if y is not None else 0} unserved)")
            continue

        beta, se, t, pv, r2, n = ols_hc1(X, y)
        coefs = []
        for i, (key, label) in enumerate(PREDICTORS, start=1):
            coefs.append({
                "key": key, "label": label,
                "beta": round(float(beta[i]), 5),
                "se": round(float(se[i]), 5),
                "t": round(float(t[i]), 3),
                "p": float(pv[i]),
                "stars": stars(float(pv[i])),
                "ci95_lo": round(float(beta[i] - 1.96 * se[i]), 5),
                "ci95_hi": round(float(beta[i] + 1.96 * se[i]), 5),
            })
        report["cities"][city] = {
            "n_blocks": n,
            "unserved_rate": round(float(y.mean()), 4),
            "r2": round(float(r2), 4),
            "coefficients": coefs,
            "distributions": {
                "resistance": summarise([b.get("resistance") for b in blocks if b.get("resistance") is not None]),
                "shelter_distance_m": summarise([b.get("shelter_distance_m") for b in blocks if b.get("shelter_distance_m") is not None]),
                "egress_width_m": summarise([b.get("egress_width_m") for b in blocks if b.get("egress_width_m")]),
            },
        }
        print(f"  {city:14s} n={n:>7,}  unserved={y.mean():6.1%}  R2={r2:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {args.output}")


def summarise(vals):
    if not vals:
        return None
    a = np.array([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "median": round(float(np.median(a)), 4),
        "q1": round(float(np.percentile(a, 25)), 4),
        "q3": round(float(np.percentile(a, 75)), 4),
        "p05": round(float(np.percentile(a, 5)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
    }


if __name__ == "__main__":
    main()
