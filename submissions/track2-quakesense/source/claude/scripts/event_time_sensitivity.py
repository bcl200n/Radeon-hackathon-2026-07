#!/usr/bin/env python3
"""Event-time sensitivity analysis: does a 02:00 / 08:00 / 15:00 / 20:00
earthquake produce different evacuation outcomes, using
ScenarioConfig.event_hour (a synthetic building-type reweighting proxy --
see simulator/model.py::TIME_OF_DAY_BUILDING_MULTIPLIERS)?

This directly tests the paper's "time-varying" claim (the title and
introduction both foreground it) at the only fidelity currently available:
event_hour is NOT WorldMove-calibrated real mobility, it is a documented,
labelled synthetic proxy (heavier Residential weighting at night, heavier
Commercial/School weighting at midday) pending a real WorldMove city
release. Reported honestly as that, not as a validated time-of-day effect.

Runs rule-only (llm_agents=0) for speed and determinism -- the mechanism
under test is population placement, not LLM behaviour, which the separate
scripts/ablation_llm_agent_layer.py already covers.

    python scripts/event_time_sensitivity.py --seeds 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/workspace/xichang-agentic-evacuation")

from geo.download_osm import make_demo_graph  # noqa: E402
from llm.mock_backend import MockBackend  # noqa: E402
from simulator.model import EvacuationModel, ScenarioConfig  # noqa: E402

EVENT_HOURS = [2, 8, 15, 20]
HOUR_LABELS = {2: "02:00 (night)", 8: "08:00 (morning)", 15: "15:00 (midday)", 20: "20:00 (evening)"}


def run_one(seed: int, event_hour: int) -> dict:
    config = ScenarioConfig(
        agents=200, llm_agents=0, shelters=5, building_count=70,
        blocked_ratio=0.12, hazard_driven_blockage=True,
        shelter_capacity_factor=1.0,
        duration_minutes=60, step_seconds=10,
        seed=seed, event_hour=event_hour,
    )
    model = EvacuationModel(config, make_demo_graph(10, 10), MockBackend())
    result = model.run()
    total = result["total_agents"]
    return {
        "evacuated_fraction": result["evacuated"] / total,
        "trapped_fraction": result["trapped"] / total,
        "mean_evacuation_minutes": result["mean_evacuation_minutes"],
        "p90_evacuation_minutes": result["p90_evacuation_minutes"],
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    for k in rows[0]:
        values = [r[k] for r in rows if r[k] is not None]
        if not values:
            out[k] = None
            continue
        out[k] = {
            "mean": round(statistics.fmean(values), 4),
            "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--output", type=Path, default=Path("results_llm_demo/event_time_sensitivity.json"))
    args = p.parse_args()

    seeds = [2000 + i for i in range(args.seeds)]
    report = {
        "seeds": seeds,
        "evidence_class": (
            "Synthetic building-type reweighting proxy (TIME_OF_DAY_BUILDING_MULTIPLIERS), "
            "NOT a WorldMove-calibrated time-of-day population distribution. Tests whether "
            "event timing changes outcomes at all, at the only fidelity currently installed."
        ),
        "hours": {},
    }

    for hour in EVENT_HOURS:
        print(f"\n=== event_hour={hour} ({HOUR_LABELS[hour]}) ===")
        rows = [run_one(seed, hour) for seed in seeds]
        for seed, row in zip(seeds, rows):
            print(f"  seed={seed} evacuated={row['evacuated_fraction']:.3f} "
                  f"trapped={row['trapped_fraction']:.3f} mean_min={row['mean_evacuation_minutes']}")
        report["hours"][str(hour)] = {"label": HOUR_LABELS[hour], "per_seed": rows, "summary": summarize(rows)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {args.output}")

    print("\n=== summary (mean_evacuation_minutes, mean +/- std) ===")
    for hour in EVENT_HOURS:
        s = report["hours"][str(hour)]["summary"]["mean_evacuation_minutes"]
        print(f"  {HOUR_LABELS[hour]:20s} {s['mean']:.2f} +/- {s['std']:.2f}  (n={s['n']})")
    print("\n=== summary (evacuated_fraction, mean +/- std) ===")
    for hour in EVENT_HOURS:
        s = report["hours"][str(hour)]["summary"]["evacuated_fraction"]
        print(f"  {HOUR_LABELS[hour]:20s} {s['mean']:.3f} +/- {s['std']:.3f}  (n={s['n']})")


if __name__ == "__main__":
    main()
