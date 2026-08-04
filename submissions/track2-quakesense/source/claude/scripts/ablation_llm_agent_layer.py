#!/usr/bin/env python3
"""Ablation study for the LLM agentic layer: does per-agent memory and RAG
knowledge retrieval actually change evacuation outcomes, or are they only
verified as "non-empty" (see scripts/demo_memory_rag_pass.py)? Also compares
LLM-controlled coordination against pure rule-based agents.

Five conditions, each run over several seeds against the real local LLM
backend (no mocking):
    A. rule_only            -- llm_runtime_enabled=False (no LLM at all)
    B. llm_full              -- LLM + memory + RAG (current default)
    C. llm_no_memory         -- LLM + RAG, memory context disabled
    D. llm_no_rag            -- LLM + memory, RAG context disabled
    E. llm_no_memory_no_rag  -- LLM with neither

Reports mean +/- std of: evacuated fraction, trapped fraction, mean/p90
evacuation minutes, early-return count, and (for LLM conditions) decision
validity/fallback rate -- honestly, whatever the numbers show, per this
project's standing discipline of reporting negative/null results plainly
rather than tuning until a difference appears.

Run llm/start-llama server first, then:
    python scripts/ablation_llm_agent_layer.py --seeds 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/workspace/xichang-agentic-evacuation")

from geo.download_osm import make_demo_graph  # noqa: E402
from llm.local_backend import LocalLlamaBackend  # noqa: E402
from simulator.model import EvacuationModel, ScenarioConfig  # noqa: E402

CONDITIONS = {
    # llm_runtime_enabled=False alone does NOT stop the initial LLM decision
    # made in EvacuationModel._initial_plans() -- only the periodic
    # re-planning cycle in step(). defer_initial_llm=True is required too,
    # for a genuine zero-LLM-call baseline (caught by this ablation's own
    # first run: rule_only showed 4 llm_calls before this fix).
    "rule_only":           dict(llm_runtime_enabled=False, defer_initial_llm=True, memory_enabled=True,  rag_enabled=True),
    "llm_full":            dict(llm_runtime_enabled=True,  memory_enabled=True,  rag_enabled=True),
    "llm_no_memory":       dict(llm_runtime_enabled=True,  memory_enabled=False, rag_enabled=True),
    "llm_no_rag":          dict(llm_runtime_enabled=True,  memory_enabled=True,  rag_enabled=False),
    "llm_no_memory_no_rag":dict(llm_runtime_enabled=True,  memory_enabled=False, rag_enabled=False),
}


def run_one(backend, seed: int, overrides: dict) -> dict:
    config = ScenarioConfig(
        agents=30, llm_agents=4, shelters=3, building_count=24,
        # Harder than the first pass (which produced a ceiling effect:
        # evacuated_fraction=1.0 and identical mean_evacuation_minutes in
        # every condition) -- more/targeted blockage and tighter shelter
        # capacity so there is real scope for a reroute or admission
        # decision to matter.
        blocked_ratio=0.18, hazard_driven_blockage=True,
        rumor_ratio=0.15, information_delay_steps=3,
        early_return_ratio=0.12,
        shelter_capacity_factor=0.8,
        duration_minutes=40, step_seconds=10,
        llm_decision_interval_steps=6, llm_decisions_per_step=2, llm_max_calls_per_agent=8,
        seed=seed,
        **overrides,
    )
    model = EvacuationModel(config, make_demo_graph(8, 8), backend)
    result = model.run()

    total = result["total_agents"]
    valid = [d for d in model.llm_decisions if d["output_valid"]]
    fallback = [d for d in model.llm_decisions if not d["output_valid"]]
    return {
        "evacuated_fraction": result["evacuated"] / total,
        "trapped_fraction": result["trapped"] / total,
        "mean_evacuation_minutes": result["mean_evacuation_minutes"],
        "p90_evacuation_minutes": result["p90_evacuation_minutes"],
        "early_returns": result["early_returns"],
        "llm_calls": model.llm_calls,
        "valid_decisions": len(valid),
        "fallback_decisions": len(fallback),
    }


def summarize(rows: list[dict]) -> dict:
    keys = [k for k in rows[0].keys()]
    out = {}
    for k in keys:
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
    p.add_argument("--server-url", default="http://127.0.0.1:8080")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("results_llm_demo/ablation_llm_agent_layer.json"))
    args = p.parse_args()

    backend = LocalLlamaBackend(server_url=args.server_url)
    health = backend.health()
    print(f"backend health: {health}")
    if not health.get("ok"):
        raise SystemExit("llama.cpp server not reachable -- start it first")

    seeds = [1000 + i for i in range(args.seeds)]
    report = {"seeds": seeds, "conditions": {}}
    started_all = time.perf_counter()

    for cond_name, overrides in CONDITIONS.items():
        print(f"\n=== {cond_name} ({overrides}) ===")
        rows = []
        for seed in seeds:
            t0 = time.perf_counter()
            row = run_one(backend, seed, overrides)
            dt = time.perf_counter() - t0
            print(f"  seed={seed} evacuated={row['evacuated_fraction']:.3f} "
                  f"trapped={row['trapped_fraction']:.3f} early_returns={row['early_returns']} "
                  f"llm_calls={row['llm_calls']} ({dt:.1f}s)")
            rows.append(row)
        report["conditions"][cond_name] = {"per_seed": rows, "summary": summarize(rows)}

    report["wall_seconds"] = round(time.perf_counter() - started_all, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nwritten to {args.output}")
    print("\n=== summary (evacuated_fraction mean +/- std) ===")
    for cond_name in CONDITIONS:
        s = report["conditions"][cond_name]["summary"]["evacuated_fraction"]
        print(f"  {cond_name:24s} {s['mean']:.3f} +/- {s['std']:.3f}  (n={s['n']})")


if __name__ == "__main__":
    main()
