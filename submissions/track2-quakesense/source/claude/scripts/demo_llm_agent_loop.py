#!/usr/bin/env python3
"""End-to-end smoke test: real local LLM (Qwen2.5 via llama.cpp/ROCm on the
Radeon GPU) actually driving agent decisions in ``simulator.model.EvacuationModel``.

This is not a new agent framework -- ``agents/planner.py`` and
``llm/local_backend.py`` already existed and already implement the
Observe -> reason -> plan -> call tool -> execute -> replan loop the AMD
hackathon Track 2 wants. What this script proves is that the loop actually
runs against a *real* local model on a *real* Radeon GPU, not a mock/scripted
backend -- swap ``ScriptedBackend`` in the existing test suite for
``LocalLlamaBackend`` and see whether real decisions still validate, still
change simulator state, and what the real latency looks like.

Run llm/start-llama server first (see scripts/start_llama_qwen7b.sh), then:

    python scripts/demo_llm_agent_loop.py --steps 10 --llm-agents 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/workspace/xichang-agentic-evacuation")  # planner/model/tools/llm live in the main repo

from geo.download_osm import make_demo_graph  # noqa: E402
from llm.local_backend import LocalLlamaBackend  # noqa: E402
from simulator.model import EvacuationModel, ScenarioConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server-url", default="http://127.0.0.1:8080")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--agents", type=int, default=40)
    p.add_argument("--llm-agents", type=int, default=4)
    p.add_argument("--shelters", type=int, default=4)
    p.add_argument("--llm-decision-interval-steps", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("results_llm_demo/llm_agent_loop.json"))
    args = p.parse_args()

    backend = LocalLlamaBackend(server_url=args.server_url)
    health = backend.health()
    print(f"[1/3] backend health: {health}")
    if not health.get("ok"):
        raise SystemExit(f"llama.cpp server not reachable at {args.server_url} -- start it first "
                         f"(scripts/start_llama_qwen7b.sh)")

    config = ScenarioConfig(
        agents=args.agents,
        llm_agents=args.llm_agents,
        shelters=args.shelters,
        building_count=30,
        llm_decision_interval_steps=args.llm_decision_interval_steps,
        llm_decisions_per_step=2,
        llm_max_calls_per_agent=6,
        step_seconds=10,
    )
    print(f"[2/3] running {args.steps} steps, {args.llm_agents} of {args.agents} agents LLM-controlled ...")
    model = EvacuationModel(config, make_demo_graph(10, 10), backend)

    initial_targets = {a.agent_id: a.target_shelter for a in model.agents if a.llm_controlled}
    started = time.perf_counter()
    for _ in range(args.steps):
        model.step()
    wall_s = time.perf_counter() - started

    changed = sum(1 for a in model.agents if a.llm_controlled and a.target_shelter != initial_targets.get(a.agent_id))
    valid = [d for d in model.llm_decisions if d["output_valid"]]
    fallback = [d for d in model.llm_decisions if not d["output_valid"]]
    latencies = model.llm_latencies_ms

    report = {
        "backend_health": health,
        "steps": args.steps,
        "wall_seconds": round(wall_s, 2),
        "llm_agents": args.llm_agents,
        "total_llm_calls": model.llm_calls,
        "valid_decisions": len(valid),
        "fallback_decisions": len(fallback),
        "targets_changed_by_llm": changed,
        "latency_ms": {
            "min": round(min(latencies), 1) if latencies else None,
            "mean": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "max": round(max(latencies), 1) if latencies else None,
        },
        "sample_decisions": model.llm_decisions[:5],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[3/3] done")
    print(json.dumps({k: v for k, v in report.items() if k != "sample_decisions"}, indent=2))
    print()
    print("sample decision:")
    print(json.dumps(report["sample_decisions"][0] if report["sample_decisions"] else {}, indent=2, ensure_ascii=False))

    if not valid:
        raise SystemExit("no LLM decision validated -- backend is reachable but produced nothing usable")
    if changed == 0:
        print("WARNING: no LLM-controlled agent's target actually changed -- decisions may be real "
              "but not doing anything observable in this short a run")


if __name__ == "__main__":
    main()
