#!/usr/bin/env python3
"""Demo-specific verification pass for per-agent memory (agents/memory.py)
and the local RAG knowledge base (rag/retrieval.py), against a real local
LLM run -- not a unit test in isolation, but evidence that both are actually
populated and consumed inside a live agent-decision loop.

Both mechanisms already exist and are already wired into
agents/planner.py::plan_with_llm (memory.compact_context and
tools.emergency_knowledge are both part of every LLM prompt's context, see
the `compact` dict there). What this script adds is a report showing they
are non-empty and meaningfully populated during an actual run, which is
what a hackathon submission needs as evidence rather than a code pointer.

Run llm/start-llama server first (see scripts/start_llama_qwen7b.sh), then:

    python scripts/demo_memory_rag_pass.py --steps 20 --llm-agents 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/workspace/xichang-agentic-evacuation")

from geo.download_osm import make_demo_graph  # noqa: E402
from llm.local_backend import LocalLlamaBackend  # noqa: E402
from rag.retrieval import LocalKnowledgeBase  # noqa: E402
from simulator.model import EvacuationModel, ScenarioConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server-url", default="http://127.0.0.1:8080")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--agents", type=int, default=40)
    p.add_argument("--llm-agents", type=int, default=4)
    p.add_argument("--shelters", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("results_llm_demo/memory_rag_pass.json"))
    args = p.parse_args()

    backend = LocalLlamaBackend(server_url=args.server_url)
    health = backend.health()
    print(f"[1/4] backend health: {health}")
    if not health.get("ok"):
        raise SystemExit(f"llama.cpp server not reachable at {args.server_url}")

    config = ScenarioConfig(
        agents=args.agents, llm_agents=args.llm_agents, shelters=args.shelters,
        building_count=30, llm_decision_interval_steps=2, llm_decisions_per_step=2,
        llm_max_calls_per_agent=8, step_seconds=10,
    )
    model = EvacuationModel(config, make_demo_graph(10, 10), backend)
    print(f"[2/4] running {args.steps} steps, {args.llm_agents} of {args.agents} agents LLM-controlled ...")
    for _ in range(args.steps):
        model.step()

    # -- memory: show it actually accumulated per-agent, not just that the
    #    class exists --
    llm_agents = [a for a in model.agents if a.llm_controlled]
    memory_report = []
    for agent in llm_agents:
        memory_report.append({
            "agent_id": agent.agent_id,
            "recent_items": len(agent.memory.recent),
            "sample_items": [
                {"text": item.text, "step": item.step, "importance": item.importance, "tags": list(item.tags)}
                for item in agent.memory.recent[:3]
            ],
            "long_term_summary": agent.memory.long_term_summary,
            "compact_context_sample": agent.memory.compact_context(model.step_count, 3),
        })
    print(f"[3/4] memory: {sum(r['recent_items'] for r in memory_report)} total items across "
          f"{len(llm_agents)} LLM-controlled agents")

    # -- RAG: show real retrieval against realistic in-scenario queries,
    #    against the just-expanded (no longer placeholder) knowledge base --
    kb = LocalKnowledgeBase(Path("/workspace/xichang-agentic-evacuation/rag/knowledge"))
    queries = [
        "post-earthquake building evacuation, road blockage and shelter selection",
        "what to do when a shelter reports full",
        "elderly or disabled person needs extra time to evacuate",
        "unverified rumor about a blocked road",
        "fire risk after earthquake",
    ]
    rag_report = []
    for q in queries:
        hits = kb.search(q, limit=2)
        rag_report.append({"query": q, "hits": [{"source": h["source"], "score": h["score"], "text": h["text"][:200]} for h in hits]})
    print(f"[4/4] RAG: {len(kb.chunks)} knowledge chunks across {len(set(c['source'] for c in kb.chunks))} files")

    report = {
        "steps": args.steps,
        "llm_agents": args.llm_agents,
        "total_llm_calls": model.llm_calls,
        "memory": memory_report,
        "rag_knowledge_chunks": len(kb.chunks),
        "rag_knowledge_files": sorted(set(c["source"] for c in kb.chunks)),
        "rag_sample_queries": rag_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {args.output}")

    if sum(r["recent_items"] for r in memory_report) == 0:
        raise SystemExit("no memory items accumulated -- memory is wired but nothing populated it in this run")
    if not any(r["hits"] for r in rag_report):
        raise SystemExit("no RAG hits returned for any sample query")


if __name__ == "__main__":
    main()
