# QuakeSense source snapshot

This directory contains the code-only implementation. Data, run outputs,
generated pages, presentations, and media are intentionally excluded.

## Main modules

- `claude/simulator/agent_swarm.py` — device-resident agent state
- `claude/simulator/block_graph.py` — block adjacency
- `claude/simulator/block_routing.py` — shelter routing
- `claude/simulator/jurisdictions.py` — leader jurisdictions
- `claude/simulator/llm_leaders.py` — local LLM planning and tool validation
- `claude/simulator/swarm_step.py` — movement and admission kernels
- `claude/scripts/bench_swarm.py` — local benchmark runner
- `claude/scripts/guard.py` — conservation and capacity checks
- `webgis/build_unified_webgis.py` — WebGIS generator

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the ROCm build of PyTorch appropriate to the local host separately.

## Private data preparation

Source rasters, road networks, shelter files, generated caches, knowledge
assets, and run outputs are not committed. Build or provide them locally using
the scripts under `claude/scripts/`.

```bash
python -m claude.scripts.build_city_caches --help
python -m claude.scripts.fetch_terrain_resistance --help
python -m claude.scripts.download_city_roads --help
python -m claude.scripts.bench_swarm --help
```

## Tests

```bash
python -m pytest claude/tests -q
```

Do not commit generated data, run evidence, WebGIS exports, decks, or media.
