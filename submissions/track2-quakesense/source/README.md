# QuakeSense source snapshot and reproduction guide

This directory contains the competition-facing source snapshot. The complete
development history is available in the public repository:

https://github.com/bcl200n/earthquake-early-warning/tree/session/quakesense-terrain-and-track2

## Main modules

- `claude/simulator/agent_swarm.py` — device-resident per-person state.
- `claude/simulator/block_graph.py` — shared-edge block adjacency.
- `claude/simulator/block_routing.py` — exact shelter routing.
- `claude/simulator/jurisdictions.py` — balanced leader jurisdictions.
- `claude/simulator/llm_leaders.py` — local LLM briefs, memory, tools, and validation.
- `claude/simulator/swarm_step.py` — departure, diffusion, movement, admission, and give-up kernels.
- `claude/scripts/bench_swarm.py` — full-run benchmark and evidence writer.
- `claude/scripts/guard.py` — conservation, capacity, and mechanism guards.
- `webgis/build_unified_webgis.py` — ten-city WebGIS generator.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the ROCm build of PyTorch appropriate to the host separately. The
submitted run used Python 3.12, PyTorch 2.13.0+rocm7.2, and ROCm 7.2.1.

## Data preparation

Large source rasters and generated cache files are not committed. Build them
from the scripts in `claude/scripts/`:

```bash
python -m claude.scripts.build_city_caches --help
python -m claude.scripts.fetch_terrain_resistance --help
python -m claude.scripts.download_city_roads --help
```

The pipeline uses openly obtainable population, OpenStreetMap road, terrain,
and shelter sources. Each generated run writes its configuration and totals to
`summary.json`.

## Full run

Start a ROCm-enabled local `llama-server`, then run:

```bash
python -m claude.scripts.bench_swarm --help
```

The submitted configuration used 350 leaders, 32 LLM slots, a ten-minute
leader interval, and 1,080 simulator steps. See
`../evidence/xian_llm_run/summary.json` and the raw log for the authoritative
measured output.

## Tests and guards

```bash
python -m pytest claude/tests -q
python -m claude.scripts.guard --help
```

The guards check population conservation, shelter capacity, legal tool
targets, configuration differences, and whether enabled mechanisms actually
change state.

## WebGIS

The committed `webgis/site/` directory is a lightweight static judge-facing
snapshot. Generate the full ten-city site from run directories with:

```bash
python webgis/build_unified_webgis.py --help
```

Then serve the generated directory with Python's static HTTP server.
