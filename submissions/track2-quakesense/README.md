# QuakeSense

**AMD AI DevMaster Hackathon 2026 — Track 2: Development & Local Deployment of Private AI Agents**

**Participant:** Bian Chunlin (`bcl200n`)

**Application:** QuakeSense
**Research question:** *When does LLM-driven leadership materially change an emergency evacuation?*

QuakeSense is a locally deployed Agentic AI system that connects LLM leaders,
millions of resident agents, real city road networks, capacity-constrained
shelters, and an inspectable WebGIS. The LLM layer does not merely describe a
simulation: leaders observe local jurisdiction briefs, reason with compact
memory, plan broadcasts, call validated tools, change resident beliefs and
routes, and replan as the simulated city evolves.

## Submission deliverables

| Requirement | Deliverable |
|---|---|
| Project specification | [`docs/QuakeSense_Project_Specification_EN.md`](docs/QuakeSense_Project_Specification_EN.md) |
| Source code | [`source/`](source/) and the [full development repository](https://github.com/bcl200n/earthquake-early-warning/tree/session/quakesense-terrain-and-track2) |
| Reproduction guide | This README and [`source/README.md`](source/README.md) |
| Demo video | [`demo/QuakeSense_Agentic_AI_AMD_Track2_Demo_EN.mp4`](demo/QuakeSense_Agentic_AI_AMD_Track2_Demo_EN.mp4) — 5:19, stable 1080p video with English narration and burned-in subtitles |
| Separate subtitles | [`demo/QuakeSense_Agentic_AI_AMD_Track2_Demo_EN.srt`](demo/QuakeSense_Agentic_AI_AMD_Track2_Demo_EN.srt) |
| Presentation | [`docs/QuakeSense_AMD_Track2_Deck_EN.pptx`](docs/QuakeSense_AMD_Track2_Deck_EN.pptx) |
| Radeon run evidence | [`evidence/xian_llm_run/`](evidence/xian_llm_run/) |

## Official Track 2 compliance map

| Official requirement | Exact review location |
|---|---|
| Application scenarios | Project specification, Section 1; presentation, Slides 1 and 8–11 |
| Agent architecture diagram | Project specification, Section 2; presentation, Slides 3–4 |
| Introduction to core capabilities | Project specification, Sections 3–4; presentation, Slides 3–7 |
| Model introduction and local deployment plan | Project specification, Section 5; this README, **Quick start** |
| AMD Radeon inference-speed optimization | Project specification, Section 6; presentation, Slides 12–13; raw run evidence |
| Complete source repository | [`source/`](source/) plus the linked full development repository |
| Environment, startup guide, dependencies | This README and [`source/README.md`](source/README.md) |
| 3–5 minute actual-operation demo | 5:19 English video; agent/WebGIS mechanism and operation at 0:37–3:17 |
| Radeon GPU execution from runtime to result | Video at 3:56–4:59; presentation, Slides 12–14; [`evidence/xian_llm_run/`](evidence/xian_llm_run/) |
| Supplementary PPT or poster | Editable English PPT, Slides 1–16 |

## What judges can verify

### Functional completeness and application value

- A ten-city pipeline covering Chengdu, Xi'an, Taipei, Noto, L'Aquila,
  Naples, Wellington, Mandalay, Kathmandu, and Los Angeles.
- A closed loop from population, roads, terrain, shelters, and capacity to
  agent decisions, validated tool execution, simulator state, metrics, and
  an English-first WebGIS.
- Baseline and LLM-leadership scenarios, resident-state layers, hierarchical
  road layers, and timeline playback.
- Explicit conservation and capacity guards: the submitted Xi'an run has zero
  population-conservation error and zero overfilled shelters.

### Agentic AI capability

- **Reasoning and planning:** jurisdiction leaders receive local briefs and
  choose shelter broadcasts under congestion and capacity constraints.
- **Tool use:** grammar-constrained JSON actions validate shelter identifiers
  before execution.
- **Memory:** compact per-leader context is updated after each decision round.
- **Task execution:** broadcasts mutate resident knowledge, destination choice,
  and movement in the simulator.
- **Multi-agent coordination:** 350 LLM leaders coordinate 12.95 million
  resident agents while exchanging jurisdiction-level information.

### AMD Radeon and ROCm

The measured Xi'an run used a local Qwen2.5-7B-Instruct Q4_K_M model through a
ROCm-enabled `llama.cpp` server on an AMD Radeon `gfx1100` GPU with 48 GiB VRAM.

| Metric | Measured value |
|---|---:|
| ROCm version | 7.2.1 |
| LLM leaders | 350 |
| Concurrent inference slots | 32 |
| Validated LLM decisions | 4,242 |
| Failed / unparseable replies | 0 / 0 |
| Decisions applied to simulator state | 100% |
| Mean LLM decision time | 0.285 s |
| Resident belief updates | 239,835 |
| Peak simulator VRAM | 8.34 GiB |
| Torch VRAM cap | 42% of 48 GiB |

The run summary and raw log are committed under
[`evidence/xian_llm_run/`](evidence/xian_llm_run/), rather than copied only
into presentation text.

## Quick start

### 1. Create the Python environment

```bash
cd submissions/track2-quakesense/source
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

ROCm-enabled PyTorch must be installed from the wheel/index appropriate to the
host ROCm version. The measured host used PyTorch `2.13.0+rocm7.2`.

### 2. Start the local LLM server

```bash
llama-server \
  -m /models/qwen2.5-7b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 999 -c 4096 --parallel 32
```

`-ngl 999` offloads all model layers to the Radeon GPU. The server is bound to
localhost so prompts, leader memory, and emergency state remain local.

### 3. Run the full-population benchmark

The benchmark expects a generated city cache and routing cache. The complete
data-building commands are documented in [`source/README.md`](source/README.md).

```bash
cd source
python -m claude.scripts.bench_swarm \
  --city xian \
  --llm-url http://127.0.0.1:8080 \
  --llm-leaders 350 \
  --llm-slots 32 \
  --leader-interval-min 10 \
  --steps 1080
```

### 4. Inspect the WebGIS

```bash
python -m http.server 8788 --directory source/webgis/site
```

Open `http://127.0.0.1:8788/`. The committed static site is a lightweight
judge-facing snapshot; `source/webgis/build_unified_webgis.py` generates the
full ten-city interface from run outputs.

## Repository layout

```text
track2-quakesense/
├── README.md
├── docs/                 specification and editable PowerPoint deck
├── demo/                 narrated MP4 and SRT subtitles
├── evidence/             raw Xi'an LLM run summary and log
└── source/
    ├── claude/
    │   ├── simulator/    resident swarm, routing, jurisdictions, LLM leaders
    │   ├── geo/          block, population, and terrain preparation
    │   ├── rl/           shelter admission controllers
    │   ├── scripts/      build, run, benchmark, guard, and replay tools
    │   └── tests/        regression and invariant tests
    ├── rag_knowledge/    local emergency guidance corpus
    └── webgis/           unified WebGIS builder and judge-facing snapshot
```

## Scope and limitations

- Leadership is an intervention mechanism, not a guarantee of better outcomes.
  Several cities improve early while others decline, and some long-run effects
  reverse.
- Most shelter capacities outside Chengdu are scenario assumptions rather than
  surveyed operational capacities.
- The submitted run is deterministic evidence for executable capability; more
  random seeds and confidence intervals are future scientific validation.
- A same-model CPU-versus-ROCm benchmark remains useful follow-up evidence; this
  submission reports measured local Radeon execution without inventing that
  comparison.

This project is a research and decision-support prototype, not an official
emergency-management system.
