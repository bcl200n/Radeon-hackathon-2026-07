# QuakeSense Project Specification

**AMD AI DevMaster Hackathon 2026 — Track 2: Development & Local Deployment of Private AI Agents**

**Participant:** Bian Chunlin
**Application:** QuakeSense

## 1. Application scenario

Emergency evacuation is shaped by two interacting systems: physical capacity
and information. Roads and shelters determine what is possible; leadership,
warning diffusion, and local knowledge influence what people attempt. Standard
accessibility maps model the first system but usually hold the second fixed.

We developed **QuakeSense** to test both systems together. It is a city-scale,
locally deployed Agentic AI platform in which LLM leaders coordinate resident
agents during an earthquake evacuation. The same application supports ten
cities and exposes its assumptions, controls, road network, agent cycle, and
outcomes through an English-first WebGIS.

The central research question is:

> When does LLM-driven leadership materially change an emergency evacuation,
> and when do physical constraints dominate the result?

## 2. System architecture

```text
Population + shelters + capacity + terrain + hierarchical roads
                              |
                              v
                 City state and jurisdiction briefs
                              |
              +---------------+---------------+
              |                               |
              v                               v
      350 local LLM leaders        millions of resident agents
      observe local state           maintain beliefs and routes
      reason with memory             move on the road graph
      plan a broadcast               queue at capacity gates
      call validated tools           exchange information
              |                               |
              +---------------+---------------+
                              v
                  simulator state mutation
                              |
                              v
          metrics + run artifacts + inspectable WebGIS
                              |
                              +---- next 10-minute replan
```

The numerical simulator and LLM leadership layer are separated deliberately.
Resident agents provide fine-grained behavior and physical execution. LLM
leaders provide reasoning, planning, memory, and validated tool use. Calling
the resident swarm alone “Agentic AI” would overstate the design; the Track 2
capability is the local leader layer that observes, decides, acts, and replans.

## 3. Agent cycle

Each simulation round follows six stages:

1. **Observe.** Every ten simulation minutes, each leader receives a compact
   jurisdiction brief: resident state, known shelters, route access, occupancy,
   congestion, vulnerable populations, and recent messages.
2. **Reason.** Local Qwen combines the current brief with compact leader memory.
3. **Plan.** The leader selects shelters and warning content for its jurisdiction.
4. **Use tools.** The model emits grammar-constrained JSON. Shelter identifiers
   and action fields are validated before execution.
5. **Execute.** Broadcast tools update resident shelter knowledge, perceived
   routes, and destination choices. Capacity and road gates remain enforced.
6. **Replan.** The result is added to compact memory and the next round observes
   the changed city state.

The zero-knowledge baseline disables leaders, broadcasts, and leader memory.
This control isolates bottom-up resident behavior and makes the effect of the
Agentic AI layer inspectable.

## 4. Core capabilities

| Capability | Implementation evidence |
|---|---|
| Reasoning and planning | Local Qwen leaders select broadcasts from jurisdiction briefs |
| Tool use | Grammar-constrained JSON and shelter-ID validation |
| Memory management | 5.07 MB compact context across the Xi'an leader layer |
| Task execution | 4,242 decisions applied to resident/simulator state |
| Multi-agent coordination | 350 leaders and 12.95 million resident agents |
| Physical grounding | Shared-edge road graph, terrain resistance, congestion, shelter capacity |
| User experience | English WebGIS, city/scenario selection, layers, playback, metrics |
| Generalization | One agent contract and interface across ten cities |

## 5. Model and local deployment

| Item | Submitted configuration |
|---|---|
| Model | Qwen2.5-7B-Instruct |
| Format | GGUF Q4_K_M |
| Serving runtime | ROCm-enabled llama.cpp |
| Endpoint | Localhost-only OpenAI-compatible server |
| GPU | AMD Radeon gfx1100, 48 GiB |
| ROCm | 7.2.1 |
| GPU offload | All model layers (`-ngl 999`) |
| Context | 4,096 tokens per request |
| Concurrency | 32 slots |

Example server command:

```bash
llama-server \
  -m /models/qwen2.5-7b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 999 -c 4096 --parallel 32
```

The LLM server is local by design. Emergency state, observations, leader
memory, and decisions are not sent to an external inference API.

## 6. Radeon and ROCm optimization

The local stack is optimized around four practical risks.

### Throughput

- Q4_K_M quantization reduces model memory and bandwidth requirements.
- All model layers are offloaded to Radeon.
- Thirty-two concurrent slots batch jurisdiction decisions.
- The measured full run averages 0.285 seconds per LLM decision.

### Tool validity

- Grammar-constrained JSON restricts the output schema.
- Shelter identifiers are checked against the current city action space.
- The submitted run has zero failures, zero unparseable replies, and zero
  hallucinated shelter identifiers.

### Shared GPU memory

- Torch is capped at 42% of the 48 GiB device so the resident simulator and
  llama.cpp retain separate working space.
- The measured simulator peak is 8.34 GiB.

### Compact memory

- Leaders receive a bounded local brief instead of the full city state.
- The Xi'an run uses 5.07 MB of aggregate compact leader context.

## 7. Measured Xi'an run

The raw summary and log are included in `evidence/xian_llm_run/`.

| Metric | Result |
|---|---:|
| Resident agents | 12,952,879 |
| Road-enclosed blocks | 36,044 |
| Shelters | 729 |
| LLM leaders | 350 |
| LLM rounds | 15 |
| LLM decisions | 4,242 |
| Failed decisions | 0 |
| Unparseable replies | 0 |
| Hallucinated shelter IDs | 0 |
| Applied decisions | 4,242 (100%) |
| Mean decision time | 0.285 s |
| Resident belief updates | 239,835 |
| Population conservation error | 0 |
| Overfilled shelters | 0 |
| Peak simulator VRAM | 8.34 GiB |

### 7.1 Maximum ten-city baseline scale

The same simulator pipeline was also measured across ten full-population city
runs on the Radeon host. The aggregate evidence is committed under
`evidence/max_scale_run/`.

| Metric | Result |
|---|---:|
| Resident agents | 51,853,775 |
| Agent-steps | 56.0 billion |
| Wall-clock | 445 s |
| Throughput | 126 million agent-steps/s |
| Peak GPU utilization | 98% |
| Peak VRAM | 27.7 GiB |

This is baseline simulator scale evidence. The local LLM leader execution is
demonstrated by the measured Xi'an run above; the specification does not claim
that every resident is an LLM instance.

## 8. Practical result

Leadership changes evacuation trajectories, but the direction depends on the
city. At fifteen minutes, six of eight global comparison cities improve under
the LLM layer, while Noto and L'Aquila decline. Some effects reverse by 120
minutes. This is useful rather than embarrassing: QuakeSense exposes where
information and coordination can help and where capacity, network redundancy,
or harmful concentration dominate.

The application therefore supports two practical questions:

1. Is a poor outcome primarily an access/capacity problem or an information
   coordination problem?
2. If leadership changes the trajectory, which jurisdictions, messages, and
   shelter choices caused the change?

## 9. User experience

The WebGIS provides:

- ten-city switching;
- planning-diagnosis and dynamic-simulation modes;
- baseline and LLM-leadership scenarios;
- the visible Observe–Reason–Plan–Tool–Execute–Replan cycle;
- motorway, primary, secondary, local, and computational road layers;
- resident state layers and timeline playback;
- LLM execution metrics beside the city state.

Roads are not decorative. Static accessibility uses the block-adjacency
network and distance; dynamic evacuation also uses routing, congestion,
information diffusion, and capacity gates.

## 10. Reproducibility and boundaries

The submission includes the core source snapshot, run summary, raw log, demo
video, subtitles, and editable presentation. Generated city rasters, caches,
and large frame arrays are excluded because they are reproducible and would
unnecessarily enlarge the official repository.

The following claims are intentionally not made:

- leadership always improves evacuation;
- assumed shelter capacities are official operational figures;
- one deterministic run establishes causal significance;
- a CPU-versus-ROCm speedup has been measured when the same-model CPU baseline
  has not yet been captured.

Future validation should add repeated seeds, confidence intervals, official
shelter capacities, damaged-road scenarios, parameter calibration, and the
same-model CPU baseline.
