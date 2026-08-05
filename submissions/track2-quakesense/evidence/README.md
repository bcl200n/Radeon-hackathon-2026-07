# Evidence

`xian_llm_run/summary.json` is the machine-readable full-run artifact and
`xian_llm_run/xian_llm000.log` is the corresponding raw console log from the
AMD Radeon/ROCm host.

The source artifacts support the metrics used in the demo and deck:

- 12,952,879 resident agents;
- 350 LLM leaders and 4,242 grammar-constrained decisions;
- zero failed, unparseable, or hallucinated decisions;
- 100% of validated decisions applied;
- 0.285 seconds mean LLM decision time with 32 slots;
- 239,835 resident belief updates;
- 8.34 GiB peak simulator VRAM;
- zero conservation error and zero overfilled shelters.

The raw artifacts are included so judges do not have to trust manually copied
presentation figures.

## Ten-city maximum-scale run

`max_scale_run/results_bigsim/bigsim_summary.json` is the aggregate measured
run summary. The ten city subdirectories contain the corresponding
machine-readable `summary.json` files, and
`max_scale_run/scenario/squeeze_cities.json` records the scenario population
and shelter inputs used by the run.

The aggregate JSON supports the submitted scale and timing claims:

- 51,853,775 resident agents across ten cities;
- 56.0 billion agent-steps;
- 445 seconds wall-clock;
- 126 million agent-steps per second.

`max_scale_run/runtime/radeon_utilization_capture.png` records the measured
host utilization trace used in the deck: 98% peak GPU utilization and
27.7 GiB peak VRAM on the Radeon `gfx1100` host.

The maximum-scale summaries are baseline simulator evidence. The local LLM
execution evidence remains the Xi'an run above; the submission does not claim
that all 51.9 million residents invoke the language model directly.
