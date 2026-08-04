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
