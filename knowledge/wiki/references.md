# References and design differences

## Planning sources

The [original project roadmap](https://websitepresentation.vercel.app/roadmap) describes 22 phases and a January 16–April 4, 2026 calendar. Its detailed plans, plus the repository's historical drafts, were reconciled into the [implementation index](implementation_phases/index.md). The calendar is historical, not a current deadline. The website was not changed during this cleanup.

## EO-1

[EO-1 paper](https://arxiv.org/abs/2508.21112), [original reference repository](https://github.com/EO-Robotics/EO1) and its linked [successor repository](https://github.com/SHAILAB-IPEC/EO1) are external research references. Verify the exact source revision before adapting code. The original repository is archived and points to the successor; the project is not a full EO-1 replication.

The retained design follows the broad idea of a shared multimodal decoder with autoregressive text and continuous flow-matching actions. Current code uses Qwen3.5-4B, a 52-D Alex state, 17-D actions and 16-step chunks. The historical EO-1 reference used Qwen2.5-VL-3B and different embodiment/chunk dimensions. Their time/noise conventions differ: this implementation integrates from noise at `t=0` to data at `t=1`. Never copy a velocity sign or solver step without checking its convention.

For targeted source reading, inspect EO-1's `modeling_eo1.py` for prefix/suffix embeddings and denoising, `processing_eo1.py` for image/state/action preparation, configuration/constants for normalization and tokens, and training/data modules for sampling and optimization. Reuse concepts through the existing local interfaces; do not vendor a second model stack or assume external example evaluation proves task success.

[Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B) is the checkpoint reference. The native Transformers processor owns image expansion and modality metadata. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) is the mathematical reference for the conditional flow-matching path. The project's specific action/control semantics remain defined by local [contracts](contracts.md).

## Physical assets and printing

IHMC/SAKE source details, pinned SDK revision and local modifications remain in [robot provenance](../../src/vla_lego/sim/assets/robots/alex/PROVENANCE.md). The optional geometric printing reference and its reuse boundary are documented once in [bricks](bricks.md). No external printable mesh is a runtime dependency.
