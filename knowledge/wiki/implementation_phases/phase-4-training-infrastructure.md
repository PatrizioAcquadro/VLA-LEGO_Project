# Phase 4 — Training infrastructure

Scope: original 4.1 and 4.2. Build a correct training path for the maintained VLA and real lightweight-brick data. Prerequisites: a validated Phase 2 minibatch, Phase 3 model integration and the [data/contracts](../data.md) boundary. Synthetic placeholder training infrastructure has been retired completely.

## 4.1 Loss function implementation

**State:** 4.1.0–4.1.2 standalone contracts/text/action modules are implemented and tested in `models/losses.py`. The VLA still computes inline weighted losses. Combined modular objective, integration and production validation in 4.1.3–4.1.4 remain unfinished.

### Implementation sequence

1. **4.1.0 Contract:** use differentiable scalar loss plus detached metric values. Keep one canonical `TokenType`; define full-sequence positions, causal shift, ignored labels, action-valid masks and reduction denominators. Shape asserts are debug helpers, not production input admission. A loader must validate its own actual data contract.
2. **4.1.1 Text loss:** retain causal cross-entropy on valid text targets and ignore other modalities/padding. Shift on the full sequence, never compact disconnected text spans into invented adjacent targets. Empty/all-ignored batches must return a finite differentiable zero. Compute detached accuracy/perplexity only over the valid denominator.
3. **4.1.2 Action loss:** retain float32 velocity regression with declared noise/time convention, mask padding and reduce over valid action dimensions/steps. Fully masked chunks must contribute no gradient. Per-dimension or per-chunk diagnostics must not change the optimization objective.
4. **4.1.3 Combined objective:** compose the maintained modules using explicit fixed text/action weights first. Reuse the VLA's public total/component interface. Preserve the authoritative total in tracking. EMA normalization or learned uncertainty balancing from the old plan is optional and requires measured scale imbalance; do not preinstall unused switches or a second optimizer owner.
5. **4.1.4 Integration/validation:** replace the VLA inline path only after nominal agreement is demonstrated. Test text-only, action-only, mixed and all-masked samples, chunk-tail perturbations, gradient routing and realistic image sequences. A bounded overfit of real replayable demonstrations must be distinguished from mathematical synthetic probes.

### Completion gate

The actual model forward must use the intended loss modules, return finite gradients under valid/masked boundary cases and agree with a hand-computed small example. Padding/non-target perturbations must not affect valid loss or leak future information. Config weights must have real consumers. Validate one real multimodal minibatch on GPU before claiming training readiness. Loss decrease alone cannot establish assembly learning.

### Handoff

Deliver the stable forward/metrics contract and mask semantics to 4.2. Keep logging detached, avoid counting total plus components twice, and define how variable valid-target counts aggregate across workers. Do not claim 4.1 completion from the existence of standalone modules alone.

## 4.2 Distributed training setup

**State:** planned. No maintained trainer, dataset loader, distributed checkpoint path or eight-GPU production script exists. Tracking and scheduler/model diagnostics are reusable components, not a training system.

### Implementation sequence

1. **4.2.0 Real training loop:** implement a minimal single-GPU VLA loop on the Phase 2 pilot. Use actual model outputs and optimizer state, bf16 where appropriate, justified gradient clipping, accumulation and a clear scheduler/step definition. Keep optimizer/zeroing/clipping/scheduler ownership singular. Add DDP or DeepSpeed only when measured scaling requires it; never wrap both or maintain an old generic-MSE fallback.
2. **4.2.1 Data pipeline:** collate real multimodal sequences, normalize from training data, mask tails and use deterministic episode-group-aware sampling. Call `set_epoch` where applicable, handle partial batches and account for rank padding/duplicates. Verify shard coverage and held-out exclusion before a large run. Throughput must report actual samples/tokens rather than accumulated microbatch counts mistaken for optimizer steps.
3. **4.2.2 Checkpoint/resume:** save the trainable model/head or adapted backbone as appropriate, optimizer, scheduler, scaler if used, RNG and data progress. For frozen pretrained weights, record the exact referenced model revision. Save on the correct ranks and reload through the same ownership path. Interrupt and resume a bounded real run; compare next-step state/loss within declared numerical tolerance. An uploaded weight file alone is not resumability.
4. **4.2.3 Scale and benchmark:** qualify one GPU first, then one node, then multi-node/eight-GPU only with available allocation. Inspect actual topology; use the cluster's permitted scheduler launcher instead of hardcoded interfaces or node SSH. Measure end-to-end load, forward/backward, communication, checkpoint time, peak memory and failure/timeout cells. Add ZeRO-1 only if its optimizer-state tradeoff is justified by actual model/freeze memory.

### Completion gates

A bounded real-data GPU run must lower the intended objective without invalid/masked leakage and resume correctly. Distributed sampling must preserve the defined sample denominator, and finite gradients/updates must agree with the single-GPU reference within justified precision limits. Every claimed device/node scale needs an actual run. The original target of eight A100 GPUs is a plan, not an available or validated capability.

### Handoff

Deliver a runnable configuration, validated launcher, restart recipe, measured resource budget and preserved failure report to Phase 5. Establish monitoring/stop criteria before launching a long run. Do not add a permanent synthetic VLA dataset merely to support tests; use small meaningful fixtures for isolated loss/collation contracts and real replayable episodes for integration.

## Source trace

Original roadmap and baseline `docs/roadmap/phase4.1.md` / `phase4.2.md`. References in the old plan to adapting `train/trainer.py`, `data/loader.py`, old ZeRO JSON or toy eight-GPU jobs are obsolete: those consumers were removed by the cleanup. The architectural intent is preserved here with a smaller real-data-first implementation.
