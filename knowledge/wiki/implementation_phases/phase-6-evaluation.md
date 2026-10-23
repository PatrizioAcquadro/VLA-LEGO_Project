# Phase 6 — Evaluation and benchmarking

Scope: original 6.1, 6.2 and 6.3. **State:** brick pose/support/stability evaluation exists as a simulation component; learned policy evaluation, LIBERO and ablations are planned. A prearranged diagnostic is not a learned rollout. Prerequisites: qualified Phase 5 checkpoints and a fixed held-out task protocol.

## 6.1 Lightweight-brick task evaluation

### Protocol and implementation

1. Define task families: pick-and-place, stable towers and planar arrangements, then additional geometry/bimanual variants only after implementation. Freeze held-out seeds, dimensions/layouts, instruction variants, lighting/background changes, timeouts and attempt limits before evaluating candidates.
2. Implement a closed-loop observation → model → normalized action → `SimRunner` runner. Declare executed chunk length/replanning frequency and include inference/actuation time. Keep any external teacher or fallback controller in a separate labeled condition; never hide failed learned actions behind corrective execution.
3. Evaluate correct pose/orientation, positive support, robot release and dwell stability using calibrated task tolerances. Retain task truth only in the evaluator. One image, transient contact or a piece still held by the gripper cannot establish success.
4. Report episode success over all attempted episodes, per-placement accuracy/error, stable-release rate, completion time, total placement attempts and recovery outcomes. Unavailable/time-limited rollouts remain visible in the denominator and failure table.
5. Analyze failure modes and, when actually exercised, bimanual coordination: collisions, arm interference, unnecessary simultaneous motion, failed handoff and synchronization. Do not claim bimanual skill from the existence of two arms.

### Completion gate

Deliver fixed-protocol rollout records, evaluator configuration, full counts, aggregate uncertainty appropriate to the sample size and representative success/failure videos. Original >50/>70/>85% website targets are historical aspirations; adopt explicit calibrated brick-specific acceptance criteria before running the final comparison. Original >98% LEGO connection quality has no direct meaning for these pieces.

Generalization claims must name the held-out axis: task arrangement, dimensions, instruction or appearance. A training-seed replay is not an unseen task. Completion time and inference latency must be measured, not inferred from a 20-Hz control declaration.

## 6.2 LIBERO benchmark

### Implementation sequence

1. Determine whether the current Alex-specific architecture and training exposure support a meaningful external embodiment evaluation. Define a separate observation/action adapter with explicit dimensions, normalization, cameras and state semantics; do not stretch Alex commands into a different robot silently.
2. Set up the official benchmark and exact version/task split. Run a known reference policy or official smoke to verify the environment/evaluator before testing this model. Keep its dependencies optional and isolated until this phase is active.
3. Evaluate all four originally proposed suites under their stated protocol, using fixed seeds/rollout counts and no task-specific test leakage. Record incompatible or unimplemented suites as unavailable rather than assigning a flattering subset average.
4. Compare reported EO-1 or other numbers only when embodiment, training data, task split and evaluation protocol are sufficiently comparable. Label non-comparable settings and additional adaptation data explicitly.

### Completion gate

Produce per-suite/per-task counts, failures, adapter definition, inference settings and a source-linked comparison with protocol limitations. LIBERO performance is a separate external benchmark; it neither qualifies Alex brick assembly nor physical transfer. No dependency, score or runnable LIBERO claim exists in the current repository.

## 6.3 Ablation studies

### Experimental design

The original comparisons remain: interleaved versus standard observation/action training; flow matching versus autoregressive actions; and action chunk size. Each requires an actual comparable alternative implementation, fixed data/splits, compute/exposure, observation boundary and evaluator. Avoid adding alternatives merely to manufacture a test suite.

Preregister cells, seeds, budgets, outcome metrics and stopping conditions. Change one interpretable factor at a time where possible. Report task success, placement/release errors, latency/memory and uncertainty, including failed runs and all seed outcomes. A larger model/data/compute budget is a confound, not evidence for the isolated architectural choice.

For interleaving, verify causal/positional semantics and compare to the simplest functioning single-observation baseline. For action decoding, account for different token counts and inference budgets. For horizon, compare planned versus executed steps and replanning, rather than only the predicted tensor length.

### Completion gate and handoff

Deliver an interpretable comparison table with all denominators, negative results and limits on causal interpretation. Keep the strongest supported simple configuration as a candidate; complexity must earn its cost. Phase 7 receives a declared baseline and adaptation needs, while Phase 8 receives observed transfer risks rather than unsupported performance predictions.
