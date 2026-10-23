# Phase 4 — Collection and coverage

## Objective

Build enough diverse, relevant data to learn robust five-block assembly and diagnose what the corpus still misses.

## Design choices

Scale from pilot evidence, measured collection cost and learning curves. Use the [initial 60/30/10 case mix](../data.md#initial-case-distribution) as a collection proposal, not a fixed loss mixture or expected success rate. Geometry, color, size, language and difficulty vary across all case categories.

Keep both bimanual coordination and correction behavior represented. Broader public data supplements task-relevant Alex experience; it must not overwhelm it by volume.

## Conceptual activities

- Expand nominal scenes across block identities, distractors, layouts, instructions, support geometry and arm roles.
- Cover missed grasps, slips, wrong pieces, off-center placements, occlusion and interrupted coordination with context and corrective continuations.
- Include justified stops for missing pieces, unreachable goals and non-recoverable instability, while retaining feasible difficult cases.
- Transform demonstrations and randomize scenes within physically justified ranges; preserve synchronization between hands.
- Reserve unseen combinations and generation families before creating training windows; measure achieved coverage and collection yield.
- Redirect collection toward errors observed in the first learned policy.

## Expected result

A growing, usable corpus with explicit source and case composition, separate held-out groups, and a visible account of failures and coverage gaps.

## Advancement conditions

The initial corpus supports Phase 5 without observation leakage or unsupported action labels. Each expansion addresses an observed coverage or performance limitation and has practical storage/compute costs. More episodes alone do not close a gap; failed attempts remain counted.
