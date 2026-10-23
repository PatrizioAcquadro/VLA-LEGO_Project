# Lightweight Brick Assembly VLA

This project develops vision-language-action components for Alex to pick up lightweight bricks and arrange them into structures. The current repository provides simulation and model components; it does not yet provide a trained assembly policy.

## Start here

| Need | Canonical page |
|---|---|
| Understand capabilities, limits and priorities | [Status](status.md) |
| Install and contribute | [Development](development.md) |
| Run or inspect the simulation | [Simulation](simulation.md) |
| Understand what counts as a brick or a successful structure | [Bricks and tasks](bricks.md) |
| Understand module ownership and execution | [Architecture](architecture.md) |
| Implement compatible observations, actions and cameras | [Contracts](contracts.md) |
| Use Qwen and the flow-matching head | [Models](models.md) |
| Develop the future dataset | [Data specification](data.md) |
| Operate containers and Gilbreth jobs | [Cluster and containers](cluster.md) |
| Integrate experiment logging | [Tracking](tracking.md) |
| Assess what has actually been validated | [Validation](validation.md) |
| Develop the next phase or review every original plan | [Implementation plans](implementation_phases/index.md) |
| Understand retired work and recover its sources | [Historical findings](history.md) |
| Find external architecture and printing references | [References](references.md) |
| Read consequential project decisions | [Decision log](log.md) |

The package and repository retain the names `vla-lego` and `VLA-LEGO_Project` for continuity. In current development, “bricks” means variable-size lightweight pieces; real LEGO connections are a [future extension](implementation_phases/phase-8-transfer-handoff.md#future-extension-real-lego).
