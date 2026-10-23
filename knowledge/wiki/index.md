# Alex003 Bimanual Assembly VLA

The target is an **ordered five-block tower**, built autonomously by Alex003 from pieces of different colors and sizes among distractors, with genuine cooperation between the arms. Develop it in Isaac Sim/Lab, then transfer it to Purdue Alex with VLA inference entirely on the onboard Jetson AGX Orin.

The repository currently contains MuJoCo/Alex V1 and untrained Qwen-based components. The Isaac/Alex003 task, data collection, trained assembly policy and real deployment are **planned**, not implemented by this documentation revision.

## Start here

| Need | Canonical page |
|---|---|
| See current capability and unresolved decisions | [Status](status.md) |
| Follow the nine development phases | [Roadmap](implementation_phases/index.md) |
| Understand the robot, sensors and ownership | [Architecture](architecture.md) |
| Define deployable observations, actions and timing | [Contracts](contracts.md) |
| Define the tower, printable pieces and progression | [Blocks and tasks](bricks.md) |
| Select public data, teachers, recovery cases and splits | [Data strategy](data.md) |
| Select a pretrained VLA and establish Orin feasibility | [Models](models.md) |
| Allocate local, Blackwell, Purdue and onboard compute | [Compute and containers](cluster.md) |
| Understand success criteria and existing evidence | [Validation](validation.md) |
| Find research rationale and primary sources | [References](references.md) |
| Install or maintain the current software | [Development](development.md) |
| Run existing MuJoCo diagnostics | [Simulation](simulation.md) |
| Use existing experiment logging | [Tracking](tracking.md) |
| Recover retired work and negative results | [History](history.md) |
| Read consequential scope decisions | [Decision log](log.md) |

Topic pages own shared details; phase plans link to them. Operating instructions describe the implementation they actually run. The names `VLA-LEGO_Project` and `vla-lego` remain for continuity; LEGO connections belong to the later [extension phase](implementation_phases/phase-8-consolidation.md).
