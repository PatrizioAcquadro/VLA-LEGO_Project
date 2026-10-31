# Alex003 Bimanual Assembly VLA

Research software for an ordered five-block tower built by Purdue Alex003, with both arms cooperating and a vision-language-action policy selecting the requested pieces among distractors. The development path uses Isaac Sim/Lab and SmolVLA, then targets inference entirely on the robot's Jetson AGX Orin.

[![CI](https://github.com/PatrizioAcquadro/VLA-LEGO_Project/actions/workflows/ci-cd.yml/badge.svg?branch=main)](https://github.com/PatrizioAcquadro/VLA-LEGO_Project/actions/workflows/ci-cd.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-2E7D32)](LICENSE)

[Project status](knowledge/wiki/status.md) · [Documentation](knowledge/wiki/index.md) · [Roadmap](knowledge/wiki/implementation_phases/index.md) · [Validation](knowledge/wiki/validation.md)

## Current capability

The contact-based Isaac teacher has produced an admitted simulation corpus of **620 episodes / 2,528,001 frames**, including **30 corrections from actual policy states**. Sensor/action recordings, error masks, family partitions and failed attempts are retained. Shared Alex assets supply the WSG32/UMI grippers and head ZED; this repository owns task scenes, control, teachers, data and policy integration.

The completed learning campaign is a **negative result**: the separate decision classifier fails calibration, the 20k-update motor control scores **0/10**, and the final comparison scores **0/20 for both the current candidate and historical reference**. All forty comparison attempts executed once. Teacher demonstrations and isolated learned motor components do not establish autonomous cooperative assembly. Orin timing, physical transfer and final qualification remain open.

| Maintained component | Purpose |
|---|---|
| `vla_lego.isaac` | Scene/control adapters, contact-based teachers, recording, replay and independent task evaluation |
| `vla_lego.pretrained` | Admitted-data readers, TRAIN-only normalization, SmolVLA motor learning, temporal decisions and inference |
| `scripts/` | Native launcher, bounded policy/panel execution, offline reports and development checks |
| `knowledge/wiki/` | Canonical operating guides, contracts, evidence and nine development phases |

## Development setup

Use Python 3.10+ in a dedicated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
make check
make test
make package
```

This installs the portable software and reporting tools. Native simulation additionally needs the compatible Isaac/Alex environment; policy execution needs the separate LeRobot environment. See [development](knowledge/wiki/development.md), [native operations](knowledge/wiki/foundations.md) and [compute](knowledge/wiki/cluster.md) for their boundaries.

Raw recordings, weights, caches and diagnostic outputs are intentionally outside Git. The [data guide](knowledge/wiki/data.md) identifies their canonical location and admission rules; cloning this repository does not download the corpus or models.

## Project boundary

Completion requires the qualified five-block system in simulation and on Purdue Alex with onboard inference. LEGO connections and additional structures are optional later work. MuJoCo/Qwen, its five CLI commands and its deployment infrastructure are retired from `main`; the immutable [v1.0.0 release](https://github.com/PatrizioAcquadro/VLA-LEGO_Project/tree/v1.0.0) preserves that baseline and its attribution. `main` contains unreleased v2 development.

Project code is MIT licensed. Imported block assets retain their [source license and provenance](src/vla_lego/isaac/assets/blocks/PROVENANCE.md); shared robot assets remain owned by the [Alex repository](https://github.com/PatrizioAcquadro/purdue-alex-assets).
