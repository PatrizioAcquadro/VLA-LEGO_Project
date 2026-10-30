"""Cache frozen CUDA observation features after complete admitted-corpus readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_lego.isaac.campaign import load_campaign
from vla_lego.pretrained.campaign_data import admitted_entries
from vla_lego.pretrained.decision_data import episode_sequences
from vla_lego.pretrained.pilot_campaign import write_json
from vla_lego.pretrained.pilot_dataset import load_observation
from vla_lego.pretrained.pilot_learning import load_policy
from vla_lego.pretrained.task_decisions import frozen_observation_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="lerobot/smolvla_base")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Feature batch size must be positive")
    entries = admitted_entries(args.corpus)
    cases = {case.name: case for case in load_campaign(args.cases)}
    normalization = json.loads(args.statistics.read_text())
    if normalization["action_representation"] != "observation_arm_offset":
        raise ValueError("Feature proprioception must use the declared training normalization")
    statistics = normalization["statistics"]
    protocol = {
        "format": "frozen-observation-features-v2",
        "model": args.model,
        "statistics": statistics,
        "sources": entries,
        "decision_hz": 2,
        "history_s": 3,
        "labels": (
            "current physical outcomes; short legacy single-block terminals use past-aligned "
            "physical replay; stop justification never waits for a scripted clock"
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Cached features belong to another model, normalization or corpus")
    else:
        write_json(protocol_path, protocol)
    import numpy as np
    import torch
    from torch.utils.data import default_collate

    policy, pre, _post = load_policy(args.model, statistics, motor_only=True)
    policy.eval()
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    index = []
    for entry in entries:
        for sequence in episode_sequences(args.corpus, entry, cases[entry["case"]]):
            name = f"{sequence.case}-{sequence.kind}.npz"
            destination = args.output / name
            tokens = len(tokenizer(sequence.instruction + "\n", truncation=False)["input_ids"])
            if tokens > policy.config.tokenizer_max_length:
                raise ValueError(f"Full instruction would be truncated: {sequence.case}, {tokens}")
            if destination.exists():
                with np.load(destination) as cached:
                    if (
                        len(cached["features"]) != len(sequence.indices)
                        or not np.array_equal(cached["labels"], sequence.labels)
                        or not np.array_equal(cached["times"], sequence.times)
                        or not np.isfinite(cached["features"]).all()
                    ):
                        raise ValueError("Incomplete or inconsistent cached decision sequence")
            else:
                pieces = []
                for begin in range(0, len(sequence.indices), args.batch_size):
                    raw = default_collate(
                        [
                            {
                                **load_observation(sequence.directory, sequence.frames[i]),
                                "task": sequence.instruction,
                            }
                            for i in sequence.indices[begin : begin + args.batch_size]
                        ]
                    )
                    features = frozen_observation_features(policy, pre(raw))
                    pieces.append(features.to(dtype=torch.float16).cpu().numpy())
                features = np.concatenate(pieces)
                if not np.isfinite(features).all():
                    raise FloatingPointError("Non-finite compact frozen features")
                pending = destination.with_suffix(".pending")
                with pending.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        features=features,
                        labels=np.asarray(sequence.labels, dtype=np.int64),
                        times=np.asarray(sequence.times, dtype=np.float64),
                    )
                pending.replace(destination)
            index.append(
                {
                    "file": name,
                    "case": sequence.case,
                    "family": sequence.family,
                    "partition": sequence.partition,
                    "kind": sequence.kind,
                    "observations": len(sequence.indices),
                    "instruction_tokens": tokens,
                }
            )
        write_json(args.output / "index.json", {"complete": False, "sequences": index})
        print(json.dumps({"case": entry["case"], "sequences_cached": len(index)}), flush=True)
    write_json(args.output / "index.json", {"complete": True, "sequences": index})


if __name__ == "__main__":
    main()
