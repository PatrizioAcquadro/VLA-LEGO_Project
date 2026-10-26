"""Train the separate compact decision head, then confirm frozen calibration once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_lego.pretrained.decision_evaluation import (
    calibrated_probabilities,
    choose_thresholds,
    evaluate_sequences,
    fit_calibration,
    probability_metrics,
    selection_rank,
)
from vla_lego.pretrained.decision_features import DecisionFeatures
from vla_lego.pretrained.pilot_campaign import write_json
from vla_lego.pretrained.task_decisions import CLASSES, make_temporal_head


def sequence_logits(head, dataset, batch_size=256):
    import torch

    result = []
    head.eval()
    with torch.inference_mode():
        for sequence in dataset.sequences:
            values = np.zeros((len(sequence["features"]), 3), dtype=np.float32)
            indices = list(sequence["histories"])
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                history = np.stack(
                    [sequence["features"][sequence["histories"][i]] for i in selected]
                ).astype(np.float32)
                logits = head(torch.from_numpy(history).to("cuda"))
                values[selected] = logits.cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite temporal-head logits")
            result.append(values)
    return result


def recorded_rows(dataset, arrays):
    return np.concatenate(
        [
            array
            for sequence, array in zip(dataset.sequences, arrays, strict=True)
            if sequence["kind"] == "recorded_instruction"
        ]
    )


def review(head, dataset):
    logits = sequence_logits(head, dataset)
    recorded = recorded_rows(dataset, logits)
    labels = recorded_rows(dataset, [row["labels"] for row in dataset.sequences])
    calibration = fit_calibration(recorded, labels)
    probabilities = [calibrated_probabilities(values, calibration) for values in logits]
    thresholds, report = choose_thresholds(dataset.sequences, probabilities)
    report.update(
        calibration=calibration,
        thresholds=thresholds,
        probability_metrics=probability_metrics(recorded_rows(dataset, probabilities), labels),
        raw_probability_metrics=probability_metrics(
            calibrated_probabilities(recorded, {"temperature": 1, "bias": [0, 0, 0]}), labels
        ),
        scope="natural calibration sequences and inherited instruction negatives; not confirmation",
    )
    return report


def save_checkpoint(path, checkpoint):
    import torch

    pending = path.with_suffix(".pending")
    torch.save(checkpoint, pending)
    pending.replace(path)


def confirm(args):
    import torch

    destination = args.output / "confirmation.json"
    if destination.exists():
        raise FileExistsError("Confirmation was already exposed; preserve its original result")
    checkpoint = torch.load(args.output / "best.pt", map_location="cpu", weights_only=True)
    if not checkpoint["review"]["passed"]:
        raise ValueError("Complete calibration selection before exposing confirmation")
    source = json.loads((args.features / "protocol.json").read_text())
    if source != checkpoint["feature_protocol"]:
        raise ValueError("Confirmation features differ from the selected model/statistics/corpus")
    dataset = DecisionFeatures(args.features, "confirmation")
    head = make_temporal_head(checkpoint["feature_width"]).to("cuda")
    head.load_state_dict(checkpoint["model"])
    logits = sequence_logits(head, dataset)
    probabilities = [
        calibrated_probabilities(values, checkpoint["review"]["calibration"]) for values in logits
    ]
    report = evaluate_sequences(
        dataset.sequences, probabilities, checkpoint["review"]["thresholds"]
    )
    report.update(
        selected_update=checkpoint["step"],
        probability_metrics=probability_metrics(
            recorded_rows(dataset, probabilities),
            recorded_rows(dataset, [row["labels"] for row in dataset.sequences]),
        ),
        thresholds=checkpoint["review"]["thresholds"],
        calibration=checkpoint["review"]["calibration"],
        scope="first confirmation exposure after frozen model, probability and threshold selection",
        limitation="teacher-observation decision evidence, not autonomous task success or Phase6",
    )
    write_json(destination, report)
    print(json.dumps({"confirmation": report["summary"], "passed": report["passed"]}), flush=True)


def train(args):
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Subset

    if (args.output / "confirmation.json").exists():
        raise ValueError("Confirmation is exposed; do not silently retune this selected experiment")
    dataset = DecisionFeatures(args.features, "train", draws=args.updates * args.batch_size)
    development = DecisionFeatures(args.features, "calibration")
    if dataset.feature_width != development.feature_width:
        raise ValueError("Training/calibration frozen features differ")
    feature_protocol = json.loads((args.features / "protocol.json").read_text())
    protocol = {
        "feature_protocol": feature_protocol,
        "learning": dataset.protocol,
        "classes": list(CLASSES),
        "history_s": 3,
        "history_stride_s": 0.5,
        "hidden_width": 256,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "review_every": args.review_every,
        "optimizer": {"name": "AdamW", "lr": 0.0003, "weight_decay": 0.0001},
        "seed": args.seed,
        "recipe": "project-specific compact head; separate from original SmolVLA motor recipe",
        "selection": "event target violations, premature calls, false abstention, recall, natural NLL",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Cannot resume a different decision experiment in this output")
    write_json(path, protocol)
    torch.manual_seed(args.seed)
    head = make_temporal_head(dataset.feature_width).to("cuda")
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.0003, weight_decay=0.0001)
    step, best_rank = 0, None
    last = args.output / "last.pt"
    if last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=True)
        head.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, best_rank = checkpoint["step"], checkpoint["best_rank"]
    data = DataLoader(
        Subset(dataset, range(step * args.batch_size, len(dataset))),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    loss_sum, loss_count = 0.0, 0
    for batch in data:
        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits = head(batch["history"].to("cuda", non_blocking=True))
        loss = F.cross_entropy(logits, batch["label"].to("cuda", non_blocking=True))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite decision learning loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        loss_sum += float(loss.detach())
        loss_count += 1
        if step % args.review_every and step != args.updates:
            continue
        report = review(head, development)
        rank = (*selection_rank(report), report["probability_metrics"]["nll"])
        checkpoint = {
            "model": head.state_dict(),
            "feature_width": dataset.feature_width,
            "feature_protocol": feature_protocol,
            "step": step,
            "review": report,
            "history_s": 3.0,
            "history_stride_s": 0.5,
            "classes": list(CLASSES),
        }
        if best_rank is None or rank < tuple(best_rank):
            best_rank = rank
            save_checkpoint(args.output / "best.pt", checkpoint)
        save_checkpoint(
            last, {**checkpoint, "optimizer": optimizer.state_dict(), "best_rank": best_rank}
        )
        report.update(update=step, mean_balanced_training_loss=loss_sum / loss_count)
        write_json(args.output / f"review-{step:06d}.json", report)
        print(json.dumps({"step": step, "calibration": report["summary"]}), flush=True)
        loss_sum, loss_count = 0.0, 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--review-every", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.updates <= 20_000 or min(args.batch_size, args.review_every) < 1:
        parser.error("Use positive batch/review intervals and at most20k compact-head updates")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Decision model learning/inference requires native CUDA")
    torch.set_num_threads(8)
    if args.confirm:
        confirm(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
