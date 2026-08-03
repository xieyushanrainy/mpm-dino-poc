#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "v2" / "src"), str(ROOT / "v3" / "src"), str(ROOT / "v4" / "src")]
from mpm_dino_v4.v43_train import build_bank, train_v43  # noqa: E402


def main(args):
    matrix = json.loads(Path(args.matrix).read_text())
    if args.top_k != matrix["top_k"] or args.memory_tokens != matrix["memory_tokens_per_source"]:
        raise ValueError("command differs from frozen rejection matrix")
    manifest = json.loads(Path(args.manifest).read_text())
    bank = build_bank(args.dataset, manifest)
    root = Path(args.runs); root.mkdir(parents=True, exist_ok=True)
    (root / "BANK_MANIFEST.json").write_text(json.dumps(bank.manifest(), indent=2) + "\n")
    mapping = {
        "compact_baseline": ("compact_memory", False),
        "compatibility_gate": ("compatibility_gate", False),
        "compatibility_gate_wrong_training": ("compatibility_gate_wrong_training", True),
    }
    for arm in args.arms:
        architecture, wrong_training = mapping[arm]
        for seed in args.seeds:
            output = root / arm / f"seed{seed}"
            if (output / "RUN_COMPLETE.json").exists():
                print(f"skip complete: {output}", flush=True); continue
            train_v43(args.dataset, manifest, args.champion, bank, output, seed,
                      "aligned_dino", device=args.device, epochs=args.epochs,
                      draws=args.draws, lr=args.lr, accumulation=args.accumulation,
                      patience=args.patience, max_batches=args.max_batches,
                      top_k=args.top_k, memory_tokens=args.memory_tokens,
                      architecture=architecture, wrong_training=wrong_training,
                      wrong_safety_weight=matrix["wrong_training"]["base_safety_weight"],
                      wrong_gate_weight=matrix["wrong_training"]["wrong_gate_weight"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument("--manifest", default="v41/manifests/v41_uid_splits.json")
    parser.add_argument("--matrix", default="v43/REJECTION_MATRIX.json")
    parser.add_argument("--champion", default="v42/checkpoints/v42_adapter_full_seed42_best.pt")
    parser.add_argument("--runs", default="v43/runs/bad_neighbour_rejection_seed42_123_456")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--arms", nargs="+", choices=("compact_baseline", "compatibility_gate", "compatibility_gate_wrong_training"), default=["compact_baseline", "compatibility_gate", "compatibility_gate_wrong_training"])
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--memory-tokens", type=int, default=32)
    parser.add_argument("--max-batches", type=int)
    main(parser.parse_args())
