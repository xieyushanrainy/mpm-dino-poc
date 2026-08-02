#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "v2" / "src"), str(ROOT / "v3" / "src"), str(ROOT / "v4" / "src")]

from mpm_dino_v4.v43_train import ARMS, build_bank, train_v43  # noqa: E402


def main(args):
    manifest = json.loads(Path(args.manifest).read_text())
    bank = build_bank(args.dataset, manifest)
    bank.save(Path(args.runs) / "retrieval_bank.pt")
    matrix = json.loads(Path(args.matrix).read_text())
    if matrix["top_k"] != args.top_k or matrix["memory_tokens_per_source"] != args.memory_tokens:
        raise ValueError("command disagrees with frozen matrix")
    for arm in args.arms:
        for seed in args.seeds:
            output = Path(args.runs) / arm / f"seed{seed}"
            if (output / "RUN_COMPLETE.json").exists():
                print(f"skip complete: {output}", flush=True); continue
            train_v43(args.dataset, manifest, args.champion, bank, output, seed, arm,
                      device=args.device, epochs=args.epochs, draws=args.draws,
                      lr=args.lr, accumulation=args.accumulation,
                      patience=args.patience, max_batches=args.max_batches,
                      top_k=args.top_k, memory_tokens=args.memory_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="v41/dataset")
    parser.add_argument("--manifest", default="v41/manifests/v41_uid_splits.json")
    parser.add_argument("--matrix", default="v43/MATRIX_CONFIG.json")
    parser.add_argument("--champion", default="v42/checkpoints/v42_adapter_full_seed42_best.pt")
    parser.add_argument("--runs", default="v43/runs/deformation_retrieval_seed42_456")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 456])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--memory-tokens", type=int, default=32)
    parser.add_argument("--max-batches", type=int)
    main(parser.parse_args())
