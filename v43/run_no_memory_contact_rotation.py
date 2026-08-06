#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "v2" / "src"), str(ROOT / "v3" / "src"), str(ROOT / "v4" / "src")]

from mpm_dino_v4.v43_rotation_contact_adapter import VARIANTS, train_contact_adapter  # noqa: E402

MVP_VARIANTS = ("physical_only", "contact_torque_basis", "contact_shuffled")


def main(args):
    matrix = json.loads(args.matrix.read_text())
    if not matrix.get("frozen") or matrix.get("memory_bank_used") or matrix.get("test_data_used"):
        raise ValueError("matrix must be frozen, memory-free and test-sealed")
    if list(args.variants) != matrix["variants"] and not args.allow_subset:
        raise ValueError("variants disagree with frozen matrix; --allow-subset is smoke-only")
    manifest = json.loads(args.manifest.read_text())
    args.runs.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        for seed in args.seeds:
            output = args.runs / variant / f"seed{seed}"
            if (output / "RUN_COMPLETE.json").exists():
                print(f"skip complete: {output}", flush=True); continue
            train_contact_adapter(
                args.dataset, manifest, args.champion, output, seed, variant,
                device=args.device, epochs=args.epochs, draws=args.draws, lr=args.lr,
                accumulation=args.accumulation, patience=args.patience,
                max_degrees=matrix["max_residual_degrees"], max_batches=args.max_batches,
            )
    complete = {"status": "full_matrix_complete", "variants": list(args.variants),
                "seeds": list(args.seeds), "memory_bank_used": False,
                "oracle_contact": True, "deployable": False, "test_data_used": False}
    (args.runs / "RUN_COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("v41/dataset"))
    parser.add_argument("--manifest", type=Path, default=Path("v41/manifests/v41_uid_splits.json"))
    parser.add_argument("--matrix", type=Path, default=Path("v43/NO_MEMORY_CONTACT_ROTATION_MATRIX.json"))
    parser.add_argument("--champion", type=Path, default=Path("v42/checkpoints/v42_adapter_full_seed42_best.pt"))
    parser.add_argument("--runs", type=Path, default=Path("v43/run/no_memory_contact_rotation"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(MVP_VARIANTS))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--allow-subset", action="store_true")
    main(parser.parse_args())
