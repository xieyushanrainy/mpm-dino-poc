#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "v2" / "src"), str(ROOT / "v3" / "src"),
                str(ROOT / "v4" / "src"), str(Path(__file__).resolve().parent)]

from mpm_dino_v4.v43_rotation_train import CONTACT_VARIANTS, train_rotation_memory  # noqa: E402
from run_rotation_memory import build_rotation_bank  # noqa: E402


def main(args):
    matrix = json.loads(args.matrix.read_text())
    if not matrix.get("frozen") or matrix.get("test_data_used"):
        raise ValueError("contact-curvature matrix must be frozen and test-sealed")
    if list(args.variants) != matrix["variants"] and not args.allow_subset:
        raise ValueError("variants disagree with frozen matrix; use --allow-subset only for smoke")
    manifest = json.loads(args.manifest.read_text())
    bank = build_rotation_bank(args.dataset, manifest)
    args.runs.mkdir(parents=True, exist_ok=True)
    bank.save(args.runs / "rotation_bank.pt")
    for variant in args.variants:
        for seed in args.seeds:
            output = args.runs / variant / f"seed{seed}"
            if (output / "RUN_COMPLETE.json").exists():
                print(f"skip complete: {output}", flush=True); continue
            train_rotation_memory(
                args.dataset, manifest, args.champion, bank, output, seed, "geometry",
                device=args.device, epochs=args.epochs, draws=args.draws, lr=args.lr,
                accumulation=args.accumulation, patience=args.patience,
                top_k=matrix["top_k"], max_degrees=matrix["max_residual_degrees"],
                smoothness_weight=args.smoothness_weight, max_batches=args.max_batches,
                contact_variant=variant,
            )
    complete = {"status": "full_matrix_complete", "variants": list(args.variants),
                "seeds": list(args.seeds), "bank_sha256": bank.content_sha256,
                "oracle_contact": True, "deployable": False, "test_data_used": False}
    (args.runs / "RUN_COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("v41/dataset"))
    parser.add_argument("--manifest", type=Path, default=Path("v41/manifests/v41_uid_splits.json"))
    parser.add_argument("--matrix", type=Path, default=Path("v43/ROTATION_CONTACT_CURVATURE_MATRIX.json"))
    parser.add_argument("--champion", type=Path, default=Path("v42/checkpoints/v42_adapter_full_seed42_best.pt"))
    parser.add_argument("--runs", type=Path, default=Path("v43/run/rotation_contact_curvature"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--variants", nargs="+", choices=CONTACT_VARIANTS, default=list(CONTACT_VARIANTS))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--draws", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--smoothness-weight", type=float, default=.01)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--allow-subset", action="store_true")
    main(parser.parse_args())
