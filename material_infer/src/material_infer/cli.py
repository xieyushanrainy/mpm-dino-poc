from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import audit_dataset
from .cross_validate import run_cross_validation
from .experiment import run_experiment


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT.parent / "v4" / "dataset" / "packaged_balanced_90_dinov2_dinov3"
DEFAULT_MANIFEST = ROOT.parent / "v4" / "splits_balanced_90.json"


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(description="Frozen-DINO material-family and soft-parameter probes")
    commands = top.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    run = commands.add_parser("run")
    cross_validate = commands.add_parser("cross-validate")
    for item in (audit, run, cross_validate):
        item.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
        item.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        item.add_argument("--dino-key", default="dinov2_reprojected_features")
    audit.add_argument("--output", type=Path)
    run.add_argument("--task", choices=("family", "soft"), required=True)
    for item in (run, cross_validate):
        item.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
        item.add_argument("--target", choices=("log10_E", "nu"), default="log10_E")
        item.add_argument("--feature-source", choices=("dino", "geometry", "valid_fraction"), default="dino")
        item.add_argument("--control", choices=("real", "scene_shuffled", "label_permuted"), default="real")
        item.add_argument("--model", choices=("linear", "mlp"), default="linear")
        item.add_argument("--pca-components", type=int, default=8)
        item.add_argument("--hidden-dim", type=int, default=8)
        item.add_argument("--dropout", type=float, default=0.2)
        item.add_argument("--lr", type=float, default=1e-3)
        item.add_argument("--weight-decay", type=float, default=1e-2)
        item.add_argument("--epochs", type=int, default=300)
        item.add_argument("--patience", type=int, default=30)
        item.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2026])
        item.add_argument("--output", type=Path, required=True)
    cross_validate.add_argument("--folds", type=int, default=6)
    return top


def main() -> None:
    args = parser().parse_args()
    if args.command == "audit":
        report = audit_dataset(args.dataset, args.manifest, args.dino_key)
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text + "\n")
        print(text)
    elif args.command == "run":
        summary = run_experiment(args)
        print(json.dumps(summary["aggregate"], indent=2))
    else:
        summary = run_cross_validation(args)
        print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
