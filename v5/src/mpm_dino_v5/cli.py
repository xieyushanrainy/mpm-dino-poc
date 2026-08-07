from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import V5Config
from .bank_builder import build_training_bank
from .data import load_manifest
from .training import (
    TrainOptions,
    train_com,
    train_deformation,
    train_interaction,
    train_memory,
    train_rotation,
    train_shared_deformation,
    train_shared_global,
    train_shared_interaction,
)


def _read_config(path: str | None) -> V5Config:
    if path is None:
        return V5Config()
    return V5Config.from_dict(json.loads(Path(path).read_text()))


def _options(args) -> TrainOptions:
    return TrainOptions(
        epochs=args.epochs,
        draws_per_epoch=args.draws,
        learning_rate=args.lr,
        patience=args.patience,
        accumulation=args.accumulation,
        max_batches=args.max_batches,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="V5 staged causal deformation")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-config", help="write the validated default config")
    init.add_argument("output")

    audit = sub.add_parser("audit", help="validate the split without loading sealed test trajectories")
    audit.add_argument("manifest")

    bank = sub.add_parser("build-bank", help="rebuild the restricted training-only mechanics bank")
    bank.add_argument("--dataset", required=True)
    bank.add_argument("--manifest", required=True)
    bank.add_argument("--output", required=True)
    bank.add_argument("--seed", type=int, default=42, choices=(42, 123, 456))

    train = sub.add_parser("train", help="train one explicitly selected V5 stage")
    train.add_argument("stage", choices=(
        "shared-global", "shared-interaction", "shared-deformation",
        "com", "rotation", "interaction", "deformation", "memory",
    ))
    train.add_argument("--dataset", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--seed", type=int, choices=(42, 123, 456), required=True)
    train.add_argument("--config")
    train.add_argument("--device", default="cpu")
    train.add_argument("--epochs", type=int, default=120)
    train.add_argument("--draws", type=int, default=40)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--accumulation", type=int, default=4)
    train.add_argument("--max-batches", type=int)
    train.add_argument("--com-checkpoint")
    train.add_argument("--rotation-checkpoint")
    train.add_argument("--interaction-checkpoint")
    train.add_argument("--deformation-checkpoint")
    train.add_argument("--bank")
    train.add_argument("--global-checkpoint")
    train.add_argument("--trunk-gradient-scale", type=float, default=0.0)
    train.add_argument("--identity-rotation", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "init-config":
        config = V5Config()
        Path(args.output).write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        return 0
    if args.command == "audit":
        manifest = load_manifest(args.manifest)
        print(json.dumps({
            "status": "pass",
            "manifest_content_sha256": manifest["manifest_content_sha256"],
            "uid_counts": {name: len(manifest["splits"][name]["uids"]) for name in ("train", "validation", "test")},
            "test_trajectories_loaded": False,
        }, indent=2))
        return 0
    if args.command == "build-bank":
        manifest = load_manifest(args.manifest)
        built = build_training_bank(args.dataset, manifest, args.output, args.seed)
        print(json.dumps(built.manifest(), indent=2))
        return 0

    config = _read_config(args.config)
    manifest = load_manifest(args.manifest)
    common = (args.dataset, manifest, args.output, args.seed)
    keyword = {"config": config, "options": _options(args), "device": args.device}
    if args.stage == "shared-global":
        score = train_shared_global(*common, **keyword)
    elif args.stage == "shared-interaction":
        if not args.global_checkpoint:
            parser.error("shared-interaction requires --global-checkpoint")
        score = train_shared_interaction(
            *common, args.global_checkpoint,
            use_identity_rotation=args.identity_rotation, **keyword,
        )
    elif args.stage == "shared-deformation":
        if not args.global_checkpoint or not args.interaction_checkpoint:
            parser.error("shared-deformation requires global and interaction checkpoints")
        score = train_shared_deformation(
            *common, args.global_checkpoint, args.interaction_checkpoint,
            trunk_gradient_scale=args.trunk_gradient_scale,
            use_identity_rotation=args.identity_rotation, **keyword,
        )
    elif args.stage == "com":
        score = train_com(*common, **keyword)
    elif args.stage == "rotation":
        score = train_rotation(*common, **keyword)
    elif args.stage == "interaction":
        if not args.com_checkpoint or (not args.identity_rotation and not args.rotation_checkpoint):
            parser.error("interaction requires --com-checkpoint and either --rotation-checkpoint or --identity-rotation")
        score = train_interaction(
            *common, args.com_checkpoint, args.rotation_checkpoint,
            use_identity_rotation=args.identity_rotation, **keyword,
        )
    elif args.stage == "deformation":
        if not args.com_checkpoint or not args.interaction_checkpoint or (not args.identity_rotation and not args.rotation_checkpoint):
            parser.error("deformation requires COM/interaction checkpoints and a rotation choice")
        score = train_deformation(
            *common, args.com_checkpoint, args.rotation_checkpoint,
            args.interaction_checkpoint, use_identity_rotation=args.identity_rotation,
            **keyword,
        )
    else:
        required = (
            args.com_checkpoint, args.interaction_checkpoint,
            args.deformation_checkpoint, args.bank,
        )
        if not all(required) or (not args.identity_rotation and not args.rotation_checkpoint):
            parser.error("memory requires the frozen base checkpoints, bank, and a rotation choice")
        score = train_memory(
            *common, args.com_checkpoint, args.rotation_checkpoint,
            args.interaction_checkpoint, args.deformation_checkpoint, args.bank,
            use_identity_rotation=args.identity_rotation, **keyword,
        )
    print(json.dumps({"stage": args.stage, "seed": args.seed, "best_validation": score}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
