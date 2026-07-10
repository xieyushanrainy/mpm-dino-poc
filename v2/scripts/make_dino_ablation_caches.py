#!/usr/bin/env python3
"""Clone V2 geometry caches while replacing only the particle DINO tensor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mpm_dino_v2.cache import load_v2_cache, validate_v2_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--zero", action="store_true")
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()
    if args.zero == (args.features_dir is not None):
        parser.error("select exactly one of --zero or --features-dir")
    for source in sorted(args.source_cache_dir.glob("*.pt")):
        payload = load_v2_cache(source)
        if args.zero:
            payload["dino"] = torch.zeros_like(payload["dino"])
        else:
            full = np.load(args.features_dir / f"{source.stem}.npz")["features"]
            replacement = torch.zeros_like(payload["dino"])
            valid = payload["particle_mask"]
            replacement[valid] = torch.from_numpy(full[payload["source_indices"][valid].numpy()])
            payload["dino"] = replacement
        payload["dino_variant"] = args.variant
        validate_v2_cache(payload)
        destination = args.output_dir / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, destination)
        print(destination, flush=True)


if __name__ == "__main__":
    main()
