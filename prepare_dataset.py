"""Batch the existing extraction and cache-preparation commands over scenes."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="../coderepos/PhysTwin/data/different_types")
    parser.add_argument("--features", default="data/features")
    parser.add_argument("--caches", default="data/cache")
    parser.add_argument("--scenes", nargs="*", help="Default: every directory containing final_data.pkl")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    scenes = args.scenes or sorted(p.name for p in root.iterdir() if (p / "final_data.pkl").exists())
    for index, name in enumerate(scenes, 1):
        scene = root / name
        feature = Path(args.features) / f"{name}.npz"
        cache = Path(args.caches) / f"{name}.pt"
        print(f"[{index}/{len(scenes)}] {name}", flush=True)
        if args.force or not feature.exists():
            command = [sys.executable, "extract_dino_features.py", str(scene), "--output", str(feature)]
            if args.device: command += ["--device", args.device]
            subprocess.run(command, check=True)
        else: print(f"  reuse {feature}")
        if args.force or not cache.exists():
            subprocess.run([sys.executable, "prepare_scene.py", "--final-data", str(scene / "final_data.pkl"), "--dino-features", str(feature), "--output", str(cache)], check=True)
        else: print(f"  reuse {cache}")


if __name__ == "__main__":
    main()
