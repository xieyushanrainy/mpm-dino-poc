#!/usr/bin/env python3
"""Validate V2 cache graphs and render them for disconnected-surface review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpm_dino_v2.cache import load_v2_cache, validate_v2_cache
from mpm_dino_v2.visualization import save_graph_visualization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("caches", type=Path, nargs="+", help="V2 cache files")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}
    for path in args.caches:
        payload = load_v2_cache(path)
        report = validate_v2_cache(payload)
        reports[path.stem] = report.to_dict()
        save_graph_visualization(payload, args.output_dir / f"{path.stem}.png", path.stem)
        print(f"{path.stem}: {report.to_dict()}")
    (args.output_dir / "graph_report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
