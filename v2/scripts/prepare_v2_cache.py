#!/usr/bin/env python3
"""Convert frozen V1 caches into the versioned V2 reference-graph schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpm_dino_v2.cache import convert_v1_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="V1 .pt file or cache directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must be a V2 data directory")
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--max-neighbours", type=int, default=8)
    parser.add_argument("--report", type=Path, help="Optional aggregate JSON audit report")
    args = parser.parse_args()

    sources = [args.input] if args.input.is_file() else sorted(args.input.glob("*.pt"))
    if not sources:
        raise SystemExit(f"No cache files found at {args.input}")
    reports = {}
    for source in sources:
        destination = args.output_dir / source.name
        report = convert_v1_cache(source, destination, args.candidate_k, args.max_neighbours)
        reports[source.stem] = report.to_dict()
        print(f"{source.name}: {report.undirected_edge_count} edges, "
              f"{report.connected_components} components, max length {report.length_max:.6g}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
