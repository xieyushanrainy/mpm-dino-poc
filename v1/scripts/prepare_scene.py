import argparse

from mpm_dino.data import prepare_scene


def main():
    parser = argparse.ArgumentParser(description="Prepare one PhysTwin scene with aligned cached DINO features.")
    parser.add_argument("--final-data", required=True)
    parser.add_argument("--dino-features", required=True, help="Numpy file shaped (number_of_tracks, feature_dim)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-points", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_scene(args.final_data, args.dino_features, args.output, args.maximum_points, args.seed)


if __name__ == "__main__":
    main()
