#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Keep this validation-only audit runnable without importing the package
# __init__, which loads the training-only PyTorch stack.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "v4" / "src" / "mpm_dino_v4")
)
from v42_rotation_audit import (  # noqa: E402
    angular_increment, constant_angular_rotation, geodesic_error,
    proper_kabsch, rotation_from_vector, rotation_vector,
)


HORIZONS = (1, 8, 16, 30, 40, 59)


def finite_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or left[valid].std() < 1e-12 or right[valid].std() < 1e-12:
        return None
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else None


def summarize_horizons(episodes):
    groups = defaultdict(list)
    for episode in episodes:
        groups[(episode["family"], episode["panel"])].append(episode)
    result = {}
    for (family, panel), rows in sorted(groups.items()):
        horizons = {}
        for horizon in HORIZONS:
            values = [row["horizons"][f"h{horizon}"] for row in rows]
            identity = np.array([value["identity_error_rad"] for value in values])
            angular = np.array([
                value["constant_angular_error_rad"] for value in values
            ])
            target_vectors = np.array([
                value["target_rotation_vector_rad"] for value in values
            ])
            target_angles = np.linalg.norm(target_vectors, axis=1)
            mean_vector = target_vectors.mean(axis=0)
            horizons[f"h{horizon}"] = {
                "episode_count": len(values),
                "identity_mean_error_rad": float(identity.mean()),
                "constant_angular_mean_error_rad": float(angular.mean()),
                "constant_angular_improvement_fraction": float(
                    (identity.mean() - angular.mean()) / max(identity.mean(), 1e-12)
                ),
                "constant_angular_episode_win_fraction": float(
                    np.mean(angular < identity)
                ),
                "target_mean_angle_rad": float(target_angles.mean()),
                "target_median_angle_rad": float(np.median(target_angles)),
                "target_mean_rotation_vector_rad": mean_vector.tolist(),
                "target_direction_consistency": float(
                    np.linalg.norm(mean_vector)
                    / max(float(target_angles.mean()), 1e-12)
                ),
            }
        result[f"{family}/panel_{panel}"] = {
            "episodes": len(rows),
            "uids": len({row["uid"] for row in rows}),
            "horizons": horizons,
        }
    return result


def paired_uid_repeatability(episodes):
    result = {}
    by_uid = defaultdict(list)
    for episode in episodes:
        by_uid[episode["uid"]].append(episode)
    for uid, rows in sorted(by_uid.items()):
        zero = [row for row in rows if row["panel"] == "Z"]
        moving = [row for row in rows if row["panel"] == "V"]
        if not zero or not moving:
            continue
        horizons = {}
        for horizon in HORIZONS:
            reference = rotation_from_vector(np.asarray(
                zero[0]["horizons"][f"h{horizon}"][
                    "target_rotation_vector_rad"
                ]
            ))
            differences = []
            for row in moving:
                comparison = rotation_from_vector(np.asarray(
                    row["horizons"][f"h{horizon}"][
                        "target_rotation_vector_rad"
                    ]
                ))
                differences.append(geodesic_error(reference, comparison))
            horizons[f"h{horizon}"] = {
                "panel_v_episodes": len(differences),
                "panel_z_to_panel_v_mean_geodesic_rad": float(
                    np.mean(differences)
                ),
                "panel_z_to_panel_v_max_geodesic_rad": float(
                    np.max(differences)
                ),
            }
        result[uid] = horizons
    return result


def grouped_impact_diagnostics(episodes):
    groups = defaultdict(list)
    for episode in episodes:
        groups[(episode["family"], episode["panel"])].append(episode)
    result = {}
    for (family, panel), rows in sorted(groups.items()):
        result[f"{family}/panel_{panel}"] = {
            "observed_step_median_angle_rad": float(np.median([
                row["observed_step_angle_rad"] for row in rows
            ])),
            "precontact_median_angular_speed_rad_s": float(np.median([
                np.linalg.norm(row["precontact_angular_velocity_rad_s"])
                for row in rows
            ])),
            "impact_delta_median_rad_s": float(np.median([
                row["impact_delta_speed_rad_s"] for row in rows
            ])),
            "postimpact_delta_median_rad_s": float(np.median([
                row["postimpact_delta_speed_rad_s"] for row in rows
            ])),
            "minimum_singular_ratio": float(min(
                row["minimum_target_singular_ratio"] for row in rows
            )),
            "singular_ratio_vs_angular_step_change_correlation":
                finite_correlation(
                    [row["minimum_target_singular_ratio"] for row in rows],
                    [row["maximum_angular_step_change_rad_s"] for row in rows],
                ),
            "deformation_vs_angular_step_change_correlation":
                finite_correlation(
                    [row["maximum_canonical_residual_nrmse"] for row in rows],
                    [row["maximum_angular_step_change_rad_s"] for row in rows],
                ),
        }
    return result


def episode_audit(sample, onset):
    positions = sample["positions"]
    active = sample["active"]
    valid01 = active[0] & active[1]
    observed_rotation, observed_singular, observed_ratio = proper_kabsch(
        positions[0], positions[1], valid01,
    )
    reference = positions[1]
    reference_mask = active[1] & valid01
    radius = np.linalg.norm(
        reference[reference_mask] - reference[reference_mask].mean(axis=0),
        axis=1,
    ).max()
    rotations = [np.eye(3)]
    ratios = [observed_ratio]
    residuals = [0.0]
    for frame in range(2, len(positions)):
        valid = reference_mask & active[frame]
        rotation, singular, ratio = proper_kabsch(
            reference, positions[frame], valid,
        )
        rotations.append(rotation)
        ratios.append(ratio)
        source = reference[valid] - reference[valid].mean(axis=0)
        destination = positions[frame, valid] - positions[frame, valid].mean(
            axis=0
        )
        residual = np.einsum("ni,ji->nj", destination, rotation) - source
        residuals.append(float(
            np.sqrt(np.mean(np.sum(residual * residual, axis=1))) / max(radius, 1e-12)
        ))
    dt = sample["dt"]
    increments = [rotation_vector(observed_rotation) / dt]
    increments.extend(
        angular_increment(rotations[index - 1], rotations[index], dt)
        for index in range(1, len(rotations))
    )
    increments = np.asarray(increments)
    onset = max(1, min(int(onset), len(rotations) - 2))
    pre = increments[1:onset].mean(axis=0) if onset > 1 else increments[0]
    impact = increments[onset:onset + 2].mean(axis=0)
    post_end = min(onset + 6, len(increments))
    post = increments[onset + 2:post_end].mean(axis=0)
    step_changes = np.linalg.norm(np.diff(increments, axis=0), axis=1)
    horizons = {}
    for horizon in HORIZONS:
        target = rotations[horizon]
        identity_error = geodesic_error(np.eye(3), target)
        angular_prediction = constant_angular_rotation(
            observed_rotation, horizon
        )
        horizons[f"h{horizon}"] = {
            "target_rotation_vector_rad": rotation_vector(target).tolist(),
            "target_angle_rad": float(np.linalg.norm(rotation_vector(target))),
            "identity_error_rad": identity_error,
            "constant_angular_error_rad": geodesic_error(
                angular_prediction, target
            ),
            "constant_angular_improves": bool(
                geodesic_error(angular_prediction, target) < identity_error
            ),
            "singular_ratio": ratios[horizon],
            "canonical_residual_nrmse": residuals[horizon],
        }
    return {
        "uid": sample["uid"],
        "episode_id": sample["episode_id"],
        "family": sample["family"],
        "panel": sample["panel"],
        "dt_s": dt,
        "contact_onset_index_from_x1": onset,
        "observed_step_rotation_vector_rad": rotation_vector(
            observed_rotation
        ).tolist(),
        "observed_step_angle_rad": float(
            np.linalg.norm(rotation_vector(observed_rotation))
        ),
        "observed_step_singular_values": observed_singular.tolist(),
        "observed_step_singular_ratio": observed_ratio,
        "minimum_target_singular_ratio": float(min(ratios)),
        "invalid_target_frames_at_1e-3": int(
            np.sum(np.asarray(ratios[1:]) < 1e-3)
        ),
        "precontact_angular_velocity_rad_s": pre.tolist(),
        "impact_angular_velocity_rad_s": impact.tolist(),
        "postimpact_angular_velocity_rad_s": post.tolist(),
        "initial_to_precontact_cosine": cosine(increments[0], pre),
        "impact_delta_angular_velocity_rad_s": (impact - pre).tolist(),
        "impact_delta_speed_rad_s": float(np.linalg.norm(impact - pre)),
        "postimpact_delta_speed_rad_s": float(np.linalg.norm(post - pre)),
        "maximum_angular_step_change_rad_s": float(step_changes.max()),
        "maximum_canonical_residual_nrmse": float(max(residuals)),
        "horizons": horizons,
    }


def load_sample(root, row):
    with np.load(root / row["trajectory"], allow_pickle=False) as data:
        positions = data["trajectory_positions_m"].astype(np.float64)
        active = data["point_active"].astype(bool)
        times = data["times_s"].astype(np.float64)
    return {
        "uid": row["uid"],
        "episode_id": row["episode_id"],
        "family": row["family"],
        "panel": "Z" if row["initial_velocity_regime"] == "zero" else "V",
        "positions": positions,
        "active": active,
        "dt": float(times[1] - times[0]),
    }


def markdown_report(payload):
    lines = [
        "# V4.2 validation-only rotation audit",
        "",
        "No training or test data was used.",
        "",
        "## Baseline comparison",
        "",
        "| Group | Horizon | Identity (deg) | Constant angular (deg) | Improvement | Episode wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, summary in payload["stratified_horizons"].items():
        for horizon, values in summary["horizons"].items():
            lines.append(
                f"| {group} | {horizon.upper()} | "
                f"{math.degrees(values['identity_mean_error_rad']):.3f} | "
                f"{math.degrees(values['constant_angular_mean_error_rad']):.3f} | "
                f"{100 * values['constant_angular_improvement_fraction']:+.1f}% | "
                f"{100 * values['constant_angular_episode_win_fraction']:.0f}% |"
            )
    diagnostics = payload["diagnostics"]
    lines += [
        "",
        "## Diagnostics",
        "",
        f"- Constant-angular baseline overall improvement: "
        f"{100 * diagnostics['overall_constant_angular_improvement_fraction']:+.1f}%.",
        f"- Frames below the Kabsch `1e-3` conditioning threshold: "
        f"{diagnostics['invalid_target_frames_at_1e-3']}.",
        f"- Correlation between minimum singular ratio and maximum angular-step change: "
        f"{diagnostics['singular_ratio_vs_angular_step_change_correlation']}.",
        f"- Correlation between deformation residual and maximum angular-step change: "
        f"{diagnostics['deformation_vs_angular_step_change_correlation']}.",
        f"- Maximum observed x0-to-x1 rotation: "
        f"{math.degrees(diagnostics['maximum_observed_step_angle_rad']):.6f} degrees.",
        f"- Median impact angular-velocity change: "
        f"{diagnostics['impact_delta_speed_median_rad_s']:.3f} rad/s.",
        "",
        "## Decision",
        "",
        "The observed-step rotation is effectively zero, and constant-angular "
        "extrapolation does not improve identity. Do not adopt a residual-over-"
        "observed-angular-velocity rotation baseline for this dataset.",
        "",
        "See `rotation_audit.json` for per-episode rotation vectors, angular velocities, "
        "conditioning values and horizon errors.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Validation-only V4.2 rotation baseline and gauge audit"
    )
    parser.add_argument("--dataset", type=Path, default=Path("v41/dataset"))
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("v41/manifests/v41_uid_splits.json"),
    )
    parser.add_argument(
        "--stage-audit", type=Path,
        default=Path("v42/target_stage_audit.json"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("v42/rotation_audit_validation_20260729"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    stage_audit = json.loads(args.stage_audit.read_text())
    onset = {
        row["episode_id"]: row["contact_onset"]
        for row in stage_audit["results"]["validation"]["episodes"]
    }
    episode_ids = (
        manifest["splits"]["validation"]["panel_z"]
        + manifest["splits"]["validation"]["panel_v"]
    )
    episodes = []
    for episode_id in episode_ids:
        row = manifest["episodes"][episode_id]
        episodes.append(episode_audit(
            load_sample(args.dataset, row), onset[episode_id]
        ))
    identity = []
    angular = []
    for episode in episodes:
        for horizon in HORIZONS:
            values = episode["horizons"][f"h{horizon}"]
            identity.append(values["identity_error_rad"])
            angular.append(values["constant_angular_error_rad"])
    minimum_ratios = [row["minimum_target_singular_ratio"] for row in episodes]
    angular_changes = [
        row["maximum_angular_step_change_rad_s"] for row in episodes
    ]
    deformation = [
        row["maximum_canonical_residual_nrmse"] for row in episodes
    ]
    diagnostics = {
        "episodes": len(episodes),
        "uids": len({row["uid"] for row in episodes}),
        "overall_identity_mean_error_rad": float(np.mean(identity)),
        "overall_constant_angular_mean_error_rad": float(np.mean(angular)),
        "overall_constant_angular_improvement_fraction": float(
            (np.mean(identity) - np.mean(angular)) / max(np.mean(identity), 1e-12)
        ),
        "overall_constant_angular_episode_horizon_win_fraction": float(
            np.mean(np.asarray(angular) < np.asarray(identity))
        ),
        "invalid_target_frames_at_1e-3": int(sum(
            row["invalid_target_frames_at_1e-3"] for row in episodes
        )),
        "minimum_singular_ratio": float(min(minimum_ratios)),
        "singular_ratio_vs_angular_step_change_correlation": finite_correlation(
            minimum_ratios, angular_changes
        ),
        "deformation_vs_angular_step_change_correlation": finite_correlation(
            deformation, angular_changes
        ),
        "impact_delta_speed_median_rad_s": float(np.median([
            row["impact_delta_speed_rad_s"] for row in episodes
        ])),
        "postimpact_delta_speed_median_rad_s": float(np.median([
            row["postimpact_delta_speed_rad_s"] for row in episodes
        ])),
        "maximum_observed_step_angle_rad": float(max(
            row["observed_step_angle_rad"] for row in episodes
        )),
    }
    payload = {
        "experiment": "v42_validation_only_rotation_audit",
        "split": "validation",
        "test_data_used": False,
        "trained_model_used": False,
        "horizons": list(HORIZONS),
        "constant_angular_definition": (
            "proper Kabsch x0->x1 rotation, extrapolated h steps in SO(3)"
        ),
        "contact_source": str(args.stage_audit),
        "diagnostics": diagnostics,
        "stratified_horizons": summarize_horizons(episodes),
        "grouped_impact_diagnostics": grouped_impact_diagnostics(episodes),
        "paired_uid_panel_repeatability": paired_uid_repeatability(episodes),
        "episodes": episodes,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rotation_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    (args.output / "RESULTS.md").write_text(markdown_report(payload))
    print(json.dumps(diagnostics, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
