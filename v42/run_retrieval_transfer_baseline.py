#!/usr/bin/env python3
"""Leakage-safe aligned-DINO retrieval baselines for V4.2 motion transfer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v4" / "src"))

from mpm_dino_v4.v41_data import MODEL_INPUT_KEYS, V41TrajectoryDataset  # noqa: E402
from mpm_dino_v4.v42_dino_audit import _dino_pair  # noqa: E402
from mpm_dino_v4.v42_geometry_audit import (  # noqa: E402
    _field_metrics, _geometry_alignment, _record,
)
from mpm_dino_v4.v42_oracle import EVENT_STAGES  # noqa: E402
from mpm_dino_v4.v42_gate2 import _batch_targets_and_stages, load_gate1e_source  # noqa: E402
from mpm_dino_v4.v42_rotation_audit import (  # noqa: E402
    geodesic_error, proper_kabsch, rotation_from_vector, rotation_vector,
)

K_VALUES = (1, 3, 5)
TEMPERATURES = (0.01, 0.03, 0.1, 0.3)
ROTATION_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)


def weights(distances, temperature):
    values = np.asarray(distances, dtype=np.float64)
    logits = -(values - values.min()) / temperature
    values = np.exp(logits - logits.max())
    return values / values.sum()


def soft_records(dataset_root, manifest, split):
    dataset = V41TrajectoryDataset(
        dataset_root, manifest, split, "real", 42, families=("soft_body",),
    )
    return [_record(dataset, i, torch.device("cpu"), dataset_root)
            for i in range(len(dataset))]


def soft_candidates(source_records, target):
    candidates = []
    for source in source_records:
        if source["uid"] == target["uid"]:
            continue
        _, sign = _geometry_alignment(source, target)
        dino = _dino_pair(source, target, sign)["aligned_point_distance"]
        fields = {}
        for stage in EVENT_STAGES:
            metric = _field_metrics(source, target, sign, int(stage))
            if metric:
                fields[int(stage)] = (metric["predicted"], metric["truth"])
        candidates.append({"uid": source["uid"], "distance": dino, "fields": fields})
    return sorted(candidates, key=lambda row: row["distance"])


def soft_prediction(candidates, k, temperature):
    selected = candidates[:k]
    selected_weights = weights([row["distance"] for row in selected], temperature)
    result = []
    common = set.intersection(*(set(row["fields"]) for row in selected))
    for stage in sorted(common):
        predicted = sum(float(w) * row["fields"][stage][0]
                        for w, row in zip(selected_weights, selected))
        result.append((predicted, selected[0]["fields"][stage][1]))
    return result


def fit_soft_scale(predictions):
    numerator = sum(float((p * t).sum() / t.square().sum().clamp_min(1e-12))
                    for rows in predictions for p, t in rows)
    denominator = sum(float(p.square().sum() / t.square().sum().clamp_min(1e-12))
                      for rows in predictions for p, t in rows)
    return numerator / max(denominator, 1e-12)


def soft_loss(predictions, scale):
    values = []
    for rows in predictions:
        stage_values = [float(((scale * p - t).square().sum()
                              / t.square().sum().clamp_min(1e-12))) for p, t in rows]
        values.append(float(np.mean(stage_values)))
    return float(np.mean(values)), values


def soft_audit(train, validation):
    train_candidates = [soft_candidates(train, target) for target in train]
    validation_candidates = [soft_candidates(train, target) for target in validation]
    conditions = {}
    for k in K_VALUES:
        choices = []
        for temperature in TEMPERATURES:
            loo = [soft_prediction(rows, k, temperature) for rows in train_candidates]
            scale = fit_soft_scale(loo)
            loss, _ = soft_loss(loo, scale)
            choices.append((loss, temperature, scale))
        train_loss, temperature, scale = min(choices)
        predicted = [soft_prediction(rows, k, temperature) for rows in validation_candidates]
        loss, per_uid = soft_loss(predicted, scale)
        unscaled_loss, _ = soft_loss(predicted, 1.0)
        conditions[f"dino_top{k}"] = {
            "temperature": temperature, "training_loo_loss": train_loss,
            "training_fitted_scale": scale, "validation_loss": loss,
            "validation_unscaled_loss": unscaled_loss,
            "per_validation_uid": dict(zip([r["uid"] for r in validation], per_uid)),
        }
    def geometry_predictions(targets, sources):
        result, oracle = [], []
        for target in targets:
            candidates = []
            for source in sources:
                if source["uid"] == target["uid"]:
                    continue
                chamfer, sign = _geometry_alignment(source, target)
                fields = [_field_metrics(source, target, sign, int(stage)) for stage in EVENT_STAGES]
                fields = [(m["predicted"], m["truth"]) for m in fields if m]
                rescaled = soft_loss([fields], fit_soft_scale([fields]))[0]
                candidates.append((chamfer, rescaled, fields))
            result.append(min(candidates, key=lambda row: row[0])[2])
            oracle.append(min(row[1] for row in candidates))
        return result, oracle
    geometry_train, _ = geometry_predictions(train, train)
    geometry_validation, oracle_losses = geometry_predictions(validation, train)
    geometry_scale = fit_soft_scale(geometry_train)
    geometry_loss, _ = soft_loss(geometry_validation, geometry_scale)
    return {
        "zero_loss": 1.0, "conditions": conditions,
        "geometry_nearest": {"validation_loss": geometry_loss,
                             "training_fitted_scale": geometry_scale},
        "oracle_best_rescaled_validation_loss": float(np.mean(oracle_losses)),
    }


def gate2c_soft_loss(dataset_root, manifest, checkpoint_path, source_path):
    model, _ = load_gate1e_source(source_path, "geometry", "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    dataset = V41TrajectoryDataset(
        dataset_root, manifest, "validation", "zero", 456,
        families=("soft_body",),
    )
    per_uid = {}
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = default_collate([dataset[index]])
            targets, stages = _batch_targets_and_stages(batch)
            output = model(**{key: batch[key] for key in MODEL_INPUT_KEYS})
            values = []
            for stage in EVENT_STAGES:
                selected = stages.labels[0, 1:].eq(int(stage))
                valid = (batch["target_mask"][0]
                         & targets.valid_rotation[0, :, None]
                         & selected[:, None])
                count = valid.sum(0)
                available = batch["input_mask"][0] & count.gt(0)
                truth = (targets.displacement[0] * valid[..., None]).sum(0)
                truth = truth / count.clamp_min(1)[..., None]
                prediction = (output.canonical_displacement[0] * valid[..., None]).sum(0)
                prediction = prediction / count.clamp_min(1)[..., None]
                values.append(float(
                    (prediction[available] - truth[available]).square().sum()
                    / truth[available].square().sum().clamp_min(1e-12)
                ))
            per_uid[batch["uid"][0]] = float(np.mean(values))
    return {"checkpoint": str(checkpoint_path), "epoch": int(checkpoint["epoch"]),
            "validation_loss": float(np.mean(list(per_uid.values()))),
            "per_validation_uid": per_uid}


def rotation_record(dataset_root, row):
    with np.load(dataset_root / row["trajectory"], allow_pickle=False) as data:
        positions = data["trajectory_positions_m"].astype(np.float64)
        active = data["point_active"].astype(bool)
    with np.load(dataset_root / row["object_static"], allow_pickle=False) as data:
        dino = torch.nn.functional.normalize(
            torch.from_numpy(data["dino_features"].astype(np.float32)), dim=-1,
        )
        dino_valid = torch.from_numpy(data["dino_valid"].astype(bool))
    valid_reference = active[0] & active[1]
    reference = positions[1]
    centered = reference[valid_reference] - reference[valid_reference].mean(0)
    radius = max(float(np.linalg.norm(centered, axis=1).max()), 1e-12)
    points = torch.from_numpy(centered / radius).double()
    covariance = points.T @ points / len(points)
    _, frame_t = torch.linalg.eigh(covariance)
    frame_t = frame_t[:, torch.tensor([2, 1, 0])]
    if torch.linalg.det(frame_t) < 0:
        frame_t[:, -1] *= -1
    frame = frame_t.numpy()
    rotations, valid_frames = [], []
    for index in range(2, len(positions)):
        valid = valid_reference & active[index]
        rotation, _, ratio = proper_kabsch(reference, positions[index], valid)
        rotations.append(frame.T @ rotation @ frame)
        valid_frames.append(ratio >= 1e-3)
    return {"uid": row["uid"], "family": row["family"],
            "points": points @ frame_t, "frame": frame_t,
            "dino": dino[valid_reference], "dino_valid": dino_valid[valid_reference],
            "pooled_dino": torch.nn.functional.normalize(
                dino[valid_reference][dino_valid[valid_reference]].mean(0), dim=0,
            ),
            "rotations": np.asarray(rotations), "valid": np.asarray(valid_frames)}


def rotation_candidates(sources, target):
    rows = []
    for source in sources:
        if source["uid"] == target["uid"] or not source["valid"].any():
            continue
        _, sign = _geometry_alignment(source, target)
        distance = _dino_pair(source, target, sign)["aligned_point_distance"]
        transform = np.diag(sign.numpy())
        rows.append({"uid": source["uid"], "distance": distance,
                     "rotations": transform @ source["rotations"] @ transform})
    return sorted(rows, key=lambda row: row["distance"])


def project_rotation(matrix):
    u, _, vh = np.linalg.svd(matrix)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def rotation_prediction(candidates, k, temperature, scale):
    selected = candidates[:k]
    ws = weights([row["distance"] for row in selected], temperature)
    result = []
    for frame in range(59):
        mean = sum(float(w) * row["rotations"][frame] for w, row in zip(ws, selected))
        rotation = project_rotation(mean)
        result.append(rotation_from_vector(rotation_vector(rotation) * scale))
    return np.asarray(result)


def rotation_loss(predictions, targets):
    per_uid = []
    for prediction, target in zip(predictions, targets):
        valid = target["valid"]
        per_uid.append(float(np.mean([
            geodesic_error(prediction[i], target["rotations"][i])
            for i in np.flatnonzero(valid)
        ])))
    return float(np.mean(per_uid)), per_uid


def rotation_audit(train, validation):
    train = [row for row in train if row["valid"].any()]
    train_candidates = [rotation_candidates(train, target) for target in train]
    validation_candidates = [rotation_candidates(train, target) for target in validation]
    conditions = {}
    for k in K_VALUES:
        choices = []
        raw_choices = []
        for temperature in TEMPERATURES:
            raw_prediction = [rotation_prediction(rows, k, temperature, 1.0)
                              for rows in train_candidates]
            raw_loss, _ = rotation_loss(raw_prediction, train)
            raw_choices.append((raw_loss, temperature))
            for scale in ROTATION_SCALES:
                prediction = [rotation_prediction(rows, k, temperature, scale)
                              for rows in train_candidates]
                loss, _ = rotation_loss(prediction, train)
                choices.append((loss, temperature, scale))
        train_loss, temperature, scale = min(choices)
        prediction = [rotation_prediction(rows, k, temperature, scale)
                      for rows in validation_candidates]
        loss, per_uid = rotation_loss(prediction, validation)
        raw_train_loss, raw_temperature = min(raw_choices)
        raw_prediction = [rotation_prediction(rows, k, raw_temperature, 1.0)
                          for rows in validation_candidates]
        raw_loss, _ = rotation_loss(raw_prediction, validation)
        conditions[f"dino_top{k}"] = {
            "temperature": temperature, "training_fitted_rotation_scale": scale,
            "training_loo_error_deg": float(np.degrees(train_loss)),
            "validation_error_deg": float(np.degrees(loss)),
            "unscaled_temperature": raw_temperature,
            "training_loo_unscaled_error_deg": float(np.degrees(raw_train_loss)),
            "validation_unscaled_error_deg": float(np.degrees(raw_loss)),
            "per_validation_uid_deg": dict(zip(
                [r["uid"] for r in validation], np.degrees(per_uid).tolist()
            )),
        }
    identity = [np.repeat(np.eye(3)[None], 59, axis=0) for _ in validation]
    identity_loss, _ = rotation_loss(identity, validation)
    oracle = []
    for target in validation:
        values = []
        for source in train:
            _, sign = _geometry_alignment(source, target)
            transform = np.diag(sign.numpy())
            prediction = transform @ source["rotations"] @ transform
            loss, _ = rotation_loss([prediction], [target])
            values.append(loss)
        oracle.append(min(values))
    return {"identity_error_deg": float(np.degrees(identity_loss)),
            "conditions": conditions,
            "oracle_best_error_deg": float(np.degrees(np.mean(oracle)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("v41/dataset"))
    parser.add_argument("--manifest", type=Path, default=Path("v41/manifests/v41_uid_splits.json"))
    parser.add_argument("--output", type=Path, default=Path("v42/run/retrieval_transfer_baseline"))
    parser.add_argument("--gate2c-checkpoint", type=Path,
                        default=Path("v42/run/gate2c_seed42_456/seed456/best.pt"))
    parser.add_argument("--gate1e-checkpoint", type=Path,
                        default=Path("v42/run/gate1e_seed42_456/seed456/best_total.pt"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    soft_train = soft_records(args.dataset, manifest, "train")
    soft_validation = soft_records(args.dataset, manifest, "validation")
    rotations = {}
    for split in ("train", "validation"):
        rows = [manifest["episodes"][eid] for eid in manifest["splits"][split]["panel_z"]]
        rotations[split] = [rotation_record(args.dataset, row) for row in rows]
    deformation = soft_audit(soft_train, soft_validation)
    deformation["current_gate2c"] = gate2c_soft_loss(
        args.dataset, manifest, args.gate2c_checkpoint, args.gate1e_checkpoint,
    )
    report = {
        "experiment": "v42_aligned_dino_retrieval_transfer_v1",
        "fit_split": "train_leave_one_uid_out", "evaluation_split": "validation",
        "test_used": False, "k_values": list(K_VALUES),
        "soft_deformation": deformation,
        "rotation": {
            family: rotation_audit(
                [r for r in rotations["train"] if r["family"] == family],
                [r for r in rotations["validation"] if r["family"] == family],
            ) for family in ("soft_body", "rigid")
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "RETRIEVAL_TRANSFER_RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
