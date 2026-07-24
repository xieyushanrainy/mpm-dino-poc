from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from mpm_dino_v2.graph import build_mutual_knn_graph
from .data import file_sha256, point_permutation


MODEL_INPUT_KEYS = (
    "x0", "x1", "input_mask", "reference", "dino", "dino_valid", "dt",
    "gravity", "floor_z", "neighbour_indices", "neighbour_mask",
    "rest_edge_vectors", "rest_edge_lengths",
)
FORBIDDEN_INPUT_KEYS = {
    "family", "material_class", "solver_route", "vlm", "vlm_parameters",
    "point_material_ids", "material_labels", "target_mask", "active",
}


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def make_v41_manifest(
    dataset_root: str | Path,
    v4_manifest: str | Path,
    output: str | Path,
) -> dict:
    """Freeze V4.1 splits by reusing V4's reviewed object split."""
    root = Path(dataset_root)
    prior = json.loads(Path(v4_manifest).read_text())
    z_rows = read_jsonl(root / "subsets/free_fall_zero_velocity_all_families.jsonl")
    v_rows = read_jsonl(root / "subsets/rigid_free_fall_initial_velocity.jsonl")
    rows = {row["episode_id"]: row for row in z_rows + v_rows}
    uid_split = {
        uid: split
        for split in ("train", "validation", "test")
        for uid in prior["splits"][split]
    }
    collection_uids = {row["uid"] for row in rows.values()}
    if collection_uids - uid_split.keys():
        raise ValueError("V4.1 contains UIDs absent from the V4 split")
    splits = {}
    for split in ("train", "validation", "test"):
        uids = sorted(uid for uid in collection_uids if uid_split[uid] == split)
        panel_z = sorted(row["episode_id"] for row in z_rows if row["uid"] in uids)
        panel_v = sorted(row["episode_id"] for row in v_rows if row["uid"] in uids)
        splits[split] = {"uids": uids, "panel_z": panel_z, "panel_v": panel_v}
    payload = {
        "schema_version": 1,
        "split_policy": "reuse_v4_balanced_90_uid_split",
        "source_split_seed": prior["split_seed"],
        "dataset_collection_sha256": file_sha256(root / "collection.json"),
        "source_v4_manifest_sha256": file_sha256(v4_manifest),
        "source_manifests": {
            "panel_z": {
                "path": "subsets/free_fall_zero_velocity_all_families.jsonl",
                "sha256": file_sha256(root / "subsets/free_fall_zero_velocity_all_families.jsonl"),
            },
            "panel_v": {
                "path": "subsets/rigid_free_fall_initial_velocity.jsonl",
                "sha256": file_sha256(root / "subsets/rigid_free_fall_initial_velocity.jsonl"),
            },
        },
        "splits": splits,
        "episodes": rows,
    }
    validate_v41_manifest(payload)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_content_sha256"] = hashlib.sha256(canonical).hexdigest()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def validate_v41_manifest(manifest: dict) -> None:
    uid_sets = [set(manifest["splits"][s]["uids"]) for s in ("train", "validation", "test")]
    if any(a & b for i, a in enumerate(uid_sets) for b in uid_sets[i + 1:]):
        raise ValueError("UID leakage")
    episode_sets = []
    for split in ("train", "validation", "test"):
        uids = set(manifest["splits"][split]["uids"])
        ids = manifest["splits"][split]["panel_z"] + manifest["splits"][split]["panel_v"]
        episode_sets.append(set(ids))
        if any(manifest["episodes"][episode]["uid"] not in uids for episode in ids):
            raise ValueError(f"episode assigned outside UID split: {split}")
    if any(a & b for i, a in enumerate(episode_sets) for b in episode_sets[i + 1:]):
        raise ValueError("episode leakage")


class V41TrajectoryDataset(Dataset):
    """Panel Z/V episodes with shared static tensors and a fixed reference graph."""

    def __init__(self, root, manifest, split, dino_mode="real", seed=42, cache_graphs=True):
        if dino_mode not in {"real", "zero", "scene_shuffled", "point_shuffled"}:
            raise ValueError(dino_mode)
        self.root, self.manifest, self.split = Path(root), manifest, split
        self.dino_mode, self.seed = dino_mode, seed
        ids = manifest["splits"][split]["panel_z"] + manifest["splits"][split]["panel_v"]
        self.rows = [manifest["episodes"][episode] for episode in ids]
        self.by_uid = defaultdict(list)
        for index, row in enumerate(self.rows):
            self.by_uid[row["uid"]].append(index)
        self.uids = sorted(self.by_uid)
        ordered = sorted(self.uids, key=lambda u: hashlib.sha256(f"donor:{seed}:{u}".encode()).hexdigest())
        self.donors = {uid: ordered[(i + 1) % len(ordered)] for i, uid in enumerate(ordered)}
        self.uid_static = {row["uid"]: row["object_static"] for row in self.rows}
        self._static, self._graphs = {}, {}
        self.cache_graphs = cache_graphs

    def __len__(self):
        return len(self.rows)

    def _load_static(self, uid):
        if uid not in self._static:
            with np.load(self.root / self.uid_static[uid], allow_pickle=False) as data:
                self._static[uid] = {
                    "reference": torch.from_numpy(data["reference_positions_m"].astype(np.float32)),
                    "dino": torch.from_numpy(data["dino_features"].astype(np.float32)),
                    "dino_valid": torch.from_numpy(data["dino_valid"].astype(bool)),
                }
        return self._static[uid]

    def _graph(self, uid, reference, input_mask):
        if uid not in self._graphs:
            self._graphs[uid] = build_mutual_knn_graph(
                reference, input_mask, candidate_k=12, max_neighbours=8
            )
        return self._graphs[uid]

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.root / row["trajectory"], allow_pickle=False) as data:
            positions = torch.from_numpy(data["trajectory_positions_m"].astype(np.float32))
            active = torch.from_numpy(data["point_active"].astype(bool))
            times = torch.from_numpy(data["times_s"].astype(np.float32))
        static = self._load_static(row["uid"])
        donor = self._load_static(self.donors[row["uid"]]) if self.dino_mode == "scene_shuffled" else static
        dino, valid = donor["dino"], donor["dino_valid"]
        if self.dino_mode == "zero":
            dino = torch.zeros_like(dino)
        elif self.dino_mode == "point_shuffled":
            order = point_permutation(row["uid"], len(dino), self.seed)
            dino, valid = dino[order], valid[order]
        input_mask = active[0] & active[1]
        graph = self._graph(row["uid"], static["reference"], input_mask)
        velocity = torch.tensor(row["initial_linear_velocity_m_s"], dtype=torch.float32)
        return {
            "uid": row["uid"], "episode_id": row["episode_id"], "family": row["family"],
            "panel": "Z" if row["initial_velocity_regime"] == "zero" else "V",
            "velocity_regime": row["initial_velocity_regime"],
            "initial_velocity": velocity,
            "x0": positions[0], "x1": positions[1], "target": positions[2:],
            "input_mask": input_mask, "target_mask": active[2:] & input_mask[None],
            "reference": static["reference"], "dino": dino, "dino_valid": valid,
            "dt": times[1] - times[0], "gravity": torch.tensor([0., 0., -9.81]),
            "floor_z": torch.tensor(0.0), **graph,
        }


class UIDBalancedSampler(Sampler[int]):
    """One draw per UID per cycle, then a uniform episode draw within UID."""

    def __init__(self, dataset: V41TrajectoryDataset, draws_per_epoch: int, seed: int):
        self.dataset, self.draws, self.seed, self.epoch = dataset, draws_per_epoch, seed, 0
        self.by_family = defaultdict(list)
        for uid in dataset.uids:
            family = dataset.rows[dataset.by_uid[uid][0]]["family"]
            self.by_family[family].append(uid)

    def __len__(self):
        return self.draws

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        families = sorted(self.by_family)
        for draw in range(self.draws):
            family = families[draw % len(families)]
            uids = self.by_family[family]
            uid = uids[int(torch.randint(len(uids), (), generator=generator))]
            episodes = self.dataset.by_uid[uid]
            yield episodes[int(torch.randint(len(episodes), (), generator=generator))]
