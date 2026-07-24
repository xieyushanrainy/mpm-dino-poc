from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from mpm_dino_v2.graph import build_mutual_knn_graph, validate_neighbour_graph


FAMILIES = ("rigid", "fluid", "soft_body")


def _category(metadata: dict) -> str:
    materials = metadata.get("vlm", {}).get("materials", [])
    if not materials:
        return "unknown"
    counts: dict[str, int] = defaultdict(int)
    for item in materials:
        counts[str(item.get("material_category", "unknown"))] += 1
    return sorted(counts, key=lambda key: (-counts[key], key))[0]


def dataset_records(root: str | Path) -> list[dict]:
    root = Path(root)
    index = json.loads((root / "dataset.json").read_text())
    records = []
    for item in index["objects"]:
        metadata = json.loads((root / item["metadata"]).read_text())
        records.append({**item, "category": _category(metadata)})
    return records


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_split_manifest(root: str | Path, seed: int = 20260722) -> dict:
    """Deterministically allocate 20/5/5 UIDs per family, category round-robin."""
    root = Path(root)
    records = dataset_records(root)
    splits = {name: [] for name in ("train", "validation", "test")}
    strata = {}
    for family in FAMILIES:
        family_rows = [row for row in records if row["solver_route"] == family]
        if len(family_rows) != 30:
            raise ValueError(f"expected 30 {family} objects, found {len(family_rows)}")
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in family_rows:
            groups[row["category"]].append(row)
        ordered = []
        for category, rows in groups.items():
            rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['uid']}".encode()).hexdigest())
        categories = sorted(groups)
        while any(groups.values()):
            for category in categories:
                if groups[category]:
                    ordered.append(groups[category].pop(0))
        allocation = {"train": ordered[:20], "validation": ordered[20:25], "test": ordered[25:]}
        for split, rows in allocation.items():
            splits[split].extend(row["uid"] for row in rows)
            for row in rows:
                strata[row["uid"]] = {"family": family, "category": row["category"], "split": split}
    return {
        "schema_version": 1,
        "dataset": str(root.resolve()),
        "dataset_index_sha256": file_sha256(root / "dataset.json"),
        "package_sha256_manifest": file_sha256(root / "PACKAGE_SHA256.json"),
        "split_seed": seed,
        "allocation_per_family": {"train": 20, "validation": 5, "test": 5},
        "splits": splits,
        "strata": strata,
    }


def validate_manifest(manifest: dict) -> None:
    sets = [set(manifest["splits"][name]) for name in ("train", "validation", "test")]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1 :]):
        raise ValueError("UID split leakage detected")
    if len(set.union(*sets)) != 90:
        raise ValueError("manifest must contain exactly 90 unique UIDs")
    for family in FAMILIES:
        expected = {"train": 20, "validation": 5, "test": 5}
        for split, count in expected.items():
            actual = sum(manifest["strata"][uid]["family"] == family for uid in manifest["splits"][split])
            if actual != count:
                raise ValueError(f"{family}/{split}: expected {count}, found {actual}")


def prepare_cache(root: str | Path, manifest: dict, output: str | Path, candidate_k: int = 12, max_neighbours: int = 8) -> list[dict]:
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = {row["uid"]: row for row in dataset_records(root)}
    reports = []
    for uid in sorted(records):
        row = records[uid]
        with np.load(root / row["sample"]) as sample:
            x = torch.from_numpy(sample["trajectory_positions_m"].astype(np.float32))
            active = torch.from_numpy(sample["point_active"].astype(bool))
            dino = torch.from_numpy(sample["dino_features"].astype(np.float32))
            dino_valid = torch.from_numpy(sample["dino_valid"].astype(bool))
        reference_mask = active[0]
        graph = build_mutual_knn_graph(x[0], reference_mask, candidate_k, max_neighbours)
        report = validate_neighbour_graph(x[0], reference_mask, **graph)
        metadata = json.loads((root / row["metadata"]).read_text())
        payload = {
            "uid": uid, "family": row["solver_route"], "category": row["category"],
            "positions": x, "active": active, "dino": dino, "dino_valid": dino_valid,
            "dt": float(metadata["time"]["time_step_s"]),
            "gravity": torch.tensor(metadata["force"]["gravity_m_s2"], dtype=torch.float32),
            "floor_z": float(metadata["coordinate_system"]["floor_z_m"]),
            **graph,
        }
        torch.save(payload, output / f"{uid}.pt")
        reports.append({"uid": uid, "family": row["solver_route"], **report.to_dict()})
    (output / "graph_audit.json").write_text(json.dumps(reports, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return reports


def donor_map(uids: list[str], seed: int) -> dict[str, str]:
    if len(uids) < 2:
        return {uid: uid for uid in uids}
    ordered = sorted(uids, key=lambda uid: hashlib.sha256(f"donor:{seed}:{uid}".encode()).hexdigest())
    return {uid: ordered[(index + 1) % len(ordered)] for index, uid in enumerate(ordered)}


def point_permutation(uid: str, count: int, seed: int) -> torch.Tensor:
    value = int(hashlib.sha256(f"points:{seed}:{uid}".encode()).hexdigest()[:16], 16) % (2**63 - 1)
    return torch.randperm(count, generator=torch.Generator().manual_seed(value))


class WindowDataset(Dataset):
    def __init__(self, cache: str | Path, manifest: dict, split: str, families=("rigid", "soft_body"), dino_mode="real", seed=42):
        self.cache, self.split, self.dino_mode, self.seed = Path(cache), split, dino_mode, seed
        self.uids = [uid for uid in manifest["splits"][split] if manifest["strata"][uid]["family"] in families]
        self.index, self.scores, self.quartiles = [], [], []
        self._scenes: dict[str, dict] = {}
        for uid in self.uids:
            scene = self._load(uid)
            residual = scene["positions"][2:] - 2 * scene["positions"][1:-1] + scene["positions"][:-2]
            masks = scene["active"][2:] & scene["active"][1:-1] & scene["active"][:-2]
            score = torch.sqrt((residual.square().sum(-1) * masks).sum(-1) / masks.sum(-1).clamp_min(1)).numpy()
            cuts = np.quantile(score, [0.25, 0.5, 0.75])
            for t, value in enumerate(score):
                self.index.append((uid, t)); self.scores.append(float(value)); self.quartiles.append(int(np.searchsorted(cuts, value, side="right")))
        self.donors = donor_map(self.uids, seed)

    def _load(self, uid: str) -> dict:
        if uid not in self._scenes:
            self._scenes[uid] = torch.load(self.cache / f"{uid}.pt", map_location="cpu", weights_only=False)
        return self._scenes[uid]

    def __len__(self): return len(self.index)

    def __getitem__(self, item):
        uid, t = self.index[item]; scene = self._load(uid)
        dino_scene = self._load(self.donors[uid]) if self.dino_mode == "scene_shuffled" else scene
        dino, valid = dino_scene["dino"], dino_scene["dino_valid"]
        if self.dino_mode == "zero": dino = torch.zeros_like(dino)
        elif self.dino_mode == "point_shuffled":
            permutation = point_permutation(uid, len(dino), self.seed); dino, valid = dino[permutation], valid[permutation]
        elif self.dino_mode not in {"real", "scene_shuffled"}: raise ValueError(f"unsupported DINO mode: {self.dino_mode}")
        return {
            "uid": uid, "family": scene["family"], "t": t,
            "x_prev": scene["positions"][t], "x_curr": scene["positions"][t + 1], "target": scene["positions"][t + 2],
            "mask_prev": scene["active"][t], "mask_curr": scene["active"][t + 1], "target_mask": scene["active"][t] & scene["active"][t + 1] & scene["active"][t + 2],
            "reference": scene["positions"][0], "dino": dino, "dino_valid": valid,
            "dt": torch.tensor(scene["dt"]), "gravity": scene["gravity"], "floor_z": torch.tensor(scene["floor_z"]),
            **{key: scene[key] for key in ("neighbour_indices", "neighbour_mask", "rest_edge_vectors", "rest_edge_lengths")},
        }


class QuartileBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: WindowDataset, batch_size: int, seed: int):
        self.dataset, self.batch_size, self.seed, self.epoch = dataset, batch_size, seed, 0
        self.bins = [[i for i, q in enumerate(dataset.quartiles) if q == value] for value in range(4)]

    def __len__(self): return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch); self.epoch += 1
        count = len(self) * self.batch_size; samples = []
        for index in range(count):
            bucket = self.bins[index % 4]
            samples.append(bucket[int(torch.randint(len(bucket), (), generator=generator))])
        for start in range(0, count, self.batch_size): yield samples[start:start + self.batch_size]
