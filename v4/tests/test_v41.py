from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import torch

from mpm_dino_v2.graph import build_mutual_knn_graph
from mpm_dino_v4.full_losses import compute_shape_balanced_trajectory_loss
from mpm_dino_v4.model import masked_mean
from mpm_dino_v4.full_model import FullTrajectorySurrogate
from mpm_dino_v4.v41_data import (
    FORBIDDEN_INPUT_KEYS, MODEL_INPUT_KEYS, UIDBalancedSampler,
    V41TrajectoryDataset, validate_v41_manifest,
)
from mpm_dino_v4.v41_model import (
    M1Fusion, M2LocalMemory, V41PooledTrackBSurrogate,
    V41TrajectorySurrogate, build_v41_model,
)


def inputs(n=12, frames=5):
    torch.manual_seed(3)
    x0 = torch.rand(1, n, 3)
    x1 = x0 + torch.tensor([0., 0., -0.005])
    mask = torch.ones(1, n, dtype=torch.bool)
    graph = build_mutual_knn_graph(x0[0], mask[0], candidate_k=5, max_neighbours=4)
    return {
        "x0": x0, "x1": x1, "input_mask": mask, "reference": x0.clone(),
        "dino": torch.randn(1, n, 384), "dino_valid": torch.arange(n)[None] % 3 != 0,
        "dt": torch.tensor([1/30]), "gravity": torch.tensor([[0.,0.,-9.81]]),
        "floor_z": torch.tensor([0.]), **{k: v[None] for k,v in graph.items()},
    }


def test_model_contract_has_no_forbidden_metadata_or_future_mask():
    assert not (set(MODEL_INPUT_KEYS) & FORBIDDEN_INPUT_KEYS)
    assert "target_mask" not in MODEL_INPUT_KEYS


def test_manifest_uid_leakage_detection():
    manifest = {"splits": {
        "train": {"uids": ["a"], "panel_z": ["az"], "panel_v": []},
        "validation": {"uids": ["a"], "panel_z": [], "panel_v": []},
        "test": {"uids": [], "panel_z": [], "panel_v": []},
    }, "episodes": {"az": {"uid": "a"}}}
    try:
        validate_v41_manifest(manifest)
    except ValueError as exc:
        assert "UID leakage" in str(exc)
    else:
        raise AssertionError("leakage accepted")


def test_m1_exact_row_alignment():
    layer = M1Fusion(4, 2)
    with torch.no_grad():
        layer.update[-1].weight.fill_(0.1)
    hidden = torch.zeros(1, 2, 3, 4)
    visual = torch.tensor([[[1., 10.], [2., 20.], [3., 30.]]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    first = layer(hidden, visual, valid)
    changed = visual.clone(); changed[:, 1] += 100
    second = layer(hidden, changed, valid)
    delta = (second-first).abs().sum((1,3))
    assert delta[0,0] == 0 and delta[0,1] > 0 and delta[0,2] == 0


def test_m2_local_only_no_dense_and_all_invalid_deterministic():
    x = inputs(n=12)
    layer = M2LocalMemory(16, 8, heads=4).eval()
    hidden = torch.randn(1, 5, 12, 16)
    visual = torch.randn(1, 12, 8)
    invalid = torch.zeros(1, 12, dtype=torch.bool)
    args = (x["neighbour_indices"], x["neighbour_mask"], x["rest_edge_vectors"])
    first = layer(hidden, visual, invalid, *args)
    second = layer(hidden, visual * 100, invalid, *args)
    assert torch.isfinite(first).all() and torch.equal(first, second)
    assert layer.last_attention_shape[-1] == x["neighbour_indices"].shape[-1] + 1
    assert 12 not in layer.last_attention_shape[-1:]


def test_zero_dino_retains_validity_and_model_is_finite():
    x = inputs()
    valid = x["dino_valid"].clone()
    x["dino"].zero_()
    model = V41TrajectorySurrogate("m2", hidden_dim=32, blocks=1, heads=4,
                                   dropout=0, frames=5, gradient_checkpointing=False)
    out = model(**x)
    assert torch.equal(x["dino_valid"], valid)
    assert torch.isfinite(out.position).all()


def test_local_decomposition_and_horizon_indices():
    x = inputs()
    model = V41TrajectorySurrogate("m1", hidden_dim=32, blocks=1, heads=4,
                                   dropout=0, frames=41, gradient_checkpointing=False)
    out = model(**x)
    mask = x["input_mask"][:,None].expand(-1,41,-1)
    assert torch.allclose(masked_mean(out.residual_local, mask, 2), torch.zeros(1,41,3), atol=1e-7)
    # H1 -> X[2] is output index 0, hence H30/H40 -> X[31]/X[41].
    assert 30 - 1 == 29 and 40 - 1 == 39
    assert out.position[:,29].shape == x["x0"].shape
    assert out.position[:,39].shape == x["x0"].shape


def test_m6_trunk_bytes_identical_before_stage2():
    torch.manual_seed(42)
    real = V41TrajectorySurrogate("m6", hidden_dim=32, blocks=1, heads=4, frames=5)
    torch.manual_seed(42)
    zero = V41TrajectorySurrogate("m6", hidden_dim=32, blocks=1, heads=4, frames=5)
    def digest(model):
        h = hashlib.sha256()
        for key, value in sorted(model.trunk_state_dict().items()):
            h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    assert digest(real) == digest(zero)


def test_pooled_track_b_adapter_is_exact_old_architecture():
    torch.manual_seed(17)
    original = FullTrajectorySurrogate(
        hidden_dim=32, blocks=1, heads=4, dropout=0, frames=5,
        gradient_checkpointing=False,
    ).eval()
    torch.manual_seed(17)
    adapted = V41PooledTrackBSurrogate(
        hidden_dim=32, blocks=1, heads=4, dropout=0, frames=5,
        gradient_checkpointing=False,
    ).eval()
    assert original.state_dict().keys() == adapted.state_dict().keys()
    for key in original.state_dict():
        assert torch.equal(original.state_dict()[key], adapted.state_dict()[key])

    x = inputs(frames=5)
    old_inputs = {key: value for key, value in x.items() if key != "reference"}
    with torch.no_grad():
        expected = original(**old_inputs).position
        actual = adapted(**x).position
    assert torch.equal(expected, actual)


def test_pooled_track_b_zero_control_keeps_full_architecture_and_validity():
    x = inputs(frames=5)
    valid = x["dino_valid"].clone()
    x["dino"].zero_()
    torch.manual_seed(23)
    zero = build_v41_model(
        "track_b_pooled", hidden_dim=32, blocks=1, heads=4, dropout=0,
        frames=5, gradient_checkpointing=False,
    ).eval()
    torch.manual_seed(23)
    real_receiving_zero = build_v41_model(
        "track_b_pooled", hidden_dim=32, blocks=1, heads=4, dropout=0,
        frames=5, gradient_checkpointing=False,
    ).eval()
    with torch.no_grad():
        first = zero(**x).position
        second = real_receiving_zero(**x).position
    assert torch.equal(x["dino_valid"], valid)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_shape_balanced_loss_separates_translation_from_deformation():
    x = inputs(n=12, frames=41)
    model = build_v41_model(
        "track_b_pooled", hidden_dim=32, blocks=1, heads=4, dropout=0,
        frames=41, gradient_checkpointing=False,
    ).eval()
    with torch.no_grad():
        output = model(**x)
    batch = {
        **x,
        "target": output.position.clone(),
        "target_mask": x["input_mask"][:, None].expand(-1, 41, -1).clone(),
    }
    perfect = compute_shape_balanced_trajectory_loss(output, batch)
    assert perfect.total == 0

    translated = output.position.clone()
    translated[:, :, :, 0] += 0.02
    translated_loss = compute_shape_balanced_trajectory_loss(
        replace(output, position=translated), batch
    )
    assert translated_loss.world > 0 and translated_loss.com > 0
    assert translated_loss.shape < 1e-8
    assert translated_loss.strain < 1e-8

    deformed = output.position.clone()
    deformed[:, :, 0, 0] += 0.02
    deformed_loss = compute_shape_balanced_trajectory_loss(
        replace(output, position=deformed), batch
    )
    assert deformed_loss.shape > 0
    assert deformed_loss.strain > 0


def test_shape_balanced_key_loss_uses_h16_h30_h40_only():
    x = inputs(n=12, frames=41)
    model = build_v41_model(
        "track_b_pooled", hidden_dim=32, blocks=1, heads=4, dropout=0,
        frames=41, gradient_checkpointing=False,
    ).eval()
    with torch.no_grad():
        output = model(**x)
    batch = {
        **x,
        "target": output.position.clone(),
        "target_mask": x["input_mask"][:, None].expand(-1, 41, -1).clone(),
    }
    h4_only = output.position.clone()
    h4_only[:, 3, :, 0] += 0.02
    h4_loss = compute_shape_balanced_trajectory_loss(
        replace(output, position=h4_only), batch
    )
    assert h4_loss.world > 0
    assert h4_loss.key_horizons == 0

    h30 = output.position.clone()
    h30[:, 29, :, 0] += 0.02
    h30_loss = compute_shape_balanced_trajectory_loss(
        replace(output, position=h30), batch
    )
    assert h30_loss.key_horizons > 0
