from __future__ import annotations

import math

import torch

from mpm_dino_v2.graph import build_mutual_knn_graph
from mpm_dino_v4.v42_geometry import (
    canonical_targets, identity_rotation_6d, rotation_6d_to_matrix,
)
from mpm_dino_v4.v42_losses import compute_v42_losses
from mpm_dino_v4.v42_model import V42RotationAwareSurrogate
from mpm_dino_v4.v42_stages import (
    ImpactStage, derive_impact_stages, lowest_active_mean_gap,
)
from mpm_dino_v4.v42_train import gate1_parameters


def inputs(n=16, frames=7):
    torch.manual_seed(20260728)
    x0 = torch.rand(1, n, 3)
    x0[..., 2] += 0.25
    x1 = x0 + torch.tensor([0.0, 0.0, -0.005])
    mask = torch.ones(1, n, dtype=torch.bool)
    graph = build_mutual_knn_graph(
        x1[0], mask[0], candidate_k=5, max_neighbours=4
    )
    return {
        "x0": x0, "x1": x1, "input_mask": mask, "reference": x0,
        "dino": torch.randn(1, n, 384), "dino_valid": mask.clone(),
        "dt": torch.tensor([1 / 30]), "gravity": torch.tensor([[0., 0., -9.81]]),
        "floor_z": torch.tensor([0.]),
        **{key: value[None] for key, value in graph.items()},
    }


def rigid_trajectory(x1, frames):
    centre = x1.mean(1)
    shape = x1 - centre[:, None]
    outputs = []
    for index in range(frames):
        angle = (index + 1) * 0.08
        rotation = torch.tensor([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        outputs.append(
            centre[:, None] + torch.tensor([0.01 * index, 0.0, -0.02 * index])
            + shape @ rotation
        )
    return torch.stack(outputs, 1)


def test_rotation_6d_identity_is_proper():
    matrix = rotation_6d_to_matrix(identity_rotation_6d(2))
    assert torch.allclose(matrix, torch.eye(3).expand(2, -1, -1))
    assert torch.allclose(torch.linalg.det(matrix), torch.ones(2))


def test_kabsch_canonical_target_removes_rigid_motion():
    sample = inputs()
    target = rigid_trajectory(sample["x1"], 7)
    mask = sample["input_mask"][:, None].expand(-1, 7, -1)
    result = canonical_targets(
        sample["x1"], target, sample["input_mask"], mask
    )
    assert result.valid_rotation.all()
    assert result.displacement.abs().max() < 2e-6


def test_rank_deficient_target_is_invalid_for_vector_rotation_loss():
    x1 = torch.zeros(1, 8, 3)
    x1[0, :, 0] = torch.linspace(-1, 1, 8)
    target = x1[:, None].expand(-1, 2, -1, -1).clone()
    mask = torch.ones(1, 8, dtype=torch.bool)
    result = canonical_targets(x1, target, mask, mask[:, None].expand(-1, 2, -1))
    assert not result.valid_rotation.any()


def test_v42_reconstruction_and_protected_local_gradient():
    sample = inputs(frames=5)
    model = V42RotationAwareSurrogate(
        hidden_dim=32, blocks=1, heads=4, dropout=0, frames=5,
        gradient_checkpointing=False, local_trunk_alpha=0,
    )
    output = model(**sample)
    assert output.position.shape == (1, 5, 16, 3)
    local_parameters = (
        list(model.dino_projection.parameters())
        + list(model.region_encoder.parameters())
        + list(model.region_adapter.parameters())
        + list(model.canonical_head.parameters())
    )
    model.zero_grad(set_to_none=True)
    output.canonical_displacement.square().sum().backward()
    assert any(parameter.grad is not None for parameter in local_parameters)
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.blocks.parameters()
    )


def test_geometry_is_architecture_identical_zero_dino_control():
    sample = inputs(frames=5)
    torch.manual_seed(42)
    geometry = V42RotationAwareSurrogate(
        local_mode="geometry", hidden_dim=32, blocks=1, heads=4,
        dropout=0, frames=5, gradient_checkpointing=False,
    ).eval()
    torch.manual_seed(42)
    real = V42RotationAwareSurrogate(
        local_mode="real_dino", hidden_dim=32, blocks=1, heads=4,
        dropout=0, frames=5, gradient_checkpointing=False,
    ).eval()
    assert geometry.state_dict().keys() == real.state_dict().keys()
    assert all(
        torch.equal(geometry.state_dict()[key], real.state_dict()[key])
        for key in geometry.state_dict()
    )
    zero_sample = {key: value for key, value in sample.items()}
    zero_sample["dino"] = torch.zeros_like(sample["dino"])
    with torch.no_grad():
        geometry_output = geometry(**sample)
        real_receiving_zero = real(**zero_sample)
    assert torch.equal(
        geometry_output.canonical_displacement,
        real_receiving_zero.canonical_displacement,
    )
    assert torch.equal(geometry_output.com, real_receiving_zero.com)
    assert torch.equal(geometry_output.rotation, real_receiving_zero.rotation)


def test_gate1_optimizer_excludes_every_local_and_visual_parameter():
    model = V42RotationAwareSurrogate(
        local_mode="zero", hidden_dim=32, blocks=1, heads=4,
        dropout=0, frames=5, gradient_checkpointing=False,
    )
    selected = {id(parameter) for parameter in gate1_parameters(model)}
    for name, parameter in model.named_parameters():
        if name.startswith((
            "dino_projection.", "region_encoder.", "region_adapter.",
            "canonical_head.", "local_head.", "com_head.",
        )):
            assert id(parameter) not in selected
    assert all(
        id(parameter) in selected
        for parameter in model.v42_com_head.parameters()
    )
    assert all(
        id(parameter) in selected
        for parameter in model.rotation_head.parameters()
    )


def test_separated_losses_are_finite_and_backward():
    sample = inputs(frames=7)
    model = V42RotationAwareSurrogate(
        hidden_dim=32, blocks=1, heads=4, dropout=0, frames=7,
        gradient_checkpointing=False,
    )
    output = model(**sample)
    target = rigid_trajectory(sample["x1"], 7)
    batch = {
        **sample, "target": target,
        "target_mask": sample["input_mask"][:, None].expand(-1, 7, -1),
        "family": ["rigid"],
    }
    losses = compute_v42_losses(output, batch)
    assert torch.isfinite(losses.total)
    losses.total.backward()


def test_stage_metadata_is_not_a_model_input_and_weights_are_bounded():
    sample = inputs(frames=9)
    target = rigid_trajectory(sample["x1"], 9)
    # Create an obvious floor impact and COM impulse.
    target[:, 3:, :, 2] -= 1.00
    target[:, 4:, :, 2] += 0.12
    target_mask = sample["input_mask"][:, None].expand(-1, 9, -1)
    stages = derive_impact_stages(
        sample["x1"], target, sample["input_mask"], target_mask,
        sample["neighbour_indices"], sample["neighbour_mask"],
        sample["rest_edge_lengths"], sample["dt"], sample["gravity"],
        sample["floor_z"],
    )
    assert stages.labels.shape == (1, 10)
    assert stages.weights.max() <= 4
    assert stages.contact_onset[0] >= 0
    assert int(ImpactStage.CONTACT_ONSET) in stages.labels


def test_floor_gap_uses_lowest_four_active_points_not_single_outlier():
    positions = torch.zeros(1, 1, 12, 3)
    positions[..., 2] = 0.1
    positions[..., 0, 2] = -1.0
    active = torch.ones(1, 1, 12, dtype=torch.bool)
    gap = lowest_active_mean_gap(
        positions, active, torch.tensor([0.0]), tail_points=4,
    )
    assert torch.allclose(gap, torch.tensor([[(-1.0 + 0.3) / 4]]))
