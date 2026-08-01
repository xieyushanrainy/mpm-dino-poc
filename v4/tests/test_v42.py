from __future__ import annotations

import math

import numpy as np
import torch

from mpm_dino_v2.graph import build_mutual_knn_graph
from mpm_dino_v4.v42_geometry import (
    canonical_targets, identity_rotation_6d, rotation_6d_to_matrix,
    rotation_chordal, rotation_matrix_to_vector, rotation_vector_to_matrix,
)
from mpm_dino_v4.v42_losses import (
    compute_v42_local_losses, compute_v42_losses,
)
from mpm_dino_v4.v42_model import V42RotationAwareSurrogate
from mpm_dino_v4.v42_stages import (
    STAGE_RAW_WEIGHTS, ImpactStage, derive_impact_stages,
    lowest_active_mean_gap, total_mass_stage_weights,
)
from mpm_dino_v4.v42_train import gate1_parameters
from mpm_dino_v4.v42_gate1d_train import (
    gate1f_screen, identity_screen, rotation_parameters,
)
from mpm_dino_v4.v42_rotation_audit import (
    constant_angular_rotation, geodesic_error, proper_kabsch,
    rotation_from_vector,
)
from mpm_dino_v4.v42_gate2 import (
    LOCAL_PREFIXES, gate2_screen, load_gate1e_source, local_parameters,
    protected_is_identical, protected_snapshot,
)
from mpm_dino_v4.v42_overfit import (
    overfit_passed, select_overfit_objective,
)


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


def test_chordal_rotation_loss_is_zero_and_has_no_geodesic_floor():
    identity = torch.eye(3).reshape(1, 3, 3)
    loss = rotation_chordal(identity, identity)
    assert torch.equal(loss, torch.zeros_like(loss))


def test_axis_angle_exponential_is_proper_and_differentiable():
    vector = torch.tensor(
        [[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]], requires_grad=True,
    )
    matrix = rotation_vector_to_matrix(vector)
    identity = torch.eye(3).expand(2, -1, -1)
    assert torch.allclose(
        matrix.transpose(-1, -2) @ matrix, identity, atol=1e-6,
    )
    assert torch.allclose(torch.linalg.det(matrix), torch.ones(2), atol=1e-6)
    matrix.square().sum().backward()
    assert vector.grad is not None
    assert torch.isfinite(vector.grad).all()


def test_axis_angle_log_exp_round_trip():
    vector = torch.tensor([[0.1, -0.2, 0.3], [-0.05, 0.02, 0.01]])
    recovered = rotation_matrix_to_vector(rotation_vector_to_matrix(vector))
    assert torch.allclose(recovered, vector, atol=1e-6)


def test_constant_angular_baseline_extrapolates_observed_rotation():
    source = np.array([
        [-1.0, -0.5, 0.2], [0.8, -0.3, 0.1],
        [0.4, 1.1, -0.2], [-0.2, 0.3, 1.0],
    ])
    step = rotation_from_vector(np.array([0.0, 0.0, 0.1]))
    observed = source @ step
    recovered, _, ratio = proper_kabsch(
        source, observed, np.ones(len(source), dtype=bool)
    )
    assert ratio > 1e-3
    assert geodesic_error(recovered, step) < 1e-7
    expected = rotation_from_vector(np.array([0.0, 0.0, 0.8]))
    assert geodesic_error(
        constant_angular_rotation(recovered, 8), expected
    ) < 1e-7


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
    assert torch.equal(output.com[:, 0], output.ballistic_com[:, 0])
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


def test_gate1c_axis_angle_head_has_exact_identity_h1():
    sample = inputs(frames=5)
    model = V42RotationAwareSurrogate(
        hidden_dim=32, blocks=1, heads=4, dropout=0, frames=5,
        gradient_checkpointing=False, rotation_parameterization="axis_angle",
    )
    output = model(**sample)
    assert output.rotation_representation.shape == (1, 5, 3)
    assert torch.equal(
        output.rotation[:, 0], torch.eye(3).reshape(1, 3, 3),
    )


def test_gate1d_attention_rotation_is_protected_from_physical_trunk():
    sample = inputs(frames=5)
    model = V42RotationAwareSurrogate(
        hidden_dim=32, blocks=1, heads=4, dropout=0, frames=5,
        gradient_checkpointing=False, rotation_parameterization="axis_angle",
        rotation_attention=True,
    )
    output = model(**sample)
    model.zero_grad(set_to_none=True)
    output.rotation[:, 1:].square().sum().backward()
    assert any(
        parameter.grad is not None
        for parameter in rotation_parameters(model)
    )
    assert all(
        parameter.grad is None
        for parameter in model.blocks.parameters()
    )


def test_gate1e_integrates_angular_velocity_and_anchors_h1():
    sample = inputs(frames=5)
    model = V42RotationAwareSurrogate(
        hidden_dim=32, blocks=1, heads=4, dropout=0, frames=5,
        gradient_checkpointing=False, rotation_parameterization="axis_angle",
        rotation_attention=True, rotation_dynamics=True,
    )
    with torch.no_grad():
        model.rotation_head[-1].bias.copy_(torch.tensor([0.0, 0.0, 0.1]))
    output = model(**sample)
    assert output.angular_velocity is not None
    assert output.angular_velocity_change is not None
    assert torch.equal(
        output.rotation[:, 0], torch.eye(3).reshape(1, 3, 3),
    )
    assert torch.equal(
        output.angular_velocity[:, 0], torch.zeros(1, 3),
    )
    assert torch.allclose(
        output.angular_velocity[:, 2] - output.angular_velocity[:, 1],
        torch.tensor([[0.0, 0.0, 0.1]]), atol=1e-6,
    )
    assert not torch.equal(
        output.rotation[:, -1], torch.eye(3).reshape(1, 3, 3),
    )


def test_gate1f_absolute_contact_gate_is_inference_only_and_anchors_h1():
    sample = inputs(frames=5)
    model = V42RotationAwareSurrogate(
        hidden_dim=32, blocks=1, heads=4, dropout=0, frames=5,
        gradient_checkpointing=False, rotation_parameterization="axis_angle",
        rotation_attention=True, contact_rotation_mode="absolute",
    )
    with torch.no_grad():
        model.rotation_head[-1].bias.copy_(torch.tensor([0.0, 0.0, 0.2]))
        model.rotation_contact_head[-1].bias.fill_(-40)
    closed = model(**sample)
    assert closed.contact_probability.shape == (1, 5)
    assert closed.contact_point.shape == (1, 5, 3)
    assert torch.equal(closed.contact_probability[:, 0], torch.zeros(1))
    assert torch.equal(
        closed.rotation[:, 0], torch.eye(3).reshape(1, 3, 3),
    )
    assert torch.allclose(
        closed.rotation, torch.eye(3).reshape(1, 1, 3, 3), atol=1e-6,
    )
    with torch.no_grad():
        model.rotation_contact_head[-1].bias.fill_(40)
    opened = model(**sample)
    assert not torch.allclose(
        opened.rotation[:, -1], torch.eye(3).reshape(1, 3, 3),
    )
    assert torch.allclose(
        torch.linalg.det(opened.rotation), torch.ones(1, 5), atol=1e-6,
    )


def test_gate1f_impulse_uses_damped_recurrence_and_protected_gradient():
    sample = inputs(frames=5)
    model = V42RotationAwareSurrogate(
        hidden_dim=32, blocks=1, heads=4, dropout=0, frames=5,
        gradient_checkpointing=False, rotation_parameterization="axis_angle",
        rotation_attention=True, contact_rotation_mode="impulse",
        angular_damping=0.8,
    )
    with torch.no_grad():
        model.rotation_head[-1].bias.copy_(torch.tensor([0.0, 0.0, 0.1]))
        model.rotation_contact_head[-1].bias.fill_(40)
    output = model(**sample)
    assert output.angular_velocity is not None
    assert output.angular_velocity_change is not None
    assert torch.equal(
        output.angular_velocity[:, 0], torch.zeros(1, 3),
    )
    assert torch.allclose(
        output.angular_velocity[:, 2],
        0.8 * output.angular_velocity[:, 1]
        + output.angular_velocity_change[:, 2],
        atol=1e-6,
    )
    model.zero_grad(set_to_none=True)
    (
        output.rotation[:, 1:].square().sum()
        + output.contact_probability[:, 1:].sum()
        + output.contact_point.square().sum()
    ).backward()
    assert any(
        parameter.grad is not None
        for parameter in rotation_parameters(model)
    )
    assert all(
        parameter.grad is None
        for parameter in model.blocks.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.v42_com_head.parameters()
    )


def test_gate1d_identity_screen_requires_every_stratum():
    strata = {}
    for group in ("rigid/panel_Z", "rigid/panel_V", "soft_body/panel_Z"):
        strata[group] = {
            f"h{horizon}": {
                "model_rotation_rad": 0.9,
                "identity_rotation_rad": 1.0,
                "count": 1,
            }
            for horizon in (1, 8, 16, 30, 40, 59)
        }
    passed, _ = identity_screen({"strata": strata})
    assert passed
    strata["soft_body/panel_Z"]["h30"]["model_rotation_rad"] = 2.0
    passed, _ = identity_screen({"strata": strata})
    assert not passed


def test_gate1f_screen_separates_active_learning_and_inactive_safety():
    validation = {"activity": {}, "strata": {}}
    for group in ("rigid/panel_Z", "rigid/panel_V"):
        validation["activity"][group] = {
            "active_model_rad": 0.08, "active_identity_rad": 0.10,
            "active_count": 10, "inactive_prediction_rad": 0.005,
            "inactive_count": 10,
        }
        validation["strata"][group] = {
            "h59": {
                "model_rotation_rad": 0.09,
                "identity_rotation_rad": 0.10,
            }
        }
    passed, _ = gate1f_screen(validation)
    assert passed
    validation["activity"]["rigid/panel_Z"][
        "inactive_prediction_rad"
    ] = 0.02
    passed, _ = gate1f_screen(validation)
    assert not passed


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


def test_decoder_overfit_gates_require_scale_and_timing_not_just_low_loss():
    frame = {
        "finite": True,
        "canonical_error_reduction": 0.96,
        "predicted_to_target_magnitude_ratio": 1.02,
    }
    assert overfit_passed("single_frame", frame)
    frame["predicted_to_target_magnitude_ratio"] = 0.1
    assert not overfit_passed("single_frame", frame)

    episode = {
        "finite": True,
        "canonical_error_reduction": 0.97,
        "magnitude_correlation": 0.98,
        "predicted_to_target_peak_ratio": 0.95,
        "peak_timing_error_frames": 1,
    }
    assert overfit_passed("single_episode", episode)
    episode["peak_timing_error_frames"] = 2
    assert not overfit_passed("single_episode", episode)


def test_canonical_overfit_objective_excludes_composite_auxiliary_terms():
    class Losses:
        total = torch.tensor(9.0)
        canonical = torch.tensor(2.0)

    assert select_overfit_objective(Losses, "composite") is Losses.total
    assert select_overfit_objective(Losses, "canonical_only") is Losses.canonical


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


def test_total_mass_stage_weights_remove_stage_length_dominance():
    labels = torch.tensor([[
        *([int(ImpactStage.FREE_FLIGHT)] * 4),
        *([int(ImpactStage.CONTACT_ONSET)] * 2),
        int(ImpactStage.PEAK_DEFORMATION),
        *([int(ImpactStage.POST_PRIMARY_EVENT)] * 5),
    ]])
    weights = total_mass_stage_weights(labels)
    assert torch.allclose(weights.mean(1), torch.ones(1))
    totals = {
        stage: weights[labels == int(stage)].sum()
        for stage in (
            ImpactStage.FREE_FLIGHT,
            ImpactStage.CONTACT_ONSET,
            ImpactStage.PEAK_DEFORMATION,
            ImpactStage.POST_PRIMARY_EVENT,
        )
    }
    reference = totals[ImpactStage.FREE_FLIGHT]
    for stage, total in totals.items():
        expected = STAGE_RAW_WEIGHTS[stage] / STAGE_RAW_WEIGHTS[
            ImpactStage.FREE_FLIGHT
        ]
        assert torch.allclose(total / reference, torch.tensor(expected))
    assert torch.allclose(
        weights[0, 0] / weights[0, -1], torch.tensor(5 / 4),
    )


def test_total_mass_peak_can_exceed_old_per_frame_cap():
    labels = torch.tensor([[
        *([int(ImpactStage.POST_PRIMARY_EVENT)] * 58),
        int(ImpactStage.PEAK_DEFORMATION),
    ]])
    weights = total_mass_stage_weights(labels)
    assert weights[0, -1] > 4
    assert torch.allclose(weights.mean(), torch.tensor(1.0))


def test_floor_gap_uses_lowest_four_active_points_not_single_outlier():
    positions = torch.zeros(1, 1, 12, 3)
    positions[..., 2] = 0.1
    positions[..., 0, 2] = -1.0
    active = torch.ones(1, 1, 12, dtype=torch.bool)
    gap = lowest_active_mean_gap(
        positions, active, torch.tensor([0.0]), tail_points=4,
    )
    assert torch.allclose(gap, torch.tensor([[(-1.0 + 0.3) / 4]]))


def test_gate2_strictly_loads_gate1e_and_only_enables_local_parameters(tmp_path):
    source = V42RotationAwareSurrogate(
        local_mode="zero", hidden_dim=32, blocks=1, heads=4,
        dropout=0, frames=59, gradient_checkpointing=False,
        rotation_parameterization="axis_angle", rotation_attention=True,
        rotation_dynamics=True,
    )
    checkpoint = tmp_path / "best_total.pt"
    torch.save({
        "model": source.state_dict(),
        "epoch": 20,
        "config": {
            "experiment": "v42_gate1e_protected_angular_dynamics",
            "model_contract_version": "gate1e_v1",
            "hidden_dim": 32, "blocks": 1, "heads": 4, "dropout": 0,
        },
    }, checkpoint)
    model, state = load_gate1e_source(checkpoint, "geometry")
    assert state["epoch"] == 20
    enabled = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert enabled
    assert all(name.startswith(LOCAL_PREFIXES) for name in enabled)


def test_gate2_local_loss_has_no_gradient_into_protected_global_path():
    sample = inputs(frames=7)
    model = V42RotationAwareSurrogate(
        local_mode="geometry", hidden_dim=32, blocks=1, heads=4,
        dropout=0, frames=7, gradient_checkpointing=False,
        local_trunk_alpha=0,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in local_parameters(model):
        parameter.requires_grad_(True)
    snapshot = protected_snapshot(model)
    output = model(**sample)
    batch = {
        **sample, "target": rigid_trajectory(sample["x1"], 7),
        "target_mask": sample["input_mask"][:, None].expand(-1, 7, -1),
        "family": ["rigid"],
    }
    targets = canonical_targets(
        batch["x1"], batch["target"], batch["input_mask"],
        batch["target_mask"],
    )
    frame_weights = torch.ones(1, 7, requires_grad=True)
    losses = compute_v42_local_losses(
        output, batch, targets=targets,
        frame_weights=frame_weights.detach(),
    )
    losses.total.backward()
    assert any(parameter.grad is not None for parameter in local_parameters(model))
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith(LOCAL_PREFIXES)
    )
    assert frame_weights.grad is None
    assert protected_is_identical(model, snapshot)


def test_gate2b_soft_scaling_amplifies_canonical_and_velocity_only():
    sample = inputs(frames=7)
    model = V42RotationAwareSurrogate(
        local_mode="geometry", hidden_dim=32, blocks=1, heads=4,
        dropout=0, frames=7, gradient_checkpointing=False,
        local_trunk_alpha=0,
    ).eval()
    output = model(**sample)
    target = rigid_trajectory(sample["x1"], 7)
    target = target.clone()
    ramp = torch.linspace(0, 1e-4, 7)
    target[:, :, 0, 0] += ramp
    batch = {
        **sample, "target": target,
        "target_mask": sample["input_mask"][:, None].expand(-1, 7, -1),
        "family": ["soft_body"],
    }
    targets = canonical_targets(
        batch["x1"], batch["target"], batch["input_mask"],
        batch["target_mask"],
    )
    common = {
        "targets": targets,
        "frame_weights": torch.ones(1, 7),
        "soft_deformation_quantile": 0.95,
        "soft_deformation_floor_fraction": 0.005,
        "family_balanced": True,
        "rigid_family_weight": 0.25,
        "rigid_zero_weight": 0.0,
    }
    x1 = compute_v42_local_losses(
        output, batch, soft_deformation_amplification_cap=1, **common,
    )
    x5 = compute_v42_local_losses(
        output, batch, soft_deformation_amplification_cap=5, **common,
    )
    assert x5.canonical > x1.canonical
    assert x5.local_velocity > x1.local_velocity
    assert torch.equal(x5.strain, x1.strain)
    assert torch.equal(x5.edge_length, x1.edge_length)
    assert torch.equal(x5.rigid_zero, x1.rigid_zero)


def test_gate2b_rigid_family_weight_and_removed_extra_zero_term():
    sample = inputs(frames=7)
    model = V42RotationAwareSurrogate(
        local_mode="geometry", hidden_dim=32, blocks=1, heads=4,
        dropout=0, frames=7, gradient_checkpointing=False,
    ).eval()
    with torch.no_grad():
        model.canonical_head[-1].weight[0].fill_(1e-3)
    output = model(**sample)
    batch = {
        **sample, "target": rigid_trajectory(sample["x1"], 7),
        "target_mask": sample["input_mask"][:, None].expand(-1, 7, -1),
        "family": ["rigid"],
    }
    common = {
        "soft_deformation_amplification_cap": 20,
        "soft_deformation_quantile": 0.95,
        "soft_deformation_floor_fraction": 0.005,
        "family_balanced": True,
        "rigid_zero_weight": 0.0,
    }
    full = compute_v42_local_losses(
        output, batch, rigid_family_weight=1.0, **common,
    )
    quarter = compute_v42_local_losses(
        output, batch, rigid_family_weight=0.25, **common,
    )
    assert torch.allclose(quarter.total, 0.25 * full.total)
    assert torch.allclose(quarter.canonical, 0.25 * full.canonical)
    assert torch.allclose(quarter.strain, 0.25 * full.strain)
    assert torch.allclose(quarter.edge_length, 0.25 * full.edge_length)
    assert torch.allclose(quarter.local_velocity, 0.25 * full.local_velocity)


def test_zero_local_baseline_is_exactly_zero_with_gate1e_rotation():
    sample = inputs(frames=5)
    model = V42RotationAwareSurrogate(
        local_mode="zero", hidden_dim=32, blocks=1, heads=4,
        dropout=0, frames=5, gradient_checkpointing=False,
        rotation_parameterization="axis_angle", rotation_attention=True,
        rotation_dynamics=True,
    ).eval()
    with torch.no_grad():
        output = model(**sample)
    assert torch.equal(
        output.canonical_displacement,
        torch.zeros_like(output.canonical_displacement),
    )
    reconstructed = output.com[:, :, None] + torch.einsum(
        "bni,btij->btnj",
        sample["x1"] - sample["x1"].mean(1, keepdim=True),
        output.rotation,
    )
    assert torch.equal(output.position, reconstructed)


def test_gate2_screen_enforces_every_threshold_without_aggregation():
    base = {}
    learned = {}
    for group in ("panel_Z/soft_body", "panel_V/soft_body"):
        base[group] = {
            "stage_weighted_canonical_nrmse": 1.0,
            "stage_weighted_strain_rmse": 1.0,
            "compression_canonical_nrmse": 1.0,
            "compression_strain_rmse": 1.0,
            "peak_deformation_canonical_nrmse": 1.0,
            "peak_deformation_strain_rmse": 1.0,
        }
        learned[group] = {
            **base[group],
            "stage_weighted_canonical_nrmse": 0.89,
            "stage_weighted_strain_rmse": 0.89,
            "compression_canonical_nrmse": 0.9,
            "uid_balanced_magnitude_correlation": 0.5,
            "identifiable_timing_episodes": 1,
            "median_onset_error_frames": 2,
            "median_peak_error_frames": 2,
            "rigid_local_rms_fraction": None,
        }
    learned["panel_Z/rigid"] = {
        "rigid_local_rms_fraction": 0.0009,
    }
    assert gate2_screen(learned, base)["passed"]
    learned["panel_V/soft_body"]["uid_balanced_magnitude_correlation"] = 0.49
    assert not gate2_screen(learned, base)["passed"]
