import torch

from mpm_dino_v4.v42_contact_curvature import DirectProbeConditionBuilder, POINT_CONDITION_DIM
from mpm_dino_v4.v42_stages import derive_impact_stages
from mpm_dino_v4.v43_rotation_contact_adapter import (
    PointwiseContactRotationAdapter, synchronized_rotation_tokens,
)
from mpm_dino_v4.v43_rotation_memory import geodesic_radians
from test_v42 import inputs


def batch_and_condition():
    batch = inputs(frames=7)
    batch["target"] = batch["x1"][:, None].expand(-1, 7, -1, -1).clone()
    batch["target"][..., 2] -= torch.linspace(0, .05, 7)[None, :, None]
    batch["target_mask"] = batch["input_mask"][:, None].expand(-1, 7, -1)
    stages = derive_impact_stages(batch["x1"], batch["target"], batch["input_mask"],
        batch["target_mask"], batch["neighbour_indices"], batch["neighbour_mask"],
        batch["rest_edge_lengths"], batch["dt"], batch["gravity"], batch["floor_z"])
    condition = DirectProbeConditionBuilder(True)(batch, stages)
    hidden = torch.randn(1, 7, 16, 128)
    return batch, condition, hidden


def test_variants_are_parameter_matched_and_preserve_point_shape():
    batch, condition, hidden = batch_and_condition()
    values = {}
    for variant in ("physical_only", "pooled_contact", "pointwise_contact",
                    "contact_lever", "contact_torque_basis"):
        tokens, valid = synchronized_rotation_tokens(batch, hidden, condition, variant, 42)
        assert tokens.shape == (1, 7, 16, 149)
        assert valid.shape == (1, 7, 16)
        values[variant] = tokens
    assert torch.count_nonzero(values["physical_only"][..., 128:]) == 0
    assert torch.count_nonzero(values["pointwise_contact"][..., -6:]) == 0
    assert torch.count_nonzero(values["contact_lever"][..., -3:]) == 0
    assert torch.count_nonzero(values["contact_torque_basis"][..., -3:]) > 0


def test_contact_shuffle_preserves_event_time_and_is_deterministic():
    batch, condition, hidden = batch_and_condition()
    full, _ = synchronized_rotation_tokens(batch, hidden, condition, "contact_torque_basis", 42)
    one, _ = synchronized_rotation_tokens(batch, hidden, condition, "contact_shuffled", 42)
    two, _ = synchronized_rotation_tokens(batch, hidden, condition, "contact_shuffled", 42)
    assert torch.equal(one, two)
    temporal = slice(128 + POINT_CONDITION_DIM, 128 + 15)
    assert torch.equal(one[..., temporal], full[..., temporal])
    assert not torch.equal(one[..., 128:128 + POINT_CONDITION_DIM],
                           full[..., 128:128 + POINT_CONDITION_DIM])


def test_zero_event_time_only_zeros_temporal_channels():
    batch, condition, hidden = batch_and_condition()
    full, _ = synchronized_rotation_tokens(batch, hidden, condition, "contact_torque_basis")
    zero, _ = synchronized_rotation_tokens(batch, hidden, condition, "zero_event_time")
    assert torch.equal(zero[..., 128:128 + POINT_CONDITION_DIM],
                       full[..., 128:128 + POINT_CONDITION_DIM])
    assert torch.count_nonzero(zero[..., 128 + POINT_CONDITION_DIM:128 + 15]) == 0


def test_adapter_starts_near_identity_and_backpropagates_finitely():
    model = PointwiseContactRotationAdapter(hidden_dim=32, heads=4)
    tokens = torch.randn(1, 7, 16, 149)
    valid = torch.ones(1, 7, 16, dtype=torch.bool)
    rotation, delta, gate = model(tokens, valid)
    identity = torch.eye(3).expand_as(rotation)
    assert gate.max() < .011
    assert torch.equal(delta, torch.zeros_like(delta))
    loss = geodesic_radians(rotation, identity).mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
