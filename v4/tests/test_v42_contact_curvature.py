import torch

from mpm_dino_v4.v42_contact_curvature import (
    CONTACT_DIM, DIRECT_CONDITION_DIM, POINT_CONDITION_DIM,
    ContactCurvatureConditionBuilder, DirectProbeConditionBuilder,
    curvature_features, oracle_floor_contact_features,
)
from mpm_dino_v4.v42_stages import derive_impact_stages
from mpm_dino_v4.v42_model import V42RotationAwareSurrogate

from test_v42 import inputs


def feature_batch():
    batch = inputs(frames=7)
    batch["target"] = batch["x1"][:, None].expand(-1, 7, -1, -1).clone()
    batch["target"][..., 2] -= torch.linspace(0, 0.05, 7)[None, :, None]
    batch["target_mask"] = batch["input_mask"][:, None].expand(-1, 7, -1)
    return batch


def test_fixed_features_are_finite_masked_and_have_expected_shape():
    batch = feature_batch()
    curvature = curvature_features(batch)
    contact = oracle_floor_contact_features(batch)
    assert curvature.shape == (1, 16, 4)
    assert contact.shape == (1, 7, 16, 3)
    assert torch.isfinite(curvature).all()
    assert torch.isfinite(contact).all()
    assert curvature.min() >= 0 and curvature.max() <= 1


def test_four_arms_share_contract_and_only_enable_declared_channels():
    batch = feature_batch()
    zero = ContactCurvatureConditionBuilder(False, False)(batch, None)
    contact = ContactCurvatureConditionBuilder(True, False)(batch, None)
    curvature = ContactCurvatureConditionBuilder(False, True)(batch, None)
    combined = ContactCurvatureConditionBuilder(True, True)(batch, None)
    assert zero.shape == (1, 7, 16, POINT_CONDITION_DIM)
    assert torch.count_nonzero(zero) == 0
    assert torch.count_nonzero(contact[..., CONTACT_DIM:]) == 0
    assert torch.count_nonzero(curvature[..., :CONTACT_DIM]) == 0
    assert torch.equal(curvature[:, 0], curvature[:, -1])
    assert torch.equal(combined[..., :CONTACT_DIM], contact[..., :CONTACT_DIM])
    assert torch.equal(combined[..., CONTACT_DIM:], curvature[..., CONTACT_DIM:])


def test_adapter_accepts_pointwise_condition_and_backpropagates():
    data = inputs(frames=7)
    model = V42RotationAwareSurrogate(
        hidden_dim=16, blocks=1, heads=4, dropout=0.0, frames=7,
        gradient_checkpointing=False,
        oracle_condition_dim=POINT_CONDITION_DIM,
        oracle_injection="adapter",
    )
    torch.nn.init.normal_(model.canonical_head[-1].weight, std=1e-3)
    condition = torch.randn(1, 7, 16, POINT_CONDITION_DIM)
    prediction = model(**data, oracle_condition=condition)
    assert prediction.canonical_displacement.shape == (1, 7, 16, 3)
    prediction.canonical_displacement.square().sum().backward()
    gradient = model.oracle_adapter_projection[1].weight.grad
    assert gradient is not None
    assert torch.linalg.vector_norm(gradient) > 0


def test_direct_probe_condition_has_matched_zero_and_full_contract():
    batch = feature_batch()
    stages = derive_impact_stages(
        batch["x1"], batch["target"], batch["input_mask"],
        batch["target_mask"], batch["neighbour_indices"],
        batch["neighbour_mask"], batch["rest_edge_lengths"], batch["dt"],
        batch["gravity"], batch["floor_z"],
    )
    zero = DirectProbeConditionBuilder(False)(batch, stages)
    full = DirectProbeConditionBuilder(True)(batch, stages)
    assert zero.shape == full.shape == (1, 7, 16, DIRECT_CONDITION_DIM)
    assert torch.count_nonzero(zero) == 0
    assert torch.count_nonzero(full) > 0


def test_direct_decoder_bypasses_adapter_and_receives_gradient():
    data = inputs(frames=7)
    model = V42RotationAwareSurrogate(
        hidden_dim=16, blocks=1, heads=4, dropout=0.0, frames=7,
        gradient_checkpointing=False,
        oracle_condition_dim=DIRECT_CONDITION_DIM,
        oracle_injection="direct",
    )
    condition = torch.randn(1, 7, 16, DIRECT_CONDITION_DIM)
    prediction = model(**data, oracle_condition=condition)
    assert prediction.canonical_displacement.shape == (1, 7, 16, 3)
    prediction.canonical_displacement.square().sum().backward()
    assert model.oracle_direct_head[1].weight.grad is not None
    assert all(
        parameter.grad is None for parameter in model.region_adapter.parameters()
    )
