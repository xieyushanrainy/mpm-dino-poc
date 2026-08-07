import torch

from mpm_dino_v5.model import (
    V5COMModel,
    V5CausalDeformationModel,
    V5DeformationDecoder,
    V5InteractionEncoder,
    V5RotationModel,
    V5SharedPhysicalModel,
    ballistic_com,
)


def inputs(batch=2, points=16, frames=7):
    torch.manual_seed(4)
    x0 = torch.randn(batch, points, 3) * .1
    x0[..., 2] += 1
    x1 = x0.clone()
    x1[..., 2] -= .01
    mask = torch.ones(batch, points, dtype=torch.bool)
    mask[:, -2:] = False
    indices = torch.stack((
        (torch.arange(points) - 1) % points,
        (torch.arange(points) + 1) % points,
    ), -1)[None].expand(batch, -1, -1)
    neighbours = torch.gather(
        x1[:, None].expand(-1, points, -1, -1), 2,
        indices[..., None].expand(-1, -1, -1, 3),
    )
    rest_vectors = neighbours - x1[:, :, None]
    return {
        "x0": x0,
        "x1": x1,
        "input_mask": mask,
        "dt": torch.full((batch,), 1 / 30),
        "gravity": torch.tensor([[0., 0., -9.81]]).expand(batch, -1),
        "floor_z": torch.zeros(batch),
        "reference": x1.clone(),
        "dino": torch.zeros(batch, points, 384),
        "dino_valid": mask.clone(),
        "neighbour_indices": indices,
        "neighbour_mask": mask[:, :, None].expand(-1, -1, 2).clone(),
        "rest_edge_vectors": rest_vectors,
        "rest_edge_lengths": torch.linalg.vector_norm(rest_vectors, dim=-1),
    }


def test_ballistic_com_uses_two_observations_and_gravity():
    values = inputs(batch=1)
    predicted = ballistic_com(
        values["x0"], values["x1"], values["input_mask"],
        values["dt"], values["gravity"], frames=3,
    )
    assert predicted.shape == (1, 3, 3)
    assert predicted[0, 1, 2] < predicted[0, 0, 2]


def test_factorized_model_shapes_proper_rotation_and_zero_mean_deformation():
    values = inputs()
    model = V5CausalDeformationModel(
        V5COMModel(hidden_dim=16, frames=7, dropout=0, blocks=1, heads=4),
        V5RotationModel(hidden_dim=16, frames=7, dropout=0),
        V5InteractionEncoder(hidden_dim=16, blocks=2, dropout=0),
        V5DeformationDecoder(interaction_dim=16, hidden_dim=16, blocks=2, dropout=0),
    )
    output = model(**values)
    assert output.position.shape == (2, 7, 16, 3)
    assert output.interaction.contact_logits.shape == (2, 7, 16)
    assert output.interaction.event_time.shape == (2, 7)
    identity = torch.eye(3).expand_as(output.rotation)
    assert torch.allclose(output.rotation.transpose(-1, -2) @ output.rotation, identity, atol=1e-5)
    assert torch.all(torch.linalg.det(output.rotation) > .999)
    valid = values["input_mask"][:, None, :, None]
    mean = (output.canonical_displacement * valid).sum(2) / valid.sum(2)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-7)
    assert torch.count_nonzero(output.canonical_displacement[:, :, -2:]) == 0


def test_inference_has_no_future_label_dependency():
    values = inputs(batch=1)
    model = V5CausalDeformationModel(
        V5COMModel(16, 7, 0, 1, 4), V5RotationModel(16, 7, 0),
        V5InteractionEncoder(16, 1, 0), V5DeformationDecoder(16, 16, 1, 0),
    ).eval()
    with torch.no_grad():
        first = model(**values).position
        poisoned = {**values, "target": torch.randn(1, 7, 16, 3)}
        poisoned.pop("target")
        second = model(**poisoned).position
    assert torch.equal(first, second)


def test_identity_rotation_path_is_exact_identity():
    values = inputs(batch=1)
    model = V5CausalDeformationModel(
        V5COMModel(16, 7, 0, 1, 4), V5RotationModel(16, 7, 0),
        V5InteractionEncoder(16, 1, 0), V5DeformationDecoder(16, 16, 1, 0),
    )
    output = model(**values, use_identity_rotation=True)
    expected = torch.eye(3).expand_as(output.rotation)
    assert torch.equal(output.rotation, expected)


def test_shared_trunk_receives_both_com_and_rotation_gradients():
    values = inputs(batch=1, points=16, frames=7)
    model = V5SharedPhysicalModel(16, 7, 0, 1, 4)
    torch.nn.init.normal_(model.core.v42_com_head[-1].weight, std=1e-3)
    torch.nn.init.normal_(model.core.rotation_head[-1].weight, std=1e-3)
    output = model(**values)
    (output.com.square().mean() + output.rotation[..., 0, 1].square().mean()).backward()
    trunk_grad = sum(
        float(parameter.grad.abs().sum()) for parameter in model.trunk_parameters()
        if parameter.grad is not None
    )
    assert trunk_grad > 0
    assert model.core.v42_com_head[-1].weight.grad.abs().sum() > 0
    assert model.core.rotation_head[-1].weight.grad.abs().sum() > 0


def test_deformation_physical_gradient_scale_can_freeze_or_finetune_trunk():
    decoder = V5DeformationDecoder(
        interaction_dim=8, hidden_dim=16, blocks=1, dropout=0,
        physical_dim=6,
    )
    shape = torch.randn(1, 10, 3)
    mask = torch.ones(1, 10, dtype=torch.bool)
    latent = torch.randn(1, 4, 10, 8)
    physical = torch.randn(1, 4, 10, 6, requires_grad=True)
    frozen = decoder(shape, latent, mask, physical, trunk_gradient_scale=0)
    frozen.square().sum().backward()
    assert physical.grad is None or torch.count_nonzero(physical.grad) == 0
    physical.grad = None
    tuned = decoder(shape, latent, mask, physical, trunk_gradient_scale=.25)
    tuned.square().sum().backward()
    assert physical.grad is not None and physical.grad.abs().sum() > 0
