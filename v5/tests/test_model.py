import torch

from mpm_dino_v5.model import (
    V5COMModel,
    V5CausalDeformationModel,
    V5DeformationDecoder,
    V5InteractionEncoder,
    V5RotationModel,
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
    return {
        "x0": x0,
        "x1": x1,
        "input_mask": mask,
        "dt": torch.full((batch,), 1 / 30),
        "gravity": torch.tensor([[0., 0., -9.81]]).expand(batch, -1),
        "floor_z": torch.zeros(batch),
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
        V5COMModel(hidden_dim=16, frames=7, dropout=0),
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
        V5COMModel(16, 7, 0), V5RotationModel(16, 7, 0),
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
        V5COMModel(16, 7, 0), V5RotationModel(16, 7, 0),
        V5InteractionEncoder(16, 1, 0), V5DeformationDecoder(16, 16, 1, 0),
    )
    output = model(**values, use_identity_rotation=True)
    expected = torch.eye(3).expand_as(output.rotation)
    assert torch.equal(output.rotation, expected)

