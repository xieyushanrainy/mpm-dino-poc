from __future__ import annotations

import torch

from mpm_dino_v2.graph import build_mutual_knn_graph
from mpm_dino_v4.full_data import ballistic_trajectory
from mpm_dino_v4.full_losses import compute_full_trajectory_loss
from mpm_dino_v4.full_model import FullTrajectorySurrogate
from mpm_dino_v4.model import masked_mean


def full_inputs(n=10, frames=7):
    torch.manual_seed(19)
    x0 = torch.rand(1, n, 3)
    x1 = x0 + torch.tensor([[[0.0, 0.0, -0.005]]])
    mask = torch.ones(1, n, dtype=torch.bool)
    graph = build_mutual_knn_graph(x0[0], mask[0], candidate_k=4, max_neighbours=3)
    return {
        "x0": x0,
        "x1": x1,
        "input_mask": mask,
        "dino": torch.randn(1, n, 384),
        "dino_valid": torch.arange(n)[None] % 3 != 0,
        "dt": torch.tensor([1 / 30]),
        "gravity": torch.tensor([[0.0, 0.0, -9.81]]),
        "floor_z": torch.tensor([0.0]),
        **{key: value[None] for key, value in graph.items()},
    }


def test_ballistic_discrete_constant_acceleration():
    x0 = torch.zeros(1, 2, 3)
    dt = torch.tensor([0.1])
    gravity = torch.tensor([[0.0, 0.0, -10.0]])
    x1 = x0 + 0.5 * gravity[:, None] * dt[:, None, None].square()
    result = ballistic_trajectory(x0, x1, gravity, dt, frames=2)
    assert torch.allclose(result[:, 0, :, 2], torch.full((1, 2), -0.2))
    assert torch.allclose(result[:, 1, :, 2], torch.full((1, 2), -0.45))


def test_full_model_shapes_zero_mean_and_backward():
    inputs = full_inputs()
    model = FullTrajectorySurrogate(
        hidden_dim=32, blocks=1, heads=4, dropout=0.0, frames=7,
        gradient_checkpointing=False,
    )
    output = model(**inputs)
    assert output.position.shape == (1, 7, 10, 3)
    expanded_mask = inputs["input_mask"][:, None].expand(-1, 7, -1)
    assert torch.allclose(
        masked_mean(output.residual_local, expanded_mask, dim=2),
        torch.zeros(1, 7, 3),
        atol=1e-7,
    )
    target = output.ballistic.detach().clone()
    target_mask = expanded_mask.clone()
    target_mask[:, 2, 0] = False
    loss = compute_full_trajectory_loss(
        output, {**inputs, "target": target, "target_mask": target_mask}
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_output_does_not_depend_on_training_target():
    inputs = full_inputs(frames=4)
    model = FullTrajectorySurrogate(
        hidden_dim=32, blocks=1, heads=4, dropout=0.0, frames=4,
        gradient_checkpointing=False,
    ).eval()
    with torch.no_grad():
        first = model(**inputs).position
        second = model(**inputs).position
    assert torch.equal(first, second)
