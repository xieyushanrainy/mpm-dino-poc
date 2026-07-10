from __future__ import annotations

import torch

from mpm_dino_v2.deformation import deformation_descriptors
from mpm_dino_v2.graph import build_mutual_knn_graph
from mpm_dino_v2.losses import one_step_loss
from mpm_dino_v2.model import ParticleGridSurrogate


def test_uniform_scaling_has_expected_stretch() -> None:
    x0 = torch.rand((1, 10, 3), generator=torch.Generator().manual_seed(2))
    mask = torch.ones(10, dtype=torch.bool)
    graph = build_mutual_knn_graph(x0[0], mask, candidate_k=4, max_neighbours=3)
    descriptors = deformation_descriptors(
        x0 * 1.1, mask[None], graph["neighbour_indices"][None], graph["neighbour_mask"][None],
        graph["rest_edge_lengths"][None],
    )
    assert torch.allclose(descriptors[0, :, 0], torch.full((10,), 0.1), atol=1e-5)
    assert torch.allclose(descriptors[0, :, 1], torch.zeros(10), atol=1e-5)


def test_model_and_edge_losses_backpropagate() -> None:
    torch.manual_seed(4)
    b, n, dino_dim = 1, 12, 8
    x0 = torch.rand((n, 3)) - 0.5
    graph = build_mutual_knn_graph(x0, torch.ones(n, dtype=torch.bool), candidate_k=4, max_neighbours=3)
    model = ParticleGridSurrogate(dino_dim=dino_dim, dino_grid_dim=4, base=4, resolution=8)
    positions = x0[None] + 0.01 * torch.randn((b, n, 3))
    mask = torch.ones((b, n), dtype=torch.bool)
    output = model(
        positions, torch.zeros_like(positions), torch.randn((b, n, dino_dim)), mask,
        torch.zeros((b, n), dtype=torch.bool), torch.zeros((b, 2, 3)), torch.zeros((b, 2, 3)),
        torch.ones((b, 2), dtype=torch.bool), torch.ones(b), torch.full((b,), 1 / 30), x0[None],
        graph["neighbour_indices"][None], graph["neighbour_mask"][None], graph["rest_edge_lengths"][None],
    )
    batch = {
        "positions": positions, "target_displacement": torch.zeros_like(positions),
        "target_velocity": torch.zeros_like(positions), "target_mask": mask,
        "neighbour_indices": graph["neighbour_indices"][None], "neighbour_mask": graph["neighbour_mask"][None],
    }
    losses = one_step_loss(output, batch, model.spec)
    losses.total.backward()
    assert torch.isfinite(losses.total)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_two_step_recurrent_gradients_are_finite() -> None:
    torch.manual_seed(8)
    n = 10
    x0 = torch.rand((1, n, 3)) - 0.5
    graph = build_mutual_knn_graph(x0[0], torch.ones(n, dtype=torch.bool), candidate_k=2, max_neighbours=1)
    model = ParticleGridSurrogate(dino_dim=4, dino_grid_dim=4, base=4, resolution=8)
    positions = x0.clone()
    velocity = torch.zeros_like(positions)
    mask = torch.ones((1, n), dtype=torch.bool)
    total = 0.0
    for _ in range(2):
        output = model(
            positions, velocity, torch.randn((1, n, 4)), mask, torch.zeros((1, n), dtype=torch.bool),
            torch.zeros((1, 2, 3)), torch.zeros((1, 2, 3)), torch.ones((1, 2), dtype=torch.bool),
            torch.ones(1), torch.full((1,), 1 / 30), x0, graph["neighbour_indices"][None],
            graph["neighbour_mask"][None], graph["rest_edge_lengths"][None],
        )
        positions = positions + output.displacement
        velocity = output.displacement * 30
        total = total + output.displacement.square().mean()
    total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def _variant_inputs(n: int = 10, controllers: int = 4):
    x0 = torch.rand((1, n, 3), generator=torch.Generator().manual_seed(11)) - 0.5
    graph = build_mutual_knn_graph(x0[0], torch.ones(n, dtype=torch.bool), candidate_k=4, max_neighbours=3)
    return x0, graph, {
        "positions": x0.clone(), "velocities": torch.zeros_like(x0), "dino": torch.randn((1, n, 8)),
        "particle_mask": torch.ones((1, n), dtype=torch.bool),
        "dino_imputed": torch.zeros((1, n), dtype=torch.bool),
        "controller_positions": torch.randn((1, controllers, 3)),
        "controller_velocity": torch.randn((1, controllers, 3)),
        "controller_mask": torch.ones((1, controllers), dtype=torch.bool),
        "scale": torch.ones(1), "dt": torch.full((1,), 1 / 30), "x0": x0,
        "neighbour_indices": graph["neighbour_indices"][None],
        "neighbour_mask": graph["neighbour_mask"][None],
        "rest_edge_lengths": graph["rest_edge_lengths"][None],
    }


def test_variant_structure_and_input_dimensions() -> None:
    fused = ParticleGridSurrogate(dino_dim=8, dino_grid_dim=4, base=4, resolution=8, variant="fused")
    grid = ParticleGridSurrogate(dino_dim=8, dino_grid_dim=4, base=4, resolution=8, variant="grid_only")
    particle = ParticleGridSurrogate(dino_dim=8, dino_grid_dim=4, base=4, resolution=8, variant="particle_only")
    assert fused.particle_input_dim == 30
    assert grid.particle_input_dim == 8
    assert particle.particle_input_dim == 86
    assert hasattr(fused, "unet") and hasattr(grid, "unet")
    assert not hasattr(particle, "unet") and not hasattr(particle, "grid_head")


def test_particle_only_controller_context_is_permutation_invariant_and_masked() -> None:
    torch.manual_seed(12)
    _, _, inputs = _variant_inputs()
    model = ParticleGridSurrogate(dino_dim=8, dino_grid_dim=4, base=4, resolution=8, variant="particle_only").eval()
    original = model(**inputs)
    assert original.next_occupancy is None and original.next_velocity is None
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = dict(inputs)
    for key in ("controller_positions", "controller_velocity", "controller_mask"):
        permuted[key] = inputs[key][:, permutation]
    assert torch.allclose(original.displacement, model(**permuted).displacement, atol=1e-6)
    masked = dict(inputs)
    masked["controller_mask"] = torch.tensor([[True, True, False, False]])
    changed_invalid = dict(masked)
    changed_invalid["controller_positions"] = masked["controller_positions"].clone()
    changed_invalid["controller_velocity"] = masked["controller_velocity"].clone()
    changed_invalid["controller_positions"][:, 2:] += 1000
    changed_invalid["controller_velocity"][:, 2:] -= 1000
    assert torch.allclose(model(**masked).displacement, model(**changed_invalid).displacement, atol=1e-6)


def test_particle_only_responds_to_controller_motion_and_backpropagates() -> None:
    torch.manual_seed(13)
    _, graph, inputs = _variant_inputs()
    model = ParticleGridSurrogate(dino_dim=8, dino_grid_dim=4, base=4, resolution=8, variant="particle_only")
    first = model(**inputs)
    changed = dict(inputs)
    changed["controller_velocity"] = inputs["controller_velocity"] + 2
    second = model(**changed)
    assert not torch.allclose(first.displacement, second.displacement)
    batch = {
        "positions": inputs["positions"], "target_displacement": torch.zeros_like(inputs["positions"]),
        "target_velocity": torch.zeros_like(inputs["positions"]), "target_mask": inputs["particle_mask"],
        "neighbour_indices": graph["neighbour_indices"][None], "neighbour_mask": graph["neighbour_mask"][None],
    }
    losses = one_step_loss(first, batch, model.spec)
    losses.total.backward()
    assert losses.occupancy.item() == losses.grid_velocity.item() == losses.consistency.item() == 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
