from __future__ import annotations

import torch

from mpm_dino_v3.action import action_particle_features, initial_action_summary
from mpm_dino_v2.graph import build_mutual_knn_graph
from mpm_dino_v2.losses import one_step_loss
from mpm_dino_v3.model import V3ParticleSurrogate


def _inputs(n: int = 14, controllers: int = 3, dino_dim: int = 8):
    torch.manual_seed(21)
    x0 = torch.rand((1, n, 3)) - 0.5
    graph = build_mutual_knn_graph(x0[0], torch.ones(n, dtype=torch.bool), candidate_k=4, max_neighbours=3)
    positions = x0 + 0.01 * torch.randn_like(x0)
    return graph, {
        "positions": positions,
        "velocities": torch.zeros_like(positions),
        "dino": torch.randn((1, n, dino_dim)),
        "particle_mask": torch.ones((1, n), dtype=torch.bool),
        "dino_imputed": torch.zeros((1, n), dtype=torch.bool),
        "controller_positions": torch.randn((1, controllers, 3)),
        "controller_velocity": torch.randn((1, controllers, 3)),
        "controller_mask": torch.ones((1, controllers), dtype=torch.bool),
        "scale": torch.ones(1),
        "dt": torch.full((1,), 1 / 30),
        "x0": x0,
        "neighbour_indices": graph["neighbour_indices"][None],
        "neighbour_mask": graph["neighbour_mask"][None],
        "rest_edge_vectors": graph["rest_edge_vectors"][None],
        "rest_edge_lengths": graph["rest_edge_lengths"][None],
    }


def test_initial_action_summary_uses_nearest_valid_controller() -> None:
    positions = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    particle_mask = torch.tensor([[True, True]])
    controllers = torch.tensor([[[10.0, 0.0, 0.0], [0.2, 0.0, 0.0]]])
    velocity = torch.tensor([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    controller_mask = torch.tensor([[True, True]])
    action, contact = initial_action_summary(positions, particle_mask, controllers, velocity, controller_mask)
    assert torch.allclose(action, torch.tensor([[2.0, 0.0, 0.0]]))
    assert torch.allclose(contact, controllers[:, 1])
    features = action_particle_features(positions, action, contact)
    assert features.shape == (1, 2, 9)


def test_v3_variants_produce_particle_only_outputs_and_backpropagate() -> None:
    for variant in ("graph_direct", "latent_graph", "action_token_graph"):
        graph, inputs = _inputs()
        model = V3ParticleSurrogate(
            dino_dim=8, dino_embed_dim=4, hidden_dim=32, latent_dim=8,
            layers=2, variant=variant, attention_heads=2, resolution=8,
        )
        output = model(**inputs)
        assert output.displacement.shape == inputs["positions"].shape
        assert output.next_occupancy is None and output.next_velocity is None
        batch = {
            "positions": inputs["positions"],
            "target_displacement": torch.zeros_like(inputs["positions"]),
            "target_velocity": torch.zeros_like(inputs["positions"]),
            "target_mask": inputs["particle_mask"],
            "neighbour_indices": graph["neighbour_indices"][None],
            "neighbour_mask": graph["neighbour_mask"][None],
        }
        losses = one_step_loss(output, batch, model.spec)
        losses.total.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_v3_recurrent_gradients_are_finite() -> None:
    graph, inputs = _inputs(n=10)
    model = V3ParticleSurrogate(
        dino_dim=8, dino_embed_dim=4, hidden_dim=32, latent_dim=8,
        layers=2, variant="latent_graph", attention_heads=2, resolution=8,
    )
    positions = inputs["positions"]
    velocities = inputs["velocities"]
    total = 0.0
    for _ in range(2):
        step_inputs = dict(inputs)
        step_inputs["positions"] = positions
        step_inputs["velocities"] = velocities
        output = model(**step_inputs)
        positions = positions + output.displacement
        velocities = output.displacement / inputs["dt"][:, None, None]
        total = total + output.displacement.square().mean()
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_latent_graph_has_no_unused_direct_dino_branch() -> None:
    model = V3ParticleSurrogate(dino_dim=8, dino_embed_dim=4, hidden_dim=32, latent_dim=8, variant="latent_graph")
    assert not hasattr(model, "dino")
