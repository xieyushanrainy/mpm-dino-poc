import torch

from mpm_dino.grid import GridSpec, gather_grid, scatter_particles
from mpm_dino.model import ParticleGridSurrogate
from mpm_dino.data import SceneSequenceDataset
from mpm_dino.losses import one_step_loss


def test_scatter_gather_constant_feature():
    torch.manual_seed(0)
    p = torch.rand(2, 20, 3) * 1.8 - 0.9
    f = torch.ones(2, 20, 3) * torch.tensor([1.0, 2.0, 3.0])
    mask = torch.ones(2, 20, dtype=torch.bool)
    grid, occ = scatter_particles(p, f, mask, GridSpec(8))
    sampled = gather_grid(grid, p, GridSpec(8))
    assert grid.shape == (2, 3, 8, 8, 8)
    assert occ.sum().allclose(torch.tensor(40.0))
    assert torch.allclose(sampled, f, atol=1e-5)


def test_model_shapes_and_gradient():
    torch.manual_seed(0)
    b, n, nc = 1, 32, 5
    model = ParticleGridSurrogate(dino_dim=12, dino_grid_dim=4, base=8, resolution=16)
    p = torch.rand(b, n, 3) * 1.6 - 0.8
    v = torch.randn(b, n, 3) * 0.01
    d = torch.randn(b, n, 12)
    mask = torch.ones(b, n, dtype=torch.bool)
    cp = torch.rand(b, nc, 3) * 1.6 - 0.8
    cv = torch.randn(b, nc, 3) * 0.01
    out = model(p, v, d, mask, torch.zeros_like(mask), cp, cv, torch.ones(b, nc, dtype=torch.bool), torch.ones(b), torch.ones(b) / 30)
    assert out.displacement.shape == (b, n, 3)
    assert out.next_occupancy.shape == (b, 1, 16, 16, 16)
    assert out.next_velocity.shape == (b, 3, 16, 16, 16)
    out.displacement.square().mean().backward()
    assert model.dino[1].weight.grad is not None


def test_one_step_loss_is_finite():
    torch.manual_seed(1)
    b, n, nc = 1, 24, 4
    model = ParticleGridSurrogate(dino_dim=6, dino_grid_dim=4, base=8, resolution=8)
    batch = {
        "positions": torch.rand(b, n, 3) - 0.5,
        "velocities": torch.randn(b, n, 3) * 0.01,
        "dino": torch.randn(b, n, 6),
        "particle_mask": torch.ones(b, n, dtype=torch.bool),
        "target_mask": torch.ones(b, n, dtype=torch.bool),
        "dino_imputed": torch.zeros(b, n, dtype=torch.bool),
        "controller_positions": torch.rand(b, nc, 3) - 0.5,
        "controller_velocity": torch.randn(b, nc, 3) * 0.01,
        "controller_mask": torch.ones(b, nc, dtype=torch.bool),
        "scale": torch.ones(b), "dt": torch.ones(b) / 30,
        "target_displacement": torch.randn(b, n, 3) * 0.01,
    }
    batch["target_velocity"] = batch["target_displacement"] / batch["dt"][:, None, None]
    out = model(**{k: batch[k] for k in ["positions", "velocities", "dino", "particle_mask", "dino_imputed", "controller_positions", "controller_velocity", "controller_mask", "scale", "dt"]})
    losses = one_step_loss(out, batch, model.spec)
    assert torch.isfinite(losses.total)


def test_particle_smooth_l1_has_expected_normalization():
    # For coordinate errors below beta, Smooth-L1(beta=.01) is ordinary
    # quadratic Huber divided by .01, i.e. 100x its numerical scale.
    from mpm_dino.losses import _masked_huber, _masked_smooth_l1
    pred = torch.full((1, 2, 3), 0.005)
    target = torch.zeros_like(pred)
    mask = torch.ones(1, 2, dtype=torch.bool)
    ordinary = _masked_huber(pred, target, mask)
    normalized = _masked_smooth_l1(pred, target, mask, beta=0.01)
    assert torch.allclose(normalized, ordinary * 100, rtol=1e-5)


def test_sequence_dataset_contract(tmp_path):
    frames, particles, controller = 6, 5, 2
    scene = {
        "points": torch.randn(frames, particles, 3),
        "visible": torch.ones(frames, particles, dtype=torch.bool),
        "motion_valid": torch.ones(frames, particles, dtype=torch.bool),
        "controller": torch.randn(frames, controller, 3),
        "dino": torch.randn(particles, 7),
        "dino_imputed": torch.zeros(particles, dtype=torch.bool),
        "scale": torch.tensor(0.2), "dt": torch.tensor(1 / 30),
    }
    path = tmp_path / "scene.pt"; torch.save(scene, path)
    dataset = SceneSequenceDataset([path], steps=2)
    assert len(dataset) == 3
    item = dataset[0]
    assert item["points"].shape == (3, particles, 3)
    assert item["previous_points"].shape == (particles, 3)
    assert len(dataset.motion_scores) == len(dataset)
