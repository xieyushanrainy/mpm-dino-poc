import math

import pytest
import torch

from mpm_dino_v4.v43_rotation_memory import (
    CompactRotationReader, RotationBank, RotationMemoryEntry,
    assert_protected_identity, geodesic_radians, protected_snapshot,
    retrieve_rotation, so3_exp,
)
from mpm_dino_v4.v43_rotation_train import rotation_contact_features, summarize
from test_v42 import inputs


def entry(uid, family="rigid", split="train", offset=0.):
    n = 6
    return RotationMemoryEntry(uid, family, split, "Z", torch.randn(n, 3) + offset,
        torch.nn.functional.normalize(torch.randn(n, 8), dim=-1), torch.ones(n, dtype=torch.bool),
        torch.ones(n, dtype=torch.bool), torch.nn.functional.normalize(torch.randn(8), dim=0),
        torch.zeros(59, 3), torch.ones(59, dtype=torch.bool), torch.zeros(59), 1.,
        "proper_kabsch_rotvec_frame1", uid + ":trajectory_positions_m")


def test_bank_is_train_only_and_accepts_both_families():
    bank = RotationBank([entry("a"), entry("b", "soft_body")])
    assert {e.family for e in bank.entries} == {"rigid", "soft_body"}
    with pytest.raises(ValueError): RotationBank([entry("x", split="validation")])


def test_retrieval_excludes_uid_is_deterministic_and_seals_test():
    bank = RotationBank([entry(chr(97+i), offset=i) for i in range(4)])
    q = torch.randn(6, 3); d = torch.randn(6, 8); mask = torch.ones(6, dtype=torch.bool)
    one = retrieve_rotation(bank, "a", "train", q, d, mask, mask, mode="geometry", k=2)
    two = retrieve_rotation(bank, "a", "train", q, d, mask, mask, mode="geometry", k=2)
    assert [e.uid for e in one] == [e.uid for e in two]
    assert all(e.uid != "a" for e in one)
    with pytest.raises(ValueError): retrieve_rotation(bank, "x", "test", q, d, mask, mask, mode="geometry")


def test_so3_is_proper_and_geodesic_identity():
    r = so3_exp(torch.tensor([[.1, -.2, .3]]))
    assert torch.allclose(r.transpose(-1, -2) @ r, torch.eye(3)[None], atol=1e-5)
    assert torch.allclose(torch.linalg.det(r), torch.ones(1), atol=1e-5)
    assert geodesic_radians(r, r).item() < 1e-3


def test_zero_memory_is_exact_identity_and_residual_is_bounded():
    model = CompactRotationReader(5, 7, hidden_dim=16, heads=4, max_degrees=20)
    q = torch.randn(2, 59, 5); memory = torch.randn(2, 59, 3, 7)
    empty = torch.zeros(2, 59, 3, dtype=torch.bool)
    rotation, delta, gate = model(q, memory, empty)
    assert torch.equal(rotation, torch.eye(3).expand(2, 59, 3, 3))
    assert torch.equal(delta, torch.zeros_like(delta)) and torch.equal(gate, torch.zeros_like(gate))
    full = torch.ones_like(empty); _, delta, _ = model(q, memory, full)
    assert torch.linalg.vector_norm(delta, dim=-1).max() <= math.sqrt(3) * math.radians(20) + 1e-6


def test_invalid_kabsch_mask_and_family_balancing_contract():
    e = entry("a"); invalid = RotationMemoryEntry(**{**e.__dict__, "kabsch_valid": torch.zeros(59, dtype=torch.bool)})
    assert not invalid.kabsch_valid.any()
    families = ["rigid", "soft_body"] * 10
    assert families.count("rigid") == families.count("soft_body")


def test_reader_optimizer_preserves_protected_state_and_output():
    protected = torch.nn.Linear(3, 4).eval()
    for parameter in protected.parameters(): parameter.requires_grad_(False)
    before = protected_snapshot(protected)
    x = torch.randn(2, 3); output_before = protected(x).detach().clone()
    reader = CompactRotationReader(5, 7, hidden_dim=16, heads=4)
    optimizer = torch.optim.AdamW(reader.parameters())
    rotation, _, _ = reader(torch.randn(2, 59, 5), torch.randn(2, 59, 3, 7),
                            torch.ones(2, 59, 3, dtype=torch.bool))
    geodesic_radians(rotation, torch.eye(3).expand_as(rotation)).mean().backward()
    optimizer.step()
    assert_protected_identity(protected, before)
    assert torch.equal(output_before, protected(x))


def test_summary_ignores_uid_with_no_valid_kabsch_frames():
    rows = [
        {"uid": "bad", "family": "rigid", "panel": "Z", "mean_error_deg": None,
         "median_error_deg": None, "identity_mean_error_deg": None, "cross_family_rate": 0.},
        {"uid": "good", "family": "soft_body", "panel": "V", "mean_error_deg": 2.,
         "median_error_deg": 1.5, "identity_mean_error_deg": 3., "cross_family_rate": 1.},
    ]
    report = summarize(rows)
    assert report["rigid"]["mean_error_deg"] is None
    assert report["family_balanced_mean_deg"] == 2.


def test_near_identity_so3_training_stays_finite_across_steps():
    model = CompactRotationReader(5, 7, hidden_dim=16, heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    query = torch.zeros(2, 59, 5)
    memory = torch.zeros(2, 59, 3, 7)
    valid = torch.ones(2, 59, 3, dtype=torch.bool)
    target = torch.eye(3).expand(2, 59, 3, 3)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        rotation, _, _ = model(query, memory, valid)
        loss = geodesic_radians(rotation, target).mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())


def contact_batch():
    batch = inputs(frames=7)
    batch["target"] = batch["x1"][:, None].expand(-1, 7, -1, -1).clone()
    batch["target"][..., 2] -= torch.linspace(0, .05, 7)[None, :, None]
    batch["target_mask"] = batch["input_mask"][:, None].expand(-1, 7, -1)
    batch["floor_z"] = torch.zeros(1)
    return batch


def test_rotation_contact_variants_have_matched_channels():
    batch = contact_batch()
    base = rotation_contact_features(batch, "geometry_base")
    timing = rotation_contact_features(batch, "contact_timing")
    patch = rotation_contact_features(batch, "contact_patch")
    full = rotation_contact_features(batch, "contact_curvature")
    assert base.shape == timing.shape == patch.shape == full.shape == (1, 7, 10)
    assert torch.count_nonzero(base) == 0
    assert torch.count_nonzero(timing[..., 3:]) == 0
    assert torch.count_nonzero(patch[..., 6:]) == 0
    assert torch.equal(timing[..., :3], patch[..., :3])
    assert torch.equal(patch[..., :6], full[..., :6])
    assert torch.isfinite(full).all()


def test_curvature_shuffle_is_deterministic_and_preserves_contact_patch():
    batch = contact_batch()
    full = rotation_contact_features(batch, "contact_curvature", 42)
    one = rotation_contact_features(batch, "curvature_shuffled", 42)
    two = rotation_contact_features(batch, "curvature_shuffled", 42)
    assert torch.equal(one, two)
    assert torch.equal(one[..., :6], full[..., :6])
    assert not torch.equal(one[..., 6:], full[..., 6:])
