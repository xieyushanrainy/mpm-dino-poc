import math

import pytest
import torch

from mpm_dino_v4.v43_rotation_memory import (
    CompactRotationReader, RotationBank, RotationMemoryEntry,
    assert_protected_identity, geodesic_radians, protected_snapshot,
    retrieve_rotation, so3_exp,
)


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
    assert geodesic_radians(r, r).item() < 1e-4


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
