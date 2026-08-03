import torch

from mpm_dino_v4.v43_rejection import RejectingMechanicalMemory


def test_rejection_reader_has_normalized_source_and_null_weights():
    module = RejectingMechanicalMemory(12, 10, hidden_dim=16, heads=4)
    base = torch.zeros(1, 2, 4, 3)
    query = torch.randn(1, 2, 4, 12)
    memory = torch.randn(1, 2, 3, 5, 10, requires_grad=True)
    valid = torch.ones(1, 2, 3, 5, dtype=torch.bool)
    points = torch.ones(1, 4, dtype=torch.bool)
    output, gate, weights, null = module(
        base, query, memory, valid, points, return_compatibility=True,
    )
    assert torch.allclose(weights.sum(-1) + null, torch.ones_like(null))
    output.square().mean().backward()
    assert module.compatibility[-1].weight.grad is not None
    assert memory.grad is not None and memory.grad.abs().sum() > 0


def test_rejection_reader_empty_memory_is_exact_base_and_full_null():
    module = RejectingMechanicalMemory(12, 10, hidden_dim=16, heads=4)
    base = torch.randn(1, 2, 4, 3)
    output, gate, weights, null = module(
        base, torch.randn(1, 2, 4, 12), torch.zeros(1, 2, 3, 5, 10),
        torch.zeros(1, 2, 3, 5, dtype=torch.bool),
        torch.ones(1, 4, dtype=torch.bool), return_compatibility=True,
    )
    assert torch.equal(output, base)
    assert not gate.any() and not weights.any() and null.eq(1).all()
