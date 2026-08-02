import torch
from torch import nn

from mpm_dino_v4.v43_retrieval import (
    AttendedMechanicalMemory, MemoryEntry, RetrievalBank, align_points,
    materialize_aligned, parameter_snapshot, parameters_identical, retrieve,
)


def entry(uid, offset=0.0, split="train", stage=2):
    coordinates = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]) + offset
    return MemoryEntry(
        uid, split, stage, 0.5, coordinates,
        torch.tensor([[1., 0.], [0., 1.], [1., 1.]]) + offset,
        torch.ones(3, 3) * (offset + 1),
        torch.tensor([True, True, False]), torch.tensor([True, True, True]),
        torch.ones(3, 3), 1.0, f"train/{uid}/canonical_target.npz",
    )


def query():
    return (torch.tensor([[0., 0., 0.], [1.1, 0., 0.], [9., 0., 0.]]),
            torch.tensor([[1., 0.], [0., 1.], [1., 1.]]),
            torch.tensor([True, True, False]), torch.tensor([True, True, True]))


def test_bank_rejects_split_leakage_and_hashes_content():
    try:
        RetrievalBank([entry("bad", split="validation")])
        assert False, "validation memory accepted"
    except ValueError:
        pass
    assert RetrievalBank([entry("a")]).content_sha256 == RetrievalBank([entry("a")]).content_sha256


def test_same_uid_exclusion_and_deterministic_topk_tie_break():
    bank = RetrievalBank([entry("query"), entry("b"), entry("a"), entry("c")])
    xyz, dino, valid, dvalid = query()
    first = retrieve(bank, "query", "train", xyz, dino, valid, dvalid,
                     stage=2, mode="geometry", k=2)
    second = retrieve(bank, "query", "train", xyz, dino, valid, dvalid,
                      stage=2, mode="geometry", k=2)
    assert [e.uid for e in first] == [e.uid for e in second]
    assert "query" not in [e.uid for e in first]


def test_point_alignment_respects_query_and_source_masks():
    xyz, _, valid, _ = query()
    mapping, aligned_valid = align_points(xyz, valid, entry("a"))
    assert mapping.tolist()[:2] == [0, 1]
    assert aligned_valid.tolist() == [True, True, False]


def test_shuffle_controls_are_deterministic_and_change_correspondence():
    xyz, _, valid, _ = query()
    memories = [entry("a"), entry("b", .2), entry("c", .4)]
    real, mask = materialize_aligned(xyz, valid, memories, "aligned_dino", 7)
    shuffled1, mask1 = materialize_aligned(xyz, valid, memories, "point_shuffled", 7)
    shuffled2, mask2 = materialize_aligned(xyz, valid, memories, "point_shuffled", 7)
    assert torch.equal(shuffled1, shuffled2) and torch.equal(mask1, mask2)
    assert torch.equal(mask, mask1)
    assert not torch.equal(real[..., 3:5], shuffled1[..., 3:5])


def test_zero_memory_is_exact_base_equivalence():
    module = AttendedMechanicalMemory(8, 10, hidden_dim=16, heads=4)
    base = torch.randn(1, 2, 3, 3)
    output = module(base, torch.randn(1, 2, 3, 8),
                    torch.zeros(1, 2, 3, 3, 10),
                    torch.zeros(1, 2, 3, 3, dtype=torch.bool),
                    torch.tensor([[True, True, False]]))
    assert torch.equal(output, base)


def test_attention_gradients_and_fixed_weight_memory_ablation():
    torch.manual_seed(3)
    module = AttendedMechanicalMemory(8, 10, hidden_dim=16, heads=4)
    base = torch.zeros(1, 2, 3, 3)
    query_tokens = torch.randn(1, 2, 3, 8, requires_grad=True)
    memory = torch.randn(1, 2, 2, 3, 10, requires_grad=True)
    memory_valid = torch.ones(1, 2, 2, 3, dtype=torch.bool)
    points = torch.ones(1, 3, dtype=torch.bool)
    real = module(base, query_tokens, memory, memory_valid, points)
    zero = module(base, query_tokens, torch.zeros_like(memory),
                  torch.zeros_like(memory_valid), points)
    assert not torch.equal(real, zero)
    real.square().sum().backward()
    assert memory.grad is not None and memory.grad.abs().sum() > 0
    assert module.attention.in_proj_weight.grad.abs().sum() > 0


def test_protected_parameter_identity_after_local_update():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.physical_trunk = nn.Linear(2, 2)
            self.com_head = nn.Linear(2, 2)
            self.rotation_branch = nn.Linear(2, 2)
            self.retrieval = nn.Linear(2, 2)
    model = Model()
    prefixes = ("physical_trunk.", "com_head.", "rotation_branch.")
    snapshot = parameter_snapshot(model, prefixes)
    optimizer = torch.optim.SGD(model.retrieval.parameters(), lr=.1)
    model.retrieval(torch.ones(1, 2)).sum().backward(); optimizer.step()
    assert parameters_identical(model, snapshot)
