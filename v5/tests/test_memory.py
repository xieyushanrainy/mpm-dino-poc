import pytest
import torch

from mpm_dino_v5.memory import (
    CompactMemoryResidual,
    CompactSourceEncoder,
    SourcePhase,
    V5MemoryBank,
    V5MemoryModule,
    interpolate_phase_tokens,
)


UIDS = [f"uid{i:02d}" for i in range(20)]


def source(uid, phase, descriptor=None, split="train"):
    n = 8
    if descriptor is None:
        descriptor = torch.tensor([1., 0., 0., 0.])
    return SourcePhase(
        uid=uid,
        split=split,
        phase=phase,
        coordinates=torch.randn(n, 3),
        deformation=torch.randn(n, 3) * .01,
        contact=torch.linspace(0, 1, n),
        point_valid=torch.ones(n, dtype=torch.bool),
        dino=descriptor[None].expand(n, -1).clone(),
        dino_valid=torch.ones(n, dtype=torch.bool),
    )


def bank():
    entries = []
    for index, uid in enumerate(UIDS):
        descriptor = torch.tensor([1., float(index) / 20, 0., 0.])
        entries.extend(source(uid, phase, descriptor) for phase in (2, 3, 4))
    return V5MemoryBank(entries, UIDS)


def test_bank_scope_and_leave_one_uid_out_top3(tmp_path):
    memory = bank()
    selected = memory.retrieve_uids(
        "uid00", torch.tensor([[1., 0., 0., 0.]]), torch.tensor([True]), "train",
    )
    assert len(selected) == 3 and "uid00" not in selected
    path = tmp_path / "bank.pt"
    memory.save(path)
    loaded = V5MemoryBank.load(path, UIDS)
    assert loaded.content_sha256 == memory.content_sha256
    json_path = path.with_suffix(".pt.manifest.json")
    assert json_path.exists()
    bad = [source(uid, phase) for uid in UIDS for phase in (2, 3, 4)]
    bad[0] = source(UIDS[0], 2, split="validation")
    with pytest.raises(ValueError, match="train"):
        V5MemoryBank(bad, UIDS)


def test_compact_encoder_returns_32_tokens_without_dino():
    encoder = CompactSourceEncoder(hidden_dim=16, heads=4)
    entry = source("uid00", 2)
    tokens = encoder(
        entry.coordinates[None], entry.deformation[None], entry.contact[None],
        entry.point_valid[None],
    )
    assert tokens.shape == (1, 32, 16)
    empty = encoder(
        entry.coordinates[None], entry.deformation[None], entry.contact[None],
        torch.zeros_like(entry.point_valid)[None],
    )
    assert torch.isfinite(empty).all()


def test_phase_interpolation_is_linear_and_clamped():
    tokens = torch.tensor([0., 1., 2.]).reshape(1, 1, 3, 1, 1)
    time = torch.tensor([[-2., -0.5, 0., 0.5, 2.]])
    result = interpolate_phase_tokens(tokens, time).flatten()
    assert torch.allclose(result, torch.tensor([0., .5, 1., 1.5, 2.]))


def test_zero_memory_is_exact_base_and_residual_is_bounded_zero_mean():
    module = CompactMemoryResidual(12, hidden_dim=16, heads=4, residual_bound=.05)
    base = torch.zeros(2, 3, 6, 3)
    query = torch.randn(2, 3, 6, 12)
    mask = torch.ones(2, 6, dtype=torch.bool)
    assert torch.equal(module(base, query, None, mask), base)
    memory = torch.randn(2, 3, 3, 32, 16)
    output, gate = module(base, query, memory, mask, return_gate=True)
    assert 0 < gate < .1
    assert output.abs().max() <= .05 + 1e-6
    assert torch.allclose(output.mean(2), torch.zeros_like(output.mean(2)), atol=1e-7)


def test_end_to_end_memory_module_encodes_top3_and_interpolates():
    memory_bank = bank()
    module = V5MemoryModule(12, hidden_dim=16, heads=4)
    base = torch.zeros(1, 3, 6, 3)
    query = torch.randn(1, 3, 6, 12)
    event_time = torch.tensor([[-1., 0., 1.]])
    mask = torch.ones(1, 6, dtype=torch.bool)
    output = module(
        base, query, event_time, mask, memory_bank,
        [["uid00", "uid01", "uid02"]],
    )
    assert output.shape == base.shape
    assert torch.isfinite(output).all()
