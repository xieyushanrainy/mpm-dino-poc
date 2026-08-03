import torch

from mpm_dino_v4.v43_field_attention import (
    FIELD_MEMORY_DIM, FIELD_QUERY_DIM, FieldBank, FieldEntry,
    PointwiseFieldAttention, materialize_fields,
)


def source(uid="a"):
    dino = torch.zeros(3, 384)
    dino[0, 0] = 1.; dino[1, 1] = 1.; dino[2, :2] = 1.
    return FieldEntry(
        uid=uid, split="train",
        coordinates=torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
        dino=dino,
        point_valid=torch.ones(3, dtype=torch.bool),
        dino_valid=torch.ones(3, dtype=torch.bool),
        displacement=torch.stack((torch.ones(3, 3), torch.ones(3, 3) * 2)),
        contact=torch.zeros(2, 3, 3), stages=torch.tensor([2, 2]),
        event_time=torch.tensor([0., 1.]), provenance="train:a",
    )


def test_field_bank_rejects_nontraining_entry():
    bad = source()
    object.__setattr__(bad, "split", "validation")
    try:
        FieldBank([bad]); assert False
    except ValueError:
        pass


def test_field_materialization_uses_nearest_event_time_and_zero_field_ablation():
    qxyz = source().coordinates
    tokens, valid = materialize_fields(
        qxyz, source().dino, torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool), torch.tensor([2]), torch.tensor([.9]),
        [source()],
    )
    # deformation occupies the three channels before distance/confidence.
    assert torch.allclose(tokens[0, :, 0, -5:-2], torch.full((3, 3), 2.))
    zero, _ = materialize_fields(
        qxyz, source().dino, torch.ones(3, dtype=torch.bool),
        torch.ones(3, dtype=torch.bool), torch.tensor([2]), torch.tensor([.9]),
        [source()], zero_field=True,
    )
    assert torch.equal(zero[0, :, 0, -5:-2], torch.zeros(3, 3))
    assert valid.all()


def test_pointwise_field_attention_has_memory_gradients_and_zero_equivalence():
    module = PointwiseFieldAttention(hidden=16)
    base = torch.randn(1, 2, 3, 3)
    query = torch.randn(1, 2, 3, FIELD_QUERY_DIM)
    memory = torch.randn(1, 2, 3, 2, FIELD_MEMORY_DIM, requires_grad=True)
    valid = torch.ones(1, 2, 3, 2, dtype=torch.bool)
    points = torch.ones(1, 3, dtype=torch.bool)
    output = module(base, query, memory, valid, points)
    output.square().mean().backward()
    assert memory.grad is not None and memory.grad.abs().sum() > 0
    zero = module(base, query, torch.zeros_like(memory),
                  torch.zeros_like(valid), points)
    assert torch.equal(zero, base)
