from types import SimpleNamespace

import torch

from mpm_dino_v4.model import masked_mean
from mpm_dino_v4.v43_causal_contact import (
    CausalContactConditionBuilder, rigid_proxy_features,
)

from test_v42 import inputs


class FakeRigidPredictor:
    frames = 7

    def __call__(self, **values):
        centre = masked_mean(values["x1"], values["input_mask"])
        fall = torch.linspace(0, 0.35, self.frames, device=centre.device)
        com = centre[:, None].expand(-1, self.frames, -1).clone()
        com[..., 2] -= fall[None]
        rotation = torch.eye(3, device=centre.device).expand(
            centre.shape[0], self.frames, -1, -1,
        )
        return SimpleNamespace(com=com, rotation=rotation)


def test_rigid_proxy_features_are_causal_finite_and_pointwise():
    batch = inputs(frames=7)
    contact, relative = rigid_proxy_features(batch, FakeRigidPredictor())
    assert contact.shape == (1, 7, 16, 3)
    assert relative.shape == (1, 7)
    assert torch.isfinite(contact).all()
    assert torch.isfinite(relative).all()
    assert contact[..., 1].min() >= 0 and contact[..., 1].max() <= 1
    assert relative.abs().max() <= 1


def test_causal_variants_share_curvature_and_enable_only_declared_channels():
    batch = inputs(frames=7)
    predictor = FakeRigidPredictor()
    static = CausalContactConditionBuilder(
        predictor, "static_control",
    )(batch, None)
    timing = CausalContactConditionBuilder(
        predictor, "causal_timing_only",
    )(batch, None)
    continuous = CausalContactConditionBuilder(
        predictor, "causal_continuous",
    )(batch, None)
    assert static.shape == timing.shape == continuous.shape == (1, 7, 16, 15)
    assert torch.equal(static[..., 3:7], timing[..., 3:7])
    assert torch.equal(static[..., 3:7], continuous[..., 3:7])
    assert torch.count_nonzero(static[..., :3]) == 0
    assert torch.count_nonzero(static[..., 7:]) == 0
    assert torch.count_nonzero(timing[..., :3]) == 0
    assert torch.count_nonzero(timing[..., 7:14]) == 0
    assert torch.count_nonzero(timing[..., 14]) > 0
    assert torch.count_nonzero(continuous[..., :3]) > 0
    assert torch.equal(timing[..., 14], continuous[..., 14])


def test_causal_builder_does_not_read_future_targets_or_stages():
    batch = inputs(frames=7)
    predictor = FakeRigidPredictor()
    builder = CausalContactConditionBuilder(predictor, "causal_continuous")
    first = builder(batch, SimpleNamespace(labels=torch.zeros(1, 8)))
    poisoned = {
        **batch,
        "target": torch.randn(1, 7, 16, 3),
        "target_mask": torch.zeros(1, 7, 16, dtype=torch.bool),
    }
    second = builder(poisoned, SimpleNamespace(labels=torch.full((1, 8), 6)))
    assert torch.equal(first, second)
