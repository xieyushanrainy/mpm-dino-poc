import copy

import pytest
import torch
from torch import nn

from mpm_dino_v5.config import V5Config
from mpm_dino_v5.losses import event_normalized_deformation_mse
from mpm_dino_v5.provenance import validate_inference_keys, validate_split_manifest
from mpm_dino_v5.staging import (
    assert_unchanged,
    choose_rotation,
    deformation_promotion,
    freeze,
    snapshot,
)


def test_default_config_and_checkpoint_rejection():
    config = V5Config()
    config.validate()
    with pytest.raises(ValueError, match="random weights"):
        config.reject_checkpoint("v42/best.pt")


def test_inference_key_contract_rejects_labels_and_family():
    validate_inference_keys({"x0", "x1", "input_mask", "reference", "dt", "gravity", "floor_z"})
    with pytest.raises(ValueError, match="forbidden"):
        validate_inference_keys({"x0", "target"})
    with pytest.raises(ValueError, match="forbidden"):
        validate_inference_keys({"x0", "family"})


def test_split_validation_checks_hash_counts_and_disjointness():
    groups = {
        "train": {"uids": [f"a{i}" for i in range(40)]},
        "validation": {"uids": [f"b{i}" for i in range(10)]},
        "test": {"uids": [f"c{i}" for i in range(10)]},
    }
    manifest = {"manifest_content_sha256": "ok", "splits": groups}
    validate_split_manifest(manifest, "ok")
    leaked = copy.deepcopy(manifest)
    leaked["splits"]["test"]["uids"][0] = "a0"
    with pytest.raises(ValueError, match="leakage"):
        validate_split_manifest(leaked, "ok")


def test_zero_prediction_is_one_under_event_normalization():
    target = torch.randn(2, 3, 5, 3)
    predicted = torch.zeros_like(target)
    mask = torch.ones(2, 3, 5, dtype=torch.bool)
    stages = torch.tensor([[2, 3, 4], [2, 3, 4]])
    value = event_normalized_deformation_mse(
        predicted, target, mask, torch.ones(2, 3, dtype=torch.bool), stages,
        torch.ones(2),
    )
    assert torch.allclose(value, torch.tensor(1.0), atol=1e-6)


def test_promotion_thresholds_and_rotation_fallback_are_global():
    base = deformation_promotion({42: .8, 123: .9, 456: .9})
    assert base.qualified and not base.success and base.use_memory
    success = deformation_promotion({42: .6, 123: .7, 456: .8})
    assert success.success and not success.use_memory
    assert choose_rotation(
        {42: 2., 123: 3., 456: 2.}, {42: 2.5, 123: 2.5, 456: 2.5},
    ) == "learned"
    assert choose_rotation(
        {42: 2., 123: 4., 456: 2.}, {42: 2.5, 123: 2.5, 456: 2.5},
    ) == "identity"


def test_frozen_stage_hash_detects_mutation():
    module = nn.Linear(3, 3)
    freeze(module)
    expected = snapshot((("com", module),))
    assert not any(parameter.requires_grad for parameter in module.parameters())
    assert_unchanged((("com", module),), expected)
    with torch.no_grad():
        module.weight.add_(1)
    with pytest.raises(RuntimeError, match="com"):
        assert_unchanged((("com", module),), expected)

