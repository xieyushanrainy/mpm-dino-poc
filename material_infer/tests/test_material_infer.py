import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from material_infer.data import pooled_dino
from material_infer.cross_validate import _balanced_target_folds
from material_infer.metrics import classification_metrics, regression_metrics
from material_infer.model import FeatureTransform, Probe


def test_masked_pooling(tmp_path):
    path = tmp_path / "sample.npz"
    features = np.asarray([[1, 2], [100, 100], [3, 4]], dtype=np.float16)
    np.savez(path, dinov2_reprojected_features=features, dinov2_reprojected_valid=[True, False, True])
    pooled, fraction = pooled_dino(path)
    np.testing.assert_allclose(pooled, [2, 3, 3, 4])
    assert fraction == 2 / 3


def test_train_only_pca_shape():
    x = np.arange(60, dtype=np.float32).reshape(10, 6)
    transform = FeatureTransform.fit(x, 4)
    assert transform.transform(x).shape == (10, 4)


def test_metrics_and_models():
    cls = classification_metrics(np.asarray([0, 1, 2]), np.asarray([0, 1, 2]), ["a", "b", "c"])
    assert cls["macro_f1"] == 1.0
    reg = regression_metrics(np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, 2.0, 3.0]), "log10_E")
    assert reg["mae"] == 0.0 and reg["spearman"] == 1.0
    assert Probe(4, 3, "linear", 8, 0.0).network.in_features == 4


def test_balanced_target_folds():
    class Record:
        family = "soft_body"
        def __init__(self, value): self.log10_e, self.nu = value, value
    records = {f"u{i:02d}": Record(float(i)) for i in range(30)}
    folds = _balanced_target_folds(records, "log10_E", 6)
    assert [len(fold) for fold in folds] == [5] * 6
    assert len(set().union(*map(set, folds))) == 30
