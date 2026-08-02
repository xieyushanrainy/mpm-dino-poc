import numpy as np
import torch

from v42.run_retrieval_transfer_baseline import (
    fit_soft_scale, project_rotation, weights,
)


def test_retrieval_weights_are_normalized_and_prefer_nearest():
    value = weights([0.1, 0.2, 0.4], temperature=0.1)
    assert np.isclose(value.sum(), 1.0)
    assert value[0] > value[1] > value[2]


def test_normalized_soft_scale_recovers_shared_amplitude():
    truth_a = torch.tensor([[2.0, 0.0, 0.0]])
    truth_b = torch.tensor([[0.0, 0.5, 0.0]])
    predictions = [[(truth_a / 4, truth_a)], [(truth_b / 4, truth_b)]]
    assert np.isclose(fit_soft_scale(predictions), 4.0)


def test_rotation_projection_is_proper_orthogonal():
    matrix = np.array([[1.0, 0.1, 0.0], [0.0, 0.9, -0.2], [0.0, 0.2, 1.1]])
    rotation = project_rotation(matrix)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(rotation), 1.0)
