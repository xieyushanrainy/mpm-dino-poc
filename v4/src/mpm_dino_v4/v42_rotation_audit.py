from __future__ import annotations

import math

import numpy as np


def proper_kabsch(
    source: np.ndarray,
    destination: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the proper row-vector rotation mapping source to destination."""
    source = source[valid]
    destination = destination[valid]
    source = source - source.mean(axis=0, keepdims=True)
    destination = destination - destination.mean(axis=0, keepdims=True)
    # einsum avoids spurious Accelerate/BLAS floating-point warnings observed
    # for small, fully finite covariance products on macOS.
    covariance = np.einsum("ni,nj->ij", source, destination)
    u, singular, vh = np.linalg.svd(covariance)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    ratio = float(singular[1] / max(float(singular[0]), 1e-12))
    return rotation, singular, ratio


def rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """SO(3) logarithm as an axis-angle vector, robust near zero and pi."""
    angle = rotation_angle(rotation)
    if angle < 1e-8:
        return 0.5 * np.array([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])
    if math.pi - angle < 1e-5:
        diagonal = np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        if axis[largest] > 1e-8:
            if largest == 0:
                axis[1] = np.copysign(axis[1], rotation[0, 1] + rotation[1, 0])
                axis[2] = np.copysign(axis[2], rotation[0, 2] + rotation[2, 0])
            elif largest == 1:
                axis[0] = np.copysign(axis[0], rotation[0, 1] + rotation[1, 0])
                axis[2] = np.copysign(axis[2], rotation[1, 2] + rotation[2, 1])
            else:
                axis[0] = np.copysign(axis[0], rotation[0, 2] + rotation[2, 0])
                axis[1] = np.copysign(axis[1], rotation[1, 2] + rotation[2, 1])
        norm = np.linalg.norm(axis)
        return angle * axis / max(float(norm), 1e-12)
    axis = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]) / (2.0 * math.sin(angle))
    return angle * axis


def rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    cross = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return (
        np.eye(3) + math.sin(angle) * cross
        + (1.0 - math.cos(angle)) * (cross @ cross)
    )


def constant_angular_rotation(observed_step: np.ndarray, horizon: int) -> np.ndarray:
    """Extrapolate the observed x0->x1 rotation for `horizon` more steps."""
    return rotation_from_vector(rotation_vector(observed_step) * horizon)


def geodesic_error(predicted: np.ndarray, target: np.ndarray) -> float:
    return rotation_angle(predicted.T @ target)


def angular_increment(previous: np.ndarray, current: np.ndarray, dt: float) -> np.ndarray:
    return rotation_vector(previous.T @ current) / dt
