"""Pose error and change-detection scores. Scene-agnostic."""

from __future__ import annotations

import numpy as np


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic angle (degrees) between two rotation matrices."""
    R_est = np.asarray(R_est, dtype=np.float64)
    R_gt = np.asarray(R_gt, dtype=np.float64)
    R_err = R_est @ R_gt.T
    cos_theta = (np.trace(R_err) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def translation_error(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(t_est) - np.asarray(t_gt)))


def transform_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def split_transform(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float64)
    return T[:3, :3].copy(), T[:3, 3].copy()


def pose_errors_from_source_to_target(
    T_est_source_to_target: np.ndarray,
    R_gt_target_to_source: np.ndarray,
    t_gt_target_to_source: np.ndarray,
) -> tuple[float, float]:
    """Compare ICP (T1→T0) with ground-truth pose that maps T0 points into T1.

    Demo convention: p_T1 = R_gt @ p_T0 + t_gt.
    ICP estimates T such that T @ p_T1 ≈ p_T0, i.e. inv(T_gt).
    """
    T_gt = transform_from_Rt(R_gt_target_to_source, t_gt_target_to_source)
    T_gt_inv = invert_transform(T_gt)
    R_est, t_est = split_transform(T_est_source_to_target)
    R_ref, t_ref = split_transform(T_gt_inv)
    return rotation_error_deg(R_est, R_ref), translation_error(t_est, t_ref)


def _as_bool_pair(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    if predicted.shape != ground_truth.shape:
        raise ValueError("predicted and ground_truth must have the same shape")
    return predicted, ground_truth


def confusion_counts(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> tuple[int, int, int, int]:
    predicted, ground_truth = _as_bool_pair(predicted, ground_truth)
    tp = int(np.count_nonzero(predicted & ground_truth))
    fp = int(np.count_nonzero(predicted & ~ground_truth))
    fn = int(np.count_nonzero(~predicted & ground_truth))
    tn = int(np.count_nonzero(~predicted & ~ground_truth))
    return tp, fp, fn, tn


def binary_recall(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    predicted, ground_truth = _as_bool_pair(predicted, ground_truth)
    n_pos = int(np.count_nonzero(ground_truth))
    if n_pos == 0:
        return 1.0
    return float(np.count_nonzero(predicted & ground_truth) / n_pos)


def binary_precision(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    predicted, ground_truth = _as_bool_pair(predicted, ground_truth)
    n_hat = int(np.count_nonzero(predicted))
    if n_hat == 0:
        return 1.0 if int(np.count_nonzero(ground_truth)) == 0 else 0.0
    return float(np.count_nonzero(predicted & ground_truth) / n_hat)


def excavation_recall(changed_mask: np.ndarray, excavation_mask: np.ndarray) -> float:
    return binary_recall(changed_mask, excavation_mask)


def excavation_precision(changed_mask: np.ndarray, excavation_mask: np.ndarray) -> float:
    return binary_precision(changed_mask, excavation_mask)
