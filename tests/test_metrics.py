from __future__ import annotations

import numpy as np

from pcchange.metrics import (
    confusion_counts,
    excavation_precision,
    excavation_recall,
    invert_transform,
    pose_errors_from_source_to_target,
    rotation_error_deg,
    transform_from_Rt,
    translation_error,
)


def test_rotation_error_zero_and_known_angle() -> None:
    R = np.eye(3)
    assert rotation_error_deg(R, R) == 0.0
    yaw = np.deg2rad(5.0)
    R5 = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(rotation_error_deg(R5, np.eye(3)), 5.0, atol=1e-10)


def test_translation_error() -> None:
    assert translation_error([0.4, -0.2, 0.1], [0.4, -0.2, 0.1]) == 0.0
    np.testing.assert_allclose(translation_error([1.0, 0.0, 0.0], [0.0, 0.0, 0.0]), 1.0)


def test_pose_errors_identity_recovery() -> None:
    yaw = np.deg2rad(5.0)
    R_gt = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    t_gt = np.array([0.4, -0.2, 0.1])
    T_icp = invert_transform(transform_from_Rt(R_gt, t_gt))
    rot_err, trans_err = pose_errors_from_source_to_target(T_icp, R_gt, t_gt)
    assert rot_err < 1e-10
    assert trans_err < 1e-12


def test_excavation_recall() -> None:
    gt = np.array([True, True, True, False, False])
    pred = np.array([True, True, False, True, False])
    np.testing.assert_allclose(excavation_recall(pred, gt), 2.0 / 3.0)
    assert excavation_recall(gt, np.zeros_like(gt)) == 1.0


def test_excavation_precision_and_confusion() -> None:
    gt = np.array([True, True, True, False, False])
    pred = np.array([True, True, False, True, False])
    np.testing.assert_allclose(excavation_precision(pred, gt), 2.0 / 3.0)
    assert confusion_counts(pred, gt) == (2, 1, 1, 1)
    assert excavation_precision(np.zeros_like(gt), gt) == 0.0
    assert excavation_precision(np.zeros_like(gt), np.zeros_like(gt)) == 1.0
