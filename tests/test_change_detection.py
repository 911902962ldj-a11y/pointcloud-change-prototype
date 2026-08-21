from __future__ import annotations

import copy

import numpy as np

from pcchange.change_detection import detect_change
from pcchange.metrics import (
    excavation_recall,
    invert_transform,
    transform_from_Rt,
)


def test_c2c_zero_on_identical_cloud(synthetic_tunnel_pcds) -> None:
    t0, _t1 = synthetic_tunnel_pcds
    result = detect_change(t0, t0, threshold=0.05)
    assert result.median < 1e-9
    assert result.fraction_changed < 0.01


def test_excavation_recall_with_ground_truth_pose(
    synthetic_tunnel, synthetic_tunnel_pcds
) -> None:
    t0, t1 = synthetic_tunnel_pcds
    T = invert_transform(transform_from_Rt(synthetic_tunnel.R_gt, synthetic_tunnel.t_gt))
    t1_aligned = copy.deepcopy(t1)
    t1_aligned.transform(T)
    result = detect_change(t1_aligned, t0, threshold=0.20)
    recall = excavation_recall(result.changed_mask, synthetic_tunnel.t1_excavation_mask)
    assert recall >= 0.8, f"excavation recall {recall:.3f} < 0.8"
    assert result.changed_mask.shape[0] == synthetic_tunnel.t1_excavation_mask.shape[0]
