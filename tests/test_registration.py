from __future__ import annotations

import numpy as np

from pcchange.metrics import pose_errors_from_source_to_target
from pcchange.preprocessing import PreprocessConfig, preprocess
from pcchange.registration import RegistrationConfig, register_point_clouds, residual_inlier_mask


def test_residual_mask_keeps_close_points(synthetic_tunnel_pcds) -> None:
    t0, t1 = synthetic_tunnel_pcds
    mask, dist = residual_inlier_mask(t0, t0, np.eye(4), k_mad=3.0, min_threshold=0.02)
    assert mask.mean() > 0.9
    assert dist.shape[0] == len(t0.points)


def test_gicp_recovers_ground_truth_pose(synthetic_tunnel, synthetic_tunnel_pcds) -> None:
    t0, t1 = synthetic_tunnel_pcds
    cfg_pre = PreprocessConfig(nb_neighbors=20, std_ratio=2.0, voxel_size=0.08)
    t0_p = preprocess(t0, cfg_pre)
    t1_p = preprocess(t1, cfg_pre)
    result = register_point_clouds(
        t1_p,
        t0_p,
        RegistrationConfig(voxel_size=0.08, n_scales=4, max_iteration=60),
    )
    rot_err, trans_err = pose_errors_from_source_to_target(
        result.transformation, synthetic_tunnel.R_gt, synthetic_tunnel.t_gt
    )
    assert rot_err < 0.5, f"rotation error {rot_err:.3f} deg"
    assert trans_err < 0.05, f"translation error {trans_err:.4f} m"
