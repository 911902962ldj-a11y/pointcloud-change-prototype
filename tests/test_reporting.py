from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pcchange.reporting import write_markdown_report


def test_write_markdown_report(tmp_path: Path) -> None:
    T = np.eye(4)
    T[:3, 3] = [0.1, 0.0, 0.0]
    change = SimpleNamespace(
        mean=0.1,
        std=0.2,
        median=0.05,
        p95=0.4,
        p99=0.6,
        threshold=0.2,
        n_changed=10,
        distances=np.zeros(100),
        fraction_changed=0.1,
    )
    reg = SimpleNamespace(
        voxel_sizes=[0.4, 0.2],
        fitness=0.9,
        inlier_rmse=0.02,
        transformation=T,
    )
    pre = SimpleNamespace(nb_neighbors=20, std_ratio=2.0, voxel_size=0.05)
    path = write_markdown_report(
        tmp_path / "report.md",
        {
            "source_path": "T1.ply",
            "target_path": "T0.ply",
            "output_dir": str(tmp_path),
            "preprocess": pre,
            "registration": reg,
            "change": change,
            "pose_errors": {"rotation_deg": 0.1, "translation_m": 0.01},
            "excavation_recall": 0.9,
            "excavation_precision": 0.75,
            "excavation_counts": {"tp": 9, "fp": 3, "fn": 1},
            "map_path": "map.png",
            "map_yz_path": "map_yz.png",
            "map_3d_path": "map_3d.png",
            "hist_path": "hist.png",
            "colored_ply_path": "c.ply",
            "aligned_ply_path": "a.ply",
        },
    )
    text = path.read_text(encoding="utf-8")
    assert "0.1°" in text or "0.1000°" in text
    assert "Recall" in text
    assert "Precision: **0.750**" in text
    assert "TP / FP / FN: 9 / 3 / 1" in text
