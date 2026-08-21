from __future__ import annotations

from pathlib import Path

from generate_synthetic_tunnel import save_synthetic_tunnel
from pcchange.pipeline import PipelineConfig, run_pipeline
from pcchange.preprocessing import PreprocessConfig
from pcchange.registration import RegistrationConfig


def test_pipeline_meets_ground_truth_thresholds(
    synthetic_tunnel, tmp_path: Path
) -> None:
    paths = save_synthetic_tunnel(synthetic_tunnel, tmp_path / "data")
    out = tmp_path / "output"
    result = run_pipeline(
        paths["T1"],
        paths["T0"],
        out,
        config=PipelineConfig(
            preprocess=PreprocessConfig(voxel_size=0.08),
            registration=RegistrationConfig(voxel_size=0.08, n_scales=4, max_iteration=60),
            change_threshold=0.20,
        ),
        ground_truth_path=paths["ground_truth"],
        excavation_mask_path=paths["mask"],
    )
    assert result.rotation_error_deg is not None
    assert result.translation_error is not None
    assert result.excavation_recall is not None
    assert result.excavation_precision is not None
    assert result.rotation_error_deg < 0.5, f"rotation {result.rotation_error_deg:.3f} deg"
    assert result.translation_error < 0.05, f"translation {result.translation_error:.4f} m"
    assert result.excavation_recall >= 0.8, f"recall {result.excavation_recall:.3f}"
    assert (out / "report.md").is_file()
    assert (out / "change_red_blue.ply").is_file()
    assert (out / "T1_aligned.ply").is_file()
    assert (out / "c2c_histogram.png").is_file()
    assert (out / "change_map.png").is_file()
    assert (out / "change_map_yz.png").is_file()
    assert (out / "change_map_3d.png").is_file()
    report = result.report_path.read_text(encoding="utf-8")
    assert "Generalized ICP" in report
    assert "Plane fitting: not used" in report
    assert "Precision:" in report
    assert "TP / FP / FN:" in report
