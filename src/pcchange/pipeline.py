"""End-to-end PLY change-detection pipeline."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pcchange.change_detection import ChangeResult, detect_change
from pcchange.io import load_point_cloud, save_point_cloud
from pcchange.metrics import (
    confusion_counts,
    excavation_precision,
    excavation_recall,
    pose_errors_from_source_to_target,
)
from pcchange.preprocessing import PreprocessConfig, preprocess
from pcchange.registration import RegistrationConfig, RegistrationResult, register_point_clouds
from pcchange.reporting import write_markdown_report
from pcchange.visualization import (
    change_vmax,
    color_point_cloud_signed,
    save_distance_histogram,
    save_red_blue_map,
    save_red_blue_map_3d,
)


@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    change_threshold: float = 0.20
    normal_radius: float = 0.25


@dataclass
class PipelineResult:
    transformation: np.ndarray
    registration: RegistrationResult
    change: ChangeResult
    rotation_error_deg: float | None
    translation_error: float | None
    excavation_recall: float | None
    excavation_precision: float | None
    excavation_tp: int | None
    excavation_fp: int | None
    excavation_fn: int | None
    output_dir: Path
    report_path: Path


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_pipeline(
    source_path: str | Path,
    target_path: str | Path,
    output_dir: str | Path,
    *,
    config: PipelineConfig | None = None,
    ground_truth_path: str | Path | None = None,
    excavation_mask_path: str | Path | None = None,
) -> PipelineResult:
    cfg = config or PipelineConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_point_cloud(source_path)
    target = load_point_cloud(target_path)
    source_pre = preprocess(source, cfg.preprocess)
    target_pre = preprocess(target, cfg.preprocess)

    reg = register_point_clouds(source_pre, target_pre, cfg.registration)

    source_aligned = copy.deepcopy(source)
    source_aligned.transform(reg.transformation)

    change = detect_change(
        source_aligned,
        target,
        threshold=cfg.change_threshold,
        normal_radius=cfg.normal_radius,
    )

    pts = np.asarray(source_aligned.points)
    vmax = change_vmax(
        change.signed_distances,
        threshold=change.threshold,
        distances=change.distances,
    )
    colored, _vmax = color_point_cloud_signed(
        source_aligned,
        change.signed_distances,
        threshold=change.threshold,
        distances=change.distances,
        vmax=vmax,
    )
    aligned_path = save_point_cloud(source_aligned, output_dir / "T1_aligned.ply")
    colored_path = save_point_cloud(colored, output_dir / "change_red_blue.ply")
    hist_path = save_distance_histogram(
        change.distances,
        output_dir / "c2c_histogram.png",
        threshold=change.threshold,
    )
    map_kwargs = dict(
        threshold=change.threshold,
        distances=change.distances,
        changed_mask=change.changed_mask,
        vmax=vmax,
    )
    map_path = save_red_blue_map(
        pts,
        change.signed_distances,
        output_dir / "change_map.png",
        axes=(0, 1),
        title=f"Plan X–Y (gray = stable; red/blue = |C2C| > {change.threshold:.2f} m)",
        **map_kwargs,
    )
    map_yz_path = save_red_blue_map(
        pts,
        change.signed_distances,
        output_dir / "change_map_yz.png",
        axes=(1, 2),
        title=f"Side Y–Z (gray = stable; red/blue = |C2C| > {change.threshold:.2f} m)",
        **map_kwargs,
    )
    map_3d_path = save_red_blue_map_3d(
        pts,
        change.signed_distances,
        output_dir / "change_map_3d.png",
        **map_kwargs,
    )

    rot_err: float | None = None
    trans_err: float | None = None
    recall: float | None = None
    precision: float | None = None
    tp = fp = fn = None
    if ground_truth_path is not None:
        gt = _load_json(ground_truth_path)
        rot_err, trans_err = pose_errors_from_source_to_target(
            reg.transformation, np.asarray(gt["R_gt"]), np.asarray(gt["t_gt"])
        )
    if excavation_mask_path is not None:
        mask = np.load(excavation_mask_path)
        if mask.shape[0] != change.changed_mask.shape[0]:
            raise ValueError(
                "excavation mask length does not match source cloud "
                f"({mask.shape[0]} vs {change.changed_mask.shape[0]})"
            )
        recall = excavation_recall(change.changed_mask, mask)
        precision = excavation_precision(change.changed_mask, mask)
        tp, fp, fn, _tn = confusion_counts(change.changed_mask, mask)

    report_path = write_markdown_report(
        output_dir / "report.md",
        {
            "source_path": str(source_path),
            "target_path": str(target_path),
            "output_dir": str(output_dir),
            "preprocess": cfg.preprocess,
            "registration": reg,
            "change": change,
            "pose_errors": (
                {"rotation_deg": rot_err, "translation_m": trans_err}
                if rot_err is not None and trans_err is not None
                else None
            ),
            "excavation_recall": recall,
            "excavation_precision": precision,
            "excavation_counts": (
                {"tp": tp, "fp": fp, "fn": fn} if tp is not None else None
            ),
            "map_path": str(map_path),
            "map_yz_path": str(map_yz_path),
            "map_3d_path": str(map_3d_path),
            "hist_path": str(hist_path),
            "colored_ply_path": str(colored_path),
            "aligned_ply_path": str(aligned_path),
        },
    )

    return PipelineResult(
        transformation=reg.transformation,
        registration=reg,
        change=change,
        rotation_error_deg=rot_err,
        translation_error=trans_err,
        excavation_recall=recall,
        excavation_precision=precision,
        excavation_tp=tp,
        excavation_fp=fp,
        excavation_fn=fn,
        output_dir=output_dir,
        report_path=report_path,
    )
