"""Markdown report for a change-detection run."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _fmt_mat3(R) -> str:
    rows = [" ".join(f"{v: .6f}" for v in row) for row in R]
    return "\n".join(f"`{row}`" for row in rows)


def write_markdown_report(path: str | Path, ctx: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pose = ctx.get("pose_errors") or {}
    pose_block = ""
    if pose:
        pose_block = (
            "## Ground-truth pose error\n\n"
            f"- Rotation error: **{pose['rotation_deg']:.4f}°** (limit 0.5°)\n"
            f"- Translation error: **{pose['translation_m']:.4f} m** (limit 0.05 m)\n\n"
        )

    recall = ctx.get("excavation_recall")
    precision = ctx.get("excavation_precision")
    counts = ctx.get("excavation_counts") or {}
    recall_block = ""
    if recall is not None or precision is not None:
        lines = ["## Excavation detection\n"]
        if recall is not None:
            lines.append(f"- Recall: **{recall:.3f}** (limit 0.8)")
        if precision is not None:
            lines.append(f"- Precision: **{precision:.3f}**")
        if counts:
            lines.append(
                f"- TP / FP / FN: {counts.get('tp', 0)} / {counts.get('fp', 0)} / {counts.get('fn', 0)}"
            )
        lines.append(f"- Threshold: {ctx['change'].threshold:.3f} m")
        recall_block = "\n".join(lines) + "\n\n"

    change = ctx["change"]
    reg = ctx["registration"]
    pre = ctx["preprocess"]

    body = f"""# Point-cloud change report

## Inputs

- Target (T0): `{ctx['target_path']}`
- Source (T1): `{ctx['source_path']}`
- Output directory: `{ctx['output_dir']}`

## Preprocessing

- Statistical filter: `nb_neighbors={pre.nb_neighbors}`, `std_ratio={pre.std_ratio}`
- Voxel size: `{pre.voxel_size}` m
- Plane fitting: not used

## Registration (Generalized ICP)

- Voxel schedule: {reg.voxel_sizes}
- Fitness: {reg.fitness:.4f}
- Inlier RMSE: {reg.inlier_rmse:.4f} m
- Residual MAD rejection after alignment: yes

Estimated source → target rotation:

{_fmt_mat3(reg.transformation[:3, :3])}

Estimated translation (m): `{reg.transformation[:3, 3].tolist()}`

{pose_block}## C2C change statistics

- Mean: {change.mean:.4f} m
- Std: {change.std:.4f} m
- Median: {change.median:.4f} m
- P95: {change.p95:.4f} m
- P99: {change.p99:.4f} m
- Threshold: {change.threshold:.3f} m
- Changed points: {change.n_changed} / {change.distances.size} ({100 * change.fraction_changed:.2f}%)

{recall_block}## Figures

- Plan map (X–Y): `{ctx.get('map_path', '')}`
- Side map (Y–Z): `{ctx.get('map_yz_path', '')}`
- 3D map: `{ctx.get('map_3d_path', '')}`
- Histogram: `{ctx.get('hist_path', '')}`
- Coloured PLY: `{ctx.get('colored_ply_path', '')}`
- Aligned source PLY: `{ctx.get('aligned_ply_path', '')}`

Stable points are gray. Red/blue only for `|C2C| > threshold`.
Red = negative signed distance (loss / excavation). Blue = positive (gain).
Colour bar is scaled from changed-point magnitude, not the whole-cloud noise percentile.
"""
    path.write_text(body, encoding="utf-8")
    return path
