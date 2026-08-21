from __future__ import annotations

from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1] / "src" / "pcchange"


@pytest.mark.parametrize("forbidden", ["tunnel", "巷道", "horseshoe", "segment_plane"])
def test_core_has_no_scene_or_ransac_terms(forbidden: str) -> None:
    hits: list[str] = []
    for path in CORE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if forbidden.lower() in text.lower():
            hits.append(path.name)
    assert not hits, f"{forbidden!r} found in {hits}"
