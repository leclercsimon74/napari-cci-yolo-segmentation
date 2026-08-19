from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def _load_tiling_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "napari-cci-yolo-segmentation"
        / "yolo_tiling_segmentation.py"
    )
    spec = importlib.util.spec_from_file_location("yolo_tiling_segmentation", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_calculate_chunk_size_uses_tile_size_and_overlap():
    mod = _load_tiling_module()

    assert mod.LargeImageYoloSegmenter.calculate_chunk_size(image_size=1024, overlap=100) == 824
    assert mod.LargeImageYoloSegmenter.calculate_chunk_size(image_size=1024, overlap=0) == 1024


def test_calculate_chunk_size_rejects_overlap_half_tile_or_larger():
    mod = _load_tiling_module()

    with pytest.raises(ValueError, match="overlap must be smaller"):
        mod.LargeImageYoloSegmenter.calculate_chunk_size(image_size=1024, overlap=512)


def test_render_instances_central_regions_respects_overlap():
    mod = _load_tiling_module()

    instance = mod.TileInstance(
        label_id=7,
        confidence=1.0,
        tile_origin=(-2, -2),
        bbox=(0, 5, 0, 5),
        mask=np.ones((5, 5), dtype=bool),
    )

    rendered = mod.LargeImageYoloSegmenter.render_instances_central_regions(
        [instance],
        shape=(8, 8),
        image_size=6,
        overlap=2,
    )

    assert np.all(rendered[:2, :2] == 7)
    assert np.all(rendered[2:, :] == 0)
    assert np.all(rendered[:, 2:] == 0)
