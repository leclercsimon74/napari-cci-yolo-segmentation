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


def test_one_pixel_boundary_merge_links_labels_touching_across_chunks():
    mod = _load_tiling_module()

    segment_results = np.array(
        [
            [0, 7, 9, 0],
            [0, 7, 9, 0],
            [0, 0, 0, 0],
            [3, 0, 0, 4],
        ],
        dtype=np.uint32,
    )

    merged = mod.merge_segments_one_pixel_boundary(
        segment_results,
        image_size=6,
        overlap=2,
    )

    assert np.all(merged[:2, 1:3] == 7)
    assert merged[3, 0] == 3
    assert merged[3, 3] == 4
