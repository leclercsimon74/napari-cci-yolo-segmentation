from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _load_training_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "napari-cci-yolo-segmentation"
        / "_segmentation_training.py"
    )
    spec = importlib.util.spec_from_file_location("segmentation_training", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save_gray_image(path: Path, data: np.ndarray) -> None:
    Image.fromarray(data.astype(np.uint8)).save(path)


def test_collect_pairs_missing_mask_raises(tmp_path: Path):
    mod = _load_training_module()

    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()

    _save_gray_image(images / "a.png", np.zeros((8, 8), dtype=np.uint8))
    _save_gray_image(images / "b.png", np.zeros((8, 8), dtype=np.uint8))
    _save_gray_image(masks / "a.png", np.zeros((8, 8), dtype=np.uint8))

    with pytest.raises(ValueError, match="Missing matching masks"):
        mod._collect_pairs(tmp_path)


def test_split_pairs_is_deterministic_and_keeps_val_sample(tmp_path: Path):
    mod = _load_training_module()

    pairs = [
        mod.PairPaths(stem=f"img_{i}", image_path=tmp_path / f"img_{i}.png", mask_path=tmp_path / f"img_{i}.png")
        for i in range(5)
    ]

    train_1, val_1 = mod._split_pairs(pairs, val_ratio=0.2, seed=42)
    train_2, val_2 = mod._split_pairs(pairs, val_ratio=0.2, seed=42)

    assert [p.stem for p in train_1] == [p.stem for p in train_2]
    assert [p.stem for p in val_1] == [p.stem for p in val_2]
    assert len(val_1) >= 1


def test_axis_starts_has_no_out_of_order_indices():
    mod = _load_training_module()

    starts = mod._axis_starts(length=2050, tile_size=1024)

    assert starts == [0, 1024, 1026]


def test_mask_to_yolo_segmentation_lines_generates_polygons():
    mod = _load_training_module()

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1

    lines = mod._mask_to_yolo_segmentation_lines(mask, class_map={1: 0})

    assert len(lines) >= 1
    first = lines[0].split()
    assert first[0] == "0"
    coords = [float(v) for v in first[1:]]
    assert len(coords) >= 6
    assert len(coords) % 2 == 0
    assert all(0.0 <= v <= 1.0 for v in coords)


def test_mask_to_yolo_segmentation_lines_treats_components_as_objects_single_class():
    mod = _load_training_module()

    mask = np.zeros((32, 32), dtype=np.uint16)
    mask[2:8, 2:8] = 1
    mask[16:24, 16:24] = 7

    lines = mod._mask_to_yolo_segmentation_lines(mask, class_map={1: 0})

    assert len(lines) >= 2
    class_ids = {line.split()[0] for line in lines}
    assert class_ids == {"0"}


def test_tile_windows_cover_edge_aligned_tiles():
    mod = _load_training_module()

    windows = list(mod._tile_windows(height=1500, width=1500, tile_size=1024))

    assert len(windows) == 4
    assert windows[0] == (0, 1024, 0, 1024)
    assert windows[-1] == (476, 1500, 476, 1500)
