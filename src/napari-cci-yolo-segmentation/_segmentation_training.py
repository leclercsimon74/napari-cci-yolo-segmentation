from __future__ import annotations

# ruff: noqa: I001

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
from skimage.measure import find_contours, label


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class CCIYoloWrapper:
    """Small wrapper used by the widget for predict/train calls."""

    def __init__(self, model_name_or_path: str = "yolov8n.pt"):
        self.model = self._create_model(model_name_or_path)

    @staticmethod
    def _create_model(model_name_or_path: str):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - runtime guard
            raise RuntimeError(
                "Failed to import ultralytics/torch. Install a compatible build for this platform."
            ) from exc
        return YOLO(model_name_or_path)

    def load_model(self, weights_path: Path) -> None:
        self.model = self._create_model(str(weights_path))

    def predict(self, img):
        return self.model(img)

    def train(
        self,
        data_set_file: Path,
        image_size: int,
        batch: int = 8,
        epochs: int = 100,
        patience: int = 30,
        **kwargs,
    ):
        return self.model.train(
            data=str(data_set_file),
            imgsz=image_size,
            batch=batch,
            epochs=epochs,
            patience=patience,
            **kwargs,
        )


@dataclass(frozen=True)
class PairPaths:
    stem: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class RetrainConfig:
    tile_size: int = 1024
    val_ratio: float = 0.2
    seed: int = 42
    batch: int = 4
    epochs: int = 100
    patience: int = 30


def run_retraining_pipeline(
    model_path: Path,
    retrain_data_root: Path,
    output_root: Path | None,
    config: RetrainConfig | None = None,
) -> Path:
    """Create a YOLO segmentation dataset from image/mask pairs and train a model."""
    cfg = config or RetrainConfig()

    model_path = Path(model_path)
    retrain_data_root = Path(retrain_data_root)

    if output_root is None:
        run_root = model_path.parent / f"retrained_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    else:
        run_root = Path(output_root)

    dataset_root = run_root / "dataset"
    traces_root = run_root / "training_traces"

    pairs = _collect_pairs(retrain_data_root)
    train_pairs, val_pairs = _split_pairs(pairs, cfg.val_ratio, cfg.seed)

    # Single-class mode for now: all non-zero mask pixels belong to class_0.
    has_foreground = _has_any_foreground(pairs)
    if not has_foreground:
        raise ValueError("No foreground classes found in masks. Masks must contain values > 0.")

    class_map = {1: 0}
    class_names = {0: "class_0"}

    _prepare_empty_dataset_folders(dataset_root)

    stats = {
        "train_tiles": 0,
        "val_tiles": 0,
        "train_positive": 0,
        "val_positive": 0,
    }

    _write_split_tiles(train_pairs, dataset_root / "images" / "train", dataset_root / "labels" / "train", cfg.tile_size, class_map, stats, "train")
    _write_split_tiles(val_pairs, dataset_root / "images" / "val", dataset_root / "labels" / "val", cfg.tile_size, class_map, stats, "val")

    if stats["train_positive"] == 0 or stats["val_positive"] == 0:
        raise ValueError(
            "Not enough positive tiles after preprocessing. Ensure masks contain objects in both train and val splits."
        )

    dataset_yaml = _write_dataset_yaml(dataset_root, class_names)

    yolo = CCIYoloWrapper(str(model_path))
    yolo.train(
        data_set_file=dataset_yaml,
        image_size=cfg.tile_size,
        batch=cfg.batch,
        epochs=cfg.epochs,
        patience=cfg.patience,
        project=str(traces_root),
        name="run",
        exist_ok=True,
        task="segment",
    )

    best_model = traces_root / "run" / "weights" / "best.pt"
    if not best_model.exists():
        raise FileNotFoundError(f"best.pt not found at: {best_model}")

    run_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, run_root / "best.pt")
    return run_root


def _collect_pairs(retrain_data_root: Path) -> list[PairPaths]:
    images_dir = retrain_data_root / "images"
    masks_dir = retrain_data_root / "masks"

    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")
    if not masks_dir.exists() or not masks_dir.is_dir():
        raise FileNotFoundError(f"Missing masks folder: {masks_dir}")

    images = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS])
    masks = sorted([p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS])

    if not images:
        raise ValueError("No images found in retrain/images")
    if not masks:
        raise ValueError("No masks found in retrain/masks")

    masks_by_stem = {p.stem: p for p in masks}

    pairs: list[PairPaths] = []
    missing_masks: list[str] = []
    for img in images:
        mask = masks_by_stem.get(img.stem)
        if mask is None:
            missing_masks.append(img.name)
            continue
        pairs.append(PairPaths(stem=img.stem, image_path=img, mask_path=mask))

    if missing_masks:
        preview = ", ".join(missing_masks[:5])
        raise ValueError(f"Missing matching masks for image(s): {preview}")
    if len(pairs) < 2:
        raise ValueError("At least 2 image/mask pairs are required for train/val splitting.")

    return pairs


def _split_pairs(pairs: list[PairPaths], val_ratio: float, seed: int) -> tuple[list[PairPaths], list[PairPaths]]:
    rnd = random.Random(seed)
    shuffled = pairs[:]
    rnd.shuffle(shuffled)

    val_count = int(round(len(shuffled) * val_ratio))
    if len(shuffled) > 1:
        val_count = max(1, min(len(shuffled) - 1, val_count))
    else:
        val_count = 0

    val_pairs = shuffled[:val_count]
    train_pairs = shuffled[val_count:]
    return train_pairs, val_pairs


def _has_any_foreground(pairs: list[PairPaths]) -> bool:
    for pair in pairs:
        mask = _read_mask(pair.mask_path)
        if np.any(mask > 0):
            return True
    return False


def _prepare_empty_dataset_folders(dataset_root: Path) -> None:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "val").mkdir(parents=True, exist_ok=True)


def _write_split_tiles(
    pairs: list[PairPaths],
    images_out: Path,
    labels_out: Path,
    tile_size: int,
    class_map: dict[int, int],
    stats: dict[str, int],
    split_name: str,
) -> None:
    tile_index = 0
    keep_empty_every = 10

    for pair in pairs:
        image = _read_image_rgb(pair.image_path)
        mask = _read_mask(pair.mask_path)

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image/mask shape mismatch for '{pair.stem}': image={image.shape[:2]}, mask={mask.shape[:2]}"
            )

        for y0, y1, x0, x1 in _tile_windows(mask.shape[0], mask.shape[1], tile_size):
            tile_img = _extract_and_pad_image_tile(image, y0, y1, x0, x1, tile_size)
            tile_mask = _extract_and_pad_mask_tile(mask, y0, y1, x0, x1, tile_size)

            yolo_lines = _mask_to_yolo_segmentation_lines(tile_mask, class_map)
            has_positive = len(yolo_lines) > 0
            if not has_positive and (tile_index % keep_empty_every != 0):
                tile_index += 1
                continue

            tile_name = f"{pair.stem}_y{y0}_x{x0}.png"
            Image.fromarray(tile_img).save(images_out / tile_name)
            label_path = labels_out / f"{pair.stem}_y{y0}_x{x0}.txt"
            _write_lines(label_path, yolo_lines)

            stats[f"{split_name}_tiles"] += 1
            if has_positive:
                stats[f"{split_name}_positive"] += 1
            tile_index += 1


def _tile_windows(height: int, width: int, tile_size: int):
    y_starts = _axis_starts(height, tile_size)
    x_starts = _axis_starts(width, tile_size)
    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(y0 + tile_size, height)
            x1 = min(x0 + tile_size, width)
            yield y0, y1, x0, x1


def _axis_starts(length: int, tile_size: int) -> list[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, tile_size))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _extract_and_pad_image_tile(image: np.ndarray, y0: int, y1: int, x0: int, x1: int, tile_size: int) -> np.ndarray:
    tile = image[y0:y1, x0:x1]
    if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
        return tile

    out = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    out[: tile.shape[0], : tile.shape[1]] = tile
    return out


def _extract_and_pad_mask_tile(mask: np.ndarray, y0: int, y1: int, x0: int, x1: int, tile_size: int) -> np.ndarray:
    tile = mask[y0:y1, x0:x1]
    if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
        return tile

    out = np.zeros((tile_size, tile_size), dtype=mask.dtype)
    out[: tile.shape[0], : tile.shape[1]] = tile
    return out


def _mask_to_yolo_segmentation_lines(mask: np.ndarray, class_map: dict[int, int]) -> list[str]:
    lines: list[str] = []
    height, width = mask.shape
    class_id = class_map.get(1, 0)

    foreground = (mask > 0).astype(np.uint8)
    components = label(foreground, connectivity=1)

    for component_id in range(1, int(components.max()) + 1):
        binary = (components == component_id).astype(np.uint8)
        contours = find_contours(binary, level=0.5)
        for contour in contours:
            if contour.shape[0] < 3:
                continue

            coords: list[float] = []
            for y, x in contour:
                xn = float(np.clip(x / width, 0.0, 1.0))
                yn = float(np.clip(y / height, 0.0, 1.0))
                coords.extend([xn, yn])

            # YOLO segmentation needs at least 3 points => 6 coordinates.
            if len(coords) < 6:
                continue

            line = f"{class_id} " + " ".join(f"{v:.6f}" for v in coords)
            lines.append(line)

    return lines


def _write_dataset_yaml(dataset_root: Path, class_names: dict[int, str]) -> Path:
    dataset_yaml = dataset_root / "dataset.yaml"
    lines = [
        "train: ./images/train/",
        "val: ./images/val/",
        "names:",
    ]
    for class_id in sorted(class_names):
        lines.append(f"  {class_id}: {class_names[class_id]}")

    dataset_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dataset_yaml


def _read_image_rgb(path: Path) -> np.ndarray:
    img = Image.open(path)
    arr = np.asarray(img)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _read_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _write_lines(path: Path, lines: list[str]) -> None:
    if lines:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        path.write_text("", encoding="utf-8")
