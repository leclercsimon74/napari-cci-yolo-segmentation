#_gui.py

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from qtpy.QtCore import QThread, QTimer, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from skimage.measure import find_contours
from skimage.morphology import disk, label, remove_small_objects

from . import config
from ._segmentation_training import (
    CCIYoloWrapper,
    RetrainConfig,
    run_retraining_pipeline,
)
from .yolo_tiling_segmentation import (
    merge_segments_one_pixel_boundary,
    predict_segments_with_yolo_tiling,
)


class _RetrainWorker(QThread):
    """Runs YOLO segmentation retraining in a background thread."""

    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, model_path: Path, retrain_data_path: Path, parent=None):
        super().__init__(parent)
        self._model_path = model_path
        self._retrain_data_path = retrain_data_path

    def run(self):
        try:
            retrain_root = run_retraining_pipeline(
                model_path=self._model_path,
                retrain_data_root=self._retrain_data_path,
                output_root=self._retrain_data_path / f"retrained_{datetime.now().strftime('%y%m%d_%H%M%S')}",
                config=RetrainConfig(
                    tile_size=config.TILE_SIZE,
                    val_ratio=0.2,
                    seed=42,
                    batch=4,
                    patience=30,
                ),
            )
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - GUI runtime guard
            self.failed.emit(f"Retrain failed: {exc}")
            return

        self.finished.emit(f"Retrain done. New model saved in: {retrain_root}")


class CciYoloSegmentatorQWidget(QWidget):
    """Minimal YOLO segmentator flow for napari.

    Workflow:
    1) Select model path (.pt) and load model
    2) Predict segments on the active image
    3) Retrain from a folder with image/mask pairs
    """

    PRED_LAYER_NAME = "yolo_segments"
    BBOX_LAYER_NAME = "yolo_bboxes"
    MERGED_LAYER_PREFIX = "yolo_segments_merged"
    TILE_GRID_LAYER_PREFIX = "yolo_tile_grid"
    CROP_SELECTION_LAYER_NAME = "yolo_crop_selection"
    CROP_IMAGE_LAYER_NAME = "yolo_crop_image"
    CROP_MASK_LAYER_NAME = "yolo_crop_mask"
    TILING_THRESHOLD = config.TILE_SIZE
    show_bbox = True

    def __init__(self, napari_viewer):
        super().__init__()
        self.napari_viewer = napari_viewer
        self.setWindowTitle("Yolo CCI Segmentator")

        self._yolo: CCIYoloWrapper | None = None
        self._model_path: Path | None = None
        self._retrain_data_path: Path | None = None
        self._retrain_worker: _RetrainWorker | None = None
        self._crop_source_layers = []

        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(400)
        self._spinner_timer.timeout.connect(self._tick_spinner)
        self._spinner_frames = ["Retraining .", "Retraining ..", "Retraining ...", "Retraining"]
        self._spinner_index = 0

        #Button creation and connection
        self._model_path_input = QLineEdit()
        self._model_path_input.setPlaceholderText("Path to YOLO model (.pt) or model folder")

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._on_browse_model)

        load_button = QPushButton("Load model")
        load_button.clicked.connect(self._on_load_model)

        predict_button = QPushButton("Predict Large Image")
        predict_button.clicked.connect(self._on_predict)

        merge_button = QPushButton("Merge Segments")
        merge_button.clicked.connect(self._on_merge_segments)

        select_crop_button = QPushButton("Select Crop")
        select_crop_button.clicked.connect(self._on_select_crop)

        new_crop_button = QPushButton("Create Crop")
        new_crop_button.clicked.connect(self._on_new_crop)

        self.retrain_data_path_input = QLineEdit()
        self.retrain_data_path_input.setPlaceholderText("Retrain folder containing images/ and masks/")

        browse_retrain_data_button = QPushButton("Browse")
        browse_retrain_data_button.clicked.connect(self._on_browse_retrain_data)

        self._add_to_retrain_button = QPushButton("Save Crop to Retrain")
        self._add_to_retrain_button.clicked.connect(self._on_add_to_retrain)

        self._retrain_button = QPushButton("Retrain")
        self._retrain_button.clicked.connect(self._on_retrain)

        #Gui Layout
        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("Model"))
        row_model.addWidget(self._model_path_input)
        row_model.addWidget(browse_button)

        row_merging = QHBoxLayout()
        row_merging.addWidget(predict_button)
        row_merging.addWidget(merge_button)

        row_crop = QHBoxLayout()
        row_crop.addWidget(select_crop_button)
        row_crop.addWidget(new_crop_button)

        row_retrain_data = QHBoxLayout()
        row_retrain_data.addWidget(QLabel("Retrain data"))
        row_retrain_data.addWidget(self.retrain_data_path_input)
        row_retrain_data.addWidget(browse_retrain_data_button)

        row_train = QHBoxLayout()
        row_train.addWidget(self._add_to_retrain_button)
        row_train.addWidget(self._retrain_button)

        self.stats_label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(QLabel("<b>Model</b>"))
        layout.addLayout(row_model)
        layout.addWidget(load_button)
        layout.addWidget(QLabel("<b>Prediction</b>"))
        layout.addWidget(QLabel(f"Tile of {config.TILE_SIZE} pxl with {config.OVERLAP} pxl overlap for large images"))
        layout.addLayout(row_merging)
        layout.addWidget(QLabel("<b>Review</b>"))
        layout.addLayout(row_crop)
        layout.addWidget(QLabel("<b>Correction</b>"))
        layout.addLayout(row_retrain_data)
        layout.addWidget(self.stats_label)
        layout.addLayout(row_train)
        layout.addStretch(1)
        self.setLayout(layout)

    def _show_info(self, text: str) -> None:
        QMessageBox.information(self, "Yolo CCI Segmentator", text)

    def _show_error(self, text: str) -> None:
        QMessageBox.critical(self, "Yolo CCI Segmentator", text)

    def _on_browse_model(self) -> None:
        model_dir = QFileDialog.getExistingDirectory(
            self,
            "Select model folder (.pt will be loaded or yolov8n.pt will be copied)"
        )
        if model_dir:
            self._model_path_input.setText(model_dir)

    def _on_browse_retrain_data(self) -> None:
        retrain_dir = QFileDialog.getExistingDirectory(self, "Select retrain folder with images/ and masks/")
        if retrain_dir:
            self._set_retrain_data_path(Path(retrain_dir))

    def _set_retrain_data_path(self, path: Path) -> None:
        self._retrain_data_path = Path(path)
        self.retrain_data_path_input.setText(str(self._retrain_data_path))
        self.update_stats_label()

    def _maybe_preselect_retrain_data_path(self) -> None:
        if self._model_path is None or self.retrain_data_path_input.text().strip():
            return

        candidate = self._model_path.parent
        if (candidate / "images").is_dir() and (candidate / "masks").is_dir():
            self._set_retrain_data_path(candidate)

    def _resolve_retrain_data_path(self, create: bool) -> Path | None:
        raw_path = self.retrain_data_path_input.text().strip()
        if raw_path:
            retrain_root = Path(raw_path)
        elif self._model_path is not None:
            retrain_root = self._model_path.parent
        else:
            self._show_error("Load a model or select a retrain folder first.")
            return None

        if create:
            (retrain_root / "images").mkdir(parents=True, exist_ok=True)
            (retrain_root / "masks").mkdir(parents=True, exist_ok=True)
        elif not retrain_root.exists():
            self._show_error("Retrain folder does not exist.")
            return None

        self._set_retrain_data_path(retrain_root)
        return retrain_root

    def _on_select_crop(self) -> None:
        # generate a rectangular selection pixels on the active image layer
        image_layer = self._get_active_image_layer()
        if image_layer is None:
            return

        image_data = np.asarray(image_layer.data)
        if image_data.ndim < 2:
            self._show_error("Active image must be at least 2D.")
            return

        height, width = image_data.shape[:2]
        crop_size = min(self.TILING_THRESHOLD, height, width)
        center_y, center_x = self._current_view_center(height, width)
        y0, y1, x0, x1 = self._clamped_crop_bounds(
            center_y,
            center_x,
            crop_size,
            height,
            width,
        )
        rectangle = np.array(
            [[y0, x0], [y0, x1], [y1, x1], [y1, x0]],
            dtype=float,
        )

        existing = self._get_layer_by_name(self.CROP_SELECTION_LAYER_NAME)
        if existing is not None:
            self.napari_viewer.layers.remove(existing)

        layer = self.napari_viewer.add_shapes(
            [rectangle],
            name=self.CROP_SELECTION_LAYER_NAME,
            shape_type="rectangle",
            edge_width=3,
            edge_color="cyan",
            face_color="transparent",
        )
        layer.metadata["crop_size"] = crop_size
        layer.metadata["source_image_layer_name"] = image_layer.name
        layer.mode = "select"

        if crop_size < self.TILING_THRESHOLD:
            self._show_info(
                f"Image is smaller than {config.TILE_SIZE}x{config.TILE_SIZE} in at least one dimension. "
                f"Created a {crop_size}x{crop_size} crop selection instead."
            )

    def _on_new_crop(self) -> None:
        selection_layer = self._get_layer_by_name(self.CROP_SELECTION_LAYER_NAME)
        if selection_layer is None or len(selection_layer.data) == 0:
            self._show_error("Create a crop selection first.")
            return

        image_layer = self._get_crop_source_image_layer(selection_layer)
        if image_layer is None:
            return

        image_data = np.asarray(image_layer.data)
        if image_data.ndim not in {2, 3}:
            self._show_error("Source image must be 2D or RGB.")
            return

        if image_data.ndim == 3 and image_data.shape[-1] > 3:
            image_data = image_data[..., :3]

        segmentation_layer = self._get_latest_segments_layer()
        if segmentation_layer is None:
            self._show_error("No segmentation layer found to crop.")
            return

        segmentation_data = np.asarray(segmentation_layer.data)
        if segmentation_data.ndim != 2:
            self._show_error("Segmentation layer must be a 2D labels layer.")
            return
        if segmentation_data.shape != image_data.shape[:2]:
            self._show_error(
                "Source image and segmentation shapes do not match: "
                f"image={image_data.shape[:2]}, segmentation={segmentation_data.shape}."
            )
            return

        try:
            y0, y1, x0, x1 = self._crop_bounds_from_selection(
                selection_layer,
                image_data.shape[0],
                image_data.shape[1],
            )
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not read crop selection: {exc}")
            return

        crop_image = self._normalize_to_uint8(image_data[y0:y1, x0:x1])
        crop_mask = segmentation_data[y0:y1, x0:x1].astype(np.uint16, copy=False)

        self._hide_existing_crop_layers()
        self._crop_source_layers = [image_layer, segmentation_layer, selection_layer]

        for layer in self._crop_source_layers:
            layer.visible = False

        crop_image_layer = self.napari_viewer.add_image(
            crop_image,
            name=self.CROP_IMAGE_LAYER_NAME,
        )
        crop_mask_layer = self.napari_viewer.add_labels(
            crop_mask,
            name=self.CROP_MASK_LAYER_NAME,
        )
        crop_image_layer.metadata["source_image_layer_name"] = image_layer.name
        crop_image_layer.metadata["crop_bounds_yx"] = (y0, y1, x0, x1)
        crop_mask_layer.metadata["source_segmentation_layer_name"] = segmentation_layer.name
        crop_mask_layer.metadata["crop_bounds_yx"] = (y0, y1, x0, x1)

        self.napari_viewer.layers.selection.active = crop_mask_layer
        self._show_info(f"Created crop: y={y0}:{y1}, x={x0}:{x1}.")

    def update_stats_label(self) -> None:
        # add some stats to this label for information
        # number of images
        # number of masks
        # number of unique labels in masks if possible - require to open each mask!
        raw_path = self.retrain_data_path_input.text().strip()
        if raw_path:
            self._retrain_data_path = Path(raw_path)

        if self._retrain_data_path is None:
            self.stats_label.setText("No retrain data selected.")
            return
        # number of images
        images_dir = self._retrain_data_path / "images"
        masks_dir = self._retrain_data_path / "masks"
        num_images = len(os.listdir(images_dir)) if images_dir.exists() else 0
        num_masks = len(os.listdir(masks_dir)) if masks_dir.exists() else 0

        self.stats_label.setText(f"Retrain data: {num_images} images, {num_masks} masks.")

    def _hide_existing_crop_layers(self) -> None:
        for layer in self._crop_source_layers:
            if layer in self.napari_viewer.layers:
                layer.visible = True

        for name in (self.CROP_IMAGE_LAYER_NAME, self.CROP_MASK_LAYER_NAME):
            layer = self._get_layer_by_name(name)
            if layer is not None:
                self.napari_viewer.layers.remove(layer)

    def _get_crop_source_image_layer(self, selection_layer):
        source_name = selection_layer.metadata.get("source_image_layer_name")
        if source_name:
            source_layer = self._get_layer_by_name(source_name)
            if source_layer is not None and self._is_candidate_image_layer(source_layer):
                data = np.asarray(source_layer.data)
                if data.ndim >= 2:
                    return source_layer

        return self._get_active_image_layer()

    def _current_view_center(self, height: int, width: int) -> tuple[float, float]:
        center = getattr(getattr(self.napari_viewer, "camera", None), "center", None)
        if center is None:
            return height / 2.0, width / 2.0

        if len(center) >= 2:
            return float(center[-2]), float(center[-1])
        return height / 2.0, width / 2.0

    def _merge_segments(self) -> None:
        pred_layer = self._get_latest_segments_layer()
        if pred_layer is None:
            self._show_error("No prediction layer found to merge.")
            return

        pred_data = np.asarray(pred_layer.data)
        if pred_data.ndim != 2:
            self._show_error("Prediction layer must be a 2D labels layer.")
            return

        image_size, overlap = self._tiling_metadata(pred_layer)
        merged_mask = merge_segments_one_pixel_boundary(
            pred_data,
            image_size=image_size,
            overlap=overlap,
        )

        merge_method = "One pixel boundary"
        layer_name = self._next_merged_layer_name(merge_method)
        merged_layer = self.napari_viewer.add_labels(
            merged_mask,
            name=layer_name,
        )
        merged_layer.metadata.update(pred_layer.metadata)
        merged_layer.metadata["merge_method"] = merge_method
        merged_layer.metadata["merge_source_layer_name"] = pred_layer.name
        self.napari_viewer.layers.selection.active = merged_layer
        if self.show_bbox:
            self._add_bbox_layer(self._label_mask_to_bbox_rectangles(merged_mask))
        self._show_info(f"Merged segments using one-pixel boundary. Created layer: {layer_name}")

    def _on_merge_segments(self) -> None:
        self._merge_segments()

    def _merge_segments_one_pixel_boundary(self, pred_data: np.ndarray) -> np.ndarray:
        return merge_segments_one_pixel_boundary(
            pred_data,
            image_size=self.TILING_THRESHOLD,
            overlap=config.OVERLAP,
        )

    def _get_latest_segments_layer(self):
        for layer in reversed(self.napari_viewer.layers):
            if self._is_segments_layer(layer):
                return layer
        return None

    def _is_segments_layer(self, layer) -> bool:
        name = getattr(layer, "name", None)
        if name is None:
            return False
        if name != self.PRED_LAYER_NAME and not name.startswith(f"{self.MERGED_LAYER_PREFIX}_"):
            return False
        data = getattr(layer, "data", None)
        return data is not None and np.asarray(data).ndim == 2

    def _next_merged_layer_name(self, method: str) -> str:
        method_name = self._sanitize_stem(method).lower()
        prefix = f"{self.MERGED_LAYER_PREFIX}_{method_name}"
        return self._next_numbered_layer_name(prefix)

    def _next_numbered_layer_name(self, prefix: str) -> str:
        existing_names = {getattr(layer, "name", "") for layer in self.napari_viewer.layers}
        index = 1
        while True:
            name = f"{prefix}_{index:03d}"
            if name not in existing_names:
                return name
            index += 1

    def _add_tile_grid_layer(self, image_shape: tuple[int, int], image_size: int, overlap: int) -> None:
        height, width = image_shape
        chunk_size = image_size - (2 * overlap)
        lines = []

        for x in range(chunk_size, width, chunk_size):
            lines.append(np.array([[0, x], [height - 1, x]], dtype=float))
        for y in range(chunk_size, height, chunk_size):
            lines.append(np.array([[y, 0], [y, width - 1]], dtype=float))

        if not lines:
            return

        name = self._next_numbered_layer_name(self.TILE_GRID_LAYER_PREFIX)
        layer = self.napari_viewer.add_shapes(
            lines,
            name=name,
            shape_type="path",
            edge_width=1,
            edge_color="magenta",
            face_color="transparent",
        )
        layer.metadata["tile_size"] = image_size
        layer.metadata["overlap"] = overlap
        layer.metadata["chunk_size"] = chunk_size

    @staticmethod
    def _tiling_metadata(layer) -> tuple[int, int]:
        metadata = getattr(layer, "metadata", {}) or {}
        image_size = int(metadata.get("tile_size", config.TILE_SIZE))
        overlap = int(metadata.get("overlap", config.OVERLAP))
        return image_size, overlap

    def _get_annotation_layer(self, mask_shape: tuple[int, int]):
        pred_layer = self._get_latest_segments_layer()
        if pred_layer is not None:
            if self._is_shapes_like_layer(pred_layer):
                return pred_layer
            pred_data = np.asarray(getattr(pred_layer, "data", None))
            if pred_data.ndim == 2 and pred_data.shape == mask_shape:
                return pred_layer

        return self._get_shapes_layer()

    @staticmethod
    def _clamped_crop_bounds(
        center_y: float,
        center_x: float,
        crop_size: int,
        height: int,
        width: int,
    ) -> tuple[int, int, int, int]:
        y0 = int(round(center_y - crop_size / 2))
        x0 = int(round(center_x - crop_size / 2))
        y0 = max(0, min(y0, height - crop_size))
        x0 = max(0, min(x0, width - crop_size))
        return y0, y0 + crop_size, x0, x0 + crop_size

    def _crop_bounds_from_selection(
        self,
        selection_layer,
        height: int,
        width: int,
    ) -> tuple[int, int, int, int]:
        data = np.asarray(selection_layer.data[0], dtype=float)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError("Crop selection is not a 2D rectangle.")

        y_min = float(np.min(data[:, 0]))
        y_max = float(np.max(data[:, 0]))
        x_min = float(np.min(data[:, 1]))
        x_max = float(np.max(data[:, 1]))
        crop_size = int(selection_layer.metadata.get("crop_size", self.TILING_THRESHOLD))
        crop_size = min(crop_size, height, width)

        selected_height = int(round(y_max - y_min))
        selected_width = int(round(x_max - x_min))
        if selected_height != crop_size or selected_width != crop_size:
            center_y = (y_min + y_max) / 2.0
            center_x = (x_min + x_max) / 2.0
            y0, y1, x0, x1 = self._clamped_crop_bounds(
                center_y,
                center_x,
                crop_size,
                height,
                width,
            )
            selection_layer.data = [
                np.array(
                    [[y0, x0], [y0, x1], [y1, x1], [y1, x0]],
                    dtype=float,
                )
            ]
            return y0, y1, x0, x1

        return self._clamped_crop_bounds(
            (y_min + y_max) / 2.0,
            (x_min + x_max) / 2.0,
            crop_size,
            height,
            width,
        )

    @staticmethod
    def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image

        img = image.astype(np.float32)
        finite_mask = np.isfinite(img)
        if not np.any(finite_mask):
            return np.zeros_like(image, dtype=np.uint8)

        finite_values = img[finite_mask]
        img_min = float(finite_values.min())
        img_max = float(finite_values.max())
        if img_max <= 1.0 and img_min >= 0.0:
            img = img * 255.0
        elif img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255.0

        img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def _sanitize_stem(text: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text.strip())
        return cleaned or "sample"

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[-1] == 1:
            return image[..., 0]
        return np.asarray(Image.fromarray(image).convert("L"))

    @staticmethod
    def _to_pil_rgb(image: np.ndarray) -> Image.Image:
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        return Image.fromarray(image).convert("RGB")

    @staticmethod
    def _label_mask_to_polygons(label_mask: np.ndarray) -> list[np.ndarray]:
        polygons = []
        for label_id in np.unique(label_mask):
            if label_id == 0:
                continue

            binary = label_mask == label_id
            for contour in find_contours(binary.astype(np.uint8), level=0.5):
                if contour.shape[0] < 3:
                    continue
                polygons.append(np.asarray(contour, dtype=float))
        return polygons

    @staticmethod
    def _is_shapes_like_layer(layer) -> bool:
        return hasattr(layer, "to_masks") and hasattr(layer, "shape_type")

    def _get_shapes_layer(self):
        active = self.napari_viewer.layers.selection.active
        if (
            active is not None
            and self._is_shapes_like_layer(active)
            and getattr(active, "name", None) != self.CROP_SELECTION_LAYER_NAME
        ):
            return active

        pred = self._get_layer_by_name(self.PRED_LAYER_NAME)
        if pred is not None and self._is_shapes_like_layer(pred):
            return pred

        for layer in self.napari_viewer.layers:
            if (
                self._is_shapes_like_layer(layer)
                and getattr(layer, "name", None) != self.CROP_SELECTION_LAYER_NAME
            ):
                return layer
        return None

    def _build_instance_mask(self, shapes_layer, mask_shape: tuple[int, int]) -> np.ndarray:
        shape_masks = np.asarray(shapes_layer.to_masks(mask_shape=mask_shape))
        if shape_masks.size == 0:
            raise ValueError("No shapes found in the selected shapes layer.")

        instance_mask = np.zeros(mask_shape, dtype=np.uint16)
        for idx, obj_mask in enumerate(shape_masks, start=1):
            instance_mask[np.asarray(obj_mask, dtype=bool)] = idx
        return instance_mask

    def _on_add_to_retrain(self) -> None:
        if self._model_path is None:
            self._show_error("Load a model first.")
            return

        has_crop_layers = (
            self._get_layer_by_name(self.CROP_IMAGE_LAYER_NAME) is not None
            or self._get_layer_by_name(self.CROP_MASK_LAYER_NAME) is not None
        )
        if has_crop_layers:
            crop_pair = self._get_current_crop_retrain_pair()
            if crop_pair is None:
                return
            image_u8, instance_mask, stem = crop_pair
            self._save_retrain_pair(image_u8, instance_mask, stem)
            return

        image_layer = self._get_active_image_layer()
        if image_layer is None:
            return

        shapes_layer = self._get_shapes_layer()
        if shapes_layer is None:
            self._show_error("Add or select a Shapes layer with annotations first.")
            return

        image_data = np.asarray(image_layer.data)
        if image_data.ndim not in {2, 3}:
            self._show_error("Active image must be 2D or RGB.")
            return

        if image_data.ndim == 3 and image_data.shape[-1] > 3:
            image_data = image_data[..., :3]

        image_u8 = self._normalize_to_uint8(image_data)
        mask_shape = tuple(image_u8.shape[:2])

        try:
            instance_mask = self._build_instance_mask(shapes_layer, mask_shape)
        except Exception as exc:  # noqa: BLE001
            self._show_error(f"Could not create mask from shapes: {exc}")
            return

        stem = self._build_retrain_stem(
            source_name=image_layer.name,
            crop_bounds=None,
        )
        self._save_retrain_pair(image_u8, instance_mask, stem)

    def _get_current_crop_retrain_pair(self):
        crop_image_layer = self._get_layer_by_name(self.CROP_IMAGE_LAYER_NAME)
        crop_mask_layer = self._get_layer_by_name(self.CROP_MASK_LAYER_NAME)
        if crop_image_layer is None and crop_mask_layer is None:
            return None
        if crop_image_layer is None or crop_mask_layer is None:
            self._show_error("Both crop image and crop mask layers are required before adding a crop to retrain.")
            return None

        image_data = np.asarray(crop_image_layer.data)
        mask_data = np.asarray(crop_mask_layer.data)
        if image_data.ndim not in {2, 3}:
            self._show_error("Crop image must be 2D or RGB.")
            return None
        if image_data.ndim == 3 and image_data.shape[-1] > 3:
            image_data = image_data[..., :3]
        if mask_data.ndim == 3:
            mask_data = mask_data[..., 0]
        if mask_data.ndim != 2:
            self._show_error("Crop mask must be a 2D labels layer.")
            return None

        image_u8 = self._normalize_to_uint8(image_data)
        if image_u8.shape[:2] != mask_data.shape[:2]:
            self._show_error(
                "Crop image and crop mask shapes do not match: "
                f"image={image_u8.shape[:2]}, mask={mask_data.shape[:2]}."
            )
            return None

        crop_bounds = crop_image_layer.metadata.get(
            "crop_bounds_yx",
            crop_mask_layer.metadata.get("crop_bounds_yx"),
        )
        source_name = crop_image_layer.metadata.get(
            "source_image_layer_name",
            crop_image_layer.name,
        )
        stem = self._build_retrain_stem(
            source_name=source_name,
            crop_bounds=crop_bounds,
        )
        return image_u8, mask_data.astype(np.uint16, copy=False), stem

    def _build_retrain_stem(self, source_name: str, crop_bounds) -> str:
        parts = [self._sanitize_stem(source_name)]
        if crop_bounds is not None and len(crop_bounds) == 4:
            y0, y1, x0, x1 = [int(v) for v in crop_bounds]
            parts.append(f"y{y0}-{y1}_x{x0}-{x1}")
        parts.append(datetime.now().strftime("%y%m%d_%H%M%S_%f"))
        return "_".join(parts)

    def _save_retrain_pair(self, image_u8: np.ndarray, instance_mask: np.ndarray, stem: str) -> None:
        retrain_root = self._resolve_retrain_data_path(create=True)
        if retrain_root is None:
            return

        images_dir = retrain_root / "images"
        masks_dir = retrain_root / "masks"

        image_out = images_dir / f"{stem}.png"
        mask_out = masks_dir / f"{stem}.png"

        Image.fromarray(image_u8).save(image_out)
        Image.fromarray(instance_mask).save(mask_out)

        self.update_stats_label()
        self._show_info(
            "Saved current image/mask pair for retraining:\n"
            f"- Image: {image_out}\n"
            f"- Mask: {mask_out}"
        )

    def _on_load_model(self) -> None:
        model_input = self._model_path_input.text().strip()
        if not model_input:
            self._show_error("Model path cannot be empty. Select a .pt file or a folder.")
            return

        model_path_input = Path(model_input)
        if not model_path_input.exists():
            self._show_error("Model path does not exist.")
            return

        model_path: Path
        copied_default_model = False

        if model_path_input.is_file():
            if model_path_input.suffix.lower() != ".pt":
                self._show_error("Select a valid .pt model file or a folder.")
                return
            model_path = model_path_input
        elif model_path_input.is_dir():
            pt_files = sorted(model_path_input.glob("*.pt"))
            if pt_files:
                model_path = pt_files[0]
            else:
                default_model_source = Path(__file__).parent / "models" / "yolov8n.pt"
                if not default_model_source.exists():
                    self._show_error("No .pt found in selected folder, and bundled yolov8n.pt is missing.")
                    return

                model_path = model_path_input / "yolov8n.pt"
                shutil.copy2(default_model_source, model_path)
                copied_default_model = True
        else:
            self._show_error("Select a valid .pt model file or a folder.")
            return

        try:
            self._yolo = CCIYoloWrapper(str(model_path))
            self._model_path = model_path
            self._maybe_preselect_retrain_data_path()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - GUI runtime guard
            self._show_error(f"Could not load model: {exc}")
            return

        if copied_default_model:
            self._show_info(f"No .pt model was found in the folder. Copied bundled model to: {model_path}")

        model_task = self._yolo.model_task()
        self._show_info(f"Model loaded: {model_path.name} (task: {model_task})")

    def _get_active_image_layer(self):
        layer = self.napari_viewer.layers.selection.active
        if layer is not None and self._is_candidate_image_layer(layer):
            data = np.asarray(layer.data)
            if data.ndim >= 2:
                return layer

        for candidate in reversed(self.napari_viewer.layers):
            if not self._is_candidate_image_layer(candidate):
                continue
            data = np.asarray(candidate.data)
            if data.ndim >= 2:
                return candidate

        self._show_error("Open or select an image layer first.")
        return None

    def _is_candidate_image_layer(self, layer) -> bool:
        if self._is_shapes_like_layer(layer):
            return False
        if getattr(layer, "data", None) is None:
            return False
        name = getattr(layer, "name", None)
        if name is not None and name.startswith(f"{self.MERGED_LAYER_PREFIX}_"):
            return False
        return name not in {
            self.PRED_LAYER_NAME,
            self.BBOX_LAYER_NAME,
            self.CROP_SELECTION_LAYER_NAME,
            self.CROP_MASK_LAYER_NAME,
        }

    def _get_layer_by_name(self, name: str):
        for layer in self.napari_viewer.layers:
            if getattr(layer, "name", None) == name:
                return layer
        return None

    @staticmethod
    def _result_to_labels_mask(result, shape: tuple[int, int]) -> np.ndarray | None:
        if result is None or result.masks is None:
            return None

        masks_data = getattr(result.masks, "data", None)
        if masks_data is None:
            return None

        masks = masks_data.cpu().numpy()
        if len(masks) == 0:
            return np.zeros(shape, dtype=np.uint32)

        labels = np.zeros(shape, dtype=np.uint32)
        confidence_map = np.full(shape, -np.inf, dtype=float)
        confs = np.ones(len(masks), dtype=float)
        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None:
            confs = boxes.conf.cpu().numpy().astype(float)

        for mask_index, mask_data in enumerate(masks):
            mask_bool = mask_data > 0.5
            if mask_bool.shape != shape:
                mask_img = Image.fromarray(mask_bool.astype(np.uint8) * 255)
                nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
                mask_img = mask_img.resize((shape[1], shape[0]), resample=nearest)
                mask_bool = np.asarray(mask_img) > 0

            confidence = float(confs[mask_index]) if mask_index < len(confs) else 1.0
            update = mask_bool & (confidence >= confidence_map)
            labels[update] = mask_index + 1
            confidence_map[update] = confidence

        return labels

    @staticmethod
    def _label_mask_to_bbox_rectangles(label_mask: np.ndarray) -> list[np.ndarray]:
        rects = []
        for label_id in np.unique(label_mask):
            if label_id == 0:
                continue
            ys, xs = np.where(label_mask == label_id)
            if len(ys) == 0:
                continue
            y0 = int(ys.min())
            y1 = int(ys.max()) + 1
            x0 = int(xs.min())
            x1 = int(xs.max()) + 1
            rects.append(np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]], dtype=float))
        return rects

    def _add_bbox_layer(self, rects: list[np.ndarray]) -> None:
        existing = self._get_layer_by_name(self.BBOX_LAYER_NAME)
        if existing is not None:
            self.napari_viewer.layers.remove(existing)

        if not rects:
            return

        layer = self.napari_viewer.add_shapes(
            rects,
            name=self.BBOX_LAYER_NAME,
            shape_type="rectangle",
            edge_width=2,
            edge_color="cyan",
            face_color="transparent",
        )
        layer.metadata["prediction_role"] = "bbox_reference"

    def _on_predict(self) -> None:
        if self._yolo is None:
            self._show_error("Load a model first.")
            return

        image_layer = self._get_active_image_layer()
        if image_layer is None:
            return

        image_data = np.asarray(image_layer.data)
        if image_data.ndim < 2 or image_data.ndim > 3:
            self._show_error("Unsupported image shape.")
            return

        if image_data.ndim == 3 and image_data.shape[-1] not in {1, 3, 4}:
            self._show_error("Unsupported image shape.")
            return

        image_u8 = self._normalize_to_uint8(image_data)
        tiled_mask = None
        labels_mask = None
        bbox_rects = []
        tile_size = self.TILING_THRESHOLD
        overlap = config.OVERLAP
        if overlap * 2 >= tile_size:
            self._show_error("Tiling overlap must be smaller than half the tile size.")
            return

        try:
            height, width = image_u8.shape[:2]
            if height > tile_size or width > tile_size:
                if self._model_path is None:
                    self._show_error("Load a model first.")
                    return

                image_gray = self._to_grayscale(image_u8)
                tiled_mask = predict_segments_with_yolo_tiling(
                    image_data=image_gray,
                    model_path=str(self._model_path),
                    image_size=tile_size,
                    overlap=overlap,
                    iou=config.YOLO_IOU,
                )
                if self.show_bbox:
                    bbox_rects = self._label_mask_to_bbox_rectangles(tiled_mask)
                result = None
            else:
                image_rgb = self._to_pil_rgb(image_u8)
                prediction = self._yolo.predict(
                    image_rgb,
                    imgsz=tile_size,
                    retina_masks=True,
                    verbose=False,
                    iou=config.YOLO_IOU,
                )
                result = prediction[0] if len(prediction) else None
                labels_mask = self._result_to_labels_mask(result, image_u8.shape[:2])
                if self.show_bbox and labels_mask is not None:
                    bbox_rects = self._label_mask_to_bbox_rectangles(labels_mask)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - GUI runtime guard
            self._show_error(f"Prediction failed: {exc}")
            return

        existing = self._get_layer_by_name(self.PRED_LAYER_NAME)
        if existing is not None:
            self.napari_viewer.layers.remove(existing)
        self._add_bbox_layer(bbox_rects)

        if labels_mask is not None:
            pred_layer = self.napari_viewer.add_labels(
                labels_mask,
                name=self.PRED_LAYER_NAME,
            )
            pred_layer.metadata["prediction_mode"] = "single_image"
            pred_layer.metadata["yolo_iou_threshold"] = config.YOLO_IOU
            self._show_info(f"Prediction done: {int(labels_mask.max())} segment(s).")
            return

        if result is None and tiled_mask is not None:
            pred_layer = self.napari_viewer.add_labels(
                tiled_mask,
                name=self.PRED_LAYER_NAME,
            )
            pred_layer.metadata["yolo_iou_threshold"] = config.YOLO_IOU
            pred_layer.metadata["tile_size"] = tile_size
            pred_layer.metadata["overlap"] = overlap
            pred_layer.metadata["chunk_size"] = tile_size - (2 * overlap)
            self._add_tile_grid_layer(tiled_mask.shape, tile_size, overlap)
            self._show_info(
                "Large image prediction done. "
                "Use Merge Segments to merge labels across tile boundaries."
            )
            return

        self._show_info(f"Prediction done: no segmentation masks, {len(bbox_rects)} bbox fallback(s).")

    def _on_retrain(self) -> None:
        if self._yolo is None or self._model_path is None:
            self._show_error("Load a model first.")
            return

        loaded_task = self._yolo.model_task().lower()
        if loaded_task != "segment":
            self._show_info(
                "Loaded model is not a segmentation checkpoint "
                f"(task: {loaded_task}). Retraining will initialize a segmentation model automatically."
            )

        retrain_data_path = self._resolve_retrain_data_path(create=False)
        if retrain_data_path is None:
            self._show_error("Select a retrain folder containing images/ and masks/.")
            return

        images_dir = retrain_data_path / "images"
        masks_dir = retrain_data_path / "masks"
        if not images_dir.exists() or not masks_dir.exists():
            self._show_error("Retrain folder must contain images/ and masks/ subfolders.")
            return

        self._retrain_button.setEnabled(False)
        self._add_to_retrain_button.setEnabled(False)
        self._spinner_index = 0
        self._spinner_timer.start()

        self._retrain_worker = _RetrainWorker(
            model_path=self._model_path,
            retrain_data_path=self._retrain_data_path,
            parent=self,
        )
        self._retrain_worker.finished.connect(self._on_retrain_done)
        self._retrain_worker.failed.connect(self._on_retrain_error)
        self._retrain_worker.start()

    def _tick_spinner(self) -> None:
        self._retrain_button.setText(self._spinner_frames[self._spinner_index % len(self._spinner_frames)])
        self._spinner_index += 1

    def _on_retrain_done(self, message: str) -> None:
        self._spinner_timer.stop()
        self._retrain_button.setText("Retrain")
        self._retrain_button.setEnabled(True)
        self._add_to_retrain_button.setEnabled(True)
        self._show_info(message)

    def _on_retrain_error(self, message: str) -> None:
        self._spinner_timer.stop()
        self._retrain_button.setText("Retrain")
        self._retrain_button.setEnabled(True)
        self._add_to_retrain_button.setEnabled(True)
        self._show_error(message)
