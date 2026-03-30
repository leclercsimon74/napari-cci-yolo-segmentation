#_gui.py

from __future__ import annotations

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

from ._segmentation_training import (
    CCIYoloWrapper,
    RetrainConfig,
    run_retraining_pipeline,
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
                output_root=None,
                config=RetrainConfig(
                    tile_size=1024,
                    val_ratio=0.2,
                    seed=42,
                    batch=4,
                    epochs=100,
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

    def __init__(self, napari_viewer):
        super().__init__()
        self.napari_viewer = napari_viewer
        self.setWindowTitle("Yolo CCI Segmentator")

        self._yolo: CCIYoloWrapper | None = None
        self._model_path: Path | None = None
        self._retrain_data_path: Path | None = None
        self._retrain_worker: _RetrainWorker | None = None

        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(400)
        self._spinner_timer.timeout.connect(self._tick_spinner)
        self._spinner_frames = ["Retraining .", "Retraining ..", "Retraining ...", "Retraining"]
        self._spinner_index = 0

        self._model_path_input = QLineEdit()
        self._model_path_input.setPlaceholderText("Path to YOLO model (.pt) or model folder")

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._on_browse_model)

        load_button = QPushButton("Load model")
        load_button.clicked.connect(self._on_load_model)

        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self._on_predict)

        self.retrain_data_path_input = QLineEdit()
        self.retrain_data_path_input.setPlaceholderText("Retrain folder containing images/ and masks/")

        browse_retrain_data_button = QPushButton("Browse")
        browse_retrain_data_button.clicked.connect(self._on_browse_retrain_data)

        self._add_to_retrain_button = QPushButton("Add to Retrain")
        self._add_to_retrain_button.clicked.connect(self._on_add_to_retrain)

        self._retrain_button = QPushButton("Retrain")
        self._retrain_button.clicked.connect(self._on_retrain)

        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("Model"))
        row_model.addWidget(self._model_path_input)
        row_model.addWidget(browse_button)

        row_actions = QHBoxLayout()
        row_actions.addWidget(load_button)
        row_actions.addWidget(predict_button)

        row_retrain_data = QHBoxLayout()
        row_retrain_data.addWidget(QLabel("Retrain data"))
        row_retrain_data.addWidget(self.retrain_data_path_input)
        row_retrain_data.addWidget(browse_retrain_data_button)

        row_train = QHBoxLayout()
        row_train.addWidget(self._add_to_retrain_button)
        row_train.addWidget(self._retrain_button)

        layout = QVBoxLayout()
        layout.addLayout(row_model)
        layout.addLayout(row_actions)
        layout.addLayout(row_retrain_data)
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
            self.retrain_data_path_input.setText(retrain_dir)

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
    def _is_shapes_like_layer(layer) -> bool:
        return hasattr(layer, "to_masks") and hasattr(layer, "shape_type")

    def _get_shapes_layer(self):
        active = self.napari_viewer.layers.selection.active
        if active is not None and self._is_shapes_like_layer(active):
            return active

        pred = self._get_layer_by_name(self.PRED_LAYER_NAME)
        if pred is not None and self._is_shapes_like_layer(pred):
            return pred

        for layer in self.napari_viewer.layers:
            if self._is_shapes_like_layer(layer):
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

        model_root = self._model_path.parent
        images_dir = model_root / "images"
        masks_dir = model_root / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        stem = self._sanitize_stem(f"{image_layer.name}_{datetime.now().strftime('%y%m%d_%H%M%S_%f')}")
        image_out = images_dir / f"{stem}.png"
        mask_out = masks_dir / f"{stem}.png"

        Image.fromarray(image_u8).save(image_out)
        Image.fromarray(instance_mask).save(mask_out)

        self.retrain_data_path_input.setText(str(model_root))
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
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - GUI runtime guard
            self._show_error(f"Could not load model: {exc}")
            return

        if copied_default_model:
            self._show_info(f"No .pt model was found in the folder. Copied bundled model to: {model_path}")

        self._show_info(f"Model loaded: {model_path.name}")

    def _get_active_image_layer(self):
        layer = self.napari_viewer.layers.selection.active
        if layer is not None and not self._is_shapes_like_layer(layer) and getattr(layer, "data", None) is not None:
            data = np.asarray(layer.data)
            if data.ndim >= 2:
                return layer

        for candidate in reversed(self.napari_viewer.layers):
            if self._is_shapes_like_layer(candidate):
                continue
            if getattr(candidate, "data", None) is None:
                continue
            data = np.asarray(candidate.data)
            if data.ndim >= 2:
                return candidate

        self._show_error("Open or select an image layer first.")
        return None

    def _get_layer_by_name(self, name: str):
        for layer in self.napari_viewer.layers:
            if getattr(layer, "name", None) == name:
                return layer
        return None

    def _on_predict(self) -> None:
        if self._yolo is None:
            self._show_error("Load a model first.")
            return

        image_layer = self._get_active_image_layer()
        if image_layer is None:
            return

        image_data = np.asarray(image_layer.data)
        if image_data.ndim < 2:
            self._show_error("Unsupported image shape.")
            return

        try:
            prediction = self._yolo.predict(image_data)
            result = prediction[0] if len(prediction) else None
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - GUI runtime guard
            self._show_error(f"Prediction failed: {exc}")
            return

        polygons = []
        if result is not None and result.masks is not None and getattr(result.masks, "xy", None) is not None:
            for points in result.masks.xy:
                if len(points) < 3:
                    continue
                poly_xy = np.asarray(points, dtype=float)
                polygons.append(np.column_stack((poly_xy[:, 1], poly_xy[:, 0])))

        rects = []
        if not polygons and result is not None and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            for x1, y1, x2, y2 in boxes:
                rects.append(np.array([[y1, x1], [y1, x2], [y2, x2], [y2, x1]], dtype=float))

        existing = self._get_layer_by_name(self.PRED_LAYER_NAME)
        if existing is not None:
            self.napari_viewer.layers.remove(existing)

        if polygons:
            self.napari_viewer.add_shapes(
                polygons,
                name=self.PRED_LAYER_NAME,
                shape_type="polygon",
                edge_width=2,
                edge_color="yellow",
                face_color="transparent",
            )
            self._show_info(f"Prediction done: {len(polygons)} segment(s).")
            return

        self.napari_viewer.add_shapes(
            rects,
            name=self.PRED_LAYER_NAME,
            shape_type="rectangle",
            edge_width=2,
            edge_color="yellow",
            face_color="transparent",
        )
        self._show_info(f"Prediction done: {len(rects)} bbox fallback(s).")

    def _on_retrain(self) -> None:
        if self._yolo is None or self._model_path is None:
            self._show_error("Load a model first.")
            return

        retrain_data_text = self.retrain_data_path_input.text().strip()
        if not retrain_data_text:
            self._show_error("Select a retrain folder containing images/ and masks/.")
            return

        self._retrain_data_path = Path(retrain_data_text)
        if not self._retrain_data_path.exists():
            self._show_error("Retrain folder does not exist.")
            return

        images_dir = self._retrain_data_path / "images"
        masks_dir = self._retrain_data_path / "masks"
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
