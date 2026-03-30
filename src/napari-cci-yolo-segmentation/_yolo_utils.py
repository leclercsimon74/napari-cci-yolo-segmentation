import shutil
from pathlib import Path


def _points_to_yolo_xywh(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    """Convert normalized polygon points to normalized YOLO xywh box."""
    if not points:
        return None

    xs = [x for x, _ in points]
    ys = [y for _, y in points]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        return None

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    width = min(max(width, 0.0), 1.0)
    height = min(max(height, 0.0), 1.0)
    return x_center, y_center, width, height


def save_vectors_to_txt(vectors: list[tuple[int, list[tuple[float, float]]]], file_path: Path) -> None:
    """Save vectors to a text file in YOLO detection format: class x_center y_center width height."""
    with open(file_path, 'w') as f:
        for class_type, points in vectors:
            bbox = _points_to_yolo_xywh(points)
            if bbox is None:
                continue
            x_center, y_center, width, height = bbox
            f.write(f"{class_type} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def convert_txt_labels_to_yolo_xywh(path: Path) -> int:
    """Convert label txt files from polygon format to YOLO xywh format in-place.

    Returns:
        int: Number of files rewritten.
    """
    path = Path(path)
    txt_files = [path] if path.is_file() else sorted(path.rglob("*.txt"))
    converted = 0

    for txt_file in txt_files:
        if txt_file.name == "dataset.yaml":
            continue

        new_lines: list[str] = []
        has_polygon_line = False

        for raw_line in txt_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) == 5:
                new_lines.append(line)
                continue

            if len(parts) < 7 or len(parts[1:]) % 2 != 0:
                continue

            class_id = parts[0]
            coords = [float(v) for v in parts[1:]]
            points = list(zip(coords[0::2], coords[1::2]))
            bbox = _points_to_yolo_xywh(points)
            if bbox is None:
                continue

            has_polygon_line = True
            x_center, y_center, width, height = bbox
            new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if has_polygon_line:
            txt_file.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
            converted += 1

    return converted


def create_training_set(path_to_images: Path, path_to_vectors: Path, destination_path: Path, label_names: list[tuple[int, str]]) -> None:
    """Create a YOLO training set by copying images and annotations to the destination path.
    Args:
        path_to_images (Path): Path to the directory containing images.
        path_to_annotations (Path): Path to the directory containing annotation files.
        destination_path (Path): Path to the destination directory for the training set.
        label_names (list[tuple[int, str]]): List of tuples defining class indices and their names.
    """
    # Create destination directories
    images_dest_path = destination_path / Path("images")
    labels_dest_path = destination_path / Path("labels")

    if images_dest_path.exists():
        shutil.rmtree(images_dest_path)
    if labels_dest_path.exists():
        shutil.rmtree(labels_dest_path)

    images_dest_val = images_dest_path / Path("val")
    images_dest_train = images_dest_path / Path("train")

    labels_dest_val = labels_dest_path / Path("val")
    labels_dest_train = labels_dest_path / Path("train")

    images_dest_val.mkdir(parents=True, exist_ok=True)
    images_dest_train.mkdir(parents=True, exist_ok=True)
    labels_dest_val.mkdir(parents=True, exist_ok=True)
    labels_dest_train.mkdir(parents=True, exist_ok=True)

    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    vec_exts = {".txt"}

    images = sorted([p for p in path_to_images.iterdir() if p.suffix.lower() in img_exts])
    label_vectors = sorted([p for p in path_to_vectors.iterdir() if p.suffix.lower() in vec_exts])

    labels_by_stem = {}
    for label_path in label_vectors:
        label_stem = label_path.stem
        if label_stem.endswith("training"):
            label_stem = label_stem[:-len("training")]
        labels_by_stem[label_stem] = label_path

    for idx, img_path in enumerate(images):
        if idx % 2 == 0:
            image_destination = images_dest_train / img_path.name
            label_destination_dir = labels_dest_train
        else:
            image_destination = images_dest_val / img_path.name
            label_destination_dir = labels_dest_val

        shutil.copy2(img_path, image_destination)

        label_path = labels_by_stem.get(img_path.stem)
        if label_path is not None:
            shutil.copy2(label_path, label_destination_dir / f"{img_path.stem}.txt")

    dataset_file = destination_path / Path("dataset.yaml")
    with open(dataset_file, 'w') as dataset_file:
        dataset_file.write("train: ./images/train/\n")
        dataset_file.write("val: ./images/val/\n")
        dataset_file.write("names:\n")
        for class_index, class_name in label_names:
            dataset_file.write(f"  {class_index}: {class_name}\n")

        dataset_file.close()

class CCIYoloWrapper:

    def __init__(self, model_name_or_path: str = "yolov8n.pt"):
        self.model_name = ""
        self.res = None
        self.model = self._create_model(model_name_or_path)

    @staticmethod
    def _create_model(model_name_or_path):
        # Defer ultralytics/torch import so package import and pure helper tests
        # do not fail on systems without a working torch runtime.
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - runtime environment guard
            raise RuntimeError(
                "Failed to import ultralytics/torch. Install a compatible CPU build "
                "for this platform to run model inference or training."
            ) from exc
        return YOLO(model_name_or_path)

    # @classmethod
    # def load_model_by_name(cls, model_name: str, basedir: str = 'models'):
    #     return cls(yolomodel(None, name=model_name, basedir=basedir), model_name=model_name, basedir=basedir)

    # @classmethod
    # def new_model(cls, config=yolo.models.Config2D, model_name: str = "latest", basedir: str = 'models'):
    #     return cls(yolomodel(config, name=model_name, basedir=basedir), model_name=model_name, basedir=basedir)

    def load_model(self, weights_path: Path):
        self.model = self._create_model(weights_path)

    def predict(self, img):
        return self.model(img)

    def train(self, data_set_file: Path, image_size, batch=8, epochs=300, patience=100, ** kwargs):
        data_set_file = Path(data_set_file)
        if data_set_file.is_dir():
            data_set_file = data_set_file / "dataset.yaml"

        if "batch_size" in kwargs:
            batch_size = kwargs.pop("batch_size")
            if batch != 8 and batch != batch_size:
                raise ValueError("Pass either 'batch' or 'batch_size', not both with different values.")
            batch = batch_size

        self.res = self.model.train(data=data_set_file, batch=batch, imgsz=image_size, epochs=epochs, patience=patience, **kwargs)
        return self.res

    def get_number_of_run_epochs(self):
        trainer = getattr(self.model, "trainer", None)
        return getattr(trainer, "epoch", None)
