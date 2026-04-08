
[![License MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/CCI-GU-Sweden/simple-napari-cci-annotator/blob/main/LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)](https://python.org)
[![tests](https://github.com/CCI-GU-Sweden/simple-napari-cci-annotator/workflows/tests/badge.svg)](https://github.com/CCI-GU-Sweden/simple-napari-cci-annotator/actions)

# napari_cci_yolo_segmentation

Minimal napari plugin for YOLO segmentation workflows:

- Predict on the currently selected image layer.
- Retrain a segmentation model from image/mask folders.

## Installation

To install latest development version :

```shell
pip install git+https://github.com/CCI-GU-Sweden/napari-cci-yolo-segmentation.git
```

— or, during development —

```shell
pip install -e .
```

### Recommended Windows Setup (Conda)

`environment.yml` provides the base environment (Python, numpy, napari, etc.)
without torch — torch is installed separately so you can choose CPU or GPU.

#### GPU (NVIDIA CUDA) — recommended if you have an NVIDIA GPU

Check your maximum supported CUDA version first:
```shell
nvidia-smi
```
Then pick the matching wheel index below (`cu124` = CUDA 12.4, `cu126` = 12.6, etc.)
and run:

```shell
conda env create -f environment.yml
conda activate yolo_segmentation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics
pip install -e .
```

Verify GPU is visible:
```shell
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```

#### CPU only

```shell
conda env create -f environment.yml
conda activate yolo_segmentation
pip install torch torchvision
pip install -e . --no-deps
```

Quick smoke test:

```shell
python -c "import numpy, scipy, torch, cv2; from ultralytics import YOLO; print(numpy.__version__, scipy.__version__, torch.__version__, cv2.__version__)"
```

## Retrain Data Format

The retrain folder must contain two subfolders:

- `images/`
- `masks/`

Each image must have a mask with the same stem (same filename without extension).

Example:

- `images/sample_001.png`
- `masks/sample_001.png`

## Retrain Preprocessing

When clicking `Retrain`, the plugin will:

1. Pair images and masks by stem.
2. Split pairs into train/val (80/20, deterministic seed).
3. Tile image/mask pairs into `1024x1024` patches with no resize.
4. Convert mask regions to YOLO segmentation polygons.
5. Build a YOLO dataset and start training.

Training outputs are stored in a retrained run folder with:

- `dataset/`
- `training_traces/`
- `best.pt`
