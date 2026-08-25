OVERLAP = 108  # pixels of overlap for tiling segmentation
TILE_SIZE = 1024  # pixels of tile size for tiling segmentation
YOLO_IOU = 0.5  # IOU for Yolo
YOLO_CONFIDENCE_THRESHOLD = 0.25  # minimum bbox confidence for Yolo predictions
SHOW_BBOX = True #show bbox napari layer with yolo score

def _torch_cuda_is_available() -> bool:
    try:
        import torch
    except Exception:
        return False

    return bool(torch.cuda.is_available())


USE_GPU = _torch_cuda_is_available()
YOLO_DEVICE = "cuda" if USE_GPU else "cpu"

