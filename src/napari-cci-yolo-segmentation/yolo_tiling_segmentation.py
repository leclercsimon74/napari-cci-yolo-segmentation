import threading
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from threading import Lock
from timeit import default_timer as timer

import dask.array as da
import numpy as np
import skimage.color
import skimage.segmentation

YOLO_IOU = 0.4
YOLO_MASK_SIDE_PADDING = 10
YOLO_MASK_SIDE_HIT_MARGIN = 2
YOLO_MASK_PADDING = YOLO_MASK_SIDE_PADDING
MERGE_IOU_THRESHOLD = 0.4


def create_padded_segmentation_predictor(
    mask_padding: int | float = YOLO_MASK_PADDING,
    hit_margin: int = YOLO_MASK_SIDE_HIT_MARGIN,
):
    try:
        from ultralytics.engine.results import Results
        from ultralytics.models.yolo.segment import SegmentationPredictor
        from ultralytics.utils import ops
    except ModuleNotFoundError as exc:
        if exc.name == "ultralytics":
            raise RuntimeError(
                "Ultralytics is not installed in the Python environment "
                "running napari."
            ) from exc
        raise

    class PaddedSegmentationPredictor(SegmentationPredictor):
        @staticmethod
        def _expand_boxes_on_mask_hits(masks, boxes, image_shape):
            height, width = image_shape
            side_padding = int(round(mask_padding))
            margin = max(1, int(hit_margin))
            if side_padding <= 0:
                return boxes

            expanded = boxes.clone()
            for mask_index in range(masks.shape[0]):
                mask = masks[mask_index].bool()
                x0, y0, x1, y1 = boxes[mask_index]
                ix0 = max(0, min(width, int(x0.floor().item())))
                iy0 = max(0, min(height, int(y0.floor().item())))
                ix1 = max(0, min(width, int(x1.ceil().item())))
                iy1 = max(0, min(height, int(y1.ceil().item())))
                if ix1 <= ix0 or iy1 <= iy0:
                    continue

                left_x1 = min(ix1, ix0 + margin)
                right_x0 = max(ix0, ix1 - margin)
                top_y1 = min(iy1, iy0 + margin)
                bottom_y0 = max(iy0, iy1 - margin)

                if mask[iy0:iy1, ix0:left_x1].any().item():
                    expanded[mask_index, 0] -= side_padding
                if mask[iy0:iy1, right_x0:ix1].any().item():
                    expanded[mask_index, 2] += side_padding
                if mask[iy0:top_y1, ix0:ix1].any().item():
                    expanded[mask_index, 1] -= side_padding
                if mask[bottom_y0:iy1, ix0:ix1].any().item():
                    expanded[mask_index, 3] += side_padding

            expanded[:, 0].clamp_(0, width)
            expanded[:, 2].clamp_(0, width)
            expanded[:, 1].clamp_(0, height)
            expanded[:, 3].clamp_(0, height)
            return expanded

        def construct_result(self, pred, img, orig_img, img_path, proto):
            if pred.shape[0] == 0:
                masks = None
            elif self.args.retina_masks:
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:],
                    pred[:, :4],
                    orig_img.shape,
                )

                mask_boxes = pred[:, :4]
                masks = ops.process_mask_native(
                    proto,
                    pred[:, 6:],
                    mask_boxes,
                    orig_img.shape[:2],
                )
                expanded_boxes = self._expand_boxes_on_mask_hits(
                    masks,
                    mask_boxes,
                    orig_img.shape[:2],
                )
                if not expanded_boxes.equal(mask_boxes):
                    masks = ops.process_mask_native(
                        proto,
                        pred[:, 6:],
                        expanded_boxes,
                        orig_img.shape[:2],
                    )
            else:
                masks = ops.process_mask(
                    proto,
                    pred[:, 6:],
                    pred[:, :4],
                    img.shape[2:],
                    upsample=True,
                )

                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:],
                    pred[:, :4],
                    orig_img.shape,
                )

            return Results(
                orig_img,
                path=img_path,
                names=self.model.names,
                boxes=pred[:, :6],
                masks=masks,
            )

    return PaddedSegmentationPredictor


def _normalize_to_uint8(image_data):
    image = np.asarray(image_data)
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


@dataclass
class TileInstance:
    label_id: int
    confidence: float
    tile_origin: tuple[int, int]
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    detection_bbox: tuple[int, int, int, int] | None = None


class IntGenerator:
    def __init__(self, start_value: int = 100):
        self.lock = threading.Lock()
        self.cnt = 0
        self.start_value = start_value

    def get_next(self) -> int:
        with self.lock:
            self.cnt += 1
            return self.start_value + self.cnt


class EquivalenceList:
    def __init__(self):
        self._the_list = []
        self._mutex = Lock()
        self.group_id_map = {}

    def add_equivalence_pair(self, id1: int, id2: int):
        with self._mutex:
            self._the_list.append((id1, id2))

    def get_equivalent_id(self, idx: int) -> int:
        with self._mutex:
            return self.group_id_map.get(idx, idx)

    def group_ids(self):
        with self._mutex:
            parent = {}
            rank = {}

            def find(x):
                if x not in parent:
                    parent[x] = x
                    rank[x] = 1
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                x_root = find(x)
                y_root = find(y)
                if x_root == y_root:
                    return
                if rank[x_root] < rank[y_root]:
                    parent[x_root] = y_root
                else:
                    parent[y_root] = x_root
                    if rank[x_root] == rank[y_root]:
                        rank[x_root] += 1

            for a, b in self._the_list:
                union(a, b)

            all_ids = set()
            for a, b in self._the_list:
                all_ids.add(a)
                all_ids.add(b)

            groups = {}
            for idx in all_ids:
                root = find(idx)
                if root not in groups:
                    groups[root] = []
                groups[root].append(idx)

            for _, members in groups.items():
                group_id = min(members)
                for member in members:
                    self.group_id_map[member] = group_id


class YoloSegmenter:
    def __init__(
        self,
        model_path: str,
        image_size: int,
        iou: float = YOLO_IOU,
        mask_padding: float = YOLO_MASK_PADDING,
    ):
        self.model_mutex = threading.Lock()
        self.image_size = image_size
        self.iou = iou
        self.predictor_cls = create_padded_segmentation_predictor(mask_padding)
        self.int_gen = IntGenerator()
        self.model = self._create_model(model_path)

    @staticmethod
    def _create_model(model_path: str):
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            if exc.name == "ultralytics":
                raise RuntimeError(
                    "Ultralytics is not installed in the Python environment "
                    "running napari."
                ) from exc
            raise

        return YOLO(model_path, task="segment")

    def segment_wrapper(self, data, block_id=None):
        with self.model_mutex:
            rgb_data = skimage.color.gray2rgb(data)
            input_data = np.ascontiguousarray(rgb_data)
            result = self.model.predict(
                source=input_data,
                imgsz=self.image_size,
                retina_masks=True,
                verbose=False,
                iou=self.iou,
                predictor=self.predictor_cls,
            )

        all_masks = np.zeros(shape=data.shape, dtype=np.uint32)
        if result is None or result[0].masks is None:
            return all_masks

        result_masks = result[0].masks
        masks = result_masks.data.cpu().numpy()
        segments = result[0].masks.shape[0]

        sh1 = all_masks.shape[0]
        sh2 = all_masks.shape[1]

        for n in range(segments):
            mask = masks[n].astype(np.uint32) * self.int_gen.get_next()
            all_masks[:sh1, :sh2] = np.where(
                all_masks[:sh1, :sh2] == 0,
                mask[:sh1, :sh2],
                all_masks[:sh1, :sh2],
            )

        return all_masks

    def predict_tile_instances(
        self,
        data,
        global_y0: int,
        global_x0: int,
        output_shape: tuple[int, int],
    ) -> list[TileInstance]:
        with self.model_mutex:
            rgb_data = skimage.color.gray2rgb(data)
            input_data = np.ascontiguousarray(rgb_data)
            result = self.model.predict(
                source=input_data,
                imgsz=self.image_size,
                retina_masks=True,
                verbose=False,
                iou=self.iou,
                predictor=self.predictor_cls,
            )

        if result is None or result[0].masks is None:
            return []

        result0 = result[0]
        masks = result0.masks.data.cpu().numpy()
        confs = np.ones(len(masks), dtype=float)
        detection_boxes = None
        boxes = getattr(result0, "boxes", None)
        if boxes is not None and getattr(boxes, "conf", None) is not None:
            confs = boxes.conf.cpu().numpy().astype(float)
        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            detection_boxes = boxes.xyxy.cpu().numpy()

        instances = []
        output_height, output_width = output_shape
        for mask_index, mask_data in enumerate(masks):
            mask_bool = mask_data > 0.5
            ys, xs = np.where(mask_bool)
            if len(ys) == 0:
                continue

            global_ys = ys + global_y0
            global_xs = xs + global_x0
            valid = (
                (global_ys >= 0)
                & (global_ys < output_height)
                & (global_xs >= 0)
                & (global_xs < output_width)
            )
            if not np.any(valid):
                continue

            global_ys = global_ys[valid]
            global_xs = global_xs[valid]
            y0 = int(global_ys.min())
            y1 = int(global_ys.max()) + 1
            x0 = int(global_xs.min())
            x1 = int(global_xs.max()) + 1

            global_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
            global_mask[global_ys - y0, global_xs - x0] = True
            confidence = float(confs[mask_index]) if mask_index < len(confs) else 1.0
            detection_bbox = None
            if detection_boxes is not None and mask_index < len(detection_boxes):
                box_x0, box_y0, box_x1, box_y1 = detection_boxes[mask_index]
                detection_bbox = (
                    max(0, int(round(box_y0 + global_y0))),
                    min(output_height, int(round(box_y1 + global_y0))),
                    max(0, int(round(box_x0 + global_x0))),
                    min(output_width, int(round(box_x1 + global_x0))),
                )
            instances.append(
                TileInstance(
                    label_id=self.int_gen.get_next(),
                    confidence=confidence,
                    tile_origin=(global_y0, global_x0),
                    bbox=(y0, y1, x0, x1),
                    mask=global_mask,
                    detection_bbox=detection_bbox,
                )
            )

        return instances


class LargeImageYoloSegmenter:
    def __init__(self):
        self.table_of_ids = EquivalenceList()

    @staticmethod
    def calculate_chunk_size(image_size: int, overlap: int) -> int:
        chunk_size = int(image_size - (2 * overlap))
        if chunk_size <= 0:
            raise ValueError("overlap must be smaller than half of image_size")
        return chunk_size

    def _calculate_neighbour_equivalence_ids(
        self,
        data,
        block_id=None,
        img_size=None,
        scan_vertical=False,
        scan_far_side=False,
    ):
        x = 1 if not scan_far_side else data.shape[0]
        y = 1 if not scan_far_side else data.shape[1]
        neighbour_mod = -1 if not scan_far_side else 1

        if scan_vertical:
            neighbour_coords_mod = (neighbour_mod, 0)
            scan_size = data.shape[1]
        else:
            neighbour_coords_mod = (0, neighbour_mod)
            scan_size = data.shape[0]

        connected_table = defaultdict(lambda: defaultdict(int))
        max_neighbour_local_table = defaultdict(lambda: defaultdict(int))

        for coord in range(scan_size):
            if scan_vertical:
                y = coord
            else:
                x = coord

            local_indices = (x, y)
            neighbour_indices = (x + neighbour_coords_mod[0], y + neighbour_coords_mod[1])
            id_local = data[local_indices]
            id_neighbour = data[neighbour_indices]

            if id_local != 0 and id_neighbour != 0 and id_neighbour != id_local:
                connected_table[id_local][id_neighbour] += 1
                max_neighbour_local_table[id_neighbour][id_local] += 1

        neighbour_max = {}
        for outer_key in max_neighbour_local_table:
            id_map = max_neighbour_local_table[outer_key]
            id_map = {k: v for k, v in id_map.items() if v == max(id_map.values())}
            neighbour_max[outer_key] = list(id_map.keys())[0]

        for neighbour_id in neighbour_max:
            local_winner = neighbour_max[neighbour_id]
            for local_id in connected_table:
                if local_id != local_winner:
                    connected_table[local_id][neighbour_id] = 0

        filtered = []
        for idx, outer_key in enumerate(connected_table):
            max_cnt = 0
            filtered.append((outer_key, 0))
            for inner_key in connected_table[outer_key]:
                value = connected_table[outer_key][inner_key]
                if value > max_cnt:
                    max_cnt = value
                    filtered[idx] = (outer_key, inner_key)

        for local_id, neighbour_id in filtered:
            if neighbour_id != 0:
                self.table_of_ids.add_equivalence_pair(local_id, neighbour_id)

        return data

    def _find_and_change_ids_along_border(self, data, block_info=None):
        d1 = data.shape[0]
        d2 = data.shape[1]
        id_set = set()

        for y in [0, d2 - 1]:
            for x in range(d1):
                id_local = data[(x, y)]
                if id_local != 0:
                    id_set.add(id_local)

        for x in [0, d1 - 1]:
            for y in range(d2):
                id_local = data[(x, y)]
                if id_local != 0:
                    id_set.add(id_local)

        for idx in id_set:
            eq_id = self.table_of_ids.get_equivalent_id(idx)
            if idx != eq_id:
                positions = np.where(data == idx)
                data[positions] = eq_id

        return data

    @staticmethod
    def _pad_to_chunk_grid(image_data, chunk_size: int, mode: str = "reflect"):
        height, width = image_data.shape[:2]
        padded_height = int(np.ceil(height / chunk_size) * chunk_size)
        padded_width = int(np.ceil(width / chunk_size) * chunk_size)
        pad_height = padded_height - height
        pad_width = padded_width - width

        if pad_height == 0 and pad_width == 0:
            return image_data, height, width

        if mode == "constant":
            padded = np.pad(
                image_data,
                ((0, pad_height), (0, pad_width)),
                mode="constant",
                constant_values=0,
            )
        else:
            pad_mode = "reflect" if height > 1 and width > 1 else "edge"
            padded = np.pad(
                image_data,
                ((0, pad_height), (0, pad_width)),
                mode=pad_mode,
            )
        return padded, height, width

    def predict_segments(self, yolo_segmenter: YoloSegmenter, image_data, overlap=100):
        chunk_size = self.calculate_chunk_size(yolo_segmenter.image_size, overlap)
        padded_image, original_height, original_width = self._pad_to_chunk_grid(image_data, chunk_size)
        large_image_tmp = da.from_array(padded_image)
        height, width = large_image_tmp.shape[:2]

        large_image = large_image_tmp.reshape((height, width)).rechunk((chunk_size, chunk_size))

        meta = np.empty((chunk_size, chunk_size), dtype=np.uint32)
        segment_results = da.map_overlap(
            yolo_segmenter.segment_wrapper,
            large_image,
            meta=meta,
            chunks=(chunk_size, chunk_size),
            depth=overlap,
            boundary="reflect",
            trim=True,
            allow_rechunk=True,
        )

        result = segment_results.compute(scheduler="threads")
        return result[:original_height, :original_width]

    def predict_instances(self, yolo_segmenter: YoloSegmenter, image_data, overlap=100):
        chunk_size = self.calculate_chunk_size(yolo_segmenter.image_size, overlap)
        padded_image, original_height, original_width = self._pad_to_chunk_grid(image_data, chunk_size)
        bordered_image = np.pad(
            padded_image,
            ((overlap, overlap), (overlap, overlap)),
            mode="reflect" if min(padded_image.shape[:2]) > 1 else "edge",
        )

        instances = []
        for y0 in range(0, padded_image.shape[0], chunk_size):
            for x0 in range(0, padded_image.shape[1], chunk_size):
                tile = bordered_image[
                    y0 : y0 + yolo_segmenter.image_size,
                    x0 : x0 + yolo_segmenter.image_size,
                ]
                instances.extend(
                    yolo_segmenter.predict_tile_instances(
                        data=tile,
                        global_y0=y0 - overlap,
                        global_x0=x0 - overlap,
                        output_shape=(original_height, original_width),
                    )
                )

        return instances

    @staticmethod
    def render_instances(
        instances: list[TileInstance],
        shape: tuple[int, int],
        label_id_map=None,
    ) -> np.ndarray:
        labels = np.zeros(shape, dtype=np.uint32)
        confidence_map = np.full(shape, -np.inf, dtype=float)
        for instance in instances:
            y0, y1, x0, x1 = instance.bbox
            region_conf = confidence_map[y0:y1, x0:x1]
            region_labels = labels[y0:y1, x0:x1]
            update = instance.mask & (instance.confidence >= region_conf)
            label_id = (
                label_id_map(instance.label_id)
                if label_id_map is not None
                else instance.label_id
            )
            region_labels[update] = label_id
            region_conf[update] = instance.confidence
        return labels

    @staticmethod
    def render_instances_central_regions(
        instances: list[TileInstance],
        shape: tuple[int, int],
        image_size: int = 1024,
        overlap: int = 100,
    ) -> np.ndarray:
        labels = np.zeros(shape, dtype=np.uint32)
        confidence_map = np.full(shape, -np.inf, dtype=float)
        chunk_size = LargeImageYoloSegmenter.calculate_chunk_size(image_size, overlap)

        for instance in instances:
            central_y0 = instance.tile_origin[0] + overlap
            central_x0 = instance.tile_origin[1] + overlap
            central_y1 = central_y0 + chunk_size
            central_x1 = central_x0 + chunk_size

            y0 = max(instance.bbox[0], central_y0, 0)
            y1 = min(instance.bbox[1], central_y1, shape[0])
            x0 = max(instance.bbox[2], central_x0, 0)
            x1 = min(instance.bbox[3], central_x1, shape[1])
            if y1 <= y0 or x1 <= x0:
                continue

            mask_crop = instance.mask[
                y0 - instance.bbox[0] : y1 - instance.bbox[0],
                x0 - instance.bbox[2] : x1 - instance.bbox[2],
            ]
            region_conf = confidence_map[y0:y1, x0:x1]
            region_labels = labels[y0:y1, x0:x1]
            update = mask_crop & (instance.confidence >= region_conf)
            region_labels[update] = instance.label_id
            region_conf[update] = instance.confidence

        return labels

    def merge_segments_one_pixel_boundary(
        self,
        segment_results,
        image_size: int = 1024,
        overlap: int = 100,
        clear_borders=False,
    ):
        self.table_of_ids = EquivalenceList()

        equivalences, padded_segments, original_height, original_width = (
            self._build_one_pixel_boundary_equivalences(
                segment_results=segment_results,
                image_size=image_size,
                overlap=overlap,
            )
        )

        height, width = padded_segments.shape[:2]
        final_dask = da.from_array(padded_segments).reshape((height, width)).rechunk(
            (self.calculate_chunk_size(image_size, overlap), self.calculate_chunk_size(image_size, overlap))
        )
        self.table_of_ids = equivalences
        end_result = final_dask.map_blocks(self._find_and_change_ids_along_border, dtype=np.uint32)

        start = timer()
        result = end_result.compute()
        end = timer()
        print("segmentation+merge runtime (s):", end - start)

        if clear_borders:
            result = skimage.segmentation.clear_border(result)

        return result[:original_height, :original_width]

    def _build_one_pixel_boundary_equivalences(
        self,
        segment_results,
        image_size: int = 1024,
        overlap: int = 100,
    ):
        self.table_of_ids = EquivalenceList()
        chunk_size = self.calculate_chunk_size(image_size, overlap)
        padded_segments, original_height, original_width = self._pad_to_chunk_grid(
            segment_results,
            chunk_size,
            mode="constant",
        )
        height, width = padded_segments.shape[:2]
        segments = da.from_array(padded_segments).reshape((height, width)).rechunk((chunk_size, chunk_size))

        dep = 1
        merge_horizontal = partial(self._calculate_neighbour_equivalence_ids, img_size=chunk_size, scan_vertical=False)
        h1_result = segments.map_overlap(
            merge_horizontal,
            dtype=np.uint32,
            depth=dep,
            boundary=0,
            trim=True,
            allow_rechunk=True,
        )

        merge_vertical = partial(self._calculate_neighbour_equivalence_ids, img_size=chunk_size, scan_vertical=True)
        v1_result = h1_result.map_overlap(
            merge_vertical,
            dtype=np.uint32,
            depth=dep,
            boundary=0,
            trim=True,
            allow_rechunk=True,
        )

        res = v1_result.compute(scheduler="threads")
        self.table_of_ids.group_ids()
        return self.table_of_ids, res, original_height, original_width

    def merge_segments_iou(
        self,
        instances: list[TileInstance],
        shape: tuple[int, int],
        iou_threshold: float = MERGE_IOU_THRESHOLD,
        clear_borders=False,
    ) -> np.ndarray:
        equivalences = EquivalenceList()

        for idx, first in enumerate(instances):
            for second in instances[idx + 1 :]:
                if first.tile_origin == second.tile_origin:
                    continue
                iou = _instance_iou(first, second)
                if iou >= iou_threshold:
                    equivalences.add_equivalence_pair(first.label_id, second.label_id)

        equivalences.group_ids()
        merged = self.render_instances(
            instances,
            shape,
            label_id_map=equivalences.get_equivalent_id,
        )
        if clear_borders:
            merged = skimage.segmentation.clear_border(merged)
        return merged

    def segment_large_image_data(self, yolo_segmenter: YoloSegmenter, image_data, overlap=100, clear_borders=False):
        segment_results = self.predict_segments(
            yolo_segmenter=yolo_segmenter,
            image_data=image_data,
            overlap=overlap,
        )
        return self.merge_segments_one_pixel_boundary(
            segment_results=segment_results,
            image_size=yolo_segmenter.image_size,
            overlap=overlap,
            clear_borders=clear_borders,
        )


LargeImageYoloSegmentator = LargeImageYoloSegmenter


def segment_with_yolo_tiling(
    image_data,
    model_path: str,
    image_size: int = 1024,
    overlap: int = 100,
    clear_borders: bool = False,
    iou: float = YOLO_IOU,
    mask_padding: float = YOLO_MASK_PADDING,
):
    yolo_segmenter = YoloSegmenter(
        model_path=model_path,
        image_size=image_size,
        iou=iou,
        mask_padding=mask_padding,
    )
    segmenter = LargeImageYoloSegmenter()
    return segmenter.segment_large_image_data(
        yolo_segmenter=yolo_segmenter,
        image_data=_normalize_to_uint8(image_data),
        overlap=overlap,
        clear_borders=clear_borders,
    )


def predict_segments_with_yolo_tiling(
    image_data,
    model_path: str,
    image_size: int = 1024,
    overlap: int = 100,
    iou: float = YOLO_IOU,
    mask_padding: float = YOLO_MASK_PADDING,
):
    yolo_segmenter = YoloSegmenter(
        model_path=model_path,
        image_size=image_size,
        iou=iou,
        mask_padding=mask_padding,
    )
    segmenter = LargeImageYoloSegmenter()
    return segmenter.predict_segments(
        yolo_segmenter=yolo_segmenter,
        image_data=_normalize_to_uint8(image_data),
        overlap=overlap,
    )


def predict_instances_with_yolo_tiling(
    image_data,
    model_path: str,
    image_size: int = 1024,
    overlap: int = 100,
    iou: float = YOLO_IOU,
    mask_padding: float = YOLO_MASK_PADDING,
):
    image_data = _normalize_to_uint8(image_data)
    yolo_segmenter = YoloSegmenter(
        model_path=model_path,
        image_size=image_size,
        iou=iou,
        mask_padding=mask_padding,
    )
    segmenter = LargeImageYoloSegmenter()
    instances = segmenter.predict_instances(
        yolo_segmenter=yolo_segmenter,
        image_data=image_data,
        overlap=overlap,
    )
    labels = segmenter.render_instances(instances, image_data.shape[:2])
    return labels, instances


def merge_segments_one_pixel_boundary(segment_results, image_size: int = 1024, overlap: int = 100, clear_borders: bool = False):
    segmenter = LargeImageYoloSegmenter()
    return segmenter.merge_segments_one_pixel_boundary(
        segment_results=segment_results,
        image_size=image_size,
        overlap=overlap,
        clear_borders=clear_borders,
    )


def merge_tile_instances_one_pixel_boundary(
    instances: list[TileInstance],
    shape: tuple[int, int],
    image_size: int = 1024,
    overlap: int = 100,
    clear_borders: bool = False,
):
    segmenter = LargeImageYoloSegmenter()
    central_labels = segmenter.render_instances_central_regions(
        instances=instances,
        shape=shape,
        image_size=image_size,
        overlap=overlap,
    )
    equivalences, _, _, _ = segmenter._build_one_pixel_boundary_equivalences(
        segment_results=central_labels,
        image_size=image_size,
        overlap=overlap,
    )
    merged = segmenter.render_instances(
        instances,
        shape,
        label_id_map=equivalences.get_equivalent_id,
    )
    if clear_borders:
        merged = skimage.segmentation.clear_border(merged)
    return merged


def merge_segments_iou(
    instances: list[TileInstance],
    shape: tuple[int, int],
    iou_threshold: float = MERGE_IOU_THRESHOLD,
    clear_borders: bool = False,
):
    segmenter = LargeImageYoloSegmenter()
    return segmenter.merge_segments_iou(
        instances=instances,
        shape=shape,
        iou_threshold=iou_threshold,
        clear_borders=clear_borders,
    )


def _instance_iou(first: TileInstance, second: TileInstance) -> float:
    y0 = max(first.bbox[0], second.bbox[0])
    y1 = min(first.bbox[1], second.bbox[1])
    x0 = max(first.bbox[2], second.bbox[2])
    x1 = min(first.bbox[3], second.bbox[3])
    if y1 <= y0 or x1 <= x0:
        return 0.0

    first_crop = first.mask[
        y0 - first.bbox[0] : y1 - first.bbox[0],
        x0 - first.bbox[2] : x1 - first.bbox[2],
    ]
    second_crop = second.mask[
        y0 - second.bbox[0] : y1 - second.bbox[0],
        x0 - second.bbox[2] : x1 - second.bbox[2],
    ]
    intersection = int(np.count_nonzero(first_crop & second_crop))
    if intersection == 0:
        return 0.0

    union = int(np.count_nonzero(first.mask)) + int(np.count_nonzero(second.mask)) - intersection
    if union == 0:
        return 0.0
    return intersection / union
