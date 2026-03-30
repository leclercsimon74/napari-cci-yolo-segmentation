import threading
from collections import defaultdict
from functools import partial
from threading import Lock
from timeit import default_timer as timer

import dask as da
import numpy as np
import skimage.color
import skimage.segmentation
from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks


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
    def __init__(self, model_path: str, image_size: int):
        self.model_mutex = threading.Lock()
        self.image_size = image_size
        self.int_gen = IntGenerator()
        self.model = YOLO(model_path, task="segment")

    def segment_wrapper(self, data, block_id):
        with self.model_mutex:
            rgb_data = skimage.color.gray2rgb(data)
            input_data = np.ascontiguousarray(rgb_data)
            result = self.model.predict(source=input_data, imgsz=self.image_size, verbose=False)

        all_masks = np.zeros(shape=data.shape, dtype=np.uint32)
        if result is None or result[0].masks is None:
            return all_masks

        result_masks = result[0].masks
        masks = result_masks.data

        if data.shape[0] != self.image_size or data.shape[1] != self.image_size:
            if masks.ndim == 2:
                masks = masks.unsqueeze(0).unsqueeze(0)
            elif masks.ndim == 3:
                masks = masks.unsqueeze(0)
            masks = scale_masks(masks, result_masks.orig_shape)
            masks = masks.squeeze(0)

        masks = masks.cpu().numpy()
        segments = result[0].masks.shape[0]

        sh1 = all_masks.shape[0]
        sh2 = all_masks.shape[1]

        for n in range(segments):
            mask = masks[n] * self.int_gen.get_next()
            all_masks[:sh1, :sh2] = np.where(
                all_masks[:sh1, :sh2] == 0,
                mask[:sh1, :sh2],
                all_masks[:sh1, :sh2],
            )

        return all_masks


class LargeImageYoloSegmenter:
    def __init__(self):
        self.table_of_ids = EquivalenceList()

    @staticmethod
    def calculate_chunk_size(image_size: int, overlap: int) -> int:
        return int(image_size - (2 * overlap))

    def _calculate_neighbour_equivalence_ids(self, data, block_id, img_size, scan_vertical, scan_far_side=False):
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

    def segment_large_image_data(self, yolo_segmenter: YoloSegmenter, image_data, overlap=100, clear_borders=False):
        large_image_tmp = da.array.from_array(image_data)
        height, width = large_image_tmp.shape[:2]
        chunk_size = self.calculate_chunk_size(yolo_segmenter.image_size, overlap)

        large_image = large_image_tmp.reshape((height, width)).rechunk((chunk_size, chunk_size))

        meta = np.empty((chunk_size, chunk_size), dtype=np.uint32)
        segment_results = da.array.map_overlap(
            yolo_segmenter.segment_wrapper,
            large_image,
            meta=meta,
            chunks=(chunk_size, chunk_size),
            depth=overlap,
            boundary="reflect",
            trim=True,
            allow_rechunk=True,
        )

        dep = 1
        merge_horizontal = partial(self._calculate_neighbour_equivalence_ids, img_size=chunk_size, scan_vertical=False)
        h1_result = segment_results.map_overlap(
            merge_horizontal,
            dtype=np.uint32,
            depth=dep,
            boundary="reflect",
            trim=True,
            allow_rechunk=True,
        )

        merge_vertical = partial(self._calculate_neighbour_equivalence_ids, img_size=chunk_size, scan_vertical=True)
        v1_result = h1_result.map_overlap(
            merge_vertical,
            dtype=np.uint32,
            depth=dep,
            boundary="reflect",
            trim=True,
            allow_rechunk=True,
        )

        res = v1_result.compute(scheduler="threads")
        self.table_of_ids.group_ids()

        final_dask = da.array.from_array(res).reshape((height, width)).rechunk((chunk_size, chunk_size))
        end_result = final_dask.map_blocks(self._find_and_change_ids_along_border, dtype=np.uint32)

        start = timer()
        result = end_result.compute()
        end = timer()
        print("segmentation+merge runtime (s):", end - start)

        if clear_borders:
            result = skimage.segmentation.clear_border(result)

        return result


def segment_with_yolo_tiling(image_data, model_path: str, image_size: int = 1024, overlap: int = 100, clear_borders: bool = False):
    yolo_segmenter = YoloSegmenter(model_path=model_path, image_size=image_size)
    segmenter = LargeImageYoloSegmenter()
    return segmenter.segment_large_image_data(
        yolo_segmenter=yolo_segmenter,
        image_data=image_data,
        overlap=overlap,
        clear_borders=clear_borders,
    )
