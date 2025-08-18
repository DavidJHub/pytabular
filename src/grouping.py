"""Utilities to group detected text boxes into table rows and columns."""

from typing import Dict, List, Tuple

import numpy as np


def merge_intersecting_boxes(
    boxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.0
) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    boxes_np = np.array(boxes, dtype=np.int32)
    merged = True
    while merged:
        merged = False
        new_boxes = []
        used = np.zeros(len(boxes_np), dtype=bool)
        for i in range(len(boxes_np)):
            if used[i]:
                continue
            x1a, y1a, x2a, y2a = boxes_np[i]
            cur = np.array([x1a, y1a, x2a, y2a])
            for j in range(i + 1, len(boxes_np)):
                if used[j]:
                    continue
                x1b, y1b, x2b, y2b = boxes_np[j]
                xx1 = max(x1a, x1b)
                yy1 = max(y1a, y1b)
                xx2 = min(x2a, x2b)
                yy2 = min(y2a, y2b)
                iw = max(0, xx2 - xx1)
                ih = max(0, yy2 - yy1)
                inter = iw * ih
                if inter > 0:
                    cur = np.array(
                        [
                            min(cur[0], x1b),
                            min(cur[1], y1b),
                            max(cur[2], x2b),
                            max(cur[3], y2b),
                        ]
                    )
                    used[j] = True
                    merged = True
            used[i] = True
            new_boxes.append(tuple(map(int, cur.tolist())))
        boxes_np = np.array(new_boxes, dtype=np.int32)
    return [tuple(map(int, b)) for b in boxes_np]


def group_rows_columns(
    boxes: List[Tuple[int, int, int, int]],
    row_tol: int = 12,
    col_tol: int = 24,
    prefer_left_alignment: bool = True,
) -> Tuple[List[List[int]], List[List[int]]]:
    if not boxes:
        return [], []
    xs1 = np.array([b[0] for b in boxes])
    ys1 = np.array([b[1] for b in boxes])
    xs2 = np.array([b[2] for b in boxes])
    ys2 = np.array([b[3] for b in boxes])
    cy = (ys1 + ys2) / 2.0
    cx = (xs1 + xs2) / 2.0
    order_y = np.argsort(cy)
    row_groups: List[List[int]] = []
    current = [int(order_y[0])]
    for k in range(1, len(order_y)):
        i_prev = int(order_y[k - 1])
        i_cur = int(order_y[k])
        if abs(cy[i_cur] - cy[i_prev]) <= row_tol:
            current.append(i_cur)
        else:
            row_groups.append(sorted(current, key=lambda idx: boxes[idx][0]))
            current = [i_cur]
    row_groups.append(sorted(current, key=lambda idx: boxes[idx][0]))
    key_x = xs1 if prefer_left_alignment else cx
    order_x = np.argsort(key_x)
    col_groups: List[List[int]] = []
    current = [int(order_x[0])]
    for k in range(1, len(order_x)):
        j_prev = int(order_x[k - 1])
        j_cur = int(order_x[k])
        if abs(key_x[j_cur] - key_x[j_prev]) <= col_tol:
            current.append(j_cur)
        else:
            col_groups.append(
                sorted(current, key=lambda idx: (boxes[idx][1] + boxes[idx][3]) / 2.0)
            )
            current = [j_cur]
    col_groups.append(
        sorted(current, key=lambda idx: (boxes[idx][1] + boxes[idx][3]) / 2.0)
    )
    return row_groups, col_groups


def assign_cells_from_groups(
    boxes: List[Tuple[int, int, int, int]],
    row_groups: List[List[int]],
    col_groups: List[List[int]],
    manual_rows: int | None = None,
    manual_cols: int | None = None,
) -> Tuple[Dict[int, Tuple[int, int]], int, int]:
    if not boxes:
        return {}, 0, 0
    row_map: Dict[int, int] = {}
    for r, grp in enumerate(row_groups):
        for idx in grp:
            row_map[idx] = r
    col_map: Dict[int, int] = {}
    for c, grp in enumerate(col_groups):
        for idx in grp:
            col_map[idx] = c
    n_rows_est = len(row_groups)
    n_cols_est = len(col_groups)
    if manual_rows is not None and manual_rows > 0 and manual_rows != n_rows_est:
        cy = np.array([(boxes[i][1] + boxes[i][3]) / 2.0 for i in range(len(boxes))])
        order = np.argsort(cy)
        bin_edges = np.linspace(0, len(order), manual_rows + 1).astype(int)
        row_map = {}
        for r in range(manual_rows):
            for pos in order[bin_edges[r] : bin_edges[r + 1]]:
                row_map[int(pos)] = r
        n_rows_est = manual_rows
    if manual_cols is not None and manual_cols > 0 and manual_cols != n_cols_est:
        key_x = np.array([boxes[i][0] for i in range(len(boxes))])
        order = np.argsort(key_x)
        bin_edges = np.linspace(0, len(order), manual_cols + 1).astype(int)
        col_map = {}
        for c in range(manual_cols):
            for pos in order[bin_edges[c] : bin_edges[c + 1]]:
                col_map[int(pos)] = c
        n_cols_est = manual_cols
    assignment: Dict[int, Tuple[int, int]] = {}
    for i in range(len(boxes)):
        r = row_map.get(i, 0)
        c = col_map.get(i, 0)
        assignment[i] = (int(r), int(c))
    return assignment, n_rows_est, n_cols_est
