"""Utilities to post-process detected text boxes for table reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TextBox:
    """Lightweight representation of a detected text block."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.width / 2.0, self.y1 + self.height / 2.0)


def _vertical_overlap(a: TextBox, b: TextBox) -> float:
    top = max(a.y1, b.y1)
    bottom = min(a.y2, b.y2)
    return max(0.0, bottom - top)


def merge_intersecting_boxes(
    boxes: Sequence[Tuple[int, int, int, int]],
    scores: Sequence[float] | None = None,
    return_sources: bool = False,
) -> Tuple[List[Tuple[int, int, int, int]], List[float]] | Tuple[
    List[Tuple[int, int, int, int]],
    List[float],
    List[Tuple[int, ...]],
]:
    """Merge overlapping boxes and optionally propagate scores/original ids."""

    if not boxes:
        if return_sources:
            return [], [], []
        return [], []

    boxes_arr = np.array(boxes, dtype=np.int32)
    if scores is None or len(scores) != len(boxes):
        scores_arr = np.ones(len(boxes), dtype=np.float32)
    else:
        scores_arr = np.array(scores, dtype=np.float32)

    parent = np.arange(len(boxes_arr))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    for i in range(len(boxes_arr)):
        x1a, y1a, x2a, y2a = boxes_arr[i]
        for j in range(i + 1, len(boxes_arr)):
            x1b, y1b, x2b, y2b = boxes_arr[j]
            xx1 = max(x1a, x1b)
            yy1 = max(y1a, y1b)
            xx2 = min(x2a, x2b)
            yy2 = min(y2a, y2b)
            if xx1 < xx2 and yy1 < yy2:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for idx in range(len(boxes_arr)):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    merged_boxes: List[Tuple[int, int, int, int]] = []
    merged_scores: List[float] = []
    sources: List[Tuple[int, ...]] = []
    for indices in clusters.values():
        x1 = int(np.min(boxes_arr[indices, 0]))
        y1 = int(np.min(boxes_arr[indices, 1]))
        x2 = int(np.max(boxes_arr[indices, 2]))
        y2 = int(np.max(boxes_arr[indices, 3]))
        merged_boxes.append((x1, y1, x2, y2))
        merged_scores.append(float(np.max(scores_arr[indices])))
        sources.append(tuple(int(i) for i in indices))

    if return_sources:
        return merged_boxes, merged_scores, sources
    return merged_boxes, merged_scores


def build_box_dataframe(
    boxes: Sequence[Tuple[int, int, int, int]],
    scores: Sequence[float] | None = None,
    sources: Iterable[Sequence[int]] | None = None,
) -> pd.DataFrame:
    """Create a dataframe describing every detected/merged box."""

    if not boxes:
        return pd.DataFrame(
            columns=[
                "id",
                "x1",
                "y1",
                "x2",
                "y2",
                "width",
                "height",
                "center_x",
                "center_y",
                "score",
                "source_ids",
                "text",
            ]
        )

    if scores is None or len(scores) != len(boxes):
        scores = [1.0] * len(boxes)
    if sources is None:
        sources = [[i] for i in range(len(boxes))]

    records = []
    for idx, (box, score, src) in enumerate(zip(boxes, scores, sources)):
        tb = TextBox(*map(int, box))
        cx, cy = tb.center
        records.append(
            {
                "id": idx,
                "x1": int(tb.x1),
                "y1": int(tb.y1),
                "x2": int(tb.x2),
                "y2": int(tb.y2),
                "width": int(tb.width),
                "height": int(tb.height),
                "center_x": float(cx),
                "center_y": float(cy),
                "score": float(score),
                "source_ids": tuple(int(s) for s in src),
                "text": "",
            }
        )
    return pd.DataFrame.from_records(records)


def merge_line_aligned_boxes(
    df: pd.DataFrame,
    row_tol: float,
    gap_tol: float,
    min_vertical_overlap: float = 0.55,
) -> pd.DataFrame:
    """Fuse neighbouring boxes that likely belong to the same word/line."""

    if df.empty:
        return df

    result_rows: List[Dict] = []
    tol_y = max(row_tol, float(df["height"].median() * 0.45 if len(df) else row_tol))

    df_sorted = df.sort_values("center_y").reset_index(drop=True)
    row_clusters: List[List[int]] = []
    current = [0]
    for idx in range(1, len(df_sorted)):
        prev = df_sorted.loc[idx - 1]
        cur = df_sorted.loc[idx]
        if abs(cur["center_y"] - prev["center_y"]) <= tol_y:
            current.append(idx)
        else:
            row_clusters.append(current)
            current = [idx]
    row_clusters.append(current)

    for cluster in row_clusters:
        boxes_cluster = df_sorted.loc[cluster].sort_values("x1")
        merged_stack: List[Dict] = []
        for _, row in boxes_cluster.iterrows():
            tb = TextBox(row.x1, row.y1, row.x2, row.y2)
            if not merged_stack:
                merged_stack.append(row.to_dict())
                continue
            last = merged_stack[-1]
            last_tb = TextBox(last["x1"], last["y1"], last["x2"], last["y2"])
            vertical_overlap = _vertical_overlap(tb, last_tb)
            min_height = max(1.0, min(tb.height, last_tb.height))
            overlap_ratio = vertical_overlap / min_height
            horizontal_gap = tb.x1 - last_tb.x2
            if horizontal_gap <= gap_tol and overlap_ratio >= min_vertical_overlap:
                last["x1"] = int(min(last_tb.x1, tb.x1))
                last["y1"] = int(min(last_tb.y1, tb.y1))
                last["x2"] = int(max(last_tb.x2, tb.x2))
                last["y2"] = int(max(last_tb.y2, tb.y2))
                last["width"] = int(last["x2"] - last["x1"])
                last["height"] = int(last["y2"] - last["y1"])
                last["center_x"] = float(last["x1"] + last["width"] / 2.0)
                last["center_y"] = float(last["y1"] + last["height"] / 2.0)
                last["score"] = float(max(last["score"], row["score"]))
                last["source_ids"] = tuple(sorted(set(last["source_ids"]) | set(row["source_ids"])))
            else:
                merged_stack.append(row.to_dict())
        result_rows.extend(merged_stack)

    merged_df = pd.DataFrame(result_rows).reset_index(drop=True)
    merged_df["id"] = merged_df.index
    return merged_df


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
