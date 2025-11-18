"""High level helpers to build table-like clusters from detected text boxes.

The module consumes the CSV exported by the detection + OCR stage.  It uses a
series of geometric heuristics to infer row/column groups, merged cells and
basic metrics such as the width/height of every row and column.  The goal is to
obtain a structured view of the table without requiring a heavy ML model.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class ClusterInfo:
    """Container describing either a row or column cluster."""

    id: int
    box_indices: List[int]
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

    def to_dict(self) -> Dict[str, int | List[int]]:
        data = asdict(self)
        data["width"] = self.width
        data["height"] = self.height
        return data


@dataclass
class CellAssignment:
    """Description of how a text box maps to the inferred table grid."""

    box_id: int
    row: int
    column: int
    row_span: int
    col_span: int
    text: str
    bbox: Sequence[int]

    def to_dict(self) -> Dict:
        return {
            "box_id": self.box_id,
            "row": self.row,
            "column": self.column,
            "row_span": self.row_span,
            "col_span": self.col_span,
            "text": self.text,
            "bbox": list(map(int, self.bbox)),
        }


def _cluster_axis(coords: np.ndarray, spans: np.ndarray, base_tol: float) -> List[List[int]]:
    """Cluster 1D coordinates using adaptive tolerances.

    The tolerance between two consecutive points is the maximum between the
    provided ``base_tol`` and 60 % of their average span.  This makes tall rows
    or wide columns more permissive.
    """

    if len(coords) == 0:
        return []

    order = np.argsort(coords)
    clusters: List[List[int]] = []
    current: List[int] = [int(order[0])]

    def adaptive_tol(i_prev: int, i_cur: int) -> float:
        span_prev = spans[i_prev]
        span_cur = spans[i_cur]
        span_tol = 0.6 * 0.5 * (span_prev + span_cur)
        return max(base_tol, span_tol)

    for idx in range(1, len(order)):
        prev_idx = int(order[idx - 1])
        cur_idx = int(order[idx])
        gap = abs(float(coords[cur_idx]) - float(coords[prev_idx]))
        if gap <= adaptive_tol(prev_idx, cur_idx):
            current.append(cur_idx)
        else:
            clusters.append(current)
            current = [cur_idx]

    clusters.append(current)
    return clusters


def _build_cluster_infos(
    boxes: np.ndarray,
    clusters: List[List[int]],
    sort_key,
) -> List[ClusterInfo]:
    infos: List[ClusterInfo] = []
    for cid, indices in enumerate(clusters):
        idx_arr = np.array(indices, dtype=np.int32)
        x1 = int(np.min(boxes[idx_arr, 0]))
        y1 = int(np.min(boxes[idx_arr, 1]))
        x2 = int(np.max(boxes[idx_arr, 2]))
        y2 = int(np.max(boxes[idx_arr, 3]))
        infos.append(
            ClusterInfo(
                id=cid,
                box_indices=list(map(int, indices)),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    infos.sort(key=sort_key)
    for new_id, info in enumerate(infos):
        info.id = new_id
    return infos


def infer_row_column_clusters(
    df: pd.DataFrame,
    row_tol: float | None = None,
    col_tol: float | None = None,
) -> tuple[List[ClusterInfo], List[ClusterInfo]]:
    """Infer row/column clusters from a dataframe of text boxes."""

    if df.empty:
        return [], []

    boxes = df[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    widths = np.maximum(1.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(1.0, boxes[:, 3] - boxes[:, 1])
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0

    base_row_tol = row_tol if row_tol is not None else float(np.median(heights) * 0.55)
    base_col_tol = col_tol if col_tol is not None else float(np.median(widths) * 0.45)

    row_clusters = _cluster_axis(centers_y, heights, base_row_tol)
    col_clusters = _cluster_axis(boxes[:, 0], widths, base_col_tol)

    row_infos = _build_cluster_infos(
        boxes,
        row_clusters,
        sort_key=lambda info: info.y1,
    )
    col_infos = _build_cluster_infos(
        boxes,
        col_clusters,
        sort_key=lambda info: info.x1,
    )
    return row_infos, col_infos


def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _memberships(
    value_range: tuple[float, float],
    clusters: Sequence[ClusterInfo],
    overlap_threshold: float,
    use_height: bool,
) -> List[int]:
    if not clusters:
        return []
    v1, v2 = value_range
    span = max(1.0, v2 - v1)
    hits: List[int] = []
    for cluster in clusters:
        ref_span = cluster.height if use_height else cluster.width
        denom = max(1.0, min(span, ref_span))
        overlap = _overlap(v1, v2, cluster.y1, cluster.y2) if use_height else _overlap(v1, v2, cluster.x1, cluster.x2)
        if denom <= 0:
            continue
        ratio = overlap / denom
        if ratio >= overlap_threshold:
            hits.append(cluster.id)
    if hits:
        return hits
    # fallback to nearest cluster based on centre distance
    center = v1 + span / 2.0
    if use_height:
        candidate = min(clusters, key=lambda c: abs((c.y1 + c.y2) / 2.0 - center))
    else:
        candidate = min(clusters, key=lambda c: abs((c.x1 + c.x2) / 2.0 - center))
    return [candidate.id]


def assign_cells(
    df: pd.DataFrame,
    rows: Sequence[ClusterInfo],
    cols: Sequence[ClusterInfo],
    row_overlap: float = 0.45,
    col_overlap: float = 0.55,
) -> List[CellAssignment]:
    assignments: List[CellAssignment] = []
    for record in df.itertuples():
        box = (record.x1, record.y1, record.x2, record.y2)
        row_hits = _memberships((box[1], box[3]), rows, row_overlap, use_height=True)
        col_hits = _memberships((box[0], box[2]), cols, col_overlap, use_height=False)
        assignments.append(
            CellAssignment(
                box_id=int(record.id),
                row=int(row_hits[0]),
                column=int(col_hits[0]),
                row_span=int(len(row_hits)),
                col_span=int(len(col_hits)),
                text=str(record.text),
                bbox=box,
            )
        )
    return assignments


def summarize_table(
    df: pd.DataFrame,
    row_tol: float | None = None,
    col_tol: float | None = None,
    row_overlap: float = 0.45,
    col_overlap: float = 0.55,
) -> Dict:
    rows, cols = infer_row_column_clusters(df, row_tol=row_tol, col_tol=col_tol)
    cells = assign_cells(df, rows, cols, row_overlap=row_overlap, col_overlap=col_overlap)
    return {
        "rows": [row.to_dict() for row in rows],
        "columns": [col.to_dict() for col in cols],
        "cells": [cell.to_dict() for cell in cells],
    }


def load_detection_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected_cols = {"id", "x1", "y1", "x2", "y2", "text"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"CSV is missing required columns: {expected_cols - set(df.columns)}")
    return df


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster OCR boxes into table rows/columns")
    parser.add_argument("csv", type=Path, help="CSV file with detection results")
    parser.add_argument("--row-tol", type=float, default=None, help="Base tolerance in pixels for row clustering")
    parser.add_argument("--col-tol", type=float, default=None, help="Base tolerance in pixels for column clustering")
    parser.add_argument("--row-overlap", type=float, default=0.45, help="Minimum vertical overlap ratio to share a row")
    parser.add_argument("--col-overlap", type=float, default=0.55, help="Minimum horizontal overlap ratio to share a column")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the JSON summary")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    df = load_detection_csv(args.csv)
    summary = summarize_table(
        df,
        row_tol=args.row_tol,
        col_tol=args.col_tol,
        row_overlap=args.row_overlap,
        col_overlap=args.col_overlap,
    )
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
