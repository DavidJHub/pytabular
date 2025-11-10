"""High level pipeline to extract table structures from document images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from deskew import Deskewer
from east import EastTextDetector
from grouping import (
    assign_cells_from_groups,
    build_box_dataframe,
    group_rows_columns,
    merge_intersecting_boxes,
    merge_line_aligned_boxes,
)
from ocr import OCRTableBuilder


@dataclass
class PipelineConfig:
    """Configuration parameters for the extraction pipeline."""

    canny1: int = 50
    canny2: int = 150
    hough_thresh: int = 120
    score_thresh: float = 0.5
    nms_thresh: float = 0.3
    row_tol: int = 12
    col_tol: int = 24
    prefer_left_alignment: bool = True
    merge_gap_tol: int = 18
    merge_row_tol: int = 10
    do_ocr: bool = False
    ocr_lang: str = "eng"
    manual_rows: int | None = None
    manual_cols: int | None = None


@dataclass
class PipelineResult:
    """Container with all intermediate and final artefacts."""

    deskewed: np.ndarray
    angle_deg: float
    boxes_df: pd.DataFrame
    row_groups: List[List[int]]
    col_groups: List[List[int]]
    assignments: Dict[int, Tuple[int, int]]
    table: pd.DataFrame
    n_rows: int
    n_cols: int


class TableExtractionPipeline:
    """Composable pipeline that mirrors the manual steps from the requirements."""

    def __init__(self, east_model_path: str = "frozen_east_text_detection.pb") -> None:
        self.detector = EastTextDetector(east_model_path)

    def process(
        self,
        image_bgr: np.ndarray,
        config: PipelineConfig | None = None,
        debug: bool = False,
    ) -> Tuple[PipelineResult, Dict]:
        cfg = config or PipelineConfig()

        deskewed, dbg_h = Deskewer.hough_deskew(
            image_bgr,
            canny1=cfg.canny1,
            canny2=cfg.canny2,
            hough_thresh=cfg.hough_thresh,
            debug=debug,
        )

        boxes_raw, scores_raw, dbg_e = self.detector.detect(
            deskewed,
            score_thresh=cfg.score_thresh,
            nms_thresh=cfg.nms_thresh,
            debug=debug,
        )

        merged_boxes, merged_scores, merged_sources = merge_intersecting_boxes(
            boxes_raw, scores_raw, return_sources=True
        )
        boxes_df = build_box_dataframe(merged_boxes, merged_scores, merged_sources)
        boxes_df = merge_line_aligned_boxes(
            boxes_df,
            row_tol=cfg.merge_row_tol,
            gap_tol=cfg.merge_gap_tol,
        )

        final_boxes: List[Tuple[int, int, int, int]] = [
            (int(row.x1), int(row.y1), int(row.x2), int(row.y2))
            for row in boxes_df.itertuples()
        ]
        final_scores: List[float] = [float(score) for score in boxes_df["score"].to_list()]

        row_groups, col_groups = group_rows_columns(
            final_boxes,
            row_tol=cfg.row_tol,
            col_tol=cfg.col_tol,
            prefer_left_alignment=cfg.prefer_left_alignment,
        )

        assignments, n_rows, n_cols = assign_cells_from_groups(
            final_boxes,
            row_groups,
            col_groups,
            manual_rows=cfg.manual_rows,
            manual_cols=cfg.manual_cols,
        )

        table_df, texts_map = OCRTableBuilder.build_dataframe_from_assignments(
            deskewed,
            final_boxes,
            final_scores,
            assignments,
            n_rows,
            n_cols,
            do_ocr=cfg.do_ocr,
            lang=cfg.ocr_lang,
            return_box_texts=True,
        )

        boxes_df = boxes_df.copy()
        boxes_df["row"] = boxes_df.index.map(lambda i: assignments.get(i, (None, None))[0])
        boxes_df["col"] = boxes_df.index.map(lambda i: assignments.get(i, (None, None))[1])
        boxes_df["text"] = boxes_df.index.map(lambda i: texts_map.get(i, ""))

        result = PipelineResult(
            deskewed=deskewed,
            angle_deg=float(dbg_h.get("angle_deg", 0.0)),
            boxes_df=boxes_df,
            row_groups=row_groups,
            col_groups=col_groups,
            assignments=assignments,
            table=table_df,
            n_rows=n_rows,
            n_cols=n_cols,
        )

        debug_payload = {
            "deskew": dbg_h,
            "east": dbg_e,
            "raw_boxes": boxes_raw,
            "merged_boxes": merged_boxes,
            "final_boxes": final_boxes,
        }
        return result, debug_payload


__all__ = [
    "PipelineConfig",
    "PipelineResult",
    "TableExtractionPipeline",
]
