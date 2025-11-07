"""OCR utilities and DataFrame construction for table extraction."""

from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    import pytesseract

    TESSERACT_OK = True
except Exception:  # pragma: no cover - optional dependency
    TESSERACT_OK = False


class OCRTableBuilder:
    """Perform OCR on detected boxes and build pandas DataFrames."""

    @staticmethod
    def ocr_text_from_box(
        image_bgr: np.ndarray, box: Tuple[int, int, int, int], lang: str = "eng"
    ) -> str:
        (x1, y1, x2, y2) = box
        roi = image_bgr[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        if roi.size == 0:
            return ""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        if not TESSERACT_OK:
            return ""
        try:
            cfg = "--psm 6"
            txt = pytesseract.image_to_string(gray, lang=lang, config=cfg)
            return txt.strip()
        except Exception:
            return ""

    @staticmethod
    def build_dataframe_from_assignments(
        image_bgr: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float],
        assignment: Dict[int, Tuple[int, int]],
        n_rows: int,
        n_cols: int,
        do_ocr: bool = True,
        lang: str = "eng",
        return_box_texts: bool = False,
    ) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[int, str]]:
        table = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        cell_blocks: Dict[Tuple[int, int], List[Tuple[float, str]]] = {}
        texts_cache: Dict[int, str] = {}
        for i, box in enumerate(boxes):
            r, c = assignment[i]
            if do_ocr:
                if i not in texts_cache:
                    texts_cache[i] = OCRTableBuilder.ocr_text_from_box(
                        image_bgr, box, lang=lang
                    )
                txt = texts_cache[i]
            else:
                txt = ""
            x1, _, x2, _ = box
            cell_blocks.setdefault((r, c), []).append(((x1 + x2) / 2.0, txt))
        for (r, c), items in cell_blocks.items():
            items_sorted = sorted(items, key=lambda t: t[0])
            table[r][c] = " ".join(
                [t[1] for t in items_sorted if t[1].strip() != ""]
            ).strip()
        df = pd.DataFrame(table)
        if return_box_texts:
            return df, {i: texts_cache.get(i, "") for i in range(len(boxes))}
        return df
