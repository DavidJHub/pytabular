"""OCR utilities and DataFrame construction for table extraction."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

try:
    import pytesseract
    from pytesseract import Output

    TESSERACT_OK = True
except Exception:  # pragma: no cover - optional dependency
    TESSERACT_OK = False
    Output = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import easyocr

    EASYOCR_OK = True
except Exception:  # pragma: no cover - optional dependency
    easyocr = None  # type: ignore
    EASYOCR_OK = False

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    HF_OCR_OK = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    TrOCRProcessor = VisionEncoderDecoderModel = None  # type: ignore
    HF_OCR_OK = False


_HF_MODEL = None
_HF_PROCESSOR = None
_HF_DEVICE = None
_HF_LOAD_FAILED = False
_HF_MODEL_NAME = os.environ.get("TABULAR_HF_OCR_MODEL", "microsoft/trocr-base-printed")

_EASYOCR_READERS: Dict[str, "easyocr.Reader"] = {}


def _normalize_easyocr_lang(lang: str) -> List[str]:
    tokens = [
        token.strip().lower()
        for token in lang.replace("+", ",").replace(";", ",").split(",")
        if token.strip()
    ]
    if not tokens:
        tokens = ["en"]

    result: List[str] = []
    for token in tokens:
        if token in {"eng", "english", "en"}:
            result.append("en")
        elif token in {"spa", "spanish", "es"}:
            result.append("es")
        elif len(token) == 2:
            result.append(token)
    if not result:
        result = ["en"]
    return sorted(set(result))


def _get_easyocr_reader(lang: str):  # pragma: no cover - heavy optional dependency
    if not EASYOCR_OK:
        return None
    key = "+".join(_normalize_easyocr_lang(lang))
    if key not in _EASYOCR_READERS:
        try:
            _EASYOCR_READERS[key] = easyocr.Reader(key.split("+"), gpu=False)
        except Exception:
            return None
    return _EASYOCR_READERS[key]


class OCRTableBuilder:
    """Perform OCR on detected boxes and build pandas DataFrames."""

    @staticmethod
    def _prep_roi(image_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        (x1, y1, x2, y2) = box
        roi = image_bgr[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        if roi.size == 0:
            empty = np.zeros((0, 0, 3), dtype=image_bgr.dtype)
            return empty, empty
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        binary = cv2.threshold(
            norm, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )[1]
        return roi, binary

    @staticmethod
    def _hf_image_to_text(roi_color: np.ndarray) -> str:
        """Run a Hugging Face VisionEncoderDecoder model (TrOCR) if available."""

        global _HF_MODEL, _HF_PROCESSOR, _HF_DEVICE, _HF_LOAD_FAILED, HF_OCR_OK
        if not HF_OCR_OK or _HF_LOAD_FAILED or roi_color.size == 0:
            return ""
        if _HF_MODEL is None or _HF_PROCESSOR is None or _HF_DEVICE is None:
            try:
                processor = TrOCRProcessor.from_pretrained(_HF_MODEL_NAME)
                model = VisionEncoderDecoderModel.from_pretrained(_HF_MODEL_NAME)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.eval()
            except Exception:
                _HF_LOAD_FAILED = True
                HF_OCR_OK = False
                return ""
            else:
                _HF_PROCESSOR = processor
                _HF_MODEL = model
                _HF_DEVICE = device
        if _HF_MODEL is None or _HF_PROCESSOR is None or _HF_DEVICE is None:
            return ""
        try:
            roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(roi_rgb)
            inputs = _HF_PROCESSOR(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(_HF_DEVICE)
            with torch.no_grad():
                generated_ids = _HF_MODEL.generate(pixel_values)
            text = _HF_PROCESSOR.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return text.strip()
        except Exception:
            return ""

    @staticmethod
    def ocr_text_from_box(
        image_bgr: np.ndarray, box: Tuple[int, int, int, int], lang: str = "eng"
    ) -> str:
        roi_color, roi_proc = OCRTableBuilder._prep_roi(image_bgr, box)
        if roi_color.size == 0:
            return ""

        hf_text = OCRTableBuilder._hf_image_to_text(roi_color)
        if hf_text:
            return hf_text

        if TESSERACT_OK:
            try:
                cfg = "--psm 6 --oem 1"
                data = pytesseract.image_to_data(
                    roi_proc, lang=lang, config=cfg, output_type=Output.DICT
                )
                words = [w.strip() for w in data.get("text", []) if w and w.strip()]
                if words:
                    return " ".join(words)
                txt = pytesseract.image_to_string(roi_proc, lang=lang, config=cfg)
                return txt.strip()
            except Exception:
                pass

        reader = _get_easyocr_reader(lang)
        if reader is not None:
            try:
                roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
                results = reader.readtext(roi_rgb)
                texts = [text.strip() for _, text, conf in results if text.strip()]
                if texts:
                    return " ".join(texts)
            except Exception:
                pass

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
