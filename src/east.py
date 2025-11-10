"""Text detection utilities based on the EAST model."""

import math
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import streamlit as st

EAST_URL = "https://github.com/opencv/opencv_extra/raw/4.x/testdata/dnn/text/frozen_east_text_detection.pb"


class EastTextDetector:
    """Wrapper around OpenCV's EAST text detector."""

    def __init__(self, model_path: str = "frozen_east_text_detection.pb"):
        self.model_path = self.ensure_model(model_path)

    @staticmethod
    def ensure_model(model_path: str) -> str:
        if os.path.exists(model_path):
            return model_path
        try:
            import urllib.request

            st.info("Descargando modelo EAST (~90MB) la primera vez…")
            urllib.request.urlretrieve(EAST_URL, model_path)
            st.success("Modelo EAST descargado.")
        except Exception as e:
            st.warning(
                "No fue posible descargar el modelo EAST automáticamente. "
                "Descárgalo manualmente y colócalo como 'frozen_east_text_detection.pb' en la carpeta del app.\n"
                + str(e)
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "No se encontró el modelo EAST requerido en "
                f"{os.path.abspath(model_path)}."
            )
        return model_path

    @staticmethod
    def decode(scores: np.ndarray, geometry: np.ndarray, score_thresh: float = 0.5) -> Tuple[List[Tuple[int, int, int, int]], List[float]]:
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        for y in range(numRows):
            scoresData = scores[0, 0, y]
            x0 = geometry[0, 0, y]
            x1 = geometry[0, 1, y]
            x2 = geometry[0, 2, y]
            x3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            for x in range(numCols):
                score = scoresData[x]
                if score < score_thresh:
                    continue
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = x0[x] + x2[x]
                w = x1[x] + x3[x]
                endX = int(offsetX + (cos * x1[x]) + (sin * x2[x]))
                endY = int(offsetY - (sin * x1[x]) + (cos * x2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                rects.append((startX, startY, w, h))
                confidences.append(float(score))
        return rects, confidences

    @staticmethod
    def nms_boxes(boxes: List[Tuple[int, int, int, int]], scores: List[float], nms_thresh: float = 0.3) -> List[int]:
        if not boxes:
            return []
        boxes_np = np.array(boxes, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)
        x1 = boxes_np[:, 0]
        y1 = boxes_np[:, 1]
        x2 = boxes_np[:, 0] + boxes_np[:, 2]
        y2 = boxes_np[:, 1] + boxes_np[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores_np.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def detect(
        self,
        image_bgr: np.ndarray,
        score_thresh: float = 0.5,
        nms_thresh: float = 0.3,
        debug: bool = False,
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float], Dict]:
        (H, W) = image_bgr.shape[:2]
        newW, newH = (
            int(math.ceil(W / 32) * 32),
            int(math.ceil(H / 32) * 32),
        )
        rW = newW / float(W)
        rH = newH / float(H)
        blob_img = cv2.resize(image_bgr, (newW, newH))
        blob = cv2.dnn.blobFromImage(
            blob_img,
            1.0,
            (newW, newH),
            (123.68, 116.78, 103.94),
            swapRB=True,
            crop=False,
        )
        net = cv2.dnn.readNet(self.model_path)
        net.setInput(blob)
        (scores, geometry) = net.forward([
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3",
        ])
        rects, confidences = self.decode(scores, geometry, score_thresh)
        keep = self.nms_boxes(rects, confidences, nms_thresh)
        boxes = []
        scores_kept = []
        for i in keep:
            (x, y, w, h) = rects[i]
            startX = int(x / rW)
            startY = int(y / rH)
            endX = int((x + w) / rW)
            endY = int((y + h) / rH)
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(W - 1, endX)
            endY = min(H - 1, endY)
            boxes.append((startX, startY, endX, endY))
            scores_kept.append(confidences[i])
        dbg = {"scores": scores, "geometry": geometry}
        if debug:
            overlay = image_bgr.copy()
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            dbg["east_overlay"] = overlay
        return boxes, scores_kept, dbg
