"""Deskewing functionality using Hough transform."""

import math
from typing import Dict, Tuple

import cv2
import numpy as np


class Deskewer:
    """Rotate images to correct skew using Hough lines."""

    @staticmethod
    def hough_deskew(
        image_bgr: np.ndarray,
        canny1: int = 50,
        canny2: int = 150,
        hough_thresh: int = 100,
        debug: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, canny1, canny2)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh)

        angle_deg = 0.0
        angles = []
        if lines is not None:
            for rho_theta in lines:
                rho, theta = rho_theta[0]
                deg = (theta * 180 / np.pi)
                deg = (deg + 90) % 180 - 90
                angles.append(deg)
            if angles:
                angle_deg = float(np.median(angles))

        (h, w) = image_bgr.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(
            image_bgr,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        dbg = {"angle_deg": angle_deg, "edges": edges, "lines": lines}
        if debug:
            overlay = image_bgr.copy()
            if lines is not None:
                for rho_theta in lines[:100]:
                    rho, theta = rho_theta[0]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
            dbg["hough_overlay"] = overlay

        return rotated, dbg
