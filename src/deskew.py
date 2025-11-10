"""Deskewing functionality using Hough transform and contour heuristics."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import cv2
import numpy as np


def _estimate_angle_from_contours(binary: np.ndarray) -> float:
    """Estimate skew angle from table-like contours.

    When Hough lines are scarce (e.g. faint borders), we analyse the biggest
    contour that resembles a table and use the minimum area rectangle angle as a
    fallback. This provides stability for photos where only one or two borders
    are visible.
    """

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.05 * binary.shape[0] * binary.shape[1]:
        # Too small to be meaningful.
        return 0.0
    rect = cv2.minAreaRect(contour)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return float(angle)


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
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, canny1, canny2)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_thresh)

        angle_deg = 0.0
        angles: list[float] = []
        if lines is not None:
            for rho_theta in lines:
                _, theta = rho_theta[0]
                deg = (theta * 180 / np.pi)
                deg = (deg + 90) % 180 - 90
                angles.append(deg)
        if angles:
            angle_deg = float(np.median(angles))
        else:
            # As a fallback, rely on the dominant contour's orientation.
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.mean(thresh) > 127:  # invert if text is darker than background
                thresh = cv2.bitwise_not(thresh)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contour_angle = _estimate_angle_from_contours(morph)
            angle_deg = contour_angle

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

        dbg: Dict = {"angle_deg": angle_deg, "edges": edges, "lines": lines}
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
