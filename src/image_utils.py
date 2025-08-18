"""Image utility functions for PIL and OpenCV conversions."""

import cv2
import numpy as np
from PIL import Image


class ImageUtils:
    """Utility methods for image format conversions."""

    @staticmethod
    def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
        arr = np.array(pil_img)
        if arr.ndim == 2:
            return arr
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    @staticmethod
    def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
        if len(img_bgr.shape) == 2:
            return Image.fromarray(img_bgr)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
