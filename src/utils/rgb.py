"""Tiền xử lý RGB dùng chung (một cài đặt duy nhất cho extract + dataset)."""
import cv2
import numpy as np
import torch

from pipeline_config import IMAGE_SIZE

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def square_box(x0, y0, x1, y1, w, h, margin=0.15):
    """Mở rộng bbox thành hình vuông (có margin) và kẹp vào khung hình. Trả về (x0, y0, x1, y1) pixel."""
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * (1.0 + 2.0 * margin)
    side = min(max(side, 0.5 * min(w, h)), min(w, h))
    left = int(round(np.clip(cx - side / 2.0, 0, w - side)))
    top = int(round(np.clip(cy - side / 2.0, 0, h - side)))
    s = int(round(side))
    return left, top, min(left + s, w), min(top + s, h)


def bgr_to_chw_uint8(bgr: np.ndarray, size: int = IMAGE_SIZE, crop_box=None) -> np.ndarray:
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        bgr = bgr[y0:y1, x0:x1]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    interp = cv2.INTER_AREA if max(rgb.shape[:2]) > size else cv2.INTER_LINEAR
    rgb = cv2.resize(rgb, (size, size), interpolation=interp)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def normalize_imagenet(x01: torch.Tensor) -> torch.Tensor:
    """x01: float (T, 3, H, W) trong [0, 1]."""
    return (x01 - IMAGENET_MEAN) / IMAGENET_STD