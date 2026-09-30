import os
import random

import numpy as np
import torch

from config import ROOT

# ---- Đường dẫn ----
DATA_PATH = os.path.join(ROOT, "datasets", "processed", "wlasl_features_v2")
LABEL_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL100")
RAW_VIDEO_DIR = os.path.join(ROOT, "datasets", "raw", "WLASL", "videos")
MODEL_DIR = os.path.join(ROOT, "outputs", "models")

SELECTED_RGB_DIR = os.path.join(ROOT, "outputs", "selected_rgb")
IMPORTANCE_DIR = os.path.join(ROOT, "outputs", "frame_importance")

# ---- Hằng số dữ liệu ----
IMAGE_SIZE = 112
MAX_FRAMES = 100
MIN_FRAMES = 8


def importance_path(split: str) -> str:
    return os.path.join(IMPORTANCE_DIR, f"wlasl100_{split}_frame_importance.json")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_warmup(optimizer, base_lrs, epoch: int, warmup_epochs: int) -> bool:
    """Linear warmup theo epoch. Trả về True khi còn trong warmup (nên bỏ qua scheduler.step)."""
    if warmup_epochs <= 0 or epoch > warmup_epochs:
        return False
    scale = 1.0 if epoch == warmup_epochs else (epoch + 1) / (warmup_epochs + 1)
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base * scale
    return epoch < warmup_epochs