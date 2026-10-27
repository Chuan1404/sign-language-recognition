import os

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = "D:/SignDetection"
BATCH_SIZE = 8
LR = 1e-4
IMAGE_SIZE = 224
MAX_FRAMES = 100

WLASL_RAW_DATA = os.path.join(ROOT, "datasets/raw/WLASL/videos")

_REMOVE_POSE_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
# _REMOVE_POSE_IDX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
_N_POSE = 33 - len(_REMOVE_POSE_IDX)
_N_HAND = 21
_NUM_NODE = int(_N_POSE + _N_HAND * 2)
_COORD_DIM = 2

# _REMOVE_VIDEO_IDS = ['05728', '05740', '05750', # before
#                      '65242',                    # bowling
#                      '08429',
#                     ]
_REMOVE_VIDEO_IDS = []


def set_seed(seed=42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# WEB
WINDOW_SIZE = 32
FRAME_W, FRAME_H = 640, 480
ZONE_W, ZONE_H = 300, 400

LEFT_ZONE = (int(FRAME_W * 0.15), 10, ZONE_W, ZONE_H)
RIGHT_ZONE = (int(FRAME_W * 0.85 - ZONE_W), 10, ZONE_W, ZONE_H)

COLOR_RED = (0, 0, 255)  # BGR
COLOR_GREEN = (0, 255, 0)

# Action boxes (raw frame coords, before flip)
# After flip:  BACKSPACE_ZONE appears top-LEFT,  CLEAR_ZONE appears top-RIGHT
ACTION_BOX_W, ACTION_BOX_H = 100, 100
ACTION_BOX_MARGIN = 10

# top-right corner of raw frame  → top-LEFT  on display  → BACKSPACE
BACKSPACE_ZONE = (FRAME_W - ACTION_BOX_W - ACTION_BOX_MARGIN, ACTION_BOX_MARGIN, ACTION_BOX_W, ACTION_BOX_H,)

# top-left corner of raw frame   → top-RIGHT on display  → CLEAR
CLEAR_ZONE = (ACTION_BOX_MARGIN, ACTION_BOX_MARGIN, ACTION_BOX_W, ACTION_BOX_H)

COLOR_ORANGE = (0, 165, 255)  # BGR – highlight when active
COLOR_YELLOW = (0, 215, 255)  # BGR
