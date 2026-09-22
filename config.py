import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = 'D:/SignDetection'
BATCH_SIZE = 8
LR = 1e-4

TRAIN_CSV = os.path.join(ROOT, "datasets/raw/how2sign/tsv_files_how2sign/tsv_files_how2sign/cvpr23.fairseq.i3d.train.how2sign.tsv")
VAL_CSV = os.path.join(ROOT, "datasets/raw/how2sign/tsv_files_how2sign/tsv_files_how2sign/cvpr23.fairseq.i3d.val.how2sign.tsv")
TEST_CSV = os.path.join(ROOT, "datasets/raw/how2sign/tsv_files_how2sign/tsv_files_how2sign/cvpr23.fairseq.i3d.test.how2sign.tsv")

BASE_I3D_TRAIN = os.path.join(ROOT, "datasets/raw/how2sign/i3d_features_how2sign/i3d_features_how2sign/train")
BASE_MP_TRAIN = os.path.join(ROOT, "datasets/raw/how2sign/mediapipe_features_how2sign/mediapipe_features/train")
BASE_I3D_VAL = os.path.join(ROOT, "datasets/raw/how2sign/i3d_features_how2sign/i3d_features_how2sign/val")
BASE_MP_VAL = os.path.join(ROOT, "datasets/raw/how2sign/mediapipe_features_how2sign/mediapipe_features/val")
BASE_I3D_TEST = os.path.join(ROOT, "datasets/raw/how2sign/i3d_features_how2sign/i3d_features_how2sign/test")
BASE_MP_TEST = os.path.join(ROOT, "datasets/raw/how2sign/mediapipe_features_how2sign/mediapipe_features/test")

HOW2SIGN_RAW_DATA = os.path.join(ROOT, "datasets/raw/how2sign_raw")
WLASL_RAW_DATA = os.path.join(ROOT, "datasets/raw/WLASL/videos")


# _REMOVE_POSE_IDX = [3, 4, 5, 6, 7, 8, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
# _REMOVE_POSE_IDX = [1, 3, 4, 6, 7, 8, 9, 10, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] #2
_REMOVE_POSE_IDX = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] #3
_N_POSE = 33 - len(_REMOVE_POSE_IDX)
_N_HAND = 21
_NUM_NODE = int(_N_POSE + _N_HAND * 2)
_COORD_DIM = 2

# WEB
WINDOW_SIZE = 32
FRAME_W, FRAME_H = 640, 480
ZONE_W, ZONE_H = 300, 400

LEFT_ZONE = (int(FRAME_W * 0.15), 10, ZONE_W, ZONE_H)
RIGHT_ZONE = (int(FRAME_W * 0.85 - ZONE_W), 10, ZONE_W, ZONE_H)

COLOR_RED = (0, 0, 255)   # BGR
COLOR_GREEN = (0, 255, 0)

# Action boxes (raw frame coords, before flip)
# After flip:  BACKSPACE_ZONE appears top-LEFT,  CLEAR_ZONE appears top-RIGHT
ACTION_BOX_W, ACTION_BOX_H = 100, 100
ACTION_BOX_MARGIN = 10

# top-right corner of raw frame  → top-LEFT  on display  → BACKSPACE
BACKSPACE_ZONE = (FRAME_W - ACTION_BOX_W - ACTION_BOX_MARGIN,
                  ACTION_BOX_MARGIN,
                  ACTION_BOX_W, ACTION_BOX_H)

# top-left corner of raw frame   → top-RIGHT on display  → CLEAR
CLEAR_ZONE     = (ACTION_BOX_MARGIN,
                  ACTION_BOX_MARGIN,
                  ACTION_BOX_W, ACTION_BOX_H)

COLOR_ORANGE  = (0, 165, 255)   # BGR – highlight when active
COLOR_YELLOW  = (0, 215, 255)   # BGR

