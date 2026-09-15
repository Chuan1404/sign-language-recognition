import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import cv2 as cv
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from config import (
    _COORD_DIM,
    _REMOVE_POSE_IDX,
)

from app.utils import (
    predict_gross,
    detect_hand,
    detect_pose,
)


class SignDetectionService:

    def __init__(
        self,
        hand_detection,
        pose_detection,
        fusion,
        model,
        pretrained_model,
        idx2gloss,
    ):
        self.hand_detection = hand_detection
        self.pose_detection = pose_detection
        self.fusion = fusion
        self.model = model
        self.pretrained_model = pretrained_model
        self.idx2gloss = idx2gloss

    def predict(self, pose_buf, left_hand_buf, right_hand_buf, predicted_text):

        pose_arr = np.stack(pose_buf)
        left_arr = np.stack(left_hand_buf)
        right_arr = np.stack(right_hand_buf)

        features = np.concatenate((pose_arr, left_arr, right_arr), axis=1)

        output = predict_gross(
            self.model,
            self.fusion,
            features,
            self.pretrained_model,
            "".join(predicted_text),
        )

        return self.idx2gloss[output]

    def detect(self, rgb_frame, timestamp_ms):

        hand_results = detect_hand(
            self.hand_detection,
            rgb_frame,
            timestamp_ms,
        )

        pose_results = detect_pose(
            self.pose_detection,
            rgb_frame,
            timestamp_ms,
        )

        return hand_results, pose_results

    def draw(self, rgb_frame, hand_results, pose_results):

        detection_hand_results, *_ = hand_results
        detection_pose_results, *_ = pose_results

        rgb_frame = self.hand_detection.draw_landmarks_on_image(
            rgb_frame,
            detection_hand_results,
        )

        rgb_frame = self.pose_detection.draw_landmarks_on_image(
            rgb_frame,
            detection_pose_results,
            remove_pose_idx=_REMOVE_POSE_IDX,
        )

        return rgb_frame


