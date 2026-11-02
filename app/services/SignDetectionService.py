import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import (
    detect_hand,
    detect_pose,
)
from config import (
    _COORD_DIM,
    _REMOVE_POSE_IDX,
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
        return self.predict_topk(pose_buf, left_hand_buf, right_hand_buf)[0]["gloss"]

    def predict_topk(self, pose_buf, left_hand_buf, right_hand_buf, k=5):
        if not pose_buf or not (len(pose_buf) == len(left_hand_buf) == len(right_hand_buf)):
            raise ValueError("Expected non-empty, aligned pose and hand buffers.")
        pose_arr = np.stack(pose_buf)[..., :_COORD_DIM]
        left_arr = np.stack(left_hand_buf)[..., :_COORD_DIM]
        right_arr = np.stack(right_hand_buf)[..., :_COORD_DIM]

        T, _, _ = pose_arr.shape

        # Match WLASLLandmarksDataset used by main.py exactly.
        shape_features = self.fusion.fuse_follow_shape(pose_arr, left_arr, right_arr)
        average_features = self.fusion.fuse(pose_arr, left_arr, right_arr)
        position_features = self.fusion.fuse_follow_position(pose_arr, left_arr, right_arr)

        fused = np.concatenate([shape_features, average_features, position_features], axis=-1)
        max_frames = self.model.pos_encoder.pe.shape[1]
        if T > max_frames:
            indices = np.linspace(0, T - 1, max_frames).round().astype(np.int64)
            fused = fused[indices]
            T = len(fused)

        device = next(self.model.parameters()).device
        features = torch.as_tensor(fused, dtype=torch.float32, device=device).unsqueeze(0)
        video_mask = torch.ones((1, T), dtype=torch.bool, device=device)

        with torch.no_grad():
            logits, _loss = self.model(features, video_mask=video_mask)

        probabilities = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probabilities, k=min(k, logits.shape[-1]), dim=-1)
        return [
            {"gloss": self.idx2gloss[index.item()], "probability": probability.item()}
            for probability, index in zip(top_probs[0], top_indices[0])
        ]

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
