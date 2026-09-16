import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys, torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (_REMOVE_POSE_IDX, )

from app.utils import (detect_hand, detect_pose, )

class SignDetectionService:

    def __init__(self, hand_detection, pose_detection, fusion, model, pretrained_model, idx2gloss, ):
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

        T, _, _ = pose_arr.shape

        position_features = self.fusion.fuse(pose_arr, left_arr, right_arr)
        shape_features = self.fusion.fuse_follow_shape(pose_arr, left_arr, right_arr)

        fused = np.concatenate([position_features, shape_features], axis=-1)

        features = torch.tensor(fused, dtype=torch.float32).unsqueeze(0).cuda()
        video_mask = torch.ones((1, T)).cuda()

        with torch.no_grad():
            logits, loss = self.model(features, video_mask=video_mask)

        top_probs, top_indices = torch.topk(logits, k=5, dim=-1)
        output = torch.argmax(logits, dim=1).item()
        print([self.idx2gloss[idx.item()] for idx in top_indices[0]])

        # if len(predicted_text) > 0:
        #     results = self.pretrained_model.predict_topk(predicted_text)
        #     print(results)

        return self.idx2gloss[output]

    def detect(self, rgb_frame, timestamp_ms):
        hand_results = detect_hand(self.hand_detection, rgb_frame, timestamp_ms, )

        pose_results = detect_pose(self.pose_detection, rgb_frame, timestamp_ms, )

        return hand_results, pose_results

    def draw(self, rgb_frame, hand_results, pose_results):
        detection_hand_results, *_ = hand_results
        detection_pose_results, *_ = pose_results

        rgb_frame = self.hand_detection.draw_landmarks_on_image(rgb_frame, detection_hand_results, )

        rgb_frame = self.pose_detection.draw_landmarks_on_image(rgb_frame, detection_pose_results,
                                                                remove_pose_idx=_REMOVE_POSE_IDX, )

        return rgb_frame
