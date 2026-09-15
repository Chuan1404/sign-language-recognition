import torch, numpy as np

from src.models import ISLR_V4
from .draw_on_frame import *

def load_model(model_path, num_classes=100):
    print("Loading model...")

    model_kwargs = dict(
        num_classes=num_classes,
    )
    model = ISLR_V4(**model_kwargs)

    checkpoint = torch.load(
        model_path,
        map_location="cuda"
    )
    model.load_state_dict(checkpoint["model"])
    model = model.cuda()
    model.eval()

    return model

def detect_hand(hand_detection, rgb_frame, timestamp_ms):
    detection_hand_results = hand_detection.detect_video(
    rgb_frame,
    timestamp_ms)

    handedness = detection_hand_results.handedness
    hand_landmarks = detection_hand_results.hand_landmarks

    right_detected, left_detected = False, False
    right_coors, left_coors = np.zeros((21, 3), dtype=np.float32),  np.zeros((21, 3), dtype=np.float32)

    for i, hand_info in enumerate(handedness):
        if i >= len(hand_landmarks):
            continue

        category = hand_info[0]
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks[i]],
            dtype=np.float32
        )

        if category.index == 0:
            right_coors = coords
            right_detected = True

        elif category.index == 1:
            left_coors = coords
            left_detected = True

    return detection_hand_results, left_coors, right_coors, left_detected, right_detected

def detect_pose(pose_detection, rgb_frame, timestamp_ms):
    detection_pose_results = pose_detection.detect_video(
        rgb_frame,
        timestamp_ms
    )

    pose_landmarks = detection_pose_results.pose_landmarks
    pose_coors = np.zeros((33, 3), dtype=np.float32)
    pose_detected = False

    if len(pose_landmarks) > 0:
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_landmarks[0]],
            dtype=np.float32
        )

        pose_coors = coords
        pose_detected = True

    return detection_pose_results,pose_coors, pose_detected

def predict_gross(model, fusion, features, pretrained_model, predicted_text=''):

    T, _, _ = features.shape
    pose_arr = features[:, :33, :]
    left_arr = features[:, 33:54, :]
    right_arr = features[:, 54:75, :]

    position_features = fusion.fuse(pose_arr, left_arr, right_arr)
    shape_features = fusion.fuse_follow_shape(pose_arr, left_arr, right_arr)

    fused = np.concatenate([position_features, shape_features], axis=-1)

    features = torch.tensor(fused, dtype=torch.float32).unsqueeze(0).cuda()
    video_mask = torch.ones((1, T)).cuda()
    with torch.no_grad():
            logits, loss = model(features, video_mask=video_mask)

    top_probs, top_indices = torch.topk(logits, k=5, dim=-1)
    print(top_probs, top_indices)
    output = torch.argmax(logits, dim=1).item()

    if len(predicted_text) > 0:
        results = pretrained_model.predict_topk(predicted_text)
        print(results)

    return output

