import os
import sys

from config import ROOT, WLASL_RAW_DATA

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import argparse
import numpy as np

from src.utils.pose_detection import PoseDetection
from src.utils.hand_detection import HandDetection
from src.utils.face_detection import FaceDetection
from src.utils.kalman_filter import restore_missing_points

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.components.containers.category import Category


class MockHandDetectionResult:
    def __init__(self, hand_landmarks, handedness):
        self.hand_landmarks = hand_landmarks
        self.handedness = handedness

class MockPoseDetectionResult:
    def __init__(self, pose_landmarks):
        self.pose_landmarks = pose_landmarks

class MockFaceDetectionResult:
    def __init__(self, face_landmarks):
        self.face_landmarks = face_landmarks

def create_mock_hands(left_points, right_points, left_visible, right_visible):
    hand_landmarks = []
    handedness = []

    # Right Hand
    if right_visible:
        r_lms = [NormalizedLandmark(x=float(p[0]), y=float(p[1]), z=float(p[2]), visibility=1.0, presence=1.0) for p in right_points]
        hand_landmarks.append(r_lms)
        handedness.append([Category(index=0, score=1.0, display_name='Right', category_name='Right')])

    # Left Hand
    if left_visible:
        l_lms = [NormalizedLandmark(x=float(p[0]), y=float(p[1]), z=float(p[2]), visibility=1.0, presence=1.0) for p in left_points]
        hand_landmarks.append(l_lms)
        handedness.append([Category(index=1, score=1.0, display_name='Left', category_name='Left')])

    return MockHandDetectionResult(hand_landmarks, handedness)

def create_mock_pose(pose_points, pose_visible):
    if not pose_visible:
        return MockPoseDetectionResult([])
    
    p_lms = [NormalizedLandmark(x=float(p[0]), y=float(p[1]), z=float(p[2]), visibility=1.0, presence=1.0) for p in pose_points]
    return MockPoseDetectionResult([p_lms])

def create_mock_face(face_points, face_visible):
    if not face_visible:
        return MockFaceDetectionResult([])
    
    f_lms = [NormalizedLandmark(x=float(p[0]), y=float(p[1]), z=float(p[2]), visibility=1.0, presence=1.0) for p in face_points]
    return MockFaceDetectionResult([f_lms])

def process_video(video_path, output_path, method, model_path):
    print(f"Processing: {video_path} with method: {method}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    hand_detector = HandDetection(min_hand_detection_confidence=0.6)
    pose_detector = PoseDetection()
    face_detector = FaceDetection()

    raw_left = []
    raw_right = []
    raw_pose = []
    raw_face = []

    frame_index = 0
    print("Step 1: Extracting keypoints from original video...")
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(frame_index * 1000 / fps)

        hand_res = hand_detector.detect_video(frame_rgb, timestamp_ms)
        pose_res = pose_detector.detect_video(frame_rgb, timestamp_ms)
        face_res = face_detector.detect_video(frame_rgb, timestamp_ms)

        # Parse Hands
        r_hand = np.zeros((21, 3), dtype=np.float32)
        l_hand = np.zeros((21, 3), dtype=np.float32)
        if hand_res.hand_landmarks:
            for i, h_info in enumerate(hand_res.handedness):
                if i >= len(hand_res.hand_landmarks): continue
                cat = h_info[0]
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_res.hand_landmarks[i]], dtype=np.float32)
                if cat.index == 0:
                    r_hand = coords
                elif cat.index == 1:
                    l_hand = coords
        raw_left.append(l_hand)
        raw_right.append(r_hand)

        # Parse Pose
        p_coords = np.zeros((5, 3), dtype=np.float32)
        if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
            lms = pose_res.pose_landmarks[0][:5]
            p_coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
        raw_pose.append(p_coords)

        # Parse Face
        f_coords = np.zeros((478, 3), dtype=np.float32)
        if face_res.face_landmarks and len(face_res.face_landmarks) > 0:
            f_coords = np.array([[lm.x, lm.y, lm.z] for lm in face_res.face_landmarks[0]], dtype=np.float32)
        raw_face.append(f_coords)

        frame_index += 1
        if frame_index % 30 == 0:
            print(f"Processed {frame_index}/{total_frames} frames for extraction.")

    cap.release()
    hand_detector.close()
    
    raw_left = np.array(raw_left)
    raw_right = np.array(raw_right)
    raw_pose = np.array(raw_pose)
    raw_face = np.array(raw_face)
    
    T = len(raw_left)

    left_mask = ~np.all(raw_left == 0, axis=-1)   # (T, 21)
    right_mask = ~np.all(raw_right == 0, axis=-1) # (T, 21)
    pose_mask = ~np.all(raw_pose == 0, axis=-1)   # (T, 5)
    
    print(f"Step 2: Restoring missing points using {method}...")
    restored_pose, restored_left, restored_right = restore_missing_points(
        raw_pose.copy(), raw_left.copy(), raw_right.copy(), fps=fps, method=method, model_path=model_path
    )
    
    print("Step 3: Generating comparison video...")
    cap = cv2.VideoCapture(video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    hand_drawer = HandDetection()
    pose_drawer = PoseDetection()
    face_drawer = FaceDetection()

    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame_before = frame.copy()
        frame_after = frame.copy()
        
        # ----- BEFORE (Original keypoints) -----
        l_vis = left_mask[frame_idx].any()
        r_vis = right_mask[frame_idx].any()
        p_vis = pose_mask[frame_idx].any()
        f_vis = ~np.all(raw_face[frame_idx] == 0)
        
        orig_hand = create_mock_hands(raw_left[frame_idx], raw_right[frame_idx], l_vis, r_vis)
        orig_pose = create_mock_pose(raw_pose[frame_idx], p_vis)
        orig_face = create_mock_face(raw_face[frame_idx], f_vis)

        frame_before = pose_drawer.draw_landmarks_on_image(frame_before, orig_pose)
        frame_before = face_drawer.draw_landmarks_on_image(frame_before, orig_face)
        frame_before = hand_drawer.draw_landmarks_on_image(frame_before, orig_hand)
        
        # ----- AFTER (Restored keypoints) -----
        r_l_vis = ~np.all(restored_left[frame_idx] == 0)
        r_r_vis = ~np.all(restored_right[frame_idx] == 0)
        r_p_vis = ~np.all(restored_pose[frame_idx] == 0)
        
        rest_hand = create_mock_hands(restored_left[frame_idx], restored_right[frame_idx], r_l_vis, r_r_vis)
        rest_pose = create_mock_pose(restored_pose[frame_idx], r_p_vis)
        
        frame_after = face_drawer.draw_landmarks_on_image(frame_after, orig_face) 
        frame_after = pose_drawer.draw_landmarks_on_image(frame_after, rest_pose)
        frame_after = hand_drawer.draw_landmarks_on_image(frame_after, rest_hand)

        cv2.putText(frame_before, "Before (Original)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_before, f"Frame index: {frame_idx + 1}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame_after, "After (Restored)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        combined_frame = np.concatenate((frame_before, frame_after), axis=1)
        out_video.write(combined_frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Written {frame_idx}/{total_frames} frames to output.")

    cap.release()
    out_video.release()
    hand_drawer.close()
    
    print(f"Finished writing output to {output_path}")

if __name__ == "__main__":
    video_id = "01327"
    parser = argparse.ArgumentParser(description="Visualize restore_missing_points on a video")
    parser.add_argument("--video", default=f"{os.path.join(WLASL_RAW_DATA, video_id)}.mp4", type=str, help="Path to input video")
    parser.add_argument("--output", default=f"{os.path.join('/', video_id)}_compare.mp4", type=str, help="Path to output video")
    parser.add_argument("--method", default="kalman", choices=["linear", "kalman", "model"], type=str, help="Method to restore missing points")
    parser.add_argument("--model_path", default="./outputs/motion_model.pt", type=str, help="Path to the trained model (required if method is 'model')")
    
    args = parser.parse_args()
    process_video(args.video, args.output, args.method, args.model_path)
