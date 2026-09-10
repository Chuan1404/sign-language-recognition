import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import cv2
import numpy as np
from tqdm import tqdm

from src.utils.pose_detection import PoseDetection
from src.utils.hand_detection import HandDetection
from src.utils.face_detection import FaceDetection
from config import ROOT, WLASL_RAW_DATA

SAVE_DIR = os.path.join(ROOT, "datasets", "processed", "wlasl_features_0_6")
os.makedirs(SAVE_DIR, exist_ok=True)

LABELS_PATH = os.path.join(ROOT, "datasets", "annotations", "wlasl_flat.json")

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    label_entries = json.load(f)

print(f"Total instances: {len(label_entries)}")

for entry in tqdm(label_entries, total=len(label_entries)):

    hand_detection = HandDetection(min_hand_detection_confidence=0.6)
    pose_detection = PoseDetection()
    face_detection = FaceDetection()


    video_id = entry["video_id"]
    gloss = entry["gloss"]
    video_path = os.path.join(
        WLASL_RAW_DATA,
        f"{video_id}.mp4"
    )

    dir_name = os.path.join(
        SAVE_DIR,
        video_id
    )

    if not os.path.exists(video_path):
        print("Missing:", video_path)
        continue

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    frame_index = 0

    right_hand_features = []
    left_hand_features = []
    pose_features = []
    lips_features = []

    while True:

        success, frame = cap.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp_ms = int(frame_index * 1000 / fps)

        detection_hand_results = hand_detection.detect_video(
            frame,
            timestamp_ms
        )

        detection_pose_results = pose_detection.detect_video(
            frame,
            timestamp_ms
        )

        detection_face_results = face_detection.detect_video(
            frame,
            timestamp_ms
        )

        # --------------------------------------------------
        # HAND
        # --------------------------------------------------

        right_hand = np.zeros((21, 3), dtype=np.float32)
        left_hand = np.zeros((21, 3), dtype=np.float32)

        handedness = detection_hand_results.handedness
        hand_landmarks = detection_hand_results.hand_landmarks

        if hand_landmarks is not None and len(hand_landmarks) > 0:

            for i, hand_info in enumerate(handedness):

                if i >= len(hand_landmarks):
                    continue

                category = hand_info[0]

                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks[i]],
                    dtype=np.float32
                )

                coords = np.nan_to_num(coords)

                # MediaPipe:
                # index == 0 -> Right
                # index == 1 -> Left

                if category.index == 0:
                    right_hand = coords

                elif category.index == 1:
                    left_hand = coords

        # --------------------------------------------------
        # POSE
        # --------------------------------------------------

        pose_coords = np.zeros((33, 3), dtype=np.float32)

        if (
            detection_pose_results.pose_landmarks is not None
            and len(detection_pose_results.pose_landmarks) > 0
        ):

            pose_landmarks = detection_pose_results.pose_landmarks[0]

            pose_coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in pose_landmarks],
                dtype=np.float32
            )

            pose_coords = np.nan_to_num(pose_coords)

        # --------------------------------------------------
        # LIPS
        # --------------------------------------------------

        lip_coords = np.zeros((40, 3), dtype=np.float32)

        if (
                detection_face_results.face_landmarks is not None
                and len(detection_face_results.face_landmarks) > 0
        ):
            face_landmarks = detection_face_results.face_landmarks[0]

            LIPS = [
                61, 146, 91, 181, 84, 17,
                314, 405, 321, 375, 291,
                185, 40, 39, 37, 0,
                267, 269, 270, 409,
                78, 95, 88, 178, 87,
                14, 317, 402, 318,
                324, 308, 191, 80,
                81, 82, 13,
                312, 311, 310, 415
            ]

            lip_coords = np.array(
                [
                    [
                        lm.x,
                        lm.y,
                        lm.z
                    ]
                    for idx in LIPS
                    for lm in [face_landmarks[idx]]
                ],
                dtype=np.float32
            )

            lip_coords = np.nan_to_num(lip_coords)

        # --------------------------------------------------
        # SAVE FRAME FEATURES
        # --------------------------------------------------

        right_hand_features.append(
            right_hand.flatten()
        )

        left_hand_features.append(
            left_hand.flatten()
        )

        pose_features.append(
            pose_coords.flatten()
        )

        lips_features.append(
            lip_coords.flatten()
        )

        frame_index += 1

    cap.release()
    hand_detection.close()

    if len(right_hand_features) == 0:
        print(f"Skip empty video: {video_id}")
        continue

    # --------------------------------------------------
    # TO NUMPY
    # --------------------------------------------------

    right_hand_features = np.array(
        right_hand_features,
        dtype=np.float32
    )

    left_hand_features = np.array(
        left_hand_features,
        dtype=np.float32
    )

    pose_features = np.array(
        pose_features,
        dtype=np.float32
    )

    lips_features = np.array(
        lips_features,
        dtype=np.float32
    )

    right_hand_features = np.nan_to_num(
        right_hand_features
    )

    left_hand_features = np.nan_to_num(
        left_hand_features
    )

    pose_features = np.nan_to_num(
        pose_features
    )


    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    os.makedirs(
        dir_name,
        exist_ok=True
    )

    np.save(
        os.path.join(
            dir_name,
            "right_hand.npy"
        ),
        right_hand_features
    )

    np.save(
        os.path.join(
            dir_name,
            "left_hand.npy"
        ),
        left_hand_features
    )

    np.save(
        os.path.join(
            dir_name,
            "pose.npy"
        ),
        pose_features
    )

    np.save(
        os.path.join(
            dir_name,
            "lips.npy"
        ),
        lips_features
    )

    # Save gloss for manual inspection / debugging convenience —
    # WLASLLandmarksDataset still reads the main label from the labels
    # JSON, not from this file, during training
    with open(
        os.path.join(
            dir_name,
            "gloss.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(gloss.strip())

    print(
        f"Saved: {video_id} ({gloss}) | "
        f"RH={right_hand_features.shape} "
        f"LH={left_hand_features.shape} "
        f"POSE={pose_features.shape}",
        f"LIPS={lips_features.shape}"
    )

print("FINISH")