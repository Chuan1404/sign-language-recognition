import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.utils.pose_detection import PoseDetection

from src.utils.face_detection import FaceDetection
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import HOW2SIGN_RAW_DATA, ROOT
from src.utils.hand_detection import HandDetection

SAVE_DIR = os.path.join(ROOT, "datasets", "processed", "full_body_how2sign",)
os.makedirs(SAVE_DIR, exist_ok=True)

csv_path = os.path.join(ROOT, r"datasets\annotations\how2sign_train.csv")
df = pd.read_csv(csv_path, sep="\t")


# cap = cv2.VideoCapture(os.path.join(r"D:\archive\videos", "00335.mp4"))
# cap = cv2.VideoCapture(os.path.join(ROOT, "datasets", "processed", "how2sign_resized", "_2u0MkRqpjA_3-5-rgb_front.mp4"))
# cap = cv2.VideoCapture(os.path.join(HOW2SIGN_RAW_DATA, "_2u0MkRqpjA_3-5-rgb_front.mp4"))

face_detection = FaceDetection()
pose_detection = PoseDetection()
# fusion_component = FusionComponent()
#
# frame_index = 0
# fps = cap.get(cv2.CAP_PROP_FPS)
# if fps is None or fps <= 0:
#     fps = 25.0
#
# while True:
#
#     success, frame = cap.read()
#     if not success:
#         break
#
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     timestamp_ms = int(frame_index * 1000 / fps)
#
#     detection_face_results = face_detection.detect_video(frame, timestamp_ms)
#     detection_hand_results = hand_detection.detect_video(frame, timestamp_ms)
#     detection_pose_results = pose_detection.detect_video(frame, timestamp_ms)
#
#     # fuse_result = fusion_component.fuse(detection_pose_results,detection_hand_results)
#
#     # frame = face_detection.draw_landmarks_on_image(frame, detection_face_results)
#     # frame = hand_detection.draw_landmarks_on_image(frame, detection_hand_results)
#     # frame = pose_detection.draw_landmarks_on_image(frame, detection_pose_results)
#
#     # frame = fusion_component.draw_landmarks_on_image(frame, fuse_result)
#     # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#
#     # cv2.imshow("test", frame)
#     frame_index += 1
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#
# cv2.destroyAllWindows()
# cap.release()

for _, row in tqdm(df.iterrows(), total=len(df)):

    hand_detection = HandDetection()
    pose_detection = PoseDetection()

    video_name = row["SENTENCE_NAME"]
    sentence = row["SENTENCE"]

    video_path = os.path.join(
        ROOT,
        "datasets",
        "processed",
        "how2sign_resized",
        f"{video_name}.mp4"
    )

    dir_name = os.path.join(
        SAVE_DIR,
        video_name
    )

    if os.path.isdir(dir_name):
        continue


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

        frame_index += 1

    cap.release()
    hand_detection.close()

    if len(right_hand_features) == 0:
        print(f"Skip empty video: {video_name}")
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

    dir_name = os.path.join(
        SAVE_DIR,
        video_name
    )

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

    with open(
        os.path.join(
            dir_name,
            "text.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            sentence.lower().strip()
        )

    print(
        f"Saved: {video_name} | "
        f"RH={right_hand_features.shape} "
        f"LH={left_hand_features.shape} "
        f"POSE={pose_features.shape}"
    )

print("FINISH")