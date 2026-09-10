import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
from matplotlib import pyplot as plt
from mediapipe.tasks.python.vision.drawing_utils import DrawingSpec
import cv2

mp_drawing_styles = vision.drawing_styles
mp_drawing_utils = vision.drawing_utils

class FaceDetection:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path=r'../../pretrained/face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.3,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            )
        self.face_detector = vision.FaceLandmarker.create_from_options(options)

    def detect_video(self, frame, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.face_detector.detect_for_video(mp_image, timestamp_ms=timestamp_ms)

        return detection_result

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)
        
        h, w, _ = annotated_image.shape
        scale = w / 640.0
        radius = max(1, int(1 * scale))
        thickness = max(1, int(1 * scale))

        small_landmark_style = DrawingSpec(
            color=(0, 255, 0),
            thickness=thickness,
            circle_radius=radius
        )

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.

            mp_drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=small_landmark_style,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=small_landmark_style,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=small_landmark_style,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
            mp_drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=small_landmark_style,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        return annotated_image

    def plot_face_blendshapes_bar_graph(self, face_blendshapes):
        # Extract the face blendshapes category names and scores.
        face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in
                                  face_blendshapes]
        face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
        # The blendshapes are ordered in decreasing score value.
        face_blendshapes_ranks = range(len(face_blendshapes_names))

        fig, ax = plt.subplots(figsize=(12, 12))
        bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
        ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
        ax.invert_yaxis()

        # Label each bar with values
        for score, patch in zip(face_blendshapes_scores, bar.patches):
            plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

        ax.set_xlabel('Score')
        ax.set_title("Face Blendshapes")
        plt.tight_layout()
        plt.show()

    def draw_lips_on_image(self, rgb_image, detection_result):
        annotated_image = np.copy(rgb_image)

        face_landmarks_list = detection_result.face_landmarks
        h, w = annotated_image.shape[:2]

        for face_landmarks in face_landmarks_list:

            # Vẽ các đường nối của môi
            for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS:
                start = face_landmarks[connection.start]
                end = face_landmarks[connection.end]

                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))

                cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)

            # Vẽ các điểm môi
            lip_indices = set()
            for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS:
                lip_indices.add(connection.start)
                lip_indices.add(connection.end)

            for idx in lip_indices:
                lm = face_landmarks[idx]
                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(annotated_image, (x, y), 1, (0, 0, 255), -1)

        return annotated_image

    def get_lips_keypoints(self, detection_result, image_shape):
        """
        Extract lip landmarks from MediaPipe Face Landmarker.

        Returns:
            np.ndarray:
                shape = (N, 3)
                [x, y, z]
        """

        if len(detection_result.face_landmarks) == 0:
            return None

        face_landmarks = detection_result.face_landmarks[0]

        h, w = image_shape[:2]

        # MediaPipe lip landmarks
        LIPS = [
            61, 146, 91, 181, 84, 17, 314, 405,
            321, 375, 291,
            185, 40, 39, 37, 0,
            267, 269, 270, 409,
            78, 95, 88, 178, 87, 14,
            317, 402, 318, 324, 308,
            191, 80, 81, 82, 13,
            312, 311, 310, 415
        ]

        lip_keypoints = []

        for idx in LIPS:
            lm = face_landmarks[idx]

            x = lm.x * w
            y = lm.y * h
            z = lm.z

            lip_keypoints.append([x, y, z])

        return np.array(lip_keypoints, dtype=np.float32)