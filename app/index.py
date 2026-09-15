from flask import render_template, Response, jsonify, send_file
import os
import cv2 as cv

from app.utils import get_start_zone, point_in_zone, draw_start_zone, draw_action_zones, wrist_in_zone
from config import FRAME_W, FRAME_H, _COORD_DIM, BACKSPACE_ZONE, CLEAR_ZONE

_ZONE_ANCHOR_POSE_IDX = (11, 12)


class SignDetectionApp:

    def __init__(self, flask_app, service, camera, gloss_list, wlasl_video_dir):
        self.app = flask_app
        self.service = service
        self.camera = camera
        self.gloss_list = gloss_list
        self.wlasl_video_dir = wlasl_video_dir

        self.predicted_text = []

        self.register_routes()

    def register_routes(self):
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule("/video", "video", self.video)
        self.app.add_url_rule("/predicted_text", "predicted_text", self.get_predicted_text)
        self.app.add_url_rule("/glosses", "glosses", self.get_glosses)
        self.app.add_url_rule("/video_file/<video_id>", "video_file", self.serve_video)
        self.app.add_url_rule("/clear_text", "clear_text", self.clear_text, methods=["POST"])

    def index(self):
        return render_template("index.html")

    def video(self):
        return Response(self.generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def get_predicted_text(self):
        return jsonify(words=self.predicted_text)

    def get_glosses(self):
        return jsonify(glosses=self.gloss_list)

    def clear_text(self):
        self.predicted_text.clear()
        return jsonify(ok=True)

    def serve_video(self, video_id):

        if not video_id.isalnum():
            return "Bad Request", 400

        path = os.path.join(self.wlasl_video_dir, f"{video_id}.mp4")

        if not os.path.exists(path):
            return "Not Found", 404

        return send_file(path, mimetype="video/mp4")

    def generate_frames(self):

        recording = False
        frame_index = 0
        wait_missing = 0

        pose_buf = []
        right_hand_buf = []
        left_hand_buf = []

        backspace_frames = 0
        clear_frames = 0

        ACTION_HOLD = 10

        backspace_triggered = False
        clear_triggered = False

        while True:

            success, frame = self.camera.read()

            if not success:
                break

            frame = cv.resize(frame, (FRAME_W, FRAME_H))

            frame_h, frame_w = frame.shape[:2]

            start_zone = get_start_zone(frame_w, frame_h)

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            timestamp_ms = int(
                cv.getTickCount()
                / cv.getTickFrequency()
                * 1000
            )

            # -------------------------
            # Detection
            # -------------------------

            hand_results, pose_results = self.service.detect(
                rgb_frame,
                timestamp_ms,
            )

            (
                detection_hand_results,
                left_coors,
                right_coors,
                left_detected,
                right_detected,
            ) = hand_results

            (
                detection_pose_results,
                pose_coors,
                pose_detected,
            ) = pose_results

            # -------------------------
            # Zone condition
            # -------------------------

            right_in_zone = (
                right_detected
                and point_in_zone(
                    right_coors[0, 0] * frame_w,
                    right_coors[0, 1] * frame_h,
                    start_zone,
                )
            )

            left_in_zone = (
                left_detected
                and point_in_zone(
                    left_coors[0, 0] * frame_w,
                    left_coors[0, 1] * frame_h,
                    start_zone,
                )
            )

            pose_anchor_in_zone = (
                pose_detected
                and all(
                    point_in_zone(
                        pose_coors[idx, 0] * frame_w,
                        pose_coors[idx, 1] * frame_h,
                        start_zone,
                    )
                    for idx in _ZONE_ANCHOR_POSE_IDX
                )
            )

            hand_in_zone = (
                (right_in_zone or left_in_zone)
                and pose_anchor_in_zone
            )

            # -------------------------
            # Start recording
            # -------------------------

            if not recording and hand_in_zone:

                recording = True
                frame_index = 0
                wait_missing = 0

                pose_buf = []
                right_hand_buf = []
                left_hand_buf = []

            # -------------------------
            # Recording
            # -------------------------

            if recording:

                pose_buf.append(pose_coors[:, :_COORD_DIM])
                right_hand_buf.append(right_coors[:, :_COORD_DIM])
                left_hand_buf.append(left_coors[:, :_COORD_DIM])

                frame_index += 1

                if not hand_in_zone:

                    if wait_missing > 9:

                        word = self.service.predict(
                            pose_buf,
                            left_hand_buf,
                            right_hand_buf,
                            self.predicted_text,
                        )

                        self.predicted_text.append(word)

                        recording = False
                        wait_missing = 0
                        frame_index = 0

                        pose_buf = []
                        right_hand_buf = []
                        left_hand_buf = []

                    wait_missing += 1

            # -------------------------
            # Action zones
            # -------------------------

            any_in_backspace = (
                wrist_in_zone(right_coors, right_detected, BACKSPACE_ZONE)
                or wrist_in_zone(left_coors, left_detected, BACKSPACE_ZONE)
            )

            any_in_clear = (
                wrist_in_zone(right_coors, right_detected, CLEAR_ZONE)
                or wrist_in_zone(left_coors, left_detected, CLEAR_ZONE)
            )

            # Backspace

            if any_in_backspace:

                backspace_frames += 1

                if backspace_frames == ACTION_HOLD and not backspace_triggered:

                    if self.predicted_text:
                        self.predicted_text.pop()

                    backspace_triggered = True

            else:

                backspace_frames = 0
                backspace_triggered = False

            # Clear

            if any_in_clear:

                clear_frames += 1

                if clear_frames == ACTION_HOLD and not clear_triggered:

                    self.predicted_text.clear()
                    clear_triggered = True

            else:

                clear_frames = 0
                clear_triggered = False

            # -------------------------
            # Draw
            # -------------------------

            rgb_frame = self.service.draw(
                rgb_frame,
                hand_results,
                pose_results,
            )

            frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)

            display_frame = cv.flip(frame, 1)

            display_frame = draw_start_zone(display_frame, start_zone, recording)

            display_frame = draw_action_zones(display_frame, any_in_backspace, any_in_clear)

            # -------------------------
            # JPEG
            # -------------------------

            ret, buffer = cv.imencode(
                ".jpg",
                display_frame,
                [cv.IMWRITE_JPEG_QUALITY, 60],
            )

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )