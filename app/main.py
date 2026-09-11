import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from app.utils import load_model, get_start_zone, point_in_zone, predict, draw_start_zone, draw_action_zones
from src import FusionComponent, PoseDetection
import sys, numpy as np, cv2 as cv, torch, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import WINDOW_SIZE, ROOT, _COORD_DIM, FRAME_H, FRAME_W, _REMOVE_POSE_IDX, BACKSPACE_ZONE, CLEAR_ZONE
from flask import Flask, render_template, Response, jsonify, send_file, request
from src.utils import HandDetection

_ZONE_ANCHOR_POSE_IDX = (11, 12)

LABEL_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL100")
MODEL_PATH = os.path.join(ROOT, 'outputs', 'models', 'contest_100_v1.pt')
WLASL_VIDEO_DIR = os.path.join(ROOT, "datasets", "raw", "WLASL", "videos")

hand_detection = HandDetection()
pose_detection = PoseDetection()
fusion = FusionComponent()
app = Flask(__name__)
cap = cv.VideoCapture(0, cv.CAP_MSMF)
predicted_text = []

with open(os.path.join(LABEL_DIR, "gloss2idx.json"), "r") as f:
    gloss2idx = json.load(f)
    idx2gloss = {v: k for k, v in gloss2idx.items()}

# Build gloss -> first available video_id mapping
def _build_gloss_list():
    all_json = []
    for split in ("train.json", "test.json", "val.json"):
        p = os.path.join(LABEL_DIR, split)
        if os.path.exists(p):
            with open(p, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_json.extend(data)
    gloss_video = {}
    for item in all_json:
        g = item["gloss"]
        vid = item["video_id"]
        if g not in gloss_video and os.path.exists(os.path.join(WLASL_VIDEO_DIR, f"{vid}.mp4")):
            gloss_video[g] = vid
    return [{"gloss": g, "video_id": v} for g, v in sorted(gloss_video.items())]

gloss_list = _build_gloss_list()

num_classes = len(gloss2idx)
model = load_model(model_path=MODEL_PATH, num_classes=num_classes)

def generate_frames():
    global predicted_text
    recording = False
    frame_index = 0
    wait_missing = 0
    right_hand_buf = []
    left_hand_buf = []

    # Action-zone debounce state
    backspace_frames    = 0
    clear_frames        = 0
    ACTION_HOLD         = 10  # consecutive frames in box before trigger
    backspace_triggered = False
    clear_triggered     = False

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv.resize(frame, (FRAME_W, FRAME_H))
        frame_h, frame_w = frame.shape[:2]

        start_zone = get_start_zone(frame_w, frame_h)

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        timestamp_ms = int(cv.getTickCount() / cv.getTickFrequency() * 1000)

        # detect_hand
        detection_hand_results = hand_detection.detect_video(
            rgb_frame,
            timestamp_ms
        )

        # detect_pose
        detection_pose_results = pose_detection.detect_video(
            rgb_frame,
            timestamp_ms
        )

        # Detect tay trên frame GỐC (chưa lật)
        pose = np.zeros((33, 3), dtype=np.float32)
        right_hand = np.zeros((21, 3), dtype=np.float32)
        left_hand = np.zeros((21, 3), dtype=np.float32)

        pose_detected = False
        right_detected = False
        left_detected = False

        handedness = detection_hand_results.handedness
        hand_landmarks = detection_hand_results.hand_landmarks

        for i, hand_info in enumerate(handedness):
            if i >= len(hand_landmarks):
                continue

            category = hand_info[0]
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks[i]],
                dtype=np.float32
            )

            if category.index == 0:
                right_hand = coords
                right_detected = True

            elif category.index == 1:
                left_hand = coords
                left_detected = True

        pose_landmarks = detection_pose_results.pose_landmarks

        if len(pose_landmarks) > 0:
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in pose_landmarks[0]],
                dtype=np.float32
            )

            pose = coords
            pose_detected = True

        right_in_zone = right_detected and point_in_zone(
            right_hand[0, 0] * frame_w, right_hand[0, 1] * frame_h, start_zone
        )

        left_in_zone = left_detected and point_in_zone(
            left_hand[0, 0] * frame_w, left_hand[0, 1] * frame_h, start_zone
        )

        pose_anchor_in_zone = pose_detected and all(
            point_in_zone(pose[idx, 0] * frame_w, pose[idx, 1] * frame_h, start_zone)
            for idx in _ZONE_ANCHOR_POSE_IDX
        )

        hand_in_zone = (right_in_zone or left_in_zone) and pose_anchor_in_zone

        if not recording and hand_in_zone:
            recording = True
            frame_index = 0
            pose_buf, right_hand_buf, left_hand_buf = [], [], []

        if recording:
            pose_buf.append(pose[:, :_COORD_DIM])
            right_hand_buf.append(right_hand[:, :_COORD_DIM])
            left_hand_buf.append(left_hand[:, :_COORD_DIM])
            frame_index += 1

            if hand_in_zone == False:
                if wait_missing > 9:
                    pose_arr = np.stack(pose_buf)
                    left_arr = np.stack(left_hand_buf)   # (T, 21, 3)
                    right_arr = np.stack(right_hand_buf)  # (T, 21, 3)

                    fused = fusion.fuse(
                        pose_feature=pose_arr,
                        left_feature=left_arr,
                        right_feature=right_arr,
                    )

                    features = torch.tensor(fused, dtype=torch.float32).unsqueeze(0).cuda()
                    video_mask = torch.ones((1, frame_index)).cuda()
                    output = predict(model, features, video_mask)

                    predicted_text.append(idx2gloss[output])

                    recording = False
                    wait_missing = 0
                    frame_index = 0
                    right_hand_buf, left_hand_buf = [], []

                wait_missing += 1


        # ---- Action zones -----------------------------------------------
        def _wrist_in_zone(hand_arr, detected, zone):
            if not detected:
                return False
            wx = hand_arr[0, 0] * frame_w
            wy = hand_arr[0, 1] * frame_h
            return point_in_zone(wx, wy, zone)

        any_in_backspace = (
            _wrist_in_zone(right_hand, right_detected, BACKSPACE_ZONE) or
            _wrist_in_zone(left_hand,  left_detected,  BACKSPACE_ZONE)
        )
        any_in_clear = (
            _wrist_in_zone(right_hand, right_detected, CLEAR_ZONE) or
            _wrist_in_zone(left_hand,  left_detected,  CLEAR_ZONE)
        )

        if any_in_backspace:
            backspace_frames += 1
            if backspace_frames == ACTION_HOLD and not backspace_triggered:
                if predicted_text:
                    predicted_text.pop()
                    print('[ACTION] backspace')
                backspace_triggered = True
        else:
            backspace_frames    = 0
            backspace_triggered = False

        if any_in_clear:
            clear_frames += 1
            if clear_frames == ACTION_HOLD and not clear_triggered:
                predicted_text.clear()
                print('[ACTION] clear')
                clear_triggered = True
        else:
            clear_frames    = 0
            clear_triggered = False
        # -----------------------------------------------------------------

        # Show on stream
        rgb_frame = hand_detection.draw_landmarks_on_image(
            rgb_frame,
            detection_hand_results
        )

        rgb_frame = pose_detection.draw_landmarks_on_image(
            rgb_frame,
            detection_pose_results,
            remove_pose_idx = _REMOVE_POSE_IDX
        )

        frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)

        display_frame = cv.flip(frame, 1)
        display_frame = draw_start_zone(display_frame, start_zone, recording)
        display_frame = draw_action_zones(display_frame, any_in_backspace, any_in_clear)

        # JPEG
        ret, buffer = cv.imencode(
            '.jpg',
            display_frame,
            [cv.IMWRITE_JPEG_QUALITY, 60]
        )

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/predicted_text')
def get_predicted_text():
    return jsonify(words=predicted_text)


@app.route('/glosses')
def get_glosses():
    return jsonify(glosses=gloss_list)


@app.route('/video_file/<video_id>')
def serve_video(video_id):
    # Sanitize: only allow alphanumeric ids
    if not video_id.isalnum():
        return ('Bad Request', 400)
    path = os.path.join(WLASL_VIDEO_DIR, f"{video_id}.mp4")
    if not os.path.exists(path):
        return ('Not Found', 404)
    return send_file(path, mimetype='video/mp4')


@app.route('/clear_text', methods=['POST'])
def clear_text():
    global predicted_text
    predicted_text.clear()
    return jsonify(ok=True)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)