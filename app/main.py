import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from app.utils import load_model, get_start_zone, point_in_zone, predict, draw_start_zone
from src import FusionComponent
import sys, numpy as np, cv2 as cv, torch, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.SLT_model import ISLR_V1
from config import WINDOW_SIZE, ROOT, _COORD_DIM, FRAME_H, FRAME_W
from flask import Flask, render_template, Response
from src.utils import HandDetection

LABEL_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL100")
MODEL_PATH = os.path.join(ROOT, 'outputs', 'models', 'contest_100_v1.pt')

hand_detection = HandDetection()
fusion = FusionComponent()
video_mask = torch.ones((1, WINDOW_SIZE)).cuda()
app = Flask(__name__)
cap = cv.VideoCapture(0, cv.CAP_MSMF)

with open(os.path.join(LABEL_DIR, "gloss2idx.json"), "r") as f:
    gloss2idx = json.load(f)
    idx2gloss = {v: k for k, v in gloss2idx.items()}

num_classes = len(gloss2idx)
model = load_model(model_path=MODEL_PATH, num_classes=num_classes)

def generate_frames():
    recording = False
    frame_index = 0
    right_hand_buf = []
    left_hand_buf = []
    predicted_text = []

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame = cv.resize(frame, (FRAME_W, FRAME_H))
        frame_h, frame_w = frame.shape[:2]

        start_zone = get_start_zone(frame_w, frame_h)

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        timestamp_ms = int(cv.getTickCount() / cv.getTickFrequency() * 1000)

        detection_hand_results = hand_detection.detect_video(
            rgb_frame,
            timestamp_ms
        )

        # Detect tay trên frame GỐC (chưa lật)
        right_hand = np.zeros((21, 3), dtype=np.float32)
        left_hand = np.zeros((21, 3), dtype=np.float32)
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

        right_in_zone = right_detected and point_in_zone(
            right_hand[0, 0] * frame_w, right_hand[0, 1] * frame_h, start_zone
        )
        left_in_zone = left_detected and point_in_zone(
            left_hand[0, 0] * frame_w, left_hand[0, 1] * frame_h, start_zone
        )
        hand_in_zone = right_in_zone or left_in_zone

        if not recording and hand_in_zone:
            recording = True
            frame_index = 0
            right_hand_buf, left_hand_buf = [], []

        if recording:
            right_hand_buf.append(right_hand[:, :_COORD_DIM])
            left_hand_buf.append(left_hand[:, :_COORD_DIM])
            frame_index += 1

            if frame_index == WINDOW_SIZE:
                left_arr = np.stack(left_hand_buf)   # (T, 21, 3)
                right_arr = np.stack(right_hand_buf)  # (T, 21, 3)

                fused = fusion.fuse_follow_hand(
                    pose_feature=None,
                    left_feature=left_arr,
                    right_feature=right_arr,
                    use_pose=False
                )  # (T, 42*_COORD_DIM)

                features = torch.tensor(fused, dtype=torch.float32).unsqueeze(0).cuda()
                output = predict(model, features, video_mask)

                predicted_text.append(idx2gloss[output])
                print(idx2gloss[output])

                recording = False
                frame_index = 0
                right_hand_buf, left_hand_buf = [], []

        # Show on stream
        rgb_frame = hand_detection.draw_landmarks_on_image(
            rgb_frame,
            detection_hand_results
        )

        frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)

        display_frame = cv.flip(frame, 1)
        display_frame = draw_start_zone(display_frame, start_zone, recording)

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


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)