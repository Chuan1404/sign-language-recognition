import os

from src import FusionComponent

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys, numpy as np, cv2 as cv, torch, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.SLT_model import ISLR_V1
from config import WINDOW_SIZE, ROOT, _COORD_DIM
from flask import Flask, render_template, Response
from src.utils import HandDetection

LABEL_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL100")
MODEL_PATH = os.path.join(ROOT, 'outputs', 'models', 'contest_100_v1.pt')


fusion = FusionComponent()

with open(os.path.join(LABEL_DIR, "gloss2idx.json"), "r") as f:
    gloss2idx = json.load(f)
    idx2gloss = {v: k for k, v in gloss2idx.items()}

num_classes = len(gloss2idx)
model_kwargs = dict(
    num_classes=num_classes,
)
model = ISLR_V1(**model_kwargs)


checkpoint = torch.load(
    MODEL_PATH,
    map_location="cuda"
)
model.load_state_dict(checkpoint["model"])
model = model.cuda()
model.eval()

video_mask = torch.ones((1, WINDOW_SIZE)).cuda()

app = Flask(__name__)
hand_detection = HandDetection()
cap = cv.VideoCapture(0, cv.CAP_MSMF)

def predict(model, features, video_mask):

    with torch.no_grad():
        logits, loss = model(features, video_mask=video_mask)
    predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class

def generate_frames():
    frame_index = 0
    right_hand_buf = []
    left_hand_buf = []
    predicted_text = []

    while True:
        if frame_index == WINDOW_SIZE:
            left_arr = np.stack(left_hand_buf)  # (T, 21, 3)
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
            frame_index = 0
            right_hand_buf, left_hand_buf = [], []
            continue  # tránh append thêm frame đang predict

            print(idx2gloss[output])

        success, frame = cap.read()

        if not success:
            break

        frame = cv.resize(frame, (640, 480))

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        timestamp_ms = int(cv.getTickCount() / cv.getTickFrequency() * 1000)

        detection_hand_results = hand_detection.detect_video(
            rgb_frame,
            timestamp_ms
        )

        # Insert frame into windows
        right_hand = np.zeros((21, 3), dtype=np.float32)
        left_hand = np.zeros((21, 3), dtype=np.float32)

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

            elif category.index == 1:
                left_hand = coords

        right_hand_buf.append(right_hand[:, :_COORD_DIM])
        left_hand_buf.append(left_hand[:, :_COORD_DIM])

        # Show on stream
        rgb_frame = hand_detection.draw_landmarks_on_image(
            rgb_frame,
            detection_hand_results
        )

        frame = cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)

        # JPEG
        ret, buffer = cv.imencode(
            '.jpg',
            frame,
            [cv.IMWRITE_JPEG_QUALITY, 60]
        )

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        frame_index += 1

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
