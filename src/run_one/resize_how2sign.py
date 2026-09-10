
import os

from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
from config import ROOT, HOW2SIGN_RAW_DATA
import pandas as pd


def center_zoom_resize(frame, zoom_factor=1.5, target_size=(320, 240)):
    """
    Crop center of frame (zoom in) then resize to target size.
    """

    h, w = frame.shape[:2]

    # size after cropping (zoom in)
    crop_w = int(w / zoom_factor)
    crop_h = int(h / zoom_factor)

    # center crop coordinates
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2

    cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]

    # resize to target
    resized = cv2.resize(
        cropped,
        target_size,
        interpolation=cv2.INTER_AREA
    )

    return resized


def process_video(input_path, output_path,
                  zoom_factor=1.5,
                  target_size=(320, 240)):

    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        target_size
    )

    frame_index = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        # convert BGR → RGB (optional for MediaPipe, not needed for output video)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # center zoom + resize
        frame = center_zoom_resize(
            frame,
            zoom_factor=zoom_factor,
            target_size=target_size
        )

        # back to BGR for saving with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out.write(frame)

        frame_index += 1

    cap.release()
    out.release()

    print("Saved to:", output_path)


if __name__ == "__main__":

    SAVE_DIR = os.path.join(ROOT, "datasets", "processed", "how2sign_resized")
    os.makedirs(SAVE_DIR, exist_ok=True)

    csv_path = os.path.join(ROOT, r"datasets\annotations\how2sign_train.csv")
    df = pd.read_csv(csv_path, sep="\t")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_name = row["SENTENCE_NAME"]
        video_path = os.path.join(HOW2SIGN_RAW_DATA, f"{video_name}.mp4")
        video_output_path = os.path.join(SAVE_DIR, f"{video_name}.mp4")

        if not os.path.exists(video_path):
            print("Missing:", video_path)
            continue

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 25.0

        process_video(
            video_path,
            video_output_path,
            zoom_factor=1.5,
            target_size=(320, 240)
        )