import os
import json
import cv2
import numpy as np

# =====================================================================
# CONFIG
# =====================================================================
JSON_PATH_TRAIN = "outputs/models/wlasl100_train_frame_importance.json"
JSON_PATH_VAL = "outputs/models/wlasl100_val_frame_importance.json"
VIDEO_DIR = "datasets/raw/WLASL/videos"  # Thư mục chứa video gốc
OUTPUT_DIR = "outputs/extracted_frames"
TOP_K = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# SCRIPT
# =====================================================================
def extract_frames():
    importance_data = {}
    if os.path.exists(JSON_PATH_TRAIN):
        with open(JSON_PATH_TRAIN, "r") as f:
            importance_data.update(json.load(f))
    else:
        print(f"Warning: Not found {JSON_PATH_TRAIN}")
        
    if os.path.exists(JSON_PATH_VAL):
        with open(JSON_PATH_VAL, "r") as f:
            importance_data.update(json.load(f))
    else:
        print(f"Warning: Not found {JSON_PATH_VAL}")

    print(f"Loaded {len(importance_data)} videos from JSON.")

    for video_id, data in importance_data.items():
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        
        if not os.path.exists(video_path):
            print(f"[Warning] Video not found: {video_path}")
            continue

        scores = np.array(data["importance"])
        valid_length = data["length"]

        # 1. Lấy Top-K index
        # Lấy tối đa TOP_K, hoặc ít hơn nếu video quá ngắn
        k = min(TOP_K, valid_length)
        topk_indices = np.argsort(scores)[-k:]
        
        # Sắp xếp lại theo trình tự thời gian
        topk_indices = np.sort(topk_indices)

        # 2. Đọc video và trích xuất đúng các frame đó
        cap = cv2.VideoCapture(video_path)
        extracted = 0
        frame_idx = 0
        
        video_out_dir = os.path.join(OUTPUT_DIR, str(video_id))
        os.makedirs(video_out_dir, exist_ok=True)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            if frame_idx in topk_indices:
                out_path = os.path.join(video_out_dir, f"frame_{frame_idx:03d}.jpg")
                cv2.imwrite(out_path, frame)
                extracted += 1
                
            frame_idx += 1
            if extracted == k:
                break
                
        cap.release()
        print(f"✓ Extracted {extracted} frames for {video_id}")

if __name__ == "__main__":
    extract_frames()
