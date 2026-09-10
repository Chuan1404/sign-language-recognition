"""
Chuẩn bị dữ liệu train cho KeypointMotionTransformer TỪ FEATURE HOW2SIGN ĐÃ
EXTRACT SẴN (pose.npy / left_hand.npy / right_hand.npy mỗi video, cùng cấu
trúc thư mục với full_body_wlasl) — KHÔNG detect lại từ video nữa.

CẮT RA các đoạn liên tục hoàn toàn sạch — không có 1 frame nào bị thiếu
pose/tay trái/tay phải — để làm dữ liệu train cho KeypointMotionTransformer.

Vì sao phải lọc "sạch 100%" thay vì giữ nguyên rồi impute như trước:
    Model này học để BIẾT chuyển động thật trông như thế nào, từ đó mới có
    thể "đoán" đúng phần bị che ở giai đoạn suy luận (inference). Nếu data
    train đã bị lẫn các đoạn thiếu/nội suy/kalman-smooth, model sẽ học nhầm
    "chuyển động giả" do chính các heuristic cũ tạo ra, thay vì chuyển động
    người ký thật. Do đó bước lọc này BẮT BUỘC dùng feature THÔ đã extract
    (chưa qua restore_missing_points/kalman_filter).

Output: mỗi đoạn sạch đủ dài được cắt thành các cửa sổ (window) có độ dài cố
định `window_size`, lưu riêng từng cửa sổ vào 1 thư mục:
    output_dir/{video_id}_{window_idx:04d}/
        pose.npy        (window_size, 33, 3)
        left_hand.npy   (window_size, 21, 3)
        right_hand.npy  (window_size, 21, 3)
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import numpy as np
from tqdm import tqdm

from config import HOW2SIGN_RAW_DATA, ROOT


DATA_DIR = os.path.join(ROOT, "datasets", "processed", "full_body_how2sign")
OUTPUT_DIR = os.path.join(ROOT, "datasets", "processed", "motion_how2sign")


def _to_joints(arr, n_joints):
    """
    Feature đã extract có thể lưu dạng phẳng (T, n_joints*3) HOẶC đã ở dạng
    (T, n_joints, 3) tuỳ pipeline extract trước đó — hàm này nhận cả 2, luôn
    trả về (T, n_joints, 3) để nhất quán với find_clean_runs/motion_dataset.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        return arr[:, :n_joints, :3]
    if arr.ndim == 2:
        T = arr.shape[0]
        return arr.reshape(T, n_joints, -1)[:, :, :3]
    raise ValueError(f"Shape feature không hợp lệ: {arr.shape}")


def find_clean_runs(pose, left, right, min_len):
    """
    Trả về list (start, end) — các đoạn [start, end) mà CẢ pose, left, right
    đều detect được ở MỌI frame trong đoạn (không thiếu dù chỉ 1 frame, ở
    bất kỳ luồng nào), độ dài >= min_len.
    """
    T = pose.shape[0]
    pose_ok = ~np.all(pose == 0, axis=(1, 2))
    left_ok = ~np.all(left == 0, axis=(1, 2))
    right_ok = ~np.all(right == 0, axis=(1, 2))

    all_ok = pose_ok & left_ok & right_ok   # (T,) True nếu frame này sạch tuyệt đối

    runs = []
    start = None
    for t in range(T):
        if all_ok[t] and start is None:
            start = t
        elif not all_ok[t] and start is not None:
            if t - start >= min_len:
                runs.append((start, t))
            start = None
    if start is not None and T - start >= min_len:
        runs.append((start, T))

    return runs


def slice_windows(start, end, window_size, stride):
    """Cắt [start, end) thành các cửa sổ độ dài cố định window_size, bước nhảy stride."""
    windows = []
    t = start
    while t + window_size <= end:
        windows.append((t, t + window_size))
        t += stride
    return windows


def process_from_features(pose, left, right, video_id, output_dir,
                           window_size=64, stride=32, min_len=64):

    if pose.shape[0] < min_len:
        return 0

    runs = find_clean_runs(pose, left, right, min_len=min_len)

    n_saved = 0
    for run_start, run_end in runs:
        windows = slice_windows(run_start, run_end, window_size, stride)
        for s, e in windows:
            seg_dir = os.path.join(output_dir, f"{video_id}_{n_saved:04d}")
            os.makedirs(seg_dir, exist_ok=True)
            np.save(os.path.join(seg_dir, "pose.npy"), pose[s:e])
            np.save(os.path.join(seg_dir, "left_hand.npy"), left[s:e])
            np.save(os.path.join(seg_dir, "right_hand.npy"), right[s:e])
            n_saved += 1

    return n_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR, help="Thư mục feature How2Sign đã extract sẵn")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--window_size", type=int, default=64, help="Số frame mỗi segment train")
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--min_len", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_video_names = sorted(os.listdir(args.data_dir))

    total_segments = 0
    skipped_missing = 0

    for video_name in tqdm(all_video_names, desc="Processing"):
        video_dir = os.path.join(args.data_dir, video_name)

        if not os.path.isdir(video_dir):
            continue

        left_hand_path = os.path.join(video_dir, "left_hand.npy")
        right_hand_path = os.path.join(video_dir, "right_hand.npy")
        pose_path = os.path.join(video_dir, "pose.npy")

        if not (os.path.exists(left_hand_path)
                and os.path.exists(right_hand_path)
                and os.path.exists(pose_path)):
            skipped_missing += 1
            continue

        left_features = _to_joints(np.load(left_hand_path), n_joints=21)
        right_features = _to_joints(np.load(right_hand_path), n_joints=21)
        pose_features = _to_joints(np.load(pose_path), n_joints=33)

        T = min(pose_features.shape[0], left_features.shape[0], right_features.shape[0])
        if T != pose_features.shape[0] or T != left_features.shape[0] or T != right_features.shape[0]:
            # pose/left/right lệch số frame với nhau (không nên xảy ra nếu extract
            # đồng bộ, nhưng cắt về min chung để an toàn thay vì crash/lệch index)
            pose_features = pose_features[:T]
            left_features = left_features[:T]
            right_features = right_features[:T]

        n_saved = process_from_features(
            pose_features, left_features, right_features,
            video_id=video_name, output_dir=args.output_dir,
            window_size=args.window_size, stride=args.stride, min_len=args.min_len,
        )
        total_segments += n_saved

    if skipped_missing:
        print(f"Bỏ qua {skipped_missing} video thiếu file feature.")
    print(f"Đã lưu {total_segments} segment sạch vào {args.output_dir}")


if __name__ == "__main__":
    main()