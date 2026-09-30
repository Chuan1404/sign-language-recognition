import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import shutil
from collections import Counter

import cv2
import numpy as np

from pipeline_config import (
    DATA_PATH, IMAGE_SIZE, MAX_FRAMES, MIN_FRAMES, RAW_VIDEO_DIR, SELECTED_RGB_DIR,
    importance_path, set_seed,
)
from src.utils.rgb import bgr_to_chw_uint8, square_box

def select_indices(strategy, scores, T, min_frames, max_frames, fixed_k, rng):
    if fixed_k:
        k = min(int(fixed_k), T)
    elif strategy == "selector":
        k = int(np.sum(scores > (1.0 / T)))
        k = int(np.clip(k, min(min_frames, T), min(max_frames, T)))
    else:
        k = min(max_frames, T)

    if strategy == "selector" and scores.sum() > 0:
        top = np.argsort(-scores, kind="stable")[:k]
        return np.sort(top)
    if strategy == "random":
        return np.sort(rng.choice(T, size=k, replace=False))
    return np.unique(np.linspace(0, T - 1, k).round().astype(np.int64))

def read_all_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames

def compute_crop_box(feat_dir, w, h, margin):
    """bbox vuông quanh người ký (upper-body pose + 2 tay), cố định cho cả video để ổn định theo thời gian.
    Giả định toạ độ landmark chuẩn hóa [0,1] theo khung hình; nếu không thoả thì trả None (dùng toàn khung)."""
    try:
        left = np.load(os.path.join(feat_dir, "left_hand.npy"))
        right = np.load(os.path.join(feat_dir, "right_hand.npy"))
        pose = np.load(os.path.join(feat_dir, "pose.npy"))
        T = left.shape[0]
        parts = [
            pose.reshape(T, 33, 3)[:, :25, :2].reshape(-1, 2),
            left.reshape(T, 21, 3)[:, :, :2].reshape(-1, 2),
            right.reshape(T, 21, 3)[:, :, :2].reshape(-1, 2),
        ]
        xy = np.concatenate(parts)
    except Exception:
        return None
    xy = xy[np.any(xy != 0, axis=1)]  # bỏ điểm thiếu (0,0)
    if len(xy) < 10:
        return None
    if np.mean((xy > -0.05) & (xy < 1.05)) < 0.9:
        return None
    lo, hi = np.percentile(xy, 2, axis=0), np.percentile(xy, 98, axis=0)
    x0, y0, x1, y1 = lo[0] * w, lo[1] * h, hi[0] * w, hi[1] * h
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return square_box(x0, y0, x1, y1, w, h, margin)

def process_video(video_id, meta, args, rng, stats):
    out_dir = os.path.join(args.out_dir, video_id)
    rgb_path = os.path.join(out_dir, "rgb_frames.npy")
    idx_path = os.path.join(out_dir, "indices.npy")

    video_path = os.path.join(args.video_dir, f"{video_id}.mp4")

    scores = np.asarray(meta["importance"], dtype=np.float64)
    T = len(scores)
    indices = select_indices(args.strategy, scores, T, args.min_frames, args.max_frames, args.fixed_k, rng)

    frames = read_all_frames(video_path)

    box = None
    if args.crop == "signer":
        h, w = frames[0].shape[:2]
        box = compute_crop_box(os.path.join(args.data_path, video_id), w, h, args.margin)
        if box is None:
            stats["crop_fallback_fullframe"] += 1

    chw = np.stack([bgr_to_chw_uint8(frames[i], IMAGE_SIZE, box) for i in indices])  # (N,3,H,W) uint8

    os.makedirs(out_dir, exist_ok=True)
    np.save(rgb_path, chw)
    np.save(idx_path, indices.astype(np.int64))

    if args.save_jpg:
        prev = os.path.join(out_dir, "preview")
        os.makedirs(prev, exist_ok=True)
        for m, arr in zip(indices, chw):
            cv2.imwrite(os.path.join(prev, f"frame_{int(m):04d}.jpg"), cv2.cvtColor(arr.transpose(1, 2, 0), cv2.COLOR_RGB2BGR))

    stats["done"] += 1
    stats["frames_total"] += T
    stats["frames_selected"] += len(indices)

def main(args):
    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # meta_cfg = {
    #     "strategy": args.strategy, "fixed_k": args.fixed_k, "min_frames": args.min_frames,
    #     "max_frames": args.max_frames, "crop": args.crop, "margin": args.margin, "image_size": IMAGE_SIZE,
    # }
    # meta_path = os.path.join(args.out_dir, "_meta.json")

    if args.overwrite and os.path.exists(args.out_dir):
        print(f"--overwrite: xoá {args.out_dir}")
        shutil.rmtree(args.out_dir)

    # if os.path.exists(meta_path):
    #     with open(meta_path) as f:
    #         old = json.load(f)
    #     if old != meta_cfg:
    #         raise SystemExit(
    #             f"[LỖI] {args.out_dir} được tạo với cấu hình khác:\n  cũ : {old}\n  mới: {meta_cfg}\n"
    #             f"Dùng --out_dir khác hoặc --overwrite."
    #         )
    os.makedirs(args.out_dir, exist_ok=True)
    # with open(meta_path, "w") as f:
    #     json.dump(meta_cfg, f, indent=2)

    importance = {}
    for split in args.splits:
        p = importance_path(split)
        if os.path.exists(p):
            with open(p) as f:
                importance.update(json.load(f))

    stats = Counter()
    for i, (video_id, meta) in enumerate(importance.items(), 1):
        process_video(str(video_id), meta, args, rng, stats)

    # print("\nTổng kết:")
    # for k, v in stats.items():
    #     print(f"  {k}: {v}")
    # if stats["frames_total"]:
    #     print(f"  tỉ lệ frame được giữ: {stats['frames_selected'] / stats['frames_total'] * 100:.1f}%")
    # print(f"Đầu ra: {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["selector", "uniform", "random"], default="selector")
    p.add_argument("--fixed_k", type=int, default=0, help="Số frame cố định cho mọi video (dùng khi so sánh chiến lược)")
    p.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    p.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    p.add_argument("--crop", choices=["none", "signer"], default="signer")
    p.add_argument("--margin", type=float, default=0.15)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--video_dir", default=RAW_VIDEO_DIR)
    p.add_argument("--out_dir", default=SELECTED_RGB_DIR)
    p.add_argument("--overwrite", action="store_true", default=True)
    p.add_argument("--save_jpg", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())