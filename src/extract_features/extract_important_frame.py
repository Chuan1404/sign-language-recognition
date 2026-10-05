import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import shutil
from collections import Counter

import cv2
import numpy as np

from pipeline_config import (DATA_PATH, IMAGE_SIZE, MAX_FRAMES, MIN_FRAMES, RAW_VIDEO_DIR, SELECTED_RGB_DIR,
                             importance_path, set_seed, )
from src.utils.rgb import bgr_to_chw_uint8, square_box


def select_indices(strategy, scores, T, min_frames, max_frames, fixed_k, rng):
    if fixed_k:
        k = min(int(fixed_k), T)
    elif strategy == "selector":
        k = int(np.sum(scores > (1.5 / T)))
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


def compute_single_hand_box(hand_frame, pose_frame, pose_indices, w, h, margin=0.25, min_size=32):
    """Tính bounding box vuông cho một bàn tay dựa trên landmark tay và/hoặc landmark pose."""
    # 1. Ưu tiên 1: Tọa độ 21 điểm của bàn tay (từ hand landmarker)
    if hand_frame is not None and np.any(hand_frame[:, :2] != 0):
        valid = hand_frame[:, :2]
        valid = valid[np.any(valid != 0, axis=1)]
        valid = valid[(valid[:, 0] >= -0.05) & (valid[:, 0] <= 1.05) & (valid[:, 1] >= -0.05) & (valid[:, 1] <= 1.05)]
        if len(valid) >= 3:
            lo, hi = valid.min(axis=0), valid.max(axis=0)
            x0, y0, x1, y1 = lo[0] * w, lo[1] * h, hi[0] * w, hi[1] * h
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
            side = max(bw, bh) * (1.0 + 2.0 * margin)
            side = max(side, min_size, int(0.12 * min(w, h)))
            side = min(side, min(w, h))
            s = int(round(side))
            left_c = max(0, min(int(round(cx - s / 2.0)), w - s))
            top_c = max(0, min(int(round(cy - s / 2.0)), h - s))
            return left_c, top_c, left_c + s, top_c + s

    # 2. Ưu tiên 2: Landmark cổ tay / ngón tay từ Pose detection (nếu hand detector trượt)
    if pose_frame is not None and pose_indices is not None:
        pts = pose_frame[pose_indices, :2]
        valid = pts[np.any(pts != 0, axis=1)]
        valid = valid[(valid[:, 0] >= 0.0) & (valid[:, 0] <= 1.0) & (valid[:, 1] >= 0.0) & (valid[:, 1] <= 1.0)]
        if len(valid) >= 1:
            cx, cy = valid.mean(axis=0) * np.array([w, h])
            if len(valid) >= 2:
                bw = max((valid[:, 0].max() - valid[:, 0].min()) * w, 1.0)
                bh = max((valid[:, 1].max() - valid[:, 1].min()) * h, 1.0)
                side = max(bw, bh) * (1.0 + 3.0 * margin)
            else:
                side = 0.18 * min(w, h)
            side = max(side, min_size, int(0.15 * min(w, h)))
            side = min(side, min(w, h))
            s = int(round(side))
            left_c = max(0, min(int(round(cx - s / 2.0)), w - s))
            top_c = max(0, min(int(round(cy - s / 2.0)), h - s))
            return left_c, top_c, left_c + s, top_c + s

    # 3. Không có keypoint hợp lệ (tay ở ngoài khung hình / không xuất hiện) -> trả về None (ảnh đen)
    return None


def compute_hand_box_frame(left_frame, right_frame, w, h, margin=0.25, min_size=32):
    pts = []
    for hand in (left_frame, right_frame):
        if hand is not None and np.any(hand[:, :2] != 0):
            valid = hand[:, :2]
            valid = valid[np.any(valid != 0, axis=1)]
            valid = valid[(valid[:, 0] >= -0.1) & (valid[:, 0] <= 1.1) & (valid[:, 1] >= -0.1) & (valid[:, 1] <= 1.1)]
            if len(valid) >= 3:
                pts.append(valid)
    if not pts:
        return None
    xy = np.concatenate(pts, axis=0)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    x0, y0, x1, y1 = lo[0] * w, lo[1] * h, hi[0] * w, hi[1] * h

    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    side = max(bw, bh) * (1.0 + 2.0 * margin)
    side = max(side, min_size)
    side = min(side, min(w, h))

    left_c = int(round(np.clip(cx - side / 2.0, 0, w - side)))
    top_c = int(round(np.clip(cy - side / 2.0, 0, h - side)))
    s = int(round(side))
    return left_c, top_c, min(left_c + s, w), min(top_c + s, h)


def compute_video_hand_box(left, right, w, h, margin=0.25, min_size=32):
    pts = []
    for hand in (left, right):
        if hand is not None:
            xy = hand[:, :, :2].reshape(-1, 2)
            xy = xy[np.any(xy != 0, axis=1)]
            xy = xy[(xy[:, 0] >= -0.1) & (xy[:, 0] <= 1.1) & (xy[:, 1] >= -0.1) & (xy[:, 1] <= 1.1)]
            if len(xy) >= 5:
                pts.append(xy)
    if not pts:
        return None
    all_xy = np.concatenate(pts, axis=0)
    lo, hi = np.percentile(all_xy, 2, axis=0), np.percentile(all_xy, 98, axis=0)
    x0, y0, x1, y1 = lo[0] * w, lo[1] * h, hi[0] * w, hi[1] * h

    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    side = max(bw, bh) * (1.0 + 2.0 * margin)
    side = max(side, min_size)
    side = min(side, min(w, h))

    left_c = int(round(np.clip(cx - side / 2.0, 0, w - side)))
    top_c = int(round(np.clip(cy - side / 2.0, 0, h - side)))
    s = int(round(side))
    return left_c, top_c, min(left_c + s, w), min(top_c + s, h)


def compute_crop_box(feat_dir, w, h, margin):
    try:
        left = np.load(os.path.join(feat_dir, "left_hand.npy"))
        right = np.load(os.path.join(feat_dir, "right_hand.npy"))
        pose = np.load(os.path.join(feat_dir, "pose.npy"))
        T = left.shape[0]
        parts = [pose.reshape(T, 33, 3)[:, :25, :2].reshape(-1, 2), left.reshape(T, 21, 3)[:, :, :2].reshape(-1, 2),
                 right.reshape(T, 21, 3)[:, :, :2].reshape(-1, 2), ]
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


def crop_hand(frame, box, image_size):
    """Crop one hand. A hand that was never detected becomes a black image."""
    if box is None:
        return np.zeros((3, image_size, image_size), dtype=np.uint8)
    return bgr_to_chw_uint8(frame, image_size, box)


def process_video(video_id, meta, args, rng, stats):
    out_dir = os.path.join(args.out_dir, video_id)
    rgb_path = os.path.join(out_dir, "rgb_frames.npy")
    idx_path = os.path.join(out_dir, "indices.npy")

    video_path = os.path.join(args.video_dir, f"{video_id}.mp4")

    scores = np.asarray(meta["importance"], dtype=np.float64)
    T = len(scores)
    indices = select_indices(args.strategy, scores, T, args.min_frames, args.max_frames, args.fixed_k, rng)

    frames = read_all_frames(video_path)
    if not frames:
        stats["skipped_empty_video"] += 1
        return

    h, w = frames[0].shape[:2]
    feat_path = os.path.join(args.data_path, video_id)
    image_size = getattr(args, "image_size", IMAGE_SIZE)
    two_hands = args.crop in ("hands", "hand")

    if two_hands:
        # One crop per hand: rgb_frames.npy has shape (N, 2, 3, H, W), index 0 = left, 1 = right
        try:
            left = np.load(os.path.join(feat_path, "left_hand.npy")).reshape(-1, 21, 3)[:, :, :2]
            right = np.load(os.path.join(feat_path, "right_hand.npy")).reshape(-1, 21, 3)[:, :, :2]
        except Exception:
            stats["skipped_no_landmarks"] += 1
            return

        pose = None
        pose_path = os.path.join(feat_path, "pose.npy")
        if os.path.exists(pose_path):
            try:
                pose = np.load(pose_path).reshape(-1, 33, 3)[:, :, :2]
            except Exception:
                pass

        left_boxes = []
        right_boxes = []
        for idx in indices:
            l_hand = left[idx] if idx < len(left) else None
            r_hand = right[idx] if idx < len(right) else None
            p_frame = pose[idx] if (pose is not None and idx < len(pose)) else None

            bl = compute_single_hand_box(l_hand, p_frame, [15, 17, 19, 21], w, h, args.margin)
            br = compute_single_hand_box(r_hand, p_frame, [16, 18, 20, 22], w, h, args.margin)

            left_boxes.append(bl)
            right_boxes.append(br)

        chw = np.stack(
            [np.stack([crop_hand(frames[i], bl, image_size), crop_hand(frames[i], br, image_size)])
             for i, bl, br in zip(indices, left_boxes, right_boxes)]
        )  # (N, 2, 3, H, W) uint8
    else:
        if args.crop == "signer":
            box = compute_crop_box(feat_path, w, h, args.margin)
            if box is None:
                stats["crop_fallback_fullframe"] += 1
            boxes = [box] * len(indices)
        else:
            boxes = [None] * len(indices)
        chw = np.stack([bgr_to_chw_uint8(frames[i], image_size, b) for i, b in zip(indices, boxes)])  # (N,3,H,W) uint8

    os.makedirs(out_dir, exist_ok=True)
    np.save(rgb_path, chw)
    np.save(idx_path, indices.astype(np.int64))

    if args.save_jpg:
        prev = os.path.join(out_dir, "preview")
        os.makedirs(prev, exist_ok=True)
        for m, arr in zip(indices, chw):
            if two_hands:
                for side, a in zip(("left", "right"), arr):
                    cv2.imwrite(os.path.join(prev, f"{int(m)}_{side}.jpg"),
                                cv2.cvtColor(a.transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(os.path.join(prev, f"frame_{int(m):04d}.jpg"),
                            cv2.cvtColor(arr.transpose(1, 2, 0), cv2.COLOR_RGB2BGR))

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

    os.makedirs(args.out_dir, exist_ok=True)

    importance = {}
    for split in args.splits:
        p = importance_path(split)
        if os.path.exists(p):
            with open(p) as f:
                importance.update(json.load(f))

    stats = Counter()
    for i, (video_id, meta) in enumerate(importance.items(), 1):
        process_video(str(video_id), meta, args, rng, stats)

    print("\nTổng kết:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if stats["frames_total"]:
        print(f"  tỉ lệ frame được giữ: {stats['frames_selected'] / stats['frames_total'] * 100:.1f}%")
    print(f"Đầu ra: {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["selector", "uniform", "random"], default="selector")
    p.add_argument("--fixed_k", type=int, default=0, help="Fixed frame count for all videos")
    p.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    p.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    p.add_argument("--crop", choices=["none", "signer", "hands", "hand"], default="hands",
                   help="Crop mode: none, signer, or hands/hand")
    p.add_argument("--image_size", type=int, default=IMAGE_SIZE,
                   help="Output square image size (default from pipeline_config.IMAGE_SIZE)")
    p.add_argument("--margin", type=float, default=0.20, help="Margin around crop region (default 0.20)")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--video_dir", default=RAW_VIDEO_DIR)
    p.add_argument("--out_dir", default=SELECTED_RGB_DIR)
    p.add_argument("--overwrite", action="store_true", default=True)
    p.add_argument("--save_jpg", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
