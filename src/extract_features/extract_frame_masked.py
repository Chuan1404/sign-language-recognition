import os
import sys

# Support running this script directly from the project root.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [PROJECT_DIR, os.path.join(PROJECT_DIR, "src")]

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import shutil
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

from pipeline_config import (DATA_PATH, IMAGE_SIZE, MAX_FRAMES, MIN_FRAMES, RAW_VIDEO_DIR, SELECTED_RGB_DIR,
                             importance_path, set_seed, )
from src.utils.rgb import bgr_to_chw_uint8, square_box
from config import DEVICE
from pipeline_config import LABEL_DIR, MODEL_DIR



def select_indices(strategy, scores, T, min_frames, max_frames, fixed_k, rng):
    if T < 1:
        return np.empty(0, dtype=np.int64)
    if fixed_k:
        k = min(int(fixed_k), T)
    elif strategy == "selector":
        k = int(np.sum(scores > (1 / T)))
        k = int(np.clip(k, min(min_frames, T), min(max_frames, T)))
    else:
        k = min(max_frames, T)

    if strategy == "selector" and scores.sum() > 0:
        top = np.argsort(-scores, kind="stable")[:k]
        return np.sort(top)
    if strategy == "random":
        return np.sort(rng.choice(T, size=k, replace=False))
    return np.unique(np.linspace(0, T - 1, k).round().astype(np.int64))


def smooth_indices(indices, max_gap=3, max_frames=100):
    """Insert real intermediate frames, keeping all selected anchors.

    Split the largest gap first. The frame budget takes priority over max_gap.
    max_gap=0 disables smoothing.
    """
    indices = np.unique(np.asarray(indices, dtype=np.int64))
    if max_gap < 0 or max_frames < 1:
        raise ValueError("max_gap must be nonnegative and max_frames positive")
    if len(indices) > max_frames:
        raise ValueError("Selected anchors exceed max_frames")
    if max_gap == 0:
        return indices
    while len(indices) > 1 and len(indices) < max_frames:
        gaps = np.diff(indices)
        i = int(np.argmax(gaps))
        if gaps[i] <= max_gap:
            break
        midpoint = (int(indices[i]) + int(indices[i + 1])) // 2
        indices = np.insert(indices, i + 1, midpoint)
    return indices


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


def crop_hand_masked(frame, box, hand, image_size, mask_margin=0.04):
    """Approximate hand foreground with a dilated landmark convex hull.

    Landmarks are normalized in original video coordinates. Missing/degenerate
    landmarks keep the original crop. Background uses the ImageNet mean color.
    """
    crop = crop_hand(frame, box, image_size)
    if box is None or hand is None:
        return crop
    points = np.asarray(hand)[:, :2]
    valid = np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
    points = points[valid]
    if len(points) < 3:
        return crop
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return crop
    xy = (points * [w, h] - [x0, y0]) / [x1 - x0, y1 - y0] * image_size
    xy = np.rint(np.clip(xy, 0, image_size - 1)).astype(np.int32)
    hull = cv2.convexHull(xy)
    if cv2.contourArea(hull) < 1:
        return crop
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    radius = int(np.ceil(mask_margin * image_size))
    if radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.dilate(mask, kernel)
    background = np.rint(np.array([0.485, 0.456, 0.406]) * 255).astype(np.uint8)
    return np.ascontiguousarray(np.where(mask[None] > 0, crop, background[:, None, None]))


def process_video(video_id, meta, args, rng, stats):
    out_dir = os.path.join(args.out_dir, video_id)
    rgb_path = os.path.join(out_dir, "rgb_frames.npy")
    idx_path = os.path.join(out_dir, "indices.npy")

    video_path = os.path.join(args.video_dir, f"{video_id}.mp4")

    scores = np.asarray(meta["importance"], dtype=np.float64)
    T = len(scores)
    indices = select_indices(args.strategy, scores, T, args.min_frames, args.max_frames, args.fixed_k, rng)
    indices = smooth_indices(indices, max_gap=args.max_frame_gap, max_frames=args.max_frames)

    frames = read_all_frames(video_path)
    if not frames:
        stats["skipped_empty_video"] += 1
        return

    if len(indices) == 0 or indices[-1] >= len(frames):
        stats["skipped_frame_length_mismatch"] += 1
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
            [np.stack([
                crop_hand_masked(frames[i], bl, left[i] if i < len(left) else None, image_size, args.mask_margin),
                crop_hand_masked(frames[i], br, right[i] if i < len(right) else None, image_size, args.mask_margin),
            ])
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

    # Remove only generated previews so reruns cannot leave old selected frames.
    prev = os.path.join(out_dir, "preview")
    if os.path.isdir(prev):
        preview_root = Path(prev).resolve()
        for pattern in ("*_left.jpg", "*_right.jpg", "frame_*.jpg"):
            for old_preview in preview_root.glob(pattern):
                if old_preview.is_file() and old_preview.resolve().parent == preview_root:
                    old_preview.unlink()
    if args.save_jpg:
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
    print(f"{video_id}: {len(indices)}/{T} frames, {len(indices) * (2 if two_hands else 1)} hand images")


def load_score_model(checkpoint_path, label_path):
    import torch
    from src.models import ISLR_Transformer_GCN
    with open(os.path.join(label_path, "gloss2idx.json"), encoding="utf-8") as f:
        num_classes = len(json.load(f))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    kwargs = dict(checkpoint.get("model_kwargs", {"num_classes": num_classes}))
    if kwargs.get("num_classes", num_classes) != num_classes:
        raise ValueError("Checkpoint class count does not match label_path")
    model = ISLR_Transformer_GCN(**kwargs)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(DEVICE).eval()


def extract_frame_scores(model, dataset, batch_size, device):
    import torch
    from torch.utils.data import DataLoader
    from src.training.train import collate_fn
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    results = {}
    # Learned importance head; no ground-truth label is used to choose frames.
    with torch.inference_mode():
        for features, _labels, video_mask, video_ids in loader:
            mask = video_mask.to(device=device, dtype=torch.bool)
            scores = model(features.to(device), video_mask=mask)["frame_importance"].cpu()
            for i, video_id in enumerate(video_ids):
                length = int(video_mask[i].sum())
                results[str(video_id)] = {
                    "importance": scores[i, :length].tolist(),
                    "length": length,
                }
    return results


def score_requested_splits(args):
    from src.data.WSASL_raw import WLASLLandmarksDataset
    from src.utils import FusionComponent
    print(f"Loading score checkpoint: {args.checkpoint}")
    model = load_score_model(args.checkpoint, args.label_path)
    combined = {}
    for split in args.splits:
        annotation_path = os.path.join(args.label_path, f"{split}.json")
        with open(annotation_path, encoding="utf-8") as f:
            annotations = json.load(f)
        if not annotations:
            print(f"Skipping empty split: {split}")
            continue
        dataset = WLASLLandmarksDataset(
            args.data_path, args.label_path, FusionComponent(), mode=split
        )
        results = extract_frame_scores(model, dataset, args.batch_size, DEVICE)
        path = importance_path(split)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f)
        print(f"Scores: {split}, {len(results)} videos -> {path}")
        combined.update(results)
    return combined


def main(args):
    if not np.isfinite(args.mask_margin) or not 0 <= args.mask_margin <= 0.5:
        raise ValueError("mask_margin must be between 0 and 0.5")
    if args.fixed_k < 1:
        raise ValueError("top_k must be positive; use an explicit frame budget")
    if args.max_frames < args.fixed_k or args.max_frame_gap < 0:
        raise ValueError("Require max_frames >= top_k and max_frame_gap >= 0")
    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # meta_cfg = {
    #     "strategy": args.strategy, "fixed_k": args.fixed_k, "min_frames": args.min_frames,
    #     "max_frames": args.max_frames, "crop": args.crop, "margin": args.margin, "image_size": IMAGE_SIZE,
    # }
    # meta_path = os.path.join(args.out_dir, "_meta.json")

    if args.strategy != "selector":
        # Uniform/random sampling only needs sequence lengths, no selector inference.
        from src.data.WSASL_raw import WLASLLandmarksDataset
        importance = {}
        for split in args.splits:
            with open(os.path.join(args.label_path, f"{split}.json"), encoding="utf-8") as f:
                if not json.load(f):
                    continue
            dataset = WLASLLandmarksDataset(
                args.data_path, args.label_path, mode=split, cache_landmarks=False,
            )
            for item in dataset.samples:
                length = len(np.load(item["left_hand_path"], mmap_mode="r"))
                if length:
                    importance[item["video_name"]] = {"importance": np.zeros(length)}
    elif args.reuse_scores:
        importance = {}
        for split in args.splits:
            path = importance_path(split)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    importance.update(json.load(f))
    else:
        importance = score_requested_splits(args)
    if not importance:
        raise ValueError("No frame scores available for requested splits")

    if args.overwrite and os.path.exists(args.out_dir):
        print(f"--overwrite: xoá {args.out_dir}")
        shutil.rmtree(args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)

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
    p = argparse.ArgumentParser(description="Top-K + smoothing -> two-hand crops with landmark background masks")
    p.add_argument("--checkpoint", default=os.path.join(MODEL_DIR, "contest_100_v1.pt"))
    p.add_argument("--label_path", default=LABEL_DIR)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--reuse_scores", action="store_true",
                   help="Skip checkpoint inference and use existing importance JSON files")
    p.add_argument("--strategy", choices=["selector", "uniform", "random"], default="selector")
    p.add_argument("--top_k", "--fixed_k", dest="fixed_k", type=int, default=10,
                   help="Anchor count before smoothing (default: 10; final count may be larger)")
    p.add_argument("--min_frames", type=int, default=MIN_FRAMES)
    p.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    p.add_argument("--max_frame_gap", type=int, default=3,
                   help="Fill gaps with real frames up to max_frames; 0 disables smoothing")
    p.add_argument("--crop", choices=["hands", "hand"], default="hands",
                   help="Crop and mask both hands")
    p.add_argument("--mask_margin", type=float, default=0.04,
                   help="Dilate hull by this fraction of crop size (default: 0.04)")
    p.add_argument("--image_size", type=int, default=256,
                   help="Cached square hand image size (default: 256; multimodal training resizes to 64)")
    p.add_argument("--margin", type=float, default=0.20, help="Margin around crop region (default 0.20)")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--video_dir", default=RAW_VIDEO_DIR)
    p.add_argument("--out_dir", default=SELECTED_RGB_DIR + "_masked")
    p.add_argument("--overwrite", action="store_true", default=False)
    p.add_argument("--save_jpg", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
