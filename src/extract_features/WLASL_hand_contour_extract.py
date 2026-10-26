"""Extract 64 color-grown mesh + 21 original MediaPipe points per hand.

Run from the repository root:
    python src/extract_features/WLASL_hand_contour_extract.py
    python src/extract_features/WLASL_hand_contour_extract.py --limit 1
    python src/extract_features/WLASL_hand_contour_extract.py --video path/to/video.mp4

Outputs: left_hand.npy/right_hand.npy (T, 170), ordered [64 mesh, 21 MP],
keypoint_valid.npy (T, 2, 85), valid.npy/landmark_valid.npy (T, 2),
landmark_xyz.npy (T, 2, 21, 3), metadata.json. Hand coordinates are normalized
x/y; missing points are zero with independent mesh/landmark validity.
Contours are 2D silhouettes, not 3D meshes or anatomical correspondences.
Contour indices are cyclically aligned across successive valid frames, as in
hand_mesh_color_region_growing.ipynb. MediaPipe wrist is always hand index 64.
"""
import argparse
import json
import os

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N_NODES = 64
N_LANDMARKS = 21
N_HAND_POINTS = N_NODES + N_LANDMARKS
COLOR_TOLERANCE = (22, 12, 12)
ROI_PADDING = 0.18
SUPPORT_RADIUS = 0.18

EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
         (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
         (13,17),(0,17),(17,18),(18,19),(19,20)]

def segment_hand(frame, points):
    h, w = frame.shape[:2]
    span = max(float(np.ptp(points[:,0])), float(np.ptp(points[:,1])), 10.)
    pad = max(4, int(span * ROI_PADDING))
    x0, y0 = np.maximum(np.floor(points.min(0)).astype(int)-pad, 0)
    x1, y1 = np.minimum(np.ceil(points.max(0)).astype(int)+pad+1, [w,h])
    full = np.zeros((h,w), np.uint8)
    if x1 <= x0 or y1 <= y0:
        return full
    crop = frame[y0:y1, x0:x1]
    pts = np.rint(points-[x0,y0]).astype(int)
    pts[:,0] = np.clip(pts[:,0],0,crop.shape[1]-1)
    pts[:,1] = np.clip(pts[:,1],0,crop.shape[0]-1)
    support = np.zeros(crop.shape[:2], np.uint8)
    radius = max(3, int(np.linalg.norm(points[0]-points[9])*SUPPORT_RADIUS))
    for a,b in EDGES:
        cv2.line(support, tuple(pts[a]), tuple(pts[b]),255,2*radius)
    palm = cv2.convexHull(pts[[0,1,5,9,13,17]])
    cv2.fillConvexPoly(support,palm,255)
    support = cv2.dilate(support, np.ones((3,3),np.uint8))
    lab = cv2.cvtColor(cv2.GaussianBlur(crop,(3,3),0),cv2.COLOR_BGR2LAB)
    union = np.zeros(crop.shape[:2],np.uint8)
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    for x,y in pts:
        # Mỗi seed dùng mask độc lập, không chặn lẫn nhau.
        flood_mask = np.zeros((crop.shape[0]+2,crop.shape[1]+2),np.uint8)
        flood_mask[1:-1,1:-1][support==0] = 1
        image = lab.copy()
        patch = lab[max(0,y-1):y+2,max(0,x-1):x+2]
        image[y,x] = np.median(patch.reshape(-1,3),axis=0).astype(np.uint8)
        cv2.floodFill(image,flood_mask,(int(x),int(y)),(0,0,0),
                      COLOR_TOLERANCE,COLOR_TOLERANCE,flags)
        union[flood_mask[1:-1,1:-1]==255] = 255
    union = cv2.morphologyEx(union,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    union[support==0] = 0
    contours,_ = cv2.findContours(union,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if contours:
        def score(c):
            hits = sum(cv2.pointPolygonTest(c,(float(x),float(y)),False)>=0 for x,y in pts)
            return hits,cv2.contourArea(c)
        best = max(contours,key=score)
        if cv2.contourArea(best)>=10:
            clean = np.zeros_like(union)
            cv2.drawContours(clean,[best],-1,255,cv2.FILLED)
            full[y0:y1,x0:x1] = clean
    return full


def contour_nodes(mask, wrist, previous=None):
    """Sample 64 contour points, matching the notebook's cyclic alignment."""
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    p = max(contours,key=cv2.contourArea)[:,0,:].astype(np.float32)
    if len(p)<3:
        return None
    # Hướng cố định và node 0 gần cổ tay ở mọi frame.
    area = np.sum(p[:,0]*np.roll(p[:,1],-1)-np.roll(p[:,0],-1)*p[:,1])
    if area<0:
        p = p[::-1]
    p = np.roll(p,-int(np.argmin(np.linalg.norm(p-wrist,axis=1))),axis=0)
    closed = np.vstack([p,p[:1]])
    lengths = np.linalg.norm(np.diff(closed,axis=0),axis=1)
    keep = np.r_[True,lengths>1e-6]
    closed = closed[keep]
    distance = np.r_[0,np.cumsum(np.linalg.norm(np.diff(closed,axis=0),axis=1))]
    if distance[-1]<=0:
        return None
    samples = np.linspace(0,distance[-1],N_NODES,endpoint=False)
    nodes = np.column_stack([np.interp(samples,distance,closed[:,i]) for i in range(2)]).astype(np.float32)
    if previous is not None:
        nodes = min((np.roll(nodes, k, axis=0) for k in range(N_NODES)),
                    key=lambda candidate: np.mean((candidate - previous) ** 2))
    return nodes


def hand_keypoints(frame, landmark_xyz, previous=None):
    """Keep original landmarks even if color segmentation produces no contour."""
    h, w = frame.shape[:2]
    xyz = np.asarray(landmark_xyz, dtype=np.float32)
    if xyz.shape != (N_LANDMARKS, 3) or not np.isfinite(xyz).all():
        raise ValueError("Expected 21 finite MediaPipe x/y/z landmarks")
    points = xyz[:, :2] * np.array([w, h], dtype=np.float32)
    nodes = contour_nodes(segment_hand(frame, points), points[0], previous)
    combined = np.zeros((N_HAND_POINTS, 2), dtype=np.float32)
    valid = np.zeros(N_HAND_POINTS, dtype=bool)
    combined[N_NODES:] = xyz[:, :2]
    valid[N_NODES:] = True
    if nodes is not None:
        combined[:N_NODES] = nodes / np.array([w, h], dtype=np.float32)
        valid[:N_NODES] = True
    return combined, valid, nodes

def extract_video(video_path, destination, model_path):
    """Stream a video, using its original frame order and one tracker per video."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {video_path}")
    left, right, validity = [], [], []
    point_validity, landmark_validity, all_landmarks = [], [], []
    previous = [None, None]
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
        timestamp = -1
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                timestamp = max(timestamp + 1, round(len(left) * 1000 / fps))
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                result = detector.detect_for_video(image, timestamp)
                nodes_frame = np.zeros((2, N_HAND_POINTS, 2), dtype=np.float32)
                valid = np.zeros(2, dtype=bool)
                point_valid = np.zeros((2, N_HAND_POINTS), dtype=bool)
                landmark_valid = np.zeros(2, dtype=bool)
                xyz_frame = np.zeros((2, N_LANDMARKS, 3), dtype=np.float32)
                next_previous = [None, None]
                occupied = set()
                order = sorted(range(len(result.hand_landmarks)),
                    key=lambda i: result.handedness[i][0].score, reverse=True)
                for i in order:
                    # Match existing WLASL extractor convention: index 0=right, 1=left.
                    slot = 1 if result.handedness[i][0].index == 0 else 0
                    if slot in occupied:
                        continue
                    occupied.add(slot)
                    xyz = np.array([(p.x, p.y, p.z) for p in result.hand_landmarks[i]], dtype=np.float32)
                    combined, point_mask, nodes = hand_keypoints(frame, xyz, previous[slot])
                    nodes_frame[slot] = combined
                    point_valid[slot] = point_mask
                    xyz_frame[slot] = xyz
                    landmark_valid[slot] = True
                    valid[slot] = nodes is not None
                    next_previous[slot] = nodes
                previous = next_previous
                left.append(nodes_frame[0].reshape(-1))
                right.append(nodes_frame[1].reshape(-1))
                validity.append(valid)
                point_validity.append(point_valid)
                landmark_validity.append(landmark_valid)
                all_landmarks.append(xyz_frame)
    finally:
        cap.release()
    if not left:
        raise ValueError(f"No readable frames: {video_path}")
    os.makedirs(destination, exist_ok=True)
    # Keep source 21-landmark files untouched by writing to a separate output tree.
    np.save(os.path.join(destination, "left_hand.npy"), np.asarray(left, dtype=np.float32), allow_pickle=False)
    np.save(os.path.join(destination, "right_hand.npy"), np.asarray(right, dtype=np.float32), allow_pickle=False)
    np.save(os.path.join(destination, "valid.npy"), np.asarray(validity, dtype=bool), allow_pickle=False)
    np.save(os.path.join(destination, "keypoint_valid.npy"), np.asarray(point_validity, dtype=bool), allow_pickle=False)
    np.save(os.path.join(destination, "landmark_valid.npy"), np.asarray(landmark_validity, dtype=bool), allow_pickle=False)
    np.save(os.path.join(destination, "landmark_xyz.npy"), np.asarray(all_landmarks, dtype=np.float32), allow_pickle=False)
    metadata = dict(video_path=os.path.abspath(video_path), num_frames=len(left), fps=fps,
                    image_size=[w,h], nodes_per_hand=N_HAND_POINTS, coord_dim=2,
                    mesh_nodes=N_NODES, mediapipe_nodes=N_LANDMARKS, format_version=2,
                    point_order="64 mesh, then 21 original MediaPipe landmarks", wrist_index=N_NODES,
                    coordinates="normalized_xy", flatten_order="x0,y0,x1,y1,...",
                    hand_slots=["left", "right"], missing_value=0,
                    node_0="contour anchored at wrist, cyclically aligned with previous frame",
                    handedness="legacy WLASL extractor: category index 0=right, 1=left",
                    color_tolerance=list(COLOR_TOLERANCE), roi_padding=ROI_PADDING,
                    support_radius=SUPPORT_RADIUS,
                    valid_frames=np.sum(validity,axis=0).tolist(),
                    landmark_valid_frames=np.sum(landmark_validity,axis=0).tolist())
    with open(os.path.join(destination, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def discover_videos(labels_dir, video_dir, splits):
    """Only select members of WLASL100; skip missing source videos."""
    seen = set()
    for split in splits:
        annotation = os.path.join(labels_dir, split + ".json")
        with open(annotation, encoding="utf-8") as handle:
            entries = json.load(handle)
        for entry in entries:
            video_id = str(entry["video_id"])
            if video_id in seen:
                continue
            seen.add(video_id)
            path = os.path.join(video_dir, video_id + ".mp4")
            if os.path.isfile(path):
                yield path, entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="D:/SignDetection" if os.path.isdir("D:/SignDetection/datasets") else PROJECT_ROOT)
    parser.add_argument("--video", help="Process a specific video instead of selecting from WLASL100")
    parser.add_argument("--labels-dir")
    parser.add_argument("--video-dir")
    parser.add_argument("--model-path", default=os.path.join(PROJECT_ROOT, "pretrained", "hand_landmarker.task"))
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "datasets", "processed", "wlasl100_hand_mesh_85"))
    parser.add_argument("--splits", nargs="+", choices=["train","val","test"], default=["train","val","test"])
    parser.add_argument("--limit", type=int, default=0, help="Number of existing WLASL100 videos; 0 = all (default)")
    parser.add_argument("--dry-run", action="store_true", help="List selected input/output paths without processing")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    labels = args.labels_dir or os.path.join(args.data_root, "datasets", "annotations", "WLASL100")
    videos = args.video_dir or os.path.join(args.data_root, "datasets", "raw", "WLASL", "videos")
    samples = [(args.video, {})] if args.video else discover_videos(labels, videos, args.splits)
    count = 0
    for video, entry in samples:
        video_id = os.path.splitext(os.path.basename(video))[0]
        destination = os.path.join(args.output_dir, video_id)
        print(f"{video} -> {destination}")
        if not args.dry_run:
            if not os.path.isfile(args.model_path):
                raise FileNotFoundError(args.model_path)
            metadata = extract_video(video, destination, args.model_path)
            if entry.get("gloss") is not None:
                with open(os.path.join(destination, "gloss.txt"), "w", encoding="utf-8") as handle:
                    handle.write(entry["gloss"])
            print(f"Frames: {metadata['num_frames']}; valid left/right: {metadata['valid_frames']}")
        count += 1
        if args.limit and count >= args.limit:
            break
    if count == 0:
        raise FileNotFoundError(f"No WLASL100 videos found in {videos}")
    print(f"{'Selected' if args.dry_run else 'Processed'} {count} video(s).")


if __name__ == "__main__":
    main()
