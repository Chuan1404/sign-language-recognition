"""Extract WLASL-compatible landmarks from MSASL frame directories.

Run from repository root:
    python -m src.extract_features.MSASL_feature_extract --dry-run
    python -m src.extract_features.MSASL_feature_extract
Input: processdata/processed_data_MS_ASL100_{Train,Val,Test}/gloss/sample/frame_*.jpg
Output: msasl_features_0_6/{train,val,test}/gloss/sample/*.npy
FPS is a timestamp assumption for pre-extracted frames, not recovered source FPS.
"""
import argparse
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[2]
LIPS = [61,146,91,181,84,17,314,405,321,375,291,185,40,39,37,0,267,269,270,409,
        78,95,88,178,87,14,317,402,318,324,308,191,80,81,82,13,312,311,310,415]
SPLITS = {"train": "Train", "val": "Val", "test": "Test"}
WIDTHS = {"right_hand": 63, "left_hand": 63, "pose": 99}
# Temporarily disabled: "lips": 120


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def discover_samples(input_dir, splits):
    for split in splits:
        directory = input_dir / "processdata" / f"processed_data_MS_ASL100_{SPLITS[split]}"
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for gloss_dir in sorted(directory.iterdir(), key=natural_key):
            if not gloss_dir.is_dir():
                continue
            for sample in sorted(gloss_dir.iterdir(), key=natural_key):
                if not sample.is_dir():
                    continue
                frames = sorted((p for p in sample.iterdir()
                                 if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
                                key=natural_key)
                if frames:
                    yield split, gloss_dir.name, sample, frames


def complete(destination, frame_count, fps):
    import numpy as np
    try:
        meta = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
        if meta["num_frames"] != frame_count or meta["fps"] != fps:
            return False
        for name, width in WIDTHS.items():
            array = np.load(destination / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            if array.shape != (frame_count, width) or array.dtype != np.float32:
                return False
        return (destination / "gloss.txt").is_file()
    except (OSError, ValueError, KeyError):
        return False


def extract_sample(frames, fps, model_dir, confidence):
    import cv2
    import numpy as np
    import mediapipe as mp
    from contextlib import ExitStack
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # New VIDEO detectors for each sample reset temporal tracking and timestamps.
    with ExitStack() as stack:
        hand = stack.enter_context(vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_dir / "hand_landmarker.task")),
                running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=confidence)))
        pose = stack.enter_context(vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model_dir / "pose_landmarker_heavy.task")),
                running_mode=vision.RunningMode.VIDEO, min_pose_detection_confidence=0.3)))
        # face = stack.enter_context(vision.FaceLandmarker.create_from_options(
        #     vision.FaceLandmarkerOptions(
        #         base_options=python.BaseOptions(model_asset_path=str(model_dir / "face_landmarker.task")),
        #         running_mode=vision.RunningMode.VIDEO, num_faces=1,
        #         min_face_detection_confidence=0.3)))
        output = {name: [] for name in WIDTHS}
        previous_timestamp = -1
        for index, path in enumerate(frames):
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"Unreadable frame: {path}")
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp = max(previous_timestamp + 1, round(index * 1000 / fps))
            previous_timestamp = timestamp
            hands = hand.detect_for_video(image, timestamp)
            poses = pose.detect_for_video(image, timestamp)
            # faces = face.detect_for_video(image, timestamp)
            right = np.zeros((21, 3), dtype=np.float32)
            left = np.zeros((21, 3), dtype=np.float32)
            for categories, landmarks in zip(hands.handedness, hands.hand_landmarks):
                if not categories:
                    continue
                coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
                # Preserve WLASL extractor convention: category 0=right, 1=left.
                if categories[0].index == 0:
                    right = coords
                elif categories[0].index == 1:
                    left = coords
            body = np.zeros((33, 3), dtype=np.float32)
            if poses.pose_landmarks:
                body = np.array([[lm.x, lm.y, lm.z] for lm in poses.pose_landmarks[0]], dtype=np.float32)
            # lips = np.zeros((40, 3), dtype=np.float32)
            # if faces.face_landmarks:
            #     landmarks = faces.face_landmarks[0]
            #     lips = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z]
            #                      for i in LIPS], dtype=np.float32)
            for name, coords in zip(WIDTHS, (right, left, body)):
                output[name].append(np.nan_to_num(coords, nan=0, posinf=0, neginf=0).reshape(-1))
    return {name: np.asarray(values, dtype=np.float32) for name, values in output.items()}


def save_sample(destination, arrays, metadata):
    import numpy as np
    destination.mkdir(parents=True, exist_ok=True)
    # Metadata is the completion marker; write it last.
    (destination / "metadata.json").unlink(missing_ok=True)
    for name, values in arrays.items():
        temporary = destination / f"{name}.tmp.npy"
        np.save(temporary, values, allow_pickle=False)
        temporary.replace(destination / f"{name}.npy")
    (destination / "gloss.txt").write_text(metadata["gloss"], encoding="utf-8")
    temporary = destination / "metadata.tmp.json"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination / "metadata.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "datasets/raw/MSASL")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/processed/msasl_features_0_6")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "pretrained")
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--hand-confidence", type=float, default=0.6)
    parser.add_argument("--limit", type=int, help="Maximum samples to process across selected splits")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    if not 0 <= args.hand_confidence <= 1:
        parser.error("--hand-confidence must be between 0 and 1")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.model_dir = args.model_dir.resolve()
    if args.output_dir == args.input_dir or args.input_dir in args.output_dir.parents:
        parser.error("--output-dir must be outside the input directory")
    from itertools import islice
    discovered = discover_samples(args.input_dir, args.splits)
    samples = list(islice(discovered, args.limit) if args.limit else discovered)
    counts = {split: sum(sample[0] == split for sample in samples) for split in args.splits}
    print(f"Samples: {counts}; total={len(samples)}")
    if args.dry_run:
        for split, gloss, sample, frames in samples[:5]:
            print(f"{split}/{gloss}/{sample.name}: {len(frames)} frames")
        return 0
    # Temporarily disabled model: "face_landmarker.task"
    for filename in ("hand_landmarker.task", "pose_landmarker_heavy.task"):
        if not (args.model_dir / filename).is_file():
            raise FileNotFoundError(args.model_dir / filename)
    from tqdm import tqdm
    saved = skipped = failed = 0
    for split, gloss, sample, frames in tqdm(samples):
        destination = args.output_dir / split / gloss / sample.name
        if not args.overwrite and complete(destination, len(frames), args.fps):
            # Changed detector confidence requires re-extraction.
            meta = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
            if (meta.get("hand_confidence") == args.hand_confidence
                    and meta.get("frame_files") == [p.name for p in frames]
                    and meta.get("source") == str(sample)):
                skipped += 1
                continue
        try:
            arrays = extract_sample(frames, args.fps, args.model_dir, args.hand_confidence)
            save_sample(destination, arrays, {
                "split": split, "gloss": gloss, "sample_id": sample.name,
                "source": str(sample), "num_frames": len(frames), "fps": args.fps,
                "hand_confidence": args.hand_confidence,
                "frame_files": [p.name for p in frames],
            })
            saved += 1
        except Exception as exc:
            failed += 1
            tqdm.write(f"FAILED {sample}: {exc}")
    print(f"FINISH: saved={saved}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
