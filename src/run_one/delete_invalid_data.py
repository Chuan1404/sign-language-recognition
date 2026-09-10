"""
One-off post-processing script for ALREADY-EXTRACTED WLASL features.

Re-running WLASL_feature_extract.py from scratch just to apply the new
"drop frames with no hand detected" filter would be very slow (re-runs
MediaPipe detection on every video). This script instead loads the
already-saved right_hand.npy / left_hand.npy / pose.npy for every video,
applies the same filter in-place, and overwrites them on disk.

Safe to run multiple times: once a folder has been filtered, no frame in
it has both hands all-zero anymore, so running the filter again is a no-op
for that folder (idempotent).

Usage:
    python run_once_filter_hands.py                  # apply filter for real
    python run_once_filter_hands.py --dry-run         # preview only, no writes
    python run_once_filter_hands.py --backup          # keep a .bak copy of
                                                        # the original .npy
                                                        # files before overwriting
"""

import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from tqdm import tqdm

from config import ROOT

SAVE_DIR = os.path.join(ROOT, "datasets", "processed", "full_body_wlasl")


def filter_video_folder(dir_path, dry_run=False, backup=False):
    """Filter one video's feature folder in-place.

    Returns a status dict describing what happened, so the caller can
    print an aggregate summary at the end.
    """
    right_hand_path = os.path.join(dir_path, "right_hand.npy")
    left_hand_path  = os.path.join(dir_path, "left_hand.npy")
    pose_path       = os.path.join(dir_path, "pose.npy")

    if not (os.path.exists(right_hand_path)
            and os.path.exists(left_hand_path)
            and os.path.exists(pose_path)):
        return {"status": "missing_files"}

    right_hand = np.load(right_hand_path)
    left_hand  = np.load(left_hand_path)
    pose       = np.load(pose_path)

    if not (len(right_hand) == len(left_hand) == len(pose)):
        return {
            "status": "shape_mismatch",
            "shapes": (right_hand.shape, left_hand.shape, pose.shape),
        }

    total_frames = len(right_hand)
    if total_frames == 0:
        return {"status": "empty"}

    has_right_hand = np.any(right_hand != 0, axis=1)
    has_left_hand  = np.any(left_hand != 0, axis=1)
    valid_frame_mask = has_right_hand | has_left_hand

    num_dropped = int((~valid_frame_mask).sum())

    if not np.any(valid_frame_mask):
        return {"status": "no_hand_in_any_frame", "total_frames": total_frames}

    if num_dropped == 0:
        return {"status": "already_clean", "total_frames": total_frames}

    if dry_run:
        return {
            "status": "would_filter",
            "total_frames": total_frames,
            "num_dropped": num_dropped,
        }

    if backup:
        np.save(right_hand_path + ".bak", right_hand)
        np.save(left_hand_path + ".bak", left_hand)
        np.save(pose_path + ".bak", pose)

    np.save(right_hand_path, right_hand[valid_frame_mask])
    np.save(left_hand_path,  left_hand[valid_frame_mask])
    np.save(pose_path,       pose[valid_frame_mask])

    return {
        "status": "filtered",
        "total_frames": total_frames,
        "num_dropped": num_dropped,
    }


def run(dry_run=False, backup=False):
    video_dirs = sorted(
        d for d in os.listdir(SAVE_DIR)
        if os.path.isdir(os.path.join(SAVE_DIR, d))
    )
    print(f"Found {len(video_dirs)} video folders under: {SAVE_DIR}")

    if dry_run:
        print("Running in DRY-RUN mode — no files will be modified.\n")

    counts = {
        "filtered": 0,
        "would_filter": 0,
        "already_clean": 0,
        "no_hand_in_any_frame": 0,
        "missing_files": 0,
        "shape_mismatch": 0,
        "empty": 0,
    }
    total_frames_dropped = 0
    problem_videos = []

    for video_id in tqdm(video_dirs, desc="Filtering"):
        dir_path = os.path.join(SAVE_DIR, video_id)
        result = filter_video_folder(dir_path, dry_run=dry_run, backup=backup)

        status = result["status"]
        counts[status] = counts.get(status, 0) + 1

        if status in ("filtered", "would_filter"):
            total_frames_dropped += result["num_dropped"]

        if status in ("missing_files", "shape_mismatch", "no_hand_in_any_frame", "empty"):
            problem_videos.append((video_id, status, result))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    label = "would_filter" if dry_run else "filtered"
    print(f"Videos {label}         : {counts[label]}")
    print(f"Videos already clean   : {counts['already_clean']}")
    print(f"Videos with no hand at all in any frame : {counts['no_hand_in_any_frame']}")
    print(f"Videos with missing .npy files           : {counts['missing_files']}")
    print(f"Videos with shape mismatch between files : {counts['shape_mismatch']}")
    print(f"Videos with zero frames                  : {counts['empty']}")
    print(f"Total frames {'that would be ' if dry_run else ''}dropped : {total_frames_dropped}")

    if problem_videos:
        print("\nVideos that need manual attention:")
        for video_id, status, result in problem_videos:
            print(f"  - {video_id}: {status} {result}")

    if dry_run and counts["would_filter"] > 0:
        print("\nThis was a dry run. Re-run without --dry-run to actually overwrite the files.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Drop frames with no hand detected from already-extracted WLASL features."
    )
    parser.add_argument("--dry-run", action="store_true",
                         help="Preview what would be filtered without writing any files")
    parser.add_argument("--backup", action="store_true",
                         help="Keep a .npy.bak copy of the original files before overwriting")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(dry_run=args.dry_run, backup=args.backup)