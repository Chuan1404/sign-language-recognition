"""Group test mistakes by ground-truth gloss and export all train/test videos.

Layout: <ground_truth>/<ground_truth or predicted_gloss>/<split>_<id>.mp4.
Every exported video contains the saved original MediaPipe keypoints.
The output directory is deleted and recreated before writing each run.
"""

import argparse
import csv
import html
import json
import os
import re
import shutil
import stat
import sys
from collections import Counter, defaultdict
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEVICE, ROOT, WLASL_RAW_DATA, _COORD_DIM, _NUM_NODE
from src.data.WSASL_raw import WLASLLandmarksDataset
from src.data.video_filter import excluded_video_ids
from src.models import ISLR_GCN, ISLR_Transformer, ISLR_Transformer_GCN
from src.training.train import collate_fn

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7),
              (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13),
              (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19),
              (19, 20), (5, 9), (9, 13), (13, 17)]
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (11, 23), (12, 24), (23, 24)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(Path(ROOT) / "outputs/models/landmarks_100_v1.pt"))
    parser.add_argument("--data_path", default=str(Path(ROOT) / "datasets/processed/wlasl_features_v2"))
    parser.add_argument("--label_path", default=str(Path(ROOT) / "datasets/annotations/WLASL100"))
    parser.add_argument("--video_dir", default=WLASL_RAW_DATA)
    parser.add_argument("--output", default=str(PROJECT / "outputs/detail_wrong_video_by_gloss"),
                        help="Export directory; deleted and recreated on every run")
    parser.add_argument("--model", choices=("auto", "gcn", "transformer", "transformer_gcn"), default="auto")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default=str(DEVICE))
    parser.add_argument("--report_only", action="store_true", help="Evaluate without exporting videos")
    parser.add_argument("--predictions_json", help="Reuse predictions from a previous report.json")
    parser.add_argument("--workers", type=int, default=4, help="Parallel video exporters")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.workers < 1:
        parser.error("batch_size and workers must be positive")
    return args


def load_model(checkpoint_path, architecture, num_classes, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    if architecture == "auto":
        if "shape_projection.0.weight" in state and "gcn_block.0.gcn.phi.weight" in state:
            architecture = "transformer_gcn"
        elif "pose_projection.0.weight" in state:
            architecture = "transformer"
        elif "gcn_block.0.linear.0.weight" in state:
            architecture = "gcn"
        else:
            raise ValueError("Cannot identify a main.py architecture; supply --model explicitly")
    kwargs = dict(checkpoint.get("model_kwargs", {}))
    if kwargs.get("num_classes", num_classes) != num_classes:
        raise ValueError("Checkpoint classes do not match gloss2idx.json")
    kwargs.setdefault("num_classes", num_classes)
    if kwargs.get("num_nodes", _NUM_NODE) != _NUM_NODE:
        raise ValueError("Expected a landmark checkpoint from main.py, not a contour/mesh checkpoint")
    cls = {"gcn": ISLR_GCN, "transformer": ISLR_Transformer,
           "transformer_gcn": ISLR_Transformer_GCN}[architecture]
    model = cls(**kwargs)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), architecture, kwargs


def collate_for_model(batch, architecture, single_feature=False):
    features, labels, mask, video_ids = collate_fn(batch)
    if architecture != "transformer_gcn" or single_feature:
        dim = _NUM_NODE * _COORD_DIM
        features = features[..., dim:2 * dim]
    return features, labels, mask, video_ids


@torch.inference_mode()
def evaluate(model, loader, idx2gloss, device):
    records = []
    for features, labels, mask, video_ids in tqdm(loader, desc="Test predictions"):
        output = model(features.to(device), video_mask=mask.to(device))
        logits = output["logits"] if isinstance(output, dict) else output[0]
        probabilities = logits.softmax(dim=-1).cpu()
        scores, predictions = probabilities.max(dim=-1)
        for index, video_id in enumerate(video_ids):
            truth, predicted = int(labels[index]), int(predictions[index])
            records.append(dict(video_id=str(video_id), truth_id=truth, prediction_id=predicted,
                                truth=idx2gloss[truth], prediction=idx2gloss[predicted],
                                confidence=float(scores[index]),
                                truth_probability=float(probabilities[index, truth]),
                                correct=truth == predicted))
    if not records:
        raise RuntimeError("No usable test samples")
    return records


def draw_points(frame, points, edges, color):
    height, width = frame.shape[:2]
    xy = np.asarray(points)[..., :2]
    present = np.isfinite(xy).all(axis=-1) & np.any(xy != 0, axis=-1)
    pixels = np.rint(np.where(present[:, None], xy, 0) * [width, height]).astype(int)
    # Clip only for drawing; source MediaPipe arrays are never modified.
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    for a, b in edges:
        if present[a] and present[b]:
            cv2.line(frame, tuple(pixels[a]), tuple(pixels[b]), color, 2, cv2.LINE_AA)
    for point in pixels[present]:
        cv2.circle(frame, tuple(point), 3, color, -1, cv2.LINE_AA)


def export_video(video_id, feature_dir, video_dir, destination, lines):
    if str(video_id) in excluded_video_ids():
        raise ValueError(f"Video {video_id} is excluded by config._REMOVE_VIDEO_IDS")
    directory = Path(feature_dir) / video_id
    arrays = {}
    for key, nodes in (("pose", 33), ("left_hand", 21), ("right_hand", 21)):
        raw = np.load(directory / f"{key}.npy", allow_pickle=False)
        if raw.ndim < 2 or len(raw) == 0 or np.prod(raw.shape[1:]) not in (nodes * 2, nodes * 3):
            raise ValueError(f"{video_id}: invalid {key} landmarks: {raw.shape}")
        arrays[key] = raw.reshape(len(raw), nodes, -1)
    frames = len(arrays["pose"])
    if any(len(points) != frames for points in arrays.values()):
        raise ValueError(f"{video_id}: landmark frame counts disagree")
    source = Path(video_dir) / f"{video_id}.mp4"
    if not source.is_file():
        raise FileNotFoundError(source)
    cap = cv2.VideoCapture(str(source))
    writer = None
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 25.0
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index >= frames:
                raise ValueError(f"{video_id}: raw video has more frames than landmarks")
            height, width = frame.shape[:2]
            if writer is None:
                size = (width + width % 2, height + height % 2)
                writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create {destination}")
            draw_points(frame, arrays["pose"][index], POSE_EDGES, (255, 200, 0))
            draw_points(frame, arrays["left_hand"][index], HAND_EDGES, (0, 255, 0))
            draw_points(frame, arrays["right_hand"][index], HAND_EDGES, (0, 160, 255))
            for row, text in enumerate(lines):
                y = 24 + row * 24
                cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(cv2.copyMakeBorder(frame, 0, size[1] - height, 0, size[0] - width, cv2.BORDER_CONSTANT))
            index += 1
        if index != frames:
            raise ValueError(f"{video_id}: video has {index} frames, landmarks have {frames}; cannot align overlay")
    except Exception:
        if writer is not None:
            writer.release()
        destination.unlink(missing_ok=True)
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def safe_name(text):
    return re.sub(r"[^\w.-]+", "_", str(text), flags=re.UNICODE).strip(". ")[:60] or "unknown"


def build_groups(records, annotation_dir):
    excluded = excluded_video_ids()
    records = [row for row in records if str(row["video_id"]) not in excluded]
    mistakes = sorted((dict(row) for row in records if not row["correct"]),
                      key=lambda row: (row["truth"], row["prediction"], row["video_id"]))
    for row in mistakes:
        # Strip fields belonging to the previous pair-based report format.
        for key in ("reference_video_id", "reference_split", "ground_truth_video",
                    "predicted_class_video", "export_error"):
            row.pop(key, None)
    wanted = defaultdict(set)
    for row in mistakes:
        wanted[row["truth"]].add(row["prediction"])
    annotations = defaultdict(list)
    seen = set()
    for split in ("train", "test"):
        with (Path(annotation_dir) / f"{split}.json").open(encoding="utf-8") as handle:
            entries = json.load(handle)
        for entry in sorted(entries, key=lambda row: str(row["video_id"])):
            video_id, gloss = str(entry["video_id"]), entry["gloss"]
            if video_id in excluded:
                continue
            if (split, video_id) not in seen:
                annotations[gloss].append(dict(video_id=video_id, split=split, gloss=gloss))
                seen.add((split, video_id))
    all_glosses = set(wanted) | {gloss for values in wanted.values() for gloss in values}
    names = [safe_name(gloss).casefold() for gloss in all_glosses]
    if len(names) != len(set(names)):
        raise ValueError("Gloss names collide after conversion to Windows folder names")
    groups = []
    for truth, predictions in sorted(wanted.items()):
        folder = safe_name(truth)
        collections = []
        for gloss in [truth, *sorted(predictions)]:
            collection_folder = f"{folder}/{safe_name(gloss)}"
            videos = [dict(entry, path=f"{collection_folder}/{entry['split']}_{safe_name(entry['video_id'])}.mp4",
                           status="pending", error="") for entry in annotations[gloss]]
            collections.append(dict(gloss=gloss, role="ground_truth" if gloss == truth else "prediction",
                                    folder=collection_folder, videos=videos))
        groups.append(dict(ground_truth=truth, folder=folder,
                           mistakes=[row for row in mistakes if row["truth"] == truth],
                           collections=collections))
    return mistakes, groups


def export_groups(groups, args, output, progress_callback):
    tasks = {}
    for group in groups:
        for collection in group["collections"]:
            (output / collection["folder"]).mkdir(parents=True, exist_ok=True)
            for video in collection["videos"]:
                key = (video["split"], video["video_id"], video["gloss"])
                tasks.setdefault(key, []).append(video)

    def render(key, entries):
        split, video_id, gloss = key
        try:
            first = output / entries[0]["path"]
            export_video(video_id, args.data_path, args.video_dir, first,
                         [f"Ground truth: {gloss}", f"ID: {video_id} | Split: {split}"])
        except (OSError, RuntimeError, ValueError, cv2.error) as error:
            return [(entry, str(error)) for entry in entries]
        results = [(entries[0], "")]
        for entry in entries[1:]:
            destination = output / entry["path"]
            try:
                destination.unlink(missing_ok=True)
                try:
                    # Each folder contains an actual MP4; identical exports share storage.
                    os.link(first, destination)
                except OSError:
                    shutil.copy2(first, destination)
                results.append((entry, ""))
            except OSError as error:
                results.append((entry, str(error)))
        return results

    print(f"Exporting {len(tasks)} unique videos into "
          f"{sum(len(entries) for entries in tasks.values())} folder entries", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(render, key, entries) for key, entries in tasks.items()]
        for index, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Export train/test videos"), 1):
            for entry, error in future.result():
                entry.update(status="error" if error else "exported", error=error)
            if index % 10 == 0:
                progress_callback()
    progress_callback()


def write_reports(output, records, mistakes, groups, metadata):
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter((row["truth"], row["prediction"]) for row in mistakes)
    videos = [video for group in groups for collection in group["collections"] for video in collection["videos"]]
    report = dict(metadata, total=len(records), wrong=len(mistakes),
                  top1_accuracy=sum(row["correct"] for row in records) / len(records),
                  layout="ground_truth/gloss/split_video_id.mp4", groups=groups,
                  video_entries=len(videos), exported_entries=sum(v["status"] == "exported" for v in videos),
                  failed_entries=sum(v["status"] == "error" for v in videos),
                  confusions=[dict(truth=truth, prediction=prediction, count=count)
                              for (truth, prediction), count in counts.most_common()],
                  mistakes=mistakes, predictions=records)
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output / "report.json")
    with (output / "wrong_predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "truth", "prediction", "confidence", "truth_probability"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(mistakes)
    sections = []
    for group in groups:
        group_folder = output / group["folder"]
        group_folder.mkdir(parents=True, exist_ok=True)
        (group_folder / "confusions.json").write_text(json.dumps(group, ensure_ascii=False, indent=2), encoding="utf-8")
        collection_html = []
        for collection in group["collections"]:
            folder = output / collection["folder"]
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "manifest.json").write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding="utf-8")
            items = []
            for video in collection["videos"]:
                label = html.escape(f"{video['split']} / {video['video_id']}")
                if video["status"] == "exported":
                    url = html.escape(video["path"], quote=True)
                    items.append(f'<details><summary>{label}</summary><video controls preload="none" src="{url}"></video>'
                                 f'<p><a href="{url}">Open/download MP4</a></p></details>')
                else:
                    items.append(f"<p>{label}: {html.escape(video['error'] or video['status'])}</p>")
            role = "Ground truth" if collection["role"] == "ground_truth" else "Predicted gloss"
            collection_html.append(f"<div><h3>{role}: {html.escape(collection['gloss'])}</h3>"
                                   f"<p>{len(collection['videos'])} train/test videos</p>{''.join(items)}</div>")
        errors = ', '.join(f"{row['video_id']} → {row['prediction']}" for row in group["mistakes"])
        sections.append(f"<section><h2>{html.escape(group['ground_truth'])}</h2><p>{html.escape(errors)}</p>"
                        f"<div class='collections'>{''.join(collection_html)}</div></section>")
    document = ("<!doctype html><meta charset='utf-8'><title>Wrong gloss videos</title>"
                "<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:16px}"
                ".collections{display:flex;gap:20px;flex-wrap:wrap}.collections>div{flex:1;min-width:280px}"
                "video{width:100%}section{border-top:1px solid #ccc;padding:16px 0}details{margin:12px 0}</style>"
                f"<h1>{len(groups)} ground-truth glosses — {len(mistakes)}/{len(records)} test mistakes</h1>"
                f"<p>Top-1: {report['top1_accuracy']:.2%}. Exported: {report['exported_entries']}/{len(videos)} videos.</p>"
                "<p>Each ground-truth folder contains all train/test videos of that gloss and each confused gloss. "
                "Overlays use saved MediaPipe points. Expand a video to play, or open its MP4 externally.</p>"
                + "".join(sections))
    (output / "index.html").write_text(document, encoding="utf-8")
    return report


def reset_output_dir(output, args):
    """Delete only the resolved export folder, never project/input directories."""
    output = Path(output)
    reparse = (output.exists() and getattr(output.lstat(), "st_file_attributes", 0)
               & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    if output.is_symlink() or reparse:
        raise ValueError(f"Output must be a real export directory: {output}")
    output = output.resolve()
    project = PROJECT.resolve()
    if output == Path(output.anchor) or output == project or project.is_relative_to(output):
        raise ValueError(f"Refusing to delete project or parent directory: {output}")
    for source in (args.data_path, args.label_path, args.video_dir):
        source = Path(source).resolve()
        if output == source or source.is_relative_to(output) or output.is_relative_to(source):
            raise ValueError(f"Output overlaps an input directory: {output} / {source}")
    checkpoint = Path(args.checkpoint).resolve()
    if checkpoint == output or checkpoint.is_relative_to(output):
        raise ValueError(f"Output contains the checkpoint: {output}")
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"Output is not a directory: {output}")
        print(f"Removing previous export directory: {output}", flush=True)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def main(args):
    if args.predictions_json:
        with Path(args.predictions_json).open(encoding="utf-8") as handle:
            previous = json.load(handle)
        records = previous["predictions"]
        if previous.get("split") != "test" or not records:
            raise ValueError("predictions_json must contain nonempty test predictions")
        metadata = {key: previous[key] for key in ("checkpoint", "model", "split", "skipped_samples") if key in previous}
        metadata["predictions_source"] = str(Path(args.predictions_json).resolve())
        # Ensure the saved predictions describe the current test labels.
        with (Path(args.label_path) / "test.json").open(encoding="utf-8") as handle:
            truth = {str(row["video_id"]): row["gloss"] for row in json.load(handle)}
        for row in records:
            if truth.get(row["video_id"]) != row["truth"]:
                raise ValueError("Saved predictions do not match the selected test annotations")
    else:
        with (Path(args.label_path) / "gloss2idx.json").open(encoding="utf-8") as handle:
            gloss2idx = json.load(handle)
        idx2gloss = {int(value): gloss for gloss, value in gloss2idx.items()}
        device = torch.device(args.device)
        model, architecture, kwargs = load_model(args.checkpoint, args.model, len(gloss2idx), device)
        print(f"Checkpoint: {args.checkpoint}\nArchitecture: {architecture}; device: {device}", flush=True)
        dataset = WLASLLandmarksDataset(args.data_path, args.label_path, mode="test", cache_landmarks=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            collate_fn=partial(collate_for_model, architecture=architecture,
                                               single_feature=kwargs.get("single_feature", False)))
        records = evaluate(model, loader, idx2gloss, device)
        metadata = dict(checkpoint=str(Path(args.checkpoint).resolve()), model=architecture,
                        split="test", skipped_samples=dict(dataset.skipped))
    excluded = excluded_video_ids()
    removed_predictions = sum(str(row["video_id"]) in excluded for row in records)
    records = [row for row in records if str(row["video_id"]) not in excluded]
    if not records:
        raise RuntimeError("No test predictions remain after _REMOVE_VIDEO_IDS filtering")
    metadata.update(excluded_video_ids=sorted(excluded), excluded_saved_predictions=removed_predictions)
    mistakes, groups = build_groups(records, args.label_path)
    # Load reused predictions above before deleting an output that may contain them.
    output = reset_output_dir(args.output, args)
    metadata.update(data_path=str(Path(args.data_path).resolve()),
                    label_path=str(Path(args.label_path).resolve()), video_dir=str(Path(args.video_dir).resolve()))
    def save():
        return write_reports(output, records, mistakes, groups, metadata)
    save()
    print(f"Wrong test videos: {len(mistakes)}/{len(records)}; ground-truth gloss folders: {len(groups)}", flush=True)
    if not args.report_only:
        export_groups(groups, args, output, save)
    report = save()
    print(f"\nTop-1: {report['top1_accuracy']:.2%}; exported: {report['exported_entries']}/{report['video_entries']}; "
          f"failed: {report['failed_entries']}")
    print(f"Output: {output}\nReport: {output / 'index.html'}", flush=True)
    return report


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main(parse_args())
