import os
import json

from config import ROOT

NSLT_JSON_PATH = os.path.join(
    ROOT, "datasets", "raw", "WLASL", "nslt_1000.json"
)

WLASL_FULL_JSON_PATH = os.path.join(
    ROOT, "datasets", "raw", "WLASL", "WLASL_v0_3.json"
)

MISSING_PATH = os.path.join(
    ROOT, "datasets", "annotations", "missing.txt"
)

SAVE_DIR = os.path.join(
    ROOT, "datasets", "annotations", "WLASL1000"
)

os.makedirs(SAVE_DIR, exist_ok=True)


def build_video_id_to_gloss(wlasl_full_data):
    """video_id -> gloss, tra từ file metadata gốc WLASL_v0_3.json"""
    video_id2gloss = {}
    for gloss_entry in wlasl_full_data:
        gloss = gloss_entry["gloss"]
        for instance in gloss_entry["instances"]:
            video_id2gloss[instance["video_id"]] = gloss
    return video_id2gloss


def flatten_nslt():

    with open(NSLT_JSON_PATH, "r", encoding="utf-8") as f:
        nslt_data = json.load(f)

    with open(WLASL_FULL_JSON_PATH, "r", encoding="utf-8") as f:
        wlasl_full_data = json.load(f)

    video_id2gloss = build_video_id_to_gloss(wlasl_full_data)

    missing_ids = set()
    if os.path.exists(MISSING_PATH):
        with open(MISSING_PATH, "r", encoding="utf-8") as f:
            missing_ids = set(f.read().split())

    train_entries = []
    val_entries = []
    test_entries = []

    skipped = 0
    not_found_in_full = []
    all_glosses = set()

    for video_id, info in nslt_data.items():

        if video_id in missing_ids:
            skipped += 1
            continue

        gloss = video_id2gloss.get(video_id)
        if gloss is None:
            not_found_in_full.append(video_id)
            continue

        class_id, start_frame, end_frame = info["action"]
        all_glosses.add(gloss)

        sample = {
            "video_id": video_id,
            "gloss": gloss,
            "class_id": class_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }

        split = info["subset"].lower()

        if split == "train" or split == "val":
            train_entries.append(sample)

        # elif split == "val":
        #     val_entries.append(sample)

        elif split == "test":
            test_entries.append(sample)

        else:
            print(f"Unknown split: {split} (video_id={video_id})")

    # -------------------------------------------------
    # Build vocabulary từ gloss thật (a-z), không phải class_id số
    # -------------------------------------------------

    gloss_list = sorted(all_glosses)

    gloss2idx = {
        gloss: idx
        for idx, gloss in enumerate(gloss_list)
    }

    idx2gloss = {
        idx: gloss
        for gloss, idx in gloss2idx.items()
    }

    for sample in train_entries + test_entries:
        sample["label_id"] = gloss2idx[sample["gloss"]]

    # -------------------------------------------------
    # Save annotation
    # -------------------------------------------------

    with open(os.path.join(SAVE_DIR, "train.json"), "w", encoding="utf-8") as f:
        json.dump(train_entries, f, indent=2, ensure_ascii=False)

    with open(os.path.join(SAVE_DIR, "val.json"), "w", encoding="utf-8") as f:
        json.dump(val_entries, f, indent=2, ensure_ascii=False)

    with open(os.path.join(SAVE_DIR, "test.json"), "w", encoding="utf-8") as f:
        json.dump(test_entries, f, indent=2, ensure_ascii=False)

    with open(os.path.join(SAVE_DIR, "gloss2idx.json"), "w", encoding="utf-8") as f:
        json.dump(gloss2idx, f, indent=2, ensure_ascii=False)

    with open(os.path.join(SAVE_DIR, "idx2gloss.json"), "w", encoding="utf-8") as f:
        json.dump(idx2gloss, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("nslt_100.json flattened successfully (gloss từ WLASL_v0_3.json)")
    print("=" * 60)
    print(f"Glosses            : {len(gloss_list)}")
    print(f"Missing videos     : {skipped}")
    print(f"Not found in full  : {len(not_found_in_full)}")
    if not_found_in_full:
        print(f"  e.g. {not_found_in_full[:10]}")
    print(f"Train samples      : {len(train_entries)}")
    print(f"Val samples      : {len(val_entries)}")
    print(f"Test samples       : {len(test_entries)}")
    print(f"Vocabulary         : {len(gloss2idx)}")
    print("=" * 60)


if __name__ == "__main__":
    flatten_nslt()