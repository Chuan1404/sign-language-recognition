import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import argparse
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader

from config import ROOT, DEVICE
from src.data.WSASL_raw import WLASLLandmarksDataset
from src.models.SLT_model import ISLR_EncoderDecoder, ISLR_Transformer
from src.utils import FusionComponent
from src.training.train import collate_fn

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = os.path.join(ROOT, "datasets", "processed", "wlasl_features_v2")

LABEL_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL100")

MODEL_NAME = "contest_100_v3.pt"

MODEL_PATH = os.path.join(ROOT, "outputs", "models", MODEL_NAME)

BATCH_SIZE = 8
TOP_K = 5


# ============================================================
# TEST
# ============================================================

def test(model, loader, idx2gloss, top_k=5, device="cuda"):
    model.eval()

    total_loss = 0.0

    total_correct_top1 = 0
    total_correct_topk = 0
    total_samples = 0

    # --------------------------------------------------------
    # Statistics for each TRUE label
    #
    # wrong_count[true_label]
    #     = number of times this label was predicted incorrectly
    #
    # total_count[true_label]
    #     = total number of samples of this label
    # --------------------------------------------------------

    total_count = Counter()
    wrong_count = Counter()

    # --------------------------------------------------------
    # confusion[true_label][pred_label]
    #
    # Used to find:
    # "True SCHOOL -> predicted UNIVERSITY 10 times"
    # --------------------------------------------------------

    confusion = defaultdict(Counter)

    with torch.no_grad():

        for features, labels, video_mask in loader:

            features = features.to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)

            video_mask = video_mask.to(device, non_blocking=True)

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            logits, loss = model(features, labels=labels, video_mask=video_mask)

            total_loss += loss.item()

            # ------------------------------------------------
            # Top-1
            # ------------------------------------------------

            pred_top1 = logits.argmax(dim=-1)

            correct_top1 = (pred_top1 == labels)

            total_correct_top1 += (correct_top1.sum().item())

            # ------------------------------------------------
            # Top-K
            # ------------------------------------------------

            k = min(top_k, logits.size(-1))

            pred_topk = logits.topk(k=k, dim=-1).indices

            in_topk = (pred_topk == labels.unsqueeze(-1)).any(dim=-1)

            total_correct_topk += (in_topk.sum().item())

            total_samples += labels.size(0)

            # ------------------------------------------------
            # Collect confusion statistics
            # ------------------------------------------------

            labels_cpu = labels.cpu().tolist()
            preds_cpu = pred_top1.cpu().tolist()

            for true_label, pred_label in zip(labels_cpu, preds_cpu):

                total_count[true_label] += 1

                if true_label != pred_label:
                    wrong_count[true_label] += 1

                    confusion[true_label][pred_label] += 1

    # ========================================================
    # Metrics
    # ========================================================

    avg_loss = (total_loss / len(loader) if len(loader) > 0 else 0.0)

    top1_acc = (total_correct_top1 / total_samples if total_samples > 0 else 0.0)

    topk_acc = (total_correct_topk / total_samples if total_samples > 0 else 0.0)

    # ========================================================
    # Print overall results
    # ========================================================

    print()
    print("=" * 75)
    print("TEST RESULTS")
    print("=" * 75)

    print(f"Loss        : {avg_loss:.4f}")
    print(f"Top-1 Acc   : {top1_acc * 100:.2f}%")
    print(f"Top-{top_k} Acc : {topk_acc * 100:.2f}%")
    print(f"Samples     : {total_samples}")

    # ========================================================
    # Top 10 TRUE labels with most errors
    # ========================================================

    print()
    print("=" * 75)
    print("TOP 10 LABELS WITH MOST ERRORS")
    print("=" * 75)

    top10_wrong_labels = (wrong_count.most_common(20))

    print(f"{'Rank':<6}"
          f"{'True Label':<20}"
          f"{'Total':<10}"
          f"{'Wrong':<10}"
          f"{'Error Rate':<12}"
          f"{'Most Confused With'}")

    print("-" * 75)

    for rank, (true_id, wrong) in enumerate(top10_wrong_labels, start=1):

        total = total_count[true_id]

        error_rate = (wrong / total if total > 0 else 0.0)

        true_name = idx2gloss[true_id]

        # Most common incorrect prediction
        if true_id in confusion:

            pred_id, pred_count = (confusion[true_id].most_common(1)[0])

            pred_name = idx2gloss[pred_id]

            confused_with = (f"{pred_name} ({pred_count})")

        else:

            confused_with = "-"

        print(f"{rank:<6}"
              f"{true_name:<20}"
              f"{total:<10}"
              f"{wrong:<10}"
              f"{error_rate * 100:>8.2f}%   "
              f"{confused_with}")

    # ========================================================
    # Detailed confusion for TOP 10
    # ========================================================

    print()
    print("=" * 75)
    print("DETAILED CONFUSION FOR TOP 10")
    print("=" * 75)

    for rank, (true_id, wrong) in enumerate(top10_wrong_labels, start=1):

        true_name = idx2gloss[true_id]

        total = total_count[true_id]

        print()
        print(f"{rank}. {true_name} "
              f"(wrong {wrong}/{total})")

        if true_id not in confusion:
            continue

        # Top 5 incorrect predictions
        for pred_id, count in confusion[true_id].most_common(5):
            pred_name = idx2gloss[pred_id]

            print(f"    -> {pred_name:<20}"
                  f"{count} times")

    print()
    print("=" * 75)

    return avg_loss, top1_acc, topk_acc


# ============================================================
# MAIN
# ============================================================

def main(args):
    print(f"Device: {DEVICE}")
    print(f"Model : {args.model}")

    # --------------------------------------------------------
    # Fusion component
    # --------------------------------------------------------

    fusion_component = FusionComponent()

    # --------------------------------------------------------
    # Test dataset
    #
    # Your train code currently uses:
    #
    # base_val = WLASLLandmarksDataset(..., mode="test")
    #
    # So we use exactly the same test split here.
    # --------------------------------------------------------

    test_dataset = WLASLLandmarksDataset(args.data_path, args.label_path, fusion_component, mode="test")

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0,
        pin_memory=True)

    print(f"Test samples: {len(test_dataset)}")

    # --------------------------------------------------------
    # Load labels
    # --------------------------------------------------------

    gloss2idx_path = os.path.join(args.label_path, "gloss2idx.json")

    with open(gloss2idx_path, "r", encoding="utf-8") as f:

        gloss2idx = json.load(f)

    # JSON keys are strings
    idx2gloss = {int(idx): gloss for gloss, idx in gloss2idx.items()}

    num_classes = len(gloss2idx)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    checkpoint = torch.load(args.model, map_location=DEVICE)

    model_kwargs = checkpoint.get("model_kwargs", {"num_classes": num_classes})
    print(model_kwargs)

    model = ISLR_Transformer(**model_kwargs).to(DEVICE)

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    model.load_state_dict(checkpoint["model"])

    print(f"Loaded checkpoint from epoch "
          f"{checkpoint.get('epoch', 'unknown')}")

    if "test_loss" in checkpoint:
        print(f"Checkpoint loss     : "
              f"{checkpoint['test_loss']:.4f}")

    if "test_top1_acc" in checkpoint:
        print(f"Checkpoint top-1    : "
              f"{checkpoint['test_top1_acc'] * 100:.2f}%")

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test(model=model, loader=test_loader, idx2gloss=idx2gloss, top_k=TOP_K, device=DEVICE)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default=DATA_PATH)

    parser.add_argument("--label_path", default=LABEL_DIR)

    parser.add_argument("--model", default=MODEL_PATH)

    args = parser.parse_args()

    main(args)
