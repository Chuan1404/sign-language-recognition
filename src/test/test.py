import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
from collections import defaultdict
from src.training.train import collate_fn

import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.data.WSASL_raw import WLASLLandmarksDataset
from src.utils import FusionComponent
from src.models.SLT_model import ISLR_V1
from config import ROOT


DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 8
TOP_K       = 5
SEED        = 42


FEATURE_DIR    = os.path.join(ROOT, "datasets", "processed", "wlasl_features")
ANNOTATION_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL100")
SAVE_DIR       = os.path.join(ROOT, "outputs", "models")
MODEL_PATH     = os.path.join(SAVE_DIR, "v3_WLASL100_26_08_22.pt")

fusion_component = FusionComponent()


def evaluate(model, loader, idx2gloss, top_k=5):
    """Tính top-1, top-k accuracy và thu thập các dự đoán sai để phân tích."""

    model.eval()

    total_loss    = 0
    correct_top1  = 0
    correct_topk  = 0
    total_samples = 0
    wrong_preds   = []   # [(gt_gloss, pred_gloss), ...]

    with torch.no_grad():
        for features, labels, video_mask in tqdm(loader, desc="Evaluating"):
            features = features.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            video_mask = video_mask.to(DEVICE, non_blocking=True)

            logits, loss = model(
                features,
                # hand_normalize_features,
                labels=labels,
                video_mask=video_mask
            )
            total_loss += loss

            logits = logits                            # (B, num_classes)

            preds_top1 = logits.argmax(dim=-1)                 # (B,)
            correct_top1 += (preds_top1 == labels).sum().item()

            k = min(top_k, logits.size(-1))
            preds_topk = logits.topk(k=k, dim=-1).indices       # (B, k)
            in_topk = (preds_topk == labels.unsqueeze(-1)).any(dim=-1)
            correct_topk += in_topk.sum().item()

            total_samples += labels.size(0)

            for i in range(labels.size(0)):
                gt = idx2gloss[labels[i].item()]
                pred = idx2gloss[preds_top1[i].item()]
                mark = "OK" if labels[i] == preds_top1[i] else "X"
                print(f"  [{mark}] GT: {gt:<20s}  PRED: {pred}")

                if labels[i] != preds_top1[i]:
                    wrong_preds.append((gt, pred))

    avg_loss = total_loss / len(loader)
    top1_acc = correct_top1 / total_samples if total_samples > 0 else 0.0
    topk_acc = correct_topk / total_samples if total_samples > 0 else 0.0

    return avg_loss, top1_acc, topk_acc, wrong_preds


def print_most_confused(wrong_preds, n=10):
    """In ra các từ và các cặp (gt, pred) bị nhầm nhiều nhất."""
    counter = defaultdict(int)
    counter_gt = defaultdict(int)
    for gt, pred in wrong_preds:
        counter[(gt, pred)] += 1
        counter_gt[gt] += 1

    sorted_pairs = sorted(counter.items(), key=lambda x: -x[1])
    sorted_gts = sorted(counter_gt.items(), key=lambda x: -x[1])

    print(f"\n  Top {n} most confused pairs (GT → PRED):")
    for (gt, pred), cnt in sorted_pairs[:n]:
        print(f"    [{cnt:3d}x]  '{gt}' → '{pred}'")

    print(f"\n  Top {n} most mispredicted words:")
    for gt, cnt in sorted_gts[:n]:
        print(f"    [{cnt:3d}x]  '{gt}'")


def main():
    with open(os.path.join(ANNOTATION_DIR, "gloss2idx.json"), "r") as f:
        gloss2idx = json.load(f)
    idx2gloss = {v: k for k, v in gloss2idx.items()}
    num_classes = len(gloss2idx)

    print("=" * 60)
    print("  WLASL Test Evaluation")
    print("=" * 60)
    print(f"  Device     : {DEVICE}")
    print(f"  Checkpoint : {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Không tìm thấy checkpoint: {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model_kwargs = checkpoint["model_kwargs"]

    print(f"  Checkpoint epoch  : {checkpoint['epoch']}")
    print(f"  Val top-1 (saved) : {checkpoint.get('val_top1_acc', 'N/A')}")

    test_dataset = WLASLLandmarksDataset(
        FEATURE_DIR, ANNOTATION_DIR, fusion_component, mode="test"
    )
    print(f"Dataset — test: {len(test_dataset)} samples / {num_classes} classes")

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )

    model = ISLR_V1(**model_kwargs).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    test_loss, top1_acc, topk_acc, wrong_preds = evaluate(
        model, test_loader, idx2gloss, top_k=TOP_K
    )

    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    print(f"  Test loss  : {test_loss:.4f}")
    print(f"  Top-1 acc  : {top1_acc*100:.2f}%")
    print(f"  Top-{TOP_K} acc  : {topk_acc*100:.2f}%")
    print(f"  Wrong preds: {len(wrong_preds)} / {len(test_dataset)}")

    print_most_confused(wrong_preds, n=15)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()