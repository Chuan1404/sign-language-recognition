import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.WSASL_raw import WLASLLandmarksDataset

from src.models.SLT_model import ISLR_Transformer

from src.utils import FusionComponent

from src.training.train import collate_fn

from config import ROOT, DEVICE

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = os.path.join(ROOT, "datasets", "processed", "wlasl_features_v2")

LABEL_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL2000")

OUTPUT_DIR = os.path.join(ROOT, "outputs", "models")

MODEL_NAME = "contest_2000_v1.pt"

LR = 1e-4
BATCH_SIZE = 8
EPOCHS = 100

TOP_K = 2
PATIENCE = 10

WEIGHT_DECAY = 0.01


# ============================================================
# ARGUMENTS
# ============================================================

def default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--data_path", default=DATA_PATH)

    parser.add_argument("--label_path", default=LABEL_DIR)

    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, MODEL_NAME))

    return parser


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch_selector(model, loader, optimizer, device='cuda'):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training")

    for features, labels, video_mask, video_ids in pbar:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        video_mask = video_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        output = model(features, labels=labels, video_mask=video_mask)

        logits = output["logits"]
        loss = output["loss"]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # -------------------------
        # Loss
        # -------------------------
        total_loss += loss.item()

        # -------------------------
        # Accuracy
        # -------------------------
        pred = logits.argmax(dim=-1)

        total_correct += (pred == labels).sum().item()

        total_samples += labels.size(0)

        acc = total_correct / total_samples

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc * 100:.2f}%")

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate_selector(model, loader, device, top_k=2):
    model.eval()

    total_loss = 0.0

    total_samples = 0

    top1_correct = 0
    topk_correct = 0

    for features, labels, video_mask, video_ids in loader:
        features = features.to(device, non_blocking=True)

        labels = labels.to(device, non_blocking=True)

        video_mask = video_mask.to(device, non_blocking=True)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        output = model(features, labels=labels, video_mask=video_mask)

        logits = output["logits"]

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)

        total_loss += (loss.item() * batch_size)

        total_samples += batch_size

        # ----------------------------------------------------
        # Top-1
        # ----------------------------------------------------

        predictions = logits.argmax(dim=1)

        top1_correct += (predictions == labels).sum().item()

        # ----------------------------------------------------
        # Top-K
        # ----------------------------------------------------

        _, topk_indices = torch.topk(logits, k=min(top_k, logits.size(1)), dim=1)

        topk_correct += (topk_indices == labels.unsqueeze(1)).any(dim=1).sum().item()

    val_loss = (total_loss / max(total_samples, 1))

    top1_acc = (top1_correct / max(total_samples, 1))

    topk_acc = (topk_correct / max(total_samples, 1))

    return (val_loss, top1_acc, topk_acc)


@torch.no_grad()
def extract_frame_importance(model, loader, device, save_path):
    model.eval()

    results = {}

    for batch_idx, (features, labels, video_mask, video_ids) in enumerate(loader):

        features = features.to(device, non_blocking=True)

        video_mask = video_mask.to(device, non_blocking=True)

        output = model(features, labels=None, video_mask=video_mask)

        importance = output["frame_importance"]

        importance = importance.cpu()

        video_mask_cpu = (video_mask.cpu().bool())

        for i, video_id in enumerate(video_ids):
            valid_length = int(video_mask_cpu[i].sum())

            scores = importance[i, :valid_length].tolist()

            results[str(video_id)] = {"importance": scores, "length": valid_length}

        if batch_idx % 50 == 0:
            print(f"Extracted "
                  f"{batch_idx + 1} batches")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:

        json.dump(results, f, indent=2)

    print(f"\n✓ Saved frame importance:"
          f"\n  {save_path}")


# ============================================================
# MAIN
# ============================================================

def main(args):
    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Fusion component
    # --------------------------------------------------------

    fusion_component = FusionComponent()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading datasets...")

    base_train = WLASLLandmarksDataset(args.data_path, args.label_path, fusion_component, mode="train")

    base_val = WLASLLandmarksDataset(args.data_path, args.label_path, fusion_component, mode="test")

    # train_dataset = AugmentedSkeletonDataset(base_train, SkeletonAugmentor())
    train_dataset = base_train
    val_dataset = base_val

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0,
                            pin_memory=True)

    with open(os.path.join(args.label_path, "gloss2idx.json"), "r") as f:

        gloss2idx = json.load(f)

    num_classes = len(gloss2idx)

    print(f"Number of classes: "
          f"{num_classes}")

    model_kwargs = {"num_classes": num_classes}

    model = ISLR_Transformer(**model_kwargs).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params    : "
          f"{total_params:,}")

    print(f"Trainable params: "
          f"{trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    best_loss = float("inf")
    best_acc = 0.0
    no_improve = 0

    for epoch in range(EPOCHS):

        print("\n" + "=" * 70)

        print(f"Epoch "
              f"{epoch + 1}/{EPOCHS}")

        print("=" * 70)

        train_loss, train_acc = (train_one_epoch_selector(model, train_loader, optimizer, DEVICE))

        val_loss, val_top1_acc, val_topk_acc = (validate_selector(model, val_loader, DEVICE, top_k=TOP_K))

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n"
              f"Train loss       : "
              f"{train_loss:.4f}\n"
              f"Train top-1 acc  : "
              f"{train_acc * 100:.2f}%\n"
              f"Val loss         : "
              f"{val_loss:.4f}\n"
              f"Val top-1 acc    : "
              f"{val_top1_acc * 100:.2f}%\n"
              f"Val top-{TOP_K} acc : "
              f"{val_topk_acc * 100:.2f}%\n"
              f"Learning rate     : "
              f"{current_lr:.7f}")

        if val_loss < best_loss:

            no_improve = 0

            best_loss = val_loss
            best_acc = val_top1_acc

            checkpoint = {

                "model": model.state_dict(),

                "optimizer": optimizer.state_dict(),

                "scheduler": scheduler.state_dict(),

                "epoch": epoch,

                "val_loss": val_loss,

                "val_top1_acc": val_top1_acc,

                "val_topk_acc": val_topk_acc,

                "model_kwargs": model_kwargs, }

            os.makedirs(os.path.dirname(args.output), exist_ok=True)

            torch.save(checkpoint, args.output)

            print(f"\n✓ Saved best model"
                  f"\n  Top-1: "
                  f"{best_acc * 100:.2f}%"
                  f"\n  Loss : "
                  f"{best_loss:.4f}")

        else:

            no_improve += 1

            print(f"\n"
                  f"No improvement "
                  f"(best loss="
                  f"{best_loss:.4f}), "
                  f"patience="
                  f"{no_improve}/{PATIENCE}")

        if no_improve >= PATIENCE:
            print(f"\n"
                  f"⚑ Early stopping "
                  f"at epoch "
                  f"{epoch + 1}")

            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)

    print("Loading best model...")

    checkpoint = torch.load(args.output, map_location=DEVICE)

    model.load_state_dict(checkpoint["model"])

    print(f"Best epoch     : "
          f"{checkpoint['epoch'] + 1}")

    print(f"Best val loss  : "
          f"{checkpoint['val_loss']:.4f}")

    print(f"Best top-1 acc : "
          f"{checkpoint['val_top1_acc'] * 100:.2f}%")

    train_importance_path = os.path.join(OUTPUT_DIR, "wlasl100_train_frame_importance.json")
    val_importance_path = os.path.join(OUTPUT_DIR, "wlasl100_val_frame_importance.json")

    print("\nExtracting temporal importance for train set...")
    extract_frame_importance(model, train_loader, DEVICE, train_importance_path)
    
    print("\nExtracting temporal importance for val set...")
    extract_frame_importance(model, val_loader, DEVICE, val_importance_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("", parents=[default_args()], add_help=False)
    args = parser.parse_args()
    main(args)
