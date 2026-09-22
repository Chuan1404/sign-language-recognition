import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.multimodal_dataset import WLASLMultimodalDataset
from src.models.SLT_model import MultimodalFusionModel
from src.utils import FusionComponent
from src.training.train import collate_fn_multimodal

from config import ROOT, DEVICE

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = os.path.join(ROOT, "datasets", "processed", "wlasl_features_v2")
EXTRACTED_FRAMES_DIR = os.path.join(ROOT, "outputs", "extracted_frames")
LABEL_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL100")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "models")

MODEL_NAME = "multimodal_fusion_v1.pt"

LR = 1e-4
BATCH_SIZE = 4 # Batch size nhỏ hơn vì load cả ảnh RGB
EPOCHS = 100
TOP_K = 2
PATIENCE = 10
WEIGHT_DECAY = 0.01

def default_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_path", default=DATA_PATH)
    parser.add_argument("--frames_dir", default=EXTRACTED_FRAMES_DIR)
    parser.add_argument("--label_path", default=LABEL_DIR)
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, MODEL_NAME))
    return parser

# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch_multimodal(model, loader, optimizer, device='cuda'):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training")

    for features, images, labels, video_mask, video_ids in pbar:
        features = features.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        video_mask = video_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        output = model(landmarks=features, images=images, labels=labels, video_mask=video_mask)

        logits = output["logits"]
        loss = output["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
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
def validate_multimodal(model, loader, device, top_k=2):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    top1_correct = 0
    topk_correct = 0

    for features, images, labels, video_mask, video_ids in loader:
        features = features.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        video_mask = video_mask.to(device, non_blocking=True)

        output = model(landmarks=features, images=images, labels=labels, video_mask=video_mask)
        logits = output["logits"]
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        total_loss += (loss.item() * batch_size)
        total_samples += batch_size

        predictions = logits.argmax(dim=1)
        top1_correct += (predictions == labels).sum().item()

        _, topk_indices = torch.topk(logits, k=min(top_k, logits.size(1)), dim=1)
        topk_correct += (topk_indices == labels.unsqueeze(1)).any(dim=1).sum().item()

    val_loss = (total_loss / max(total_samples, 1))
    top1_acc = (top1_correct / max(total_samples, 1))
    topk_acc = (topk_correct / max(total_samples, 1))

    return (val_loss, top1_acc, topk_acc)

# ============================================================
# MAIN
# ============================================================

def main(args):
    print(f"Device: {DEVICE}")

    fusion_component = FusionComponent()

    print("\nLoading datasets...")
    train_dataset = WLASLMultimodalDataset(
        args.data_path, args.label_path, args.frames_dir, fusion_component, mode="train"
    )
    val_dataset = WLASLMultimodalDataset(
        args.data_path, args.label_path, args.frames_dir, fusion_component, mode="test"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=collate_fn_multimodal, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        collate_fn=collate_fn_multimodal, num_workers=0, pin_memory=True
    )

    with open(os.path.join(args.label_path, "gloss2idx.json"), "r") as f:
        gloss2idx = json.load(f)
    num_classes = len(gloss2idx)
    print(f"Number of classes: {num_classes}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model_kwargs = {"num_classes": num_classes}
    model = MultimodalFusionModel(**model_kwargs).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    best_loss = float("inf")
    best_acc = 0.0
    no_improve = 0

    for epoch in range(EPOCHS):
        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("=" * 70)

        train_loss, train_acc = train_one_epoch_multimodal(model, train_loader, optimizer, DEVICE)
        val_loss, val_top1_acc, val_topk_acc = validate_multimodal(model, val_loader, DEVICE, top_k=TOP_K)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nTrain loss       : {train_loss:.4f}\n"
              f"Train top-1 acc  : {train_acc * 100:.2f}%\n"
              f"Val loss         : {val_loss:.4f}\n"
              f"Val top-1 acc    : {val_top1_acc * 100:.2f}%\n"
              f"Val top-{TOP_K} acc : {val_topk_acc * 100:.2f}%\n"
              f"Learning rate     : {current_lr:.7f}")

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
                "model_kwargs": model_kwargs,
            }

            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(checkpoint, args.output)
            print(f"\n✓ Saved best model\n  Top-1: {best_acc * 100:.2f}%\n  Loss : {best_loss:.4f}")
        else:
            no_improve += 1
            print(f"\nNo improvement (best loss={best_loss:.4f}), patience={no_improve}/{PATIENCE}")

        if no_improve >= PATIENCE:
            print(f"\n⚑ Early stopping at epoch {epoch + 1}")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("", parents=[default_args()], add_help=False)
    args = parser.parse_args()
    main(args)
