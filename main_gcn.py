import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

import argparse
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEVICE, ROOT
from pipeline_config import DATA_PATH, LABEL_DIR, MODEL_DIR, SELECTED_RGB_DIR, set_seed
from src.data.augmentation import AugmentedSkeletonDataset, SkeletonAugmentor
from src.data.WSASL_raw import WLASLLandmarksDataset, WLASLImportantLandmarksDataset
from src.models import ISLR_GCN, ISLR_Transformer_GCN
from src.training.train import collate_fn
from src.utils import FusionComponent


def parse_args():
    parser = argparse.ArgumentParser(description="Runner for GCN models (ISLR_GCN & ISLR_Transformer_GCN)")

    # Data arguments
    parser.add_argument("--data_path", default=DATA_PATH, help="Path to landmark features (wlasl_features_v2)")
    parser.add_argument("--label_path", default=LABEL_DIR, help="Path to annotation directory (WLASL100)")
    parser.add_argument("--indices_dir", default=SELECTED_RGB_DIR, help="Directory containing indices.npy for important frames")
    parser.add_argument("--only_important", action="store_true", default=True, help="Train/evaluate only on important frames")
    parser.add_argument("--all_frames", action="store_true", default=False, help="Use all frames instead of only important frames")
    parser.add_argument("--val_mode", default="test", help="Split used for validation/testing (test or val)")
    parser.add_argument("--augment", action="store_true", default=True, help="Use skeleton data augmentation during training")
    parser.add_argument("--no_augment", action="store_true", default=False, help="Disable data augmentation")

    # Model arguments
    parser.add_argument("--model", default="islr_gcn", choices=["islr_gcn", "gcn", "transformer_gcn"], help="Model architecture")
    parser.add_argument("--channels", default="32,64,128,256", help="Comma-separated channel dimensions for GCN layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")

    # Training arguments
    parser.add_argument("--mode", default="train", choices=["train", "eval", "test"], help="Run mode")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K accuracy metric")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Checkpoint arguments
    parser.add_argument("--output", default="", help="Path to save best checkpoint. If empty, auto-names in outputs/models")
    parser.add_argument("--checkpoint", default="", help="Path to checkpoint to resume training or evaluate")

    args = parser.parse_args()
    if args.all_frames:
        args.only_important = False
    if args.no_augment:
        args.augment = False
    if not args.output:
        os.makedirs(MODEL_DIR, exist_ok=True)
        m_name = "islr_gcn" if args.model in ("islr_gcn", "gcn") else "transformer_gcn"
        suffix = "important_frames" if args.only_important else "all_frames"
        args.output = os.path.join(MODEL_DIR, f"{m_name}_{suffix}_best.pt")

    return args


def build_model(model_type, channels, num_classes, dropout=0.1):
    channels_list = [int(c.strip()) for c in channels.split(",") if c.strip()]
    if model_type in ("islr_gcn", "gcn"):
        print(f"\n[Model] Initializing ISLR_GCN (channels={channels_list}, num_classes={num_classes})")
        return ISLR_GCN(channels=channels_list, num_classes=num_classes)
    elif model_type in ("transformer_gcn", "gcn_transformer"):
        print(f"\n[Model] Initializing ISLR_Transformer_GCN (gcn_channels={channels_list}, num_classes={num_classes})")
        return ISLR_Transformer_GCN(gcn_channels=channels_list, num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@torch.no_grad()
def evaluate_model(model, loader, top_k=5, device=DEVICE):
    model.eval()
    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_topk = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for features, labels, video_mask, _ in pbar:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        video_mask = video_mask.to(device, non_blocking=True)

        logits, loss = model(features, labels=labels, video_mask=video_mask)
        if loss is not None:
            total_loss += loss.item() * labels.size(0)

        pred_top1 = logits.argmax(dim=-1)
        total_correct_top1 += (pred_top1 == labels).sum().item()

        k = min(top_k, logits.size(-1))
        pred_topk = logits.topk(k=k, dim=-1).indices
        in_topk = (pred_topk == labels.unsqueeze(-1)).any(dim=-1)
        total_correct_topk += in_topk.sum().item()

        total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    top1_acc = total_correct_top1 / max(total_samples, 1)
    topk_acc = total_correct_topk / max(total_samples, 1)

    return avg_loss, top1_acc, topk_acc


def run_train(args):
    set_seed(args.seed)
    print(f"Device: {DEVICE}")
    fusion_component = FusionComponent()

    # 1. Dataset loading
    if args.only_important:
        print(f"\nLoading Landmark Dataset with IMPORTANT FRAMES ONLY from: {args.indices_dir}")
        base_train = WLASLImportantLandmarksDataset(
            feature_dir=args.data_path,
            annotation_dir=args.label_path,
            indices_dir=args.indices_dir,
            fusion_component=fusion_component,
            mode="train",
        )
        base_val = WLASLImportantLandmarksDataset(
            feature_dir=args.data_path,
            annotation_dir=args.label_path,
            indices_dir=args.indices_dir,
            fusion_component=fusion_component,
            mode=args.val_mode,
        )
    else:
        print(f"\nLoading Landmark Dataset with ALL FRAMES...")
        base_train = WLASLLandmarksDataset(
            feature_dir=args.data_path,
            annotation_dir=args.label_path,
            fusion_component=fusion_component,
            mode="train",
        )
        base_val = WLASLLandmarksDataset(
            feature_dir=args.data_path,
            annotation_dir=args.label_path,
            fusion_component=fusion_component,
            mode=args.val_mode,
        )

    if args.augment:
        train_dataset = AugmentedSkeletonDataset(base_train, SkeletonAugmentor())
        print("Data augmentation: ENABLED (AugmentedSkeletonDataset)")
    else:
        train_dataset = base_train
        print("Data augmentation: DISABLED")

    val_dataset = base_val

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # 2. Number of classes
    with open(os.path.join(args.label_path, "gloss2idx.json"), "r", encoding="utf-8") as f:
        gloss2idx = json.load(f)
    num_classes = len(gloss2idx)
    print(f"Number of gloss classes: {num_classes}")

    # 3. Model initialization
    model = build_model(args.model, args.channels, num_classes, dropout=args.dropout).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 4. Checkpoint resume
    start_epoch = 0
    best_acc = 0.0
    best_loss = float("inf")
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"\nLoading checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("test_top1_acc", 0.0)
        best_loss = ckpt.get("test_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch} (best top-1 acc={best_acc * 100:.2f}%)")

    # 5. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # 6. Training loop
    no_improve = 0
    print(f"\nStarting training for {args.epochs} epochs (save to: {args.output})...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1:3d}/{args.epochs}")

        for features, labels, video_mask, _ in pbar:
            features = features.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            video_mask = video_mask.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, loss = model(features, labels=labels, video_mask=video_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / len(train_loader)
        val_loss, val_top1_acc, val_topk_acc = evaluate_model(model, val_loader, top_k=args.top_k, device=DEVICE)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:3d} | LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Top-1: {val_top1_acc * 100:.2f}% | "
            f"Val Top-{args.top_k}: {val_topk_acc * 100:.2f}%"
        )

        if val_top1_acc > best_acc:
            best_acc = val_top1_acc
            best_loss = val_loss
            no_improve = 0
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "test_loss": val_loss,
                    "test_top1_acc": val_top1_acc,
                    "test_topk_acc": val_topk_acc,
                    "model_name": args.model,
                    "channels": args.channels,
                    "num_classes": num_classes,
                    "only_important": args.only_important,
                },
                args.output,
            )
            print(f"  * Saved new best model checkpoint! Top-1: {best_acc * 100:.2f}%")
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve}/{args.patience} epochs (Best Top-1: {best_acc * 100:.2f}%)")
            if no_improve >= args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
                break

    print(f"\nTraining finished! Best Validation Top-1 Acc: {best_acc * 100:.2f}% (Loss: {best_loss:.4f})")
    print(f"Checkpoint saved at: {args.output}")


def run_eval(args):
    print(f"Device: {DEVICE}")
    fusion_component = FusionComponent()

    if args.only_important:
        print(f"\nLoading evaluation dataset with IMPORTANT FRAMES ONLY from: {args.indices_dir}")
        eval_dataset = WLASLImportantLandmarksDataset(
            feature_dir=args.data_path,
            annotation_dir=args.label_path,
            indices_dir=args.indices_dir,
            fusion_component=fusion_component,
            mode=args.val_mode,
        )
    else:
        print(f"\nLoading evaluation dataset with ALL FRAMES...")
        eval_dataset = WLASLLandmarksDataset(
            feature_dir=args.data_path,
            annotation_dir=args.label_path,
            fusion_component=fusion_component,
            mode=args.val_mode,
        )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    with open(os.path.join(args.label_path, "gloss2idx.json"), "r", encoding="utf-8") as f:
        gloss2idx = json.load(f)
    num_classes = len(gloss2idx)

    model = build_model(args.model, args.channels, num_classes, dropout=args.dropout).to(DEVICE)

    ckpt_path = args.checkpoint if args.checkpoint else args.output
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}. Please provide a valid --checkpoint path.")

    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)

    loss, top1, topk = evaluate_model(model, eval_loader, top_k=args.top_k, device=DEVICE)
    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS ({args.val_mode} split - {len(eval_dataset)} samples)")
    print(f"Model       : {args.model}")
    print(f"Frames mode : {'Important frames only' if args.only_important else 'All frames'}")
    print(f"Loss        : {loss:.4f}")
    print(f"Top-1 Acc   : {top1 * 100:.2f}%")
    print(f"Top-{args.top_k} Acc   : {topk * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    args = parse_args()
    if args.mode in ("eval", "test"):
        run_eval(args)
    else:
        run_train(args)
