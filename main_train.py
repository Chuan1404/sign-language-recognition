import os


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pipeline_config import SELECTED_RGB_DIR, apply_warmup
from models import TwoHandRGBModel

import torch
from torch.utils.data import DataLoader

from config import DEVICE, ROOT
from src.data.WSASL_raw import WLASLLandmarksRGBDataset
from src.models import RGBModel
from src.training.train import collate_fn_rgb, train_rgb_one_epoch, validate_rgb
from src.utils import FusionComponent

DATA_PATH = os.path.join(ROOT, "datasets", "processed", "wlasl_features_v2")
LABEL_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL100")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "models")
MODEL_NAME = "stage2_pose_rgb_contrastive.pt"
STAGE1_CKPT = os.path.join(ROOT, "outputs", "models", "stage1_pose_rgb_contrastive.pt")

LR = 1e-4
BATCH_SIZE = 8
EPOCHS = 100
TOP_K = 5
PATIENCE = 15
WEIGHT_DECAY = 0.01
WARMUP_EPOCHS = 5


def default_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_path", default=DATA_PATH)
    parser.add_argument("--rgb_dir", default=SELECTED_RGB_DIR)
    parser.add_argument("--label_path", default=LABEL_DIR)
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, MODEL_NAME))
    parser.add_argument("--val_mode", default="test", help="tên split dùng để validate (<val_mode>.json)", )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS, help="Number of linear warmup epochs")
    parser.add_argument("--stage1_ckpt", default=STAGE1_CKPT,
        help="Path to stage-1 pretrained checkpoint (.pt). Set to '' to skip.", )
    return parser


def main(args):
    print(f"Device: {DEVICE}")

    print("\nLoading datasets...")
    fusion_component = FusionComponent()

    train_dataset = WLASLLandmarksRGBDataset(args.data_path, args.label_path, args.rgb_dir, fusion_component,
        mode="train")
    val_dataset = WLASLLandmarksRGBDataset(args.data_path, args.label_path, args.rgb_dir, fusion_component,
        mode=args.val_mode)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_rgb,
        num_workers=0, pin_memory=True, )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_rgb,
        num_workers=0, pin_memory=True, )

    with open(os.path.join(args.label_path, "gloss2idx.json"), "r") as f:
        num_classes = len(json.load(f))
    print(f"Number of classes: {num_classes}")

    model_kwargs = {"num_classes": num_classes}
    # model = RGBModel(**model_kwargs).to(DEVICE)
    model = TwoHandRGBModel(rgb_backbone="resnet18", temporal="opticalflow", **model_kwargs).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # stage1_ckpt_path = args.stage1_ckpt
    # if stage1_ckpt_path and os.path.isfile(stage1_ckpt_path):
    #     print(f"\n⟳ Loading Stage-1 checkpoint: {stage1_ckpt_path}")
    #     model.load_pretrained(stage1_ckpt_path)
    #     print("✓ Stage-1 checkpoint loaded successfully.\n")
    # else:
    #     print(f"\n⚠ Stage-1 checkpoint not found at: {stage1_ckpt_path}. Training from scratch.\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=WEIGHT_DECAY)
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    best_acc = 0.0
    best_loss = float("inf")
    no_improve = 0

    for epoch in range(args.epochs):
        in_warmup = apply_warmup(optimizer, base_lrs, epoch, args.warmup_epochs)

        print("\n" + "=" * 70)
        if in_warmup:
            print(f"Epoch {epoch + 1}/{args.epochs} [WARMUP {epoch + 1}/{args.warmup_epochs}]")
        else:
            print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 70)

        train_out = train_rgb_one_epoch(model, train_loader, optimizer, DEVICE)
        val_out = validate_rgb(model, val_loader, DEVICE, top_k=args.top_k)

        train_loss = train_out["loss"]
        train_acc = train_out["acc"]
        val_loss = val_out["loss"]
        val_top1 = val_out["top1"]
        val_topk = val_out["topk"]

        if not in_warmup:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        lr_display = f"{current_lr:.7f} (Warmup)" if in_warmup else f"{current_lr:.7f}"

        print(f"\n"
              f"Train loss        : {train_loss:.4f}\n"
              f"Train top-1 acc   : {train_acc * 100:.2f}%\n"
              f"Val loss          : {val_loss:.4f}\n"
              f"Val top-1 acc     : {val_top1 * 100:.2f}%\n"
              f"Val top-{args.top_k} acc     : {val_topk * 100:.2f}%\n"
              f"Learning rate     : {lr_display}")

        improved = val_top1 > best_acc or (val_top1 == best_acc and val_loss < best_loss)
        if improved:
            no_improve = 0
            best_acc = val_top1
            best_loss = val_loss

            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epoch, "val_loss": val_loss, "val_top1_acc": val_top1,
                "val_topk_acc": val_topk, "model_kwargs": model_kwargs, }
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(checkpoint, args.output)
            print(f"\n✓ Saved best model  (top-1={best_acc * 100:.2f}%, loss={best_loss:.4f})")
        else:
            no_improve += 1
            print(f"\nNo improvement (best top-1={best_acc * 100:.2f}%), patience={no_improve}/{args.patience}")

        if no_improve >= args.patience:
            print(f"\n⚑ Early stopping at epoch {epoch + 1}")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Training complete.")
    if os.path.isfile(args.output):
        ckpt = torch.load(args.output, map_location=DEVICE)
        print(f"Best epoch     : {ckpt['epoch'] + 1}")
        print(f"Val loss       : {ckpt['val_loss']:.4f}")
        print(f"Top-1 acc      : {ckpt['val_top1_acc'] * 100:.2f}%")
        print(f"Top-{args.top_k} acc      : {ckpt['val_topk_acc'] * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("", parents=[default_args()], add_help=False)
    args = parser.parse_args()
    main(args)
