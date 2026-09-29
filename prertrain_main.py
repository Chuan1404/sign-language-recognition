import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json

import torch
from torch.utils.data import DataLoader

from config import DEVICE, ROOT
from src.data.WSASL_raw import WLASLLandmarksRGBDataset
from src.models.SLT_model import PoseRGBFusionModel
from src.utils import FusionComponent
from src.training.train import collate_fn_rgb, train_rgb_one_epoch, validate_rgb

DATA_PATH = os.path.join(ROOT, "datasets", "processed", "wlasl_features_v2")
LABEL_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL100")
VIDEO_DIR = os.path.join(ROOT, "outputs", "extracted_frames")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "models")
MODEL_NAME = "stage1_pose_rgb_contrastive.pt"

LR = 1e-4
CNN_LR_SCALE = 0.1
BATCH_SIZE = 4
EPOCHS = 100
TOP_K = 5
IMAGE_SIZE = 112
MAX_FRAMES = 64
PATIENCE = 10
WEIGHT_DECAY = 0.01


def default_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_path", default=DATA_PATH)
    parser.add_argument("--label_path", default=LABEL_DIR)
    parser.add_argument("--video_dir", default=VIDEO_DIR)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, MODEL_NAME))
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    return parser


def to_device(batch, device):
    landmarks, rgb, feature_mask, rgb_mask, _labels, video_ids = batch

    return (
        landmarks.to(device, non_blocking=True),
        rgb.to(device, non_blocking=True),
        feature_mask.to(device, non_blocking=True),
        rgb_mask.to(device, non_blocking=True),
        video_ids,
    )


def main(args):
    print(f"Device: {DEVICE}")

    fusion_component = FusionComponent()

    print("\nLoading datasets...")

    ds_kwargs = {
        "max_frames": args.max_frames,
        # RGB frames được load từ rgb_frames.npy (pre-processed bởi preprocess_rgb.py).
        # Đặt use_cache=False để load ảnh trực tiếp (chậm hơn).
        "use_cache": True,
    }

    train_dataset = WLASLLandmarksRGBDataset(
        args.data_path,
        args.label_path,
        args.video_dir,
        fusion_component,
        mode="train",
        **ds_kwargs,
    )

    val_dataset = WLASLLandmarksRGBDataset(
        args.data_path,
        args.label_path,
        args.video_dir,
        fusion_component,
        mode="test",
        **ds_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn_rgb,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_rgb,
        num_workers=0,
        pin_memory=True,
    )

    with open(os.path.join(args.label_path, "gloss2idx.json"), "r") as f:
        num_classes = len(json.load(f))

    print(f"Number of classes (Stage 2 only): {num_classes}")

    model_kwargs = {"num_classes": num_classes}

    model = PoseRGBFusionModel(**model_kwargs).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total params: {total_params:,}")

    cnn_params = list(model.rgb_encoder.backbone.parameters())

    cnn_ids = {id(p) for p in cnn_params}

    other_params = [p for p in model.parameters() if id(p) not in cnn_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": cnn_params, "lr": args.lr * CNN_LR_SCALE},
        ],
        eps=1e-8,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
    )

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(args.epochs):
        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 70)

        train_stats = train_rgb_one_epoch(model, train_loader, optimizer, DEVICE)

        val_stats = validate_rgb(model, val_loader, DEVICE, top_k=TOP_K)

        scheduler.step(val_stats["loss"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\n"
            f"Train InfoNCE               : "
            f"{train_stats['loss']:.4f}\n"
            f"Val InfoNCE (N={val_stats['n']})   : "
            f"{val_stats['loss']:.4f}\n"
            f"Val pose->rgb R@1 / R@{TOP_K}    : "
            f"{val_stats['p2r_r1'] * 100:.2f}% / "
            f"{val_stats['p2r_rk'] * 100:.2f}%\n"
            f"Val rgb->pose R@1            : "
            f"{val_stats['r2p_r1'] * 100:.2f}%\n"
            f"Learning rate                : "
            f"{current_lr:.7f}"
        )

        if val_stats["loss"] < best_loss:
            no_improve = 0
            best_loss = val_stats["loss"]

            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "val": val_stats,
                "model_kwargs": model_kwargs,
            }

            os.makedirs(os.path.dirname(args.output), exist_ok=True)

            torch.save(checkpoint, args.output)

            print(
                f"\n✓ Saved best model "
                f"(val InfoNCE={best_loss:.4f}, "
                f"R@1={val_stats['p2r_r1'] * 100:.2f}%)"
            )

        else:
            no_improve += 1

            print(
                f"\nNo improvement "
                f"(best={best_loss:.4f}), "
                f"patience={no_improve}/{PATIENCE}"
            )

        if no_improve >= PATIENCE:
            print(f"\n⚑ Early stopping at epoch {epoch + 1}")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)

    ckpt = torch.load(args.output, map_location="cpu")

    print(
        f"Best epoch: {ckpt['epoch'] + 1} | "
        f"val InfoNCE: {ckpt['val']['loss']:.4f} | "
        f"pose->rgb R@1: "
        f"{ckpt['val']['p2r_r1'] * 100:.2f}%"
    )

    print(f"Stage 2: model.load_pretrained('{args.output}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("", parents=[default_args()], add_help=False)

    args = parser.parse_args()

    main(args)
