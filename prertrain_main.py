import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ROOT, DEVICE
from src.data.WSASL_raw import WLASLLandmarksRGBDataset, collate_fn_rgb
from src.models.SLT_model import PoseRGBFusionModel
from src.utils import FusionComponent

DATA_PATH = os.path.join(ROOT, "datasets", "processed", "wlasl_features_v2")
LABEL_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL100")
VIDEO_DIR = os.path.join(ROOT, "datasets", "raw", "WLASL", "videos")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "models")
MODEL_NAME = "stage1_pose_rgb_contrastive.pt"

LR = 1e-4
CNN_LR_SCALE = 0.1  # backbone ResNet đã pretrain ImageNet -> lr nhỏ hơn
BATCH_SIZE = 4  # contrastive cần nhiều negative; giảm nếu hết VRAM
EPOCHS = 100
TOP_K = 5
IMAGE_SIZE = 112
MAX_FRAMES = 64  # giới hạn T (cả pose và RGB) để đỡ tốn VRAM; None = giữ nguyên
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
    """Batch: (landmarks (B,T,F), rgb (B,T,3,H,W), video_mask (B,T), labels, video_ids). Stage 1 bỏ qua labels."""
    landmarks, rgb, video_mask, _labels, video_ids = batch
    return (landmarks.to(device, non_blocking=True), rgb.to(device, non_blocking=True),
            video_mask.to(device, non_blocking=True), video_ids)


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss, n_batches = 0.0, 0

    pbar = tqdm(loader, desc="Pretrain")
    for batch in pbar:
        landmarks, rgb, video_mask, _ = to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        out = model.forward_pretrain(landmarks, rgb, video_mask)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += out["loss"].item()
        n_batches += 1
        pbar.set_postfix(loss=f"{out['loss'].item():.4f}")

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(model, loader, device, top_k=TOP_K):
    """Gom embedding của cả tập val rồi tính loss + retrieval trên toàn tập
    (không phụ thuộc batch size như loss theo từng batch)."""
    model.eval()

    all_pose, all_rgb = [], []
    for batch in loader:
        landmarks, rgb, video_mask, _ = to_device(batch, device)
        out = model.forward_pretrain(landmarks, rgb, video_mask)
        all_pose.append(out["p_g"].cpu())
        all_rgb.append(out["r_g"].cpu())

    pose = torch.cat(all_pose)  # (N, P) đã L2-normalize
    rgb = torch.cat(all_rgb)

    scale = model.logit_scale.exp().clamp(max=100).item()
    loss = model._info_nce(pose, rgb, scale).item()

    sim = pose @ rgb.t()  # (N, N)
    target = torch.arange(sim.size(0)).unsqueeze(1)

    def recall(s, k):
        k = min(k, s.size(1))
        return (s.topk(k, dim=1).indices == target).any(dim=1).float().mean().item()

    return {"loss": loss, "p2r_r1": recall(sim, 1), "p2r_rk": recall(sim, top_k), "r2p_r1": recall(sim.t(), 1),
            "n": sim.size(0)}


def main(args):
    print(f"Device: {DEVICE}")

    fusion_component = FusionComponent()

    print("\nLoading datasets...")
    ds_kwargs = dict(image_size=args.image_size, max_frames=args.max_frames)
    train_dataset = WLASLLandmarksRGBDataset(args.data_path, args.label_path, args.video_dir, fusion_component,
                                             mode="train", **ds_kwargs)
    val_dataset = WLASLLandmarksRGBDataset(args.data_path, args.label_path, args.video_dir, fusion_component,
                                           mode="test", **ds_kwargs)

    # drop_last=True: batch chỉ có 1 mẫu thì contrastive không có negative
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              collate_fn=collate_fn_rgb, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_rgb,
                            num_workers=0, pin_memory=True)

    with open(os.path.join(args.label_path, "gloss2idx.json"), "r") as f:
        num_classes = len(json.load(f))
    print(f"Number of classes (chỉ dùng ở Stage 2): {num_classes}")

    model_kwargs = {"num_classes": num_classes}
    model = PoseRGBFusionModel(**model_kwargs).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    # lr nhỏ hơn cho CNN backbone (đã pretrain ImageNet)
    cnn_params = list(model.rgb_encoder.backbone.parameters())
    cnn_ids = {id(p) for p in cnn_params}
    other_params = [p for p in model.parameters() if id(p) not in cnn_ids]

    optimizer = torch.optim.AdamW(
        [{"params": other_params, "lr": args.lr}, {"params": cnn_params, "lr": args.lr * CNN_LR_SCALE}], eps=1e-8,
        weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5,
                                                           min_lr=1e-7)

    best_loss = float("inf")
    no_improve = 0

    for epoch in range(args.epochs):
        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print("=" * 70)

        train_stats = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_stats = validate(model, val_loader, DEVICE, top_k=TOP_K)

        scheduler.step(val_stats["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\n"
              f"Train InfoNCE               : {train_stats['loss']:.4f}\n"
              f"Val InfoNCE (N={val_stats['n']})   : {val_stats['loss']:.4f}\n"
              f"Val pose->rgb R@1 / R@{TOP_K}    : {val_stats['p2r_r1'] * 100:.2f}% / "
              f"{val_stats['p2r_rk'] * 100:.2f}%\n"
              f"Val rgb->pose R@1            : {val_stats['r2p_r1'] * 100:.2f}%\n"
              f"Learning rate                : {current_lr:.7f}")

        if val_stats["loss"] < best_loss:
            no_improve = 0
            best_loss = val_stats["loss"]

            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "epoch": epoch, "val": val_stats,
                          "model_kwargs": model_kwargs}

            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(checkpoint, args.output)
            print(f"\n✓ Saved best model  (val InfoNCE={best_loss:.4f}, R@1={val_stats['p2r_r1'] * 100:.2f}%)")
        else:
            no_improve += 1
            print(f"\nNo improvement (best={best_loss:.4f}), patience={no_improve}/{PATIENCE}")

        if no_improve >= PATIENCE:
            print(f"\n⚑ Early stopping at epoch {epoch + 1}")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    ckpt = torch.load(args.output, map_location="cpu")
    print(f"Best epoch: {ckpt['epoch'] + 1} | val InfoNCE: {ckpt['val']['loss']:.4f} | "
          f"pose->rgb R@1: {ckpt['val']['p2r_r1'] * 100:.2f}%")
    print(f"Stage 2: model.load_pretrained('{args.output}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("", parents=[default_args()], add_help=False)
    args = parser.parse_args()
    main(args)