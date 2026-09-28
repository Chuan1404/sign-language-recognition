import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ROOT, DEVICE
from src.data.WSASL_rgb import WLASLLandmarksRGBDataset, collate_fn_rgb
from src.models.SLT_multimodal import PoseRGBFusionModel
from src.utils import FusionComponent

DATA_PATH = os.path.join(ROOT, "datasets", "processed", "wlasl_features_v2")
LABEL_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL100")
VIDEO_DIR = os.path.join(ROOT, "datasets", "raw", "wlasl_videos")  # TODO: đổi cho đúng thư mục chứa <video_id>.mp4
OUTPUT_DIR = os.path.join(ROOT, "outputs", "models")
PRETRAINED = os.path.join(OUTPUT_DIR, "stage1_pose_rgb_contrastive.pt")
MODEL_NAME = "stage2_pose_rgb_fusion.pt"

LR = 1e-4  # lr cho phần fusion + classifier
ENCODER_LR_SCALE = 0.1  # lr của encoder sau khi unfreeze = LR * scale
FREEZE_EPOCHS = 5  # số epoch đầu freeze encoder, chỉ train fusion + classifier
BATCH_SIZE = 16
EPOCHS = 100
IMAGE_SIZE = 112
MAX_FRAMES = 64  # phải giống lúc pretrain
TOP_K = 5
PATIENCE = 15
WEIGHT_DECAY = 0.01


def default_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_path", default=DATA_PATH)
    parser.add_argument("--label_path", default=LABEL_DIR)
    parser.add_argument("--video_dir", default=VIDEO_DIR)
    parser.add_argument("--pretrained", default=PRETRAINED, help="checkpoint Stage 1; để trống để train từ đầu")
    parser.add_argument("--output", default=os.path.join(OUTPUT_DIR, MODEL_NAME))
    parser.add_argument("--val_mode", default="test", help="tên split dùng để validate (<val_mode>.json)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--encoder_lr_scale", type=float, default=ENCODER_LR_SCALE)
    parser.add_argument("--freeze_epochs", type=int, default=FREEZE_EPOCHS)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    return parser


def to_device(batch, device):
    """Batch: (landmarks (B,T,F), rgb (B,T,3,H,W), video_mask (B,T), labels (B,), video_ids)."""
    landmarks, rgb, video_mask, labels, video_ids = batch
    return (landmarks.to(device, non_blocking=True), rgb.to(device, non_blocking=True),
            video_mask.to(device, non_blocking=True), labels.to(device, non_blocking=True), video_ids)


def set_train_mode(model, encoders_frozen):
    """Khi encoder đang freeze thì để eval luôn (tắt dropout, giữ nguyên BatchNorm stats của CNN)."""
    model.train()
    if encoders_frozen:
        model.pose_encoder.eval()
        model.rgb_encoder.eval()


def train_one_epoch(model, loader, optimizer, device, encoders_frozen):
    set_train_mode(model, encoders_frozen)

    total_loss, total_correct, total_samples = 0.0, 0, 0

    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        landmarks, rgb, video_mask, labels, _ = to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        out = model(landmarks, rgb, labels=labels, video_mask=video_mask)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_correct += (out["logits"].argmax(dim=-1) == labels).sum().item()
        total_samples += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * total_correct / total_samples:.2f}%")

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def validate(model, loader, device, top_k=TOP_K):
    model.eval()

    total_loss, total_samples = 0.0, 0
    top1_correct, topk_correct = 0, 0

    for batch in loader:
        landmarks, rgb, video_mask, labels, _ = to_device(batch, device)

        out = model(landmarks, rgb, labels=labels, video_mask=video_mask)
        logits = out["logits"]

        bs = labels.size(0)
        total_loss += out["loss"].item() * bs
        total_samples += bs

        top1_correct += (logits.argmax(dim=1) == labels).sum().item()
        topk_idx = torch.topk(logits, k=min(top_k, logits.size(1)), dim=1).indices
        topk_correct += (topk_idx == labels.unsqueeze(1)).any(dim=1).sum().item()

    n = max(total_samples, 1)
    return total_loss / n, top1_correct / n, topk_correct / n


def build_model(args, num_classes):
    model_kwargs = {"num_classes": num_classes}
    ckpt = None

    if args.pretrained and os.path.exists(args.pretrained):
        ckpt = torch.load(args.pretrained, map_location="cpu")
        model_kwargs = ckpt.get("model_kwargs", model_kwargs)
        assert model_kwargs["num_classes"] == num_classes, (
            f"num_classes lệch: checkpoint={model_kwargs['num_classes']} vs dataset={num_classes}")
    elif args.pretrained:
        print(f"⚠ Không tìm thấy {args.pretrained} -> train từ đầu (không pretrain)")

    model = PoseRGBFusionModel(**model_kwargs)

    if ckpt is not None:
        print("Load Stage 1:", model.load_state_dict(ckpt["model"], strict=False))
        v = ckpt.get("val", {})
        if v:
            print(f"  (Stage 1 val InfoNCE={v.get('loss', float('nan')):.4f}, "
                  f"pose->rgb R@1={v.get('p2r_r1', float('nan')) * 100:.2f}%)")

    return model.to(DEVICE), model_kwargs


def main(args):
    print(f"Device: {DEVICE}")

    fusion_component = FusionComponent()

    print("\nLoading datasets...")
    ds_kwargs = dict(image_size=args.image_size, max_frames=args.max_frames)
    train_dataset = WLASLLandmarksRGBDataset(args.data_path, args.label_path, args.video_dir, fusion_component,
                                             mode="train", **ds_kwargs)
    val_dataset = WLASLLandmarksRGBDataset(args.data_path, args.label_path, args.video_dir, fusion_component,
                                           mode=args.val_mode, **ds_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_rgb,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_rgb,
                            num_workers=0, pin_memory=True)

    with open(os.path.join(args.label_path, "gloss2idx.json"), "r") as f:
        num_classes = len(json.load(f))
    print(f"Number of classes: {num_classes}")

    model, model_kwargs = build_model(args, num_classes)

    # --- param groups: encoder (lr nhỏ) vs fusion + classifier (lr chuẩn) ---
    encoder_params = list(model.pose_encoder.parameters()) + list(model.rgb_encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    fusion_params = [p for n, p in model.named_parameters()
                     if id(p) not in encoder_ids and not n.startswith(("pose_head", "rgb_head", "logit_scale"))]

    optimizer = torch.optim.AdamW(
        [{"params": fusion_params, "lr": args.lr}, {"params": encoder_params, "lr": args.lr * args.encoder_lr_scale}],
        eps=1e-8, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5,
                                                           min_lr=1e-7)

    print(f"Total params    : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Fusion params   : {sum(p.numel() for p in fusion_params):,}")
    print(f"Encoder params  : {sum(p.numel() for p in encoder_params):,}")

    best_acc, best_loss, no_improve = 0.0, float("inf"), 0
    encoders_frozen = args.freeze_epochs > 0
    model.freeze_encoders(encoders_frozen)

    for epoch in range(args.epochs):
        if encoders_frozen and epoch >= args.freeze_epochs:
            model.freeze_encoders(False)
            encoders_frozen = False
            no_improve = 0  # reset patience vì bắt đầu giai đoạn mới
            print("\n>>> Unfreeze encoders (fine-tune toàn bộ)")

        print("\n" + "=" * 70)
        print(f"Epoch {epoch + 1}/{args.epochs}  [{'FROZEN encoders' if encoders_frozen else 'FULL fine-tune'}]")
        print("=" * 70)

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, DEVICE, encoders_frozen)
        val_loss, val_top1, val_topk = validate(model, val_loader, DEVICE, top_k=TOP_K)

        scheduler.step(val_loss)
        lrs = [g["lr"] for g in optimizer.param_groups]

        print(f"\n"
              f"Train loss      : {train_loss:.4f}\n"
              f"Train top-1 acc : {train_acc * 100:.2f}%\n"
              f"Val loss        : {val_loss:.4f}\n"
              f"Val top-1 acc   : {val_top1 * 100:.2f}%\n"
              f"Val top-{TOP_K} acc   : {val_topk * 100:.2f}%\n"
              f"LR (fusion/enc) : {lrs[0]:.7f} / {lrs[1]:.7f}")

        improved = val_top1 > best_acc or (val_top1 == best_acc and val_loss < best_loss)
        if improved:
            no_improve = 0
            best_acc, best_loss = val_top1, val_loss

            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "epoch": epoch, "val_loss": val_loss,
                          "val_top1_acc": val_top1, "val_topk_acc": val_topk, "model_kwargs": model_kwargs}
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(checkpoint, args.output)
            print(f"\n✓ Saved best model  (top-1={best_acc * 100:.2f}%, loss={best_loss:.4f})")
        else:
            no_improve += 1
            print(f"\nNo improvement (best top-1={best_acc * 100:.2f}%), patience={no_improve}/{PATIENCE}")

        if no_improve >= PATIENCE and not encoders_frozen:
            print(f"\n⚑ Early stopping at epoch {epoch + 1}")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    ckpt = torch.load(args.output, map_location="cpu")
    print(f"Best epoch : {ckpt['epoch'] + 1}")
    print(f"Val loss   : {ckpt['val_loss']:.4f}")
    print(f"Top-1 acc  : {ckpt['val_top1_acc'] * 100:.2f}%")
    print(f"Top-{TOP_K} acc  : {ckpt['val_topk_acc'] * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("", parents=[default_args()], add_help=False)
    args = parser.parse_args()
    main(args)