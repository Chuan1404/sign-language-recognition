"""Train independent pose/RGB classifiers with probability late fusion."""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

import argparse
import json
import torch
from torch.utils.data import DataLoader

from config import DEVICE
from pipeline_config import DATA_PATH, LABEL_DIR, MODEL_DIR, SELECTED_RGB_DIR, set_seed
from src.data.WSASL_raw import WLASLLandmarksRGBDataset
from src.models import ISLR_Multimodal, ISLR_Multimodal_v2
from src.training.train import collate_fn_rgb, train_rgb_one_epoch, validate_rgb
from src.utils import FusionComponent


class MultimodalDataset(WLASLLandmarksRGBDataset):
    """Keep full pose and uniformly sample spatial RGB frames from cache."""

    def __init__(self, *args, **kwargs):
        kwargs["full_frames"] = False
        kwargs["keep_full_pose"] = True
        kwargs.setdefault("max_frames", 20)
        kwargs["img_size"] = 64
        super().__init__(*args, **kwargs)


def default_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--label_path", default=LABEL_DIR)
    p.add_argument("--rgb_dir", default=SELECTED_RGB_DIR)
    p.add_argument("--model", choices=["ISLR_Multimodal", "ISLR_Multimodal_v2"], default="ISLR_Multimodal_v2")
    p.add_argument("--pose_checkpoint", default=str(Path(MODEL_DIR) / "contest_100_selector_v1.pt"))
    p.add_argument("--output", default=str(Path(MODEL_DIR) / "islr_multimodal_best.pt"))
    p.add_argument("--val_mode", default="test")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5, help="RGB fine-tuning LR")
    p.add_argument("--pose_lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--top_k", type=int, default=10, help="Number of uniformly sampled RGB frames; pose uses all frames")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=str(DEVICE))
    p.add_argument("--rgb_backbone", choices=["resnet18", "resnet34", "resnet50"], default="resnet18")
    p.add_argument("--rgb_d_model", type=int, default=256)
    p.add_argument("--rgb_dropout", type=float, default=0.1)
    p.add_argument("--separate_rgb_encoders", action="store_true")
    return p


def make_loader(dataset, args, shuffle):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      collate_fn=collate_fn_rgb, num_workers=0,
                      pin_memory=torch.device(args.device).type == "cuda")


def main(args):
    output_path = Path(args.output).resolve()
    set_seed(args.seed)

    with open(Path(args.label_path) / "gloss2idx.json", encoding="utf-8") as f:
        gloss2idx = json.load(f)

    pose_checkpoint = torch.load(args.pose_checkpoint, map_location="cpu", weights_only=True)
    pose_kwargs = dict(pose_checkpoint.get("model_kwargs", {}))


    model_kwargs = dict(
        num_classes=len(gloss2idx),
        pose_kwargs=pose_kwargs,
        pose_checkpoint=None,
        rgb_checkpoint=None,
        rgb_pretrained=True,  # Nạp weights pretrained của backbone RGB
        rgb_backbone=args.rgb_backbone,
        rgb_d_model=args.rgb_d_model,
        rgb_dropout=args.rgb_dropout,
        share_encoder=not args.separate_rgb_encoders,
        temporal="bilstm" if args.model == "ISLR_Multimodal_v2" else "none",
        rgb_pooling="mean" if args.model == "ISLR_Multimodal_v2" else "attention",
        img_size=64,
        freeze_pose=False,  # Train nhánh pose
        freeze_rgb=False,  # Train nhánh RGB
    )
    model_class = ISLR_Multimodal_v2 if args.model == "ISLR_Multimodal_v2" else ISLR_Multimodal
    model = model_class(**model_kwargs).to(args.device)

    print("Training pose/RGB from scratch with a pretrained RGB backbone")
    print(f"Model: {args.model}; RGB temporal: {model_kwargs['temporal']}")
    print(f"Pose trainable: {not model.freeze_pose}; fusion: probability late fusion (50/50)")
    print(f"Checkpoint selection split: {args.val_mode}; device: {args.device}")

    fusion = FusionComponent()
    train_ds = MultimodalDataset(args.data_path, args.label_path, args.rgb_dir, fusion, mode="train", max_frames=args.top_k)
    val_ds = MultimodalDataset(args.data_path, args.label_path, args.rgb_dir, fusion, mode=args.val_mode, augment=False, max_frames=args.top_k)
    train_loader = make_loader(train_ds, args, True)
    val_loader = make_loader(val_ds, args, False)

    groups = [{"params": [p for p in model.rgb_model.parameters() if p.requires_grad], "lr": args.lr}]

    if not model.freeze_pose:
        groups.append({"params": [p for p in model.pose_model.parameters() if p.requires_grad], "lr": args.pose_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc, best_loss, no_improve = -1.0, float("inf"), 0
    for epoch in range(args.epochs):
        train = train_rgb_one_epoch(model, train_loader, optimizer, args.device)
        val = validate_rgb(model, val_loader, args.device, top_k=5)
        scheduler.step(val["loss"])
        print(f"Epoch {epoch + 1}/{args.epochs} | train loss={train['loss']:.4f} acc={train['acc']:.2%} | "
              f"val loss={val['loss']:.4f} top1={val['top1']:.2%} top5={val['topk']:.2%} | "
              f"LR={[group['lr'] for group in optimizer.param_groups]}")
        improved = val["top1"] > best_acc or (val["top1"] == best_acc and val["loss"] < best_loss)
        if improved:
            best_acc, best_loss, no_improve = val["top1"], val["loss"], 0
            torch.save(dict(
                model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                model_class=args.model, model_kwargs=model_kwargs, epoch=epoch, val_loss=best_loss,
                val_top1_acc=best_acc, val_topk_acc=val["topk"],
                gloss2idx=gloss2idx, training_args=vars(args),
            ), output_path)
            print(f"Saved: {output_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping")
                break
    print(f"Best top1={best_acc:.2%}, loss={best_loss:.4f}; checkpoint={output_path}")


if __name__ == "__main__":
    main(default_args().parse_args())
