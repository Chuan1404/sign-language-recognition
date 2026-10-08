"""Fine-tune pretrained pose/RGB branches using probability late fusion."""
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
from src.models import ISLR_Multimodal
from src.training.train import collate_fn_rgb, train_rgb_one_epoch, validate_rgb
from src.utils import FusionComponent


class MultimodalDataset(WLASLLandmarksRGBDataset):
    """Reuse RGB loading/augmentation, but keep the full pose sequence."""

    def __getitem__(self, idx):
        _, rgb, label, video_id = super().__getitem__(idx)
        full_pose = self._load_features(self.samples[idx])
        return full_pose, rgb, label, video_id


def default_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--label_path", default=LABEL_DIR)
    p.add_argument("--rgb_dir", default=SELECTED_RGB_DIR)
    p.add_argument("--pose_checkpoint", default=str(Path(MODEL_DIR) / "contest_100_selector_v1.pt"))
    p.add_argument("--rgb_checkpoint", default=str(Path(MODEL_DIR) / "stage2_pose_rgb_contrastive.pt"))
    p.add_argument("--output", default=str(Path(MODEL_DIR) / "islr_multimodal_best.pt"))
    p.add_argument("--val_mode", default="test")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-5, help="RGB fine-tuning LR")
    p.add_argument("--pose_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=str(DEVICE))
    p.add_argument("--finetune_pose", action="store_true", help="Unfreeze pose; default trains RGB only")
    p.add_argument("--pose_weight", type=float, default=0.5)
    p.add_argument("--pose_temperature", type=float, default=1.0)
    p.add_argument("--rgb_temperature", type=float, default=1.0)
    p.add_argument("--rgb_backbone", choices=["resnet18", "resnet34", "resnet50"], default="resnet18")
    p.add_argument("--temporal", choices=["bilstm", "transformer", "none", "opticalflow"], default="bilstm")
    p.add_argument("--rgb_d_model", type=int, default=256)
    p.add_argument("--rgb_nhead", type=int, default=8)
    p.add_argument("--rgb_dropout", type=float, default=0.1)
    p.add_argument("--separate_rgb_encoders", action="store_true")
    return p


def make_loader(dataset, args, shuffle):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      collate_fn=collate_fn_rgb, num_workers=0,
                      pin_memory=torch.device(args.device).type == "cuda")


def main(args):
    if min(args.epochs, args.batch_size, args.patience, args.top_k) < 1:
        raise ValueError("epochs, batch_size, patience and top_k must be positive")
    if args.lr <= 0 or args.pose_lr <= 0:
        raise ValueError("Learning rates must be positive")
    if not 0 < args.pose_weight < 1:
        raise ValueError("Use 0 < pose_weight < 1 so both branches contribute")
    output_path = Path(args.output).resolve()
    for path in (args.pose_checkpoint, args.rgb_checkpoint):
        if output_path == Path(path).resolve():
            raise ValueError("Output must differ from the source checkpoints")

    set_seed(args.seed)
    with open(Path(args.label_path) / "gloss2idx.json", encoding="utf-8") as f:
        gloss2idx = json.load(f)
    pose_checkpoint = torch.load(args.pose_checkpoint, map_location="cpu", weights_only=True)
    pose_kwargs = dict(pose_checkpoint.get("model_kwargs", {}))
    if pose_kwargs.pop("num_classes", len(gloss2idx)) != len(gloss2idx):
        raise ValueError("Pose checkpoint class count does not match label_path")
    del pose_checkpoint

    # Complete constructor metadata, independent of source checkpoint paths.
    model_kwargs = dict(
        num_classes=len(gloss2idx), pose_kwargs=pose_kwargs,
        rgb_backbone=args.rgb_backbone, temporal=args.temporal,
        rgb_d_model=args.rgb_d_model, rgb_nhead=args.rgb_nhead,
        rgb_dropout=args.rgb_dropout, rgb_pretrained=False, img_size=112,
        share_encoder=not args.separate_rgb_encoders,
        freeze_pose=not args.finetune_pose, freeze_rgb=False,
        pose_weight=args.pose_weight,
        pose_temperature=args.pose_temperature, rgb_temperature=args.rgb_temperature,
    )
    model = ISLR_Multimodal(
        **model_kwargs, pose_checkpoint=args.pose_checkpoint,
        rgb_checkpoint=args.rgb_checkpoint,
    ).to(args.device)
    print(f"Loaded pose: {args.pose_checkpoint}")
    print(f"Loaded RGB: {args.rgb_checkpoint}")
    print(f"Pose trainable: {args.finetune_pose}; pose weight: {args.pose_weight}")
    print(f"Checkpoint selection split: {args.val_mode}; device: {args.device}")

    fusion = FusionComponent()
    train_ds = MultimodalDataset(args.data_path, args.label_path, args.rgb_dir, fusion, mode="train")
    val_ds = MultimodalDataset(args.data_path, args.label_path, args.rgb_dir, fusion, mode=args.val_mode, augment=False)
    train_loader = make_loader(train_ds, args, True)
    val_loader = make_loader(val_ds, args, False)

    groups = [{"params": [p for p in model.rgb_model.parameters() if p.requires_grad], "lr": args.lr}]
    if args.finetune_pose:
        groups.append({"params": [p for p in model.pose_model.parameters() if p.requires_grad], "lr": args.pose_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-8,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc, best_loss, no_improve = -1.0, float("inf"), 0
    for epoch in range(args.epochs):
        train = train_rgb_one_epoch(model, train_loader, optimizer, args.device)
        val = validate_rgb(model, val_loader, args.device, top_k=args.top_k)
        scheduler.step(val["loss"])
        print(f"Epoch {epoch + 1}/{args.epochs} | train loss={train['loss']:.4f} acc={train['acc']:.2%} | "
              f"val loss={val['loss']:.4f} top1={val['top1']:.2%} top{args.top_k}={val['topk']:.2%} | "
              f"LR={[group['lr'] for group in optimizer.param_groups]}")
        improved = val["top1"] > best_acc or (val["top1"] == best_acc and val["loss"] < best_loss)
        if improved:
            best_acc, best_loss, no_improve = val["top1"], val["loss"], 0
            torch.save(dict(
                model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                model_kwargs=model_kwargs, epoch=epoch, val_loss=best_loss,
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
