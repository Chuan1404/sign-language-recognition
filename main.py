import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch.utils.data import DataLoader
from src.data.WSASL_raw import WLASLLandmarksDataset
from src.data.augmentation import AugmentedSkeletonDataset, SkeletonAugmentor
from src.models.SLT_model import ISLR_V2, ISLR_V1, ISLR_V3, ISLR_V4
from src.utils import FusionComponent
from src.training.train import collate_fn, train_one_epoch, validate

import torch
import argparse
from config import ROOT, DEVICE
import json


DATA_PATH = os.path.join(ROOT, "datasets", "processed", "wlasl_features_v2")
LABEL_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL100")
# CONFIG_PATH = os.path.join(LABEL_DIR, "gloss.txt")
OUTPUT_DIR    = os.path.join(ROOT, "outputs", "models")
MODEL_NAME = f"contest_100_v4.pt"
LR = 1e-4

BATCH_SIZE = 8
EPOCHS = 100
TOP_K = 2
PATIENCE = 10

print(DEVICE)

def default_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_path", default=f"{DATA_PATH}")
    parser.add_argument("--label_path", default=f"{LABEL_DIR}")
    # parser.add_argument("--config_path", default=f"{CONFIG_PATH}", help="Path to the train dataset")
    parser.add_argument("--output", default=f"{os.path.join(OUTPUT_DIR, MODEL_NAME)}")

    return parser

def main(args):
    fusion_component = FusionComponent()

    base_train = WLASLLandmarksDataset(
        args.data_path, args.label_path, fusion_component, mode="train"
    )
    base_val = WLASLLandmarksDataset(
        args.data_path, args.label_path, fusion_component, mode="test"
    )

    feature, _ = base_train[0]

    # train_dataset = AugmentedSkeletonDataset(base_train, SkeletonAugmentor())
    train_dataset = base_train
    val_dataset = base_val

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn,
        num_workers=0, pin_memory=True
    )

    with open(os.path.join(LABEL_DIR, "gloss2idx.json"), "r") as f:
        gloss2idx = json.load(f)

    num_classes = len(gloss2idx)
    model_kwargs = dict(
        num_classes=num_classes,
    )

    model = ISLR_V4(**model_kwargs).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params    : {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        eps=1e-8,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best = 0
    best_loss = 100
    no_improve = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}")

        torch.cuda.empty_cache()

        train_loss = train_one_epoch(model, train_loader, optimizer, device=DEVICE)
        val_loss, val_top1_acc, val_topk_acc = validate(model, val_loader, top_k=TOP_K, device=DEVICE)

        print(f"Train loss      : {train_loss:.4f}")
        print(f"Test   loss      : {val_loss:.4f}")
        print(f"Test   top-1 acc : {val_top1_acc * 100:.2f}%")
        print(f"Val   top-{TOP_K} acc : {val_topk_acc*100:.2f}%")

        if val_loss < best_loss:
            no_improve = 0
            best = val_top1_acc
            best_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "test_loss": val_loss,
                "test_top1_acc": val_top1_acc,
                "test_topk_acc": val_topk_acc,
                "model_kwargs": model_kwargs,
            }, os.path.join(args.output))
            print(f"✓ Saved best model  {best * 100:.2f}% (Test loss: {best_loss:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement (best={best * 100:.2f}% loss={best_loss:.4f}), patience={no_improve}/{PATIENCE})")

            if no_improve >= PATIENCE:
                print(f"\n⚑ Early stopping at epoch {epoch + 1}")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser("", parents=[default_args()], add_help=False)
    args = parser.parse_args()
    main(args)