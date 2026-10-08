import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import numpy as np

from src.data.WSASL_raw import MultiFeatureSkeletonAugmentor

import argparse
import json

import torch
from torch.utils.data import DataLoader

from config import DEVICE, ROOT
from pipeline_config import set_seed
from src.models import ISLR_GCN, ISLR_Transformer, ISLR_Transformer_GCN
from src.data.augmentation import AugmentedSkeletonDataset
from src.data.WSASL_raw import WLASLLandmarksDataset
from src.training.train import collate_fn, train_one_epoch, validate
from src.utils import FusionComponent

DATA_PATH = os.path.join(ROOT, "datasets", "processed", "wlasl_features_v2")
LABEL_DIR = os.path.join(ROOT, "datasets", "annotations", "WLASL100")
# CONFIG_PATH = os.path.join(LABEL_DIR, "gloss.txt")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "models")
MODEL_NAME = "contest_100_v1.pt"
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
    parser.add_argument("--output", default=f"{os.path.join(OUTPUT_DIR, MODEL_NAME)}")
    parser.add_argument("--model", default="transformer_gcn", choices=["gcn", "transformer", "transformer_gcn"], help="Model architecture")
    parser.add_argument("--val_mode", default="test")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)

    return parser


def main(args):
    set_seed(args.seed)
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    fusion_component = FusionComponent()
    base_train = WLASLLandmarksDataset(
        args.data_path, args.label_path,
        fusion_component=fusion_component, mode="train"
    )
    base_val = WLASLLandmarksDataset(
        args.data_path, args.label_path,
        fusion_component=fusion_component, mode=args.val_mode
    )
    train_dataset = AugmentedSkeletonDataset(
        base_train,
        MultiFeatureSkeletonAugmentor(rng=np.random.default_rng(args.seed)),
    )
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

    with open(os.path.join(args.label_path, "gloss2idx.json"), "r") as f:
        gloss2idx = json.load(f)

    num_classes = len(gloss2idx)
    model_kwargs = {
        "num_classes": num_classes,
    }

    if args.model == "gcn":
        print("Using model: ISLR_GCN")
        model = ISLR_Transformer_GCN(**model_kwargs).to(DEVICE)
    elif args.model == "transformer_gcn":
        print("Using model: ISLR_Transformer_GCN")
        model = ISLR_Transformer_GCN(**model_kwargs).to(DEVICE)
    else:
        model = ISLR_Transformer(**model_kwargs).to(DEVICE)
        # model = ISLR_Transformer_Relative(**model_kwargs).to(DEVICE)
        # model = ISLR_Transformer_Selector(**model_kwargs).to(DEVICE)


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params    : {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best = 0
    best_loss = float("inf")
    no_improve = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")

        torch.cuda.empty_cache()

        train_loss = train_one_epoch(model, train_loader, optimizer, device=DEVICE)
        val_loss, val_top1_acc, val_topk_acc = validate(
            model, val_loader, top_k=args.top_k, device=DEVICE
        )

        scheduler.step(val_loss)

        print(f"Train loss      : {train_loss:.4f}")
        print(f"Val    loss      : {val_loss:.4f}")
        print(f"Val    top-1 acc : {val_top1_acc * 100:.2f}%")
        print(f"Val   top-{args.top_k} acc : {val_topk_acc * 100:.2f}%")

        if val_loss < best_loss:
            no_improve = 0
            best = val_top1_acc
            best_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_top1_acc": val_top1_acc,
                    "val_topk_acc": val_topk_acc,
                    "model_kwargs": model_kwargs,
                },
                os.path.join(args.output),
            )
            print(f"✓ Saved best model  {best * 100:.2f}% (Val loss: {best_loss:.4f})")
        else:
            no_improve += 1
            print(
                f"  No improvement (best={best * 100:.2f}% loss={best_loss:.4f}), patience={no_improve}/{args.patience})"
            )

            if no_improve >= args.patience:
                print(f"\n⚑ Early stopping at epoch {epoch + 1}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser("", parents=[default_args()], add_help=False)
    args = parser.parse_args()
    main(args)
