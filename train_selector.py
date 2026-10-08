import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from src.models import ISLR_Transformer_GCN

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DEVICE
from pipeline_config import (DATA_PATH, IMPORTANCE_DIR, LABEL_DIR, MODEL_DIR, importance_path, set_seed, )
from src.data.WSASL_raw import WLASLLandmarksDataset, MultiFeatureSkeletonAugmentor
from src.data.augmentation import AugmentedSkeletonDataset
from src.training.train import collate_fn, train_one_epoch_selector, validate_selector
from src.utils import FusionComponent

MODEL_NAME = "contest_100_selector_v1.pt"


def default_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--label_path", default=LABEL_DIR)
    p.add_argument("--output", default=os.path.join(MODEL_DIR, MODEL_NAME))
    p.add_argument("--val_mode", default="test", help="split dùng để early-stop/chọn checkpoint")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p


def make_loader(ds, batch_size, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=0, pin_memory=True)


def fit_selector(model_kwargs, train_ds, val_loader, args, ckpt_path, tag):
    augmented_train = AugmentedSkeletonDataset(
        train_ds,
        MultiFeatureSkeletonAugmentor(rng=np.random.default_rng(args.seed)),
    )
    loader = make_loader(augmented_train, args.batch_size, True)
    model = ISLR_Transformer_GCN(**model_kwargs).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    best_loss, best_info, no_improve = float("inf"), None, 0
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch_selector(model, loader, optimizer, DEVICE)
        val_loss, top1, topk = validate_selector(model, val_loader, DEVICE, top_k=args.top_k)
        scheduler.step(val_loss)
        print(f"[{tag}] ep {epoch + 1:3d} | train {train_loss:.4f}/{train_acc * 100:5.2f}% | "
              f"val {val_loss:.4f} top1 {top1 * 100:5.2f}% top{args.top_k} {topk * 100:5.2f}% | "
              f"lr {optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < best_loss:
            best_loss, no_improve = val_loss, 0
            best_info = {"epoch": epoch, "val_loss": val_loss, "val_top1_acc": top1, "val_topk_acc": topk}
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "model_kwargs": model_kwargs, **best_info}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[{tag}] Early stopping ở epoch {epoch + 1}")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"[{tag}] Best epoch {best_info['epoch'] + 1} | val loss {best_info['val_loss']:.4f} | "
          f"top1 {best_info['val_top1_acc'] * 100:.2f}%")
    return model, best_info


@torch.no_grad()
def extract_frame_importance(model, loader, device):
    model.eval()
    results = {}
    for batch_idx, (features, _labels, video_mask, video_ids) in enumerate(loader):
        features = features.to(device, non_blocking=True)
        video_mask = video_mask.to(device, non_blocking=True)
        importance = model(features, labels=None, video_mask=video_mask)["frame_importance"].cpu()
        mask = video_mask.cpu().bool()

        for i, vid in enumerate(video_ids):
            n = int(mask[i].sum())
            results[str(vid)] = {"importance": importance[i, :n].tolist(), "length": n}

    return results


def summarize(name, results):
    fr = [np.mean(np.asarray(r["importance"]) / max(np.sum(r["importance"]), 1e-9) > 1.0 / r["length"]) for r in
          results.values()]
    print(f"  [{name}] {len(results)} video | tỉ lệ frame vượt 1/T trung bình = {np.mean(fr) * 100:.1f}%")

def save_json(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"✓ Đã lưu {path}")


def main(args):
    set_seed(args.seed)
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    print(f"Device: {DEVICE}")
    fusion = FusionComponent()

    train_ds = WLASLLandmarksDataset(args.data_path, args.label_path, fusion, mode="train")
    test_ds = WLASLLandmarksDataset(args.data_path, args.label_path, fusion, mode="test")

    test_loader = make_loader(test_ds, args.batch_size, False)
    val_ds = test_ds if args.val_mode == "test" else WLASLLandmarksDataset(
        args.data_path, args.label_path, fusion, mode=args.val_mode
    )
    val_loader = make_loader(val_ds, args.batch_size, False)

    with open(os.path.join(args.label_path, "gloss2idx.json"), "r") as f:
        num_classes = len(json.load(f))
    model_kwargs = {"num_classes": num_classes}
    print(f"Number of classes: {num_classes}")

    model, _ = fit_selector(model_kwargs=model_kwargs, train_ds=train_ds, val_loader=val_loader, args=args,
        ckpt_path=args.output, tag="full")

    t_loss, t1, tk = validate_selector(model, test_loader, DEVICE, top_k=args.top_k)
    print(f"\nTEST: loss {t_loss:.4f} | top1 {t1 * 100:.2f}% | top{args.top_k} {tk * 100:.2f}%")

    os.makedirs(IMPORTANCE_DIR, exist_ok=True)

    print("\nTrích importance cho val/test bằng model đầy đủ...")
    # for name, ds in ("test", test_ds):
    #     res = extract_frame_importance(model, make_loader(ds, args.batch_size, False), DEVICE)
    #     summarize(name, res)
    #     save_json(res, importance_path(name)

    res = extract_frame_importance(model, make_loader(test_ds, args.batch_size, False), DEVICE)
    summarize("test", res)
    save_json(res, importance_path("test"))

    print("\nTrích importance cho train...")
    train_res = extract_frame_importance(model, make_loader(train_ds, args.batch_size, False), DEVICE)
    summarize("train", train_res)
    save_json(train_res, importance_path("train"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("", parents=[default_args()], add_help=False)
    main(parser.parse_args())
