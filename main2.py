"""Train the new 85-point hand mesh Transformer: python main2.py --help."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEVICE, ROOT, set_seed
from src.data.WSASL_mesh_raw import WLASLMeshDataset, collate_mesh
from src.models import ISLR_Transformer_Mesh
from src.utils.fuse_mesh_component import NUM_JOINTS

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def default_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmark_path", "--pose_path", dest="landmark_path",
                        default=os.path.join(ROOT, "datasets", "processed", "wlasl_features_v2"),
                        help="Pose source (<video_id>/pose.npy); old hand landmarks are not loaded")
    parser.add_argument("--mesh_path", "--data_path", dest="mesh_path",
                        default=os.path.join(PROJECT_ROOT, "datasets", "processed", "wlasl100_hand_mesh_85"))
    parser.add_argument("--label_path", default=os.path.join(ROOT, "datasets", "annotations", "WLASL100"))
    parser.add_argument("--output", default=os.path.join(ROOT, "outputs", "models", "transformer_mesh_100.pt"))
    parser.add_argument("--val_mode", default="test", choices=("val", "test"),
                        help="Evaluation split (default: test, matching main.py)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=5000)
    parser.add_argument("--no_cache", action="store_true")
    return parser


def run_epoch(model, loader, device, top_k=2, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct1 = correctk = count = 0
    with torch.set_grad_enabled(training):
        for features, labels, mask, _ in tqdm(loader, desc="Train" if training else "Validate"):
            features, labels, mask = [x.to(device, non_blocking=True)
                                     for x in (features, labels, mask)]
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, loss = model(features, labels=labels, video_mask=mask)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            size = labels.size(0)
            total_loss += loss.item() * size
            correct1 += (logits.argmax(dim=-1) == labels).sum().item()
            correctk += (logits.topk(min(top_k, logits.size(-1)), dim=-1).indices
                         == labels[:, None]).any(dim=-1).sum().item()
            count += size
    if count == 0:
        raise RuntimeError("Empty data loader")
    return total_loss / count, correct1 / count, correctk / count


def main(args):
    if min(args.epochs, args.batch_size, args.patience, args.top_k) < 1:
        raise ValueError("epochs, batch_size, patience and top_k must be positive")
    set_seed(args.seed)
    datasets = [WLASLMeshDataset(
        args.mesh_path, args.label_path, mode=mode, pose_dir=args.landmark_path,
        cache_landmarks=not args.no_cache,
    ) for mode in ("train", args.val_mode)]
    loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=(i == 0),
                          collate_fn=collate_mesh, num_workers=0,
                          pin_memory=DEVICE.type == "cuda") for i, dataset in enumerate(datasets)]
    model_kwargs = dict(num_classes=len(datasets[0].gloss2idx), num_nodes=NUM_JOINTS, d_model=args.d_model,
                        num_encoder_layers=args.num_encoder_layers, nhead=args.nhead,
                        max_seq_len=args.max_seq_len)
    model = ISLR_Transformer_Mesh(**model_kwargs).to(DEVICE)
    print(f"Device: {DEVICE}; parameters: {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    best, no_improve = -1.0, 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    for epoch in range(args.epochs):
        train_loss, _, _ = run_epoch(model, loaders[0], DEVICE, args.top_k, optimizer)
        val_loss, top1, topk = run_epoch(model, loaders[1], DEVICE, args.top_k)
        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"top1={top1:.2%}, top{args.top_k}={topk:.2%}")
        if top1 > best:
            best, no_improve = top1, 0
            torch.save(dict(
                model=model.state_dict(), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), epoch=epoch, val_loss=val_loss,
                val_top1_acc=top1, val_topk_acc=topk, model_kwargs=model_kwargs,
                model_name="ISLR_Transformer_Mesh", args=vars(args),
                gloss2idx=datasets[0].gloss2idx,
            ), args.output)
            print(f"Saved: {args.output}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


if __name__ == "__main__":
    main(default_args().parse_args())
