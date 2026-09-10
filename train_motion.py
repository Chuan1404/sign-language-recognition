import os

from config import ROOT

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.motion_dataset import MotionKeypointDataset
from src.models.motion_model import KeypointMotionTransformer, MotionReconstructionLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_pos, total_vel = 0.0, 0.0, 0.0

    for coords_input, mask_flags, target in tqdm(loader, desc="Train"):
        coords_input = coords_input.to(DEVICE)
        mask_flags = mask_flags.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        pose_pred, left_pred, right_pred = model(coords_input, mask_flags)
        loss, pos_loss, vel_loss = criterion(
            pose_pred, left_pred, right_pred, target, mask_flags,
            model.pose_dim, model.left_dim, model.right_dim,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_pos += pos_loss.item()
        total_vel += vel_loss.item()

    n = len(loader)
    return total_loss / n, total_pos / n, total_vel / n


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss, total_pos, total_vel = 0.0, 0.0, 0.0

    for coords_input, mask_flags, target in tqdm(loader, desc="Val"):
        coords_input = coords_input.to(DEVICE)
        mask_flags = mask_flags.to(DEVICE)
        target = target.to(DEVICE)

        pose_pred, left_pred, right_pred = model(coords_input, mask_flags)
        loss, pos_loss, vel_loss = criterion(
            pose_pred, left_pred, right_pred, target, mask_flags,
            model.pose_dim, model.left_dim, model.right_dim,
        )

        total_loss += loss.item()
        total_pos += pos_loss.item()
        total_vel += vel_loss.item()

    n = len(loader)
    return total_loss / n, total_pos / n, total_vel / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment_dir", default=os.path.join(ROOT, "datasets", "processed", "motion_how2sign"), help="Thư mục segment sạch từ how2sign_extract.py")
    parser.add_argument("--save_path", default="./outputs/motion_model.pt")
    parser.add_argument("--coord_dim", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    dataset = MotionKeypointDataset(args.segment_dir, coord_dim=args.coord_dim)

    val_len = max(1, int(len(dataset) * args.val_ratio))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    coords_input,mask_flags, target  = dataset[0]

    model = KeypointMotionTransformer(
        pose_dim=dataset.pose_dim,
        left_dim=dataset.left_dim,
        right_dim=dataset.right_dim,
        max_seq_len=max(dataset.window_size * 2, 512),
    ).to(DEVICE)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = MotionReconstructionLoss(velocity_weight=0.5)

    best_val = float("inf")
    no_improve = 0

    for epoch in range(args.epochs):
        train_loss, train_pos, train_vel = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_pos, val_vel = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: train={train_loss:.4f} (pos={train_pos:.4f}, vel={train_vel:.4f})  "
              f"val={val_loss:.4f} (pos={val_pos:.4f}, vel={val_vel:.4f})")

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            torch.save({
                "model": model.state_dict(),
                "model_kwargs": dict(
                    pose_dim=dataset.pose_dim,
                    left_dim=dataset.left_dim,
                    right_dim=dataset.right_dim,
                    max_seq_len=max(dataset.window_size * 2, 512),
                ),
                "epoch": epoch,
                "val_loss": val_loss,
            }, args.save_path)
            print(f"  ✓ saved best (val_loss={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping tại epoch {epoch}")
                break


if __name__ == "__main__":
    main()