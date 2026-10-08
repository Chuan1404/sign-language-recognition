"""Evaluate a main.py checkpoint and print the most confident mistakes."""
import argparse
import json
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

# Match the mixed src/package imports used by the training scripts.
PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEVICE, ROOT
from src.data.WSASL_raw import WLASLLandmarksDataset, WLASLImportantLandmarksDataset
from src.models import ISLR_Transformer, ISLR_Transformer_GCN_Relative
from src.training.train import collate_fn
from src.utils import FusionComponent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(Path(ROOT) / "outputs/models/contest_2000_v1.pt"))
    p.add_argument("--data_path", default=str(Path(ROOT) / "datasets/processed/wlasl_features_v2"))
    p.add_argument("--label_path", default=str(Path(ROOT) / "datasets/annotations/WLASL2000"))
    p.add_argument("--model", choices=["transformer", "gcn", "transformer_gcn"], default="transformer")
    p.add_argument("--split", default="test")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--top_errors", type=int, default=20)
    p.add_argument("--only_important", action="store_true")
    p.add_argument("--indices_dir", default=str(Path(ROOT) / "outputs/selected_rgb"))
    args = p.parse_args()
    if args.batch_size < 1 or args.top_errors < 1:
        p.error("batch_size and top_errors must be positive")
    return args


@torch.inference_mode()
def main(args):
    with open(Path(args.label_path) / "gloss2idx.json", encoding="utf-8") as f:
        gloss2idx = json.load(f)
    idx2gloss = {int(idx): gloss for gloss, idx in gloss2idx.items()}

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    kwargs = checkpoint.get("model_kwargs", {"num_classes": len(gloss2idx)})
    if kwargs.get("num_classes", len(gloss2idx)) != len(gloss2idx):
        raise ValueError("Checkpoint class count does not match label_path")

    # Same architecture selection as the current main.py.
    model_cls = ISLR_Transformer if args.model == "transformer" else ISLR_Transformer_GCN_Relative
    model = model_cls(**kwargs)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(DEVICE).eval()
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {model_cls.__name__}; device: {DEVICE}")
    if "epoch" in checkpoint:
        print(f"Saved epoch: {checkpoint['epoch'] + 1}")

    dataset_kwargs = dict(fusion_component=FusionComponent(), mode=args.split)
    dataset_cls = WLASLLandmarksDataset
    if args.only_important:
        dataset_cls = WLASLImportantLandmarksDataset
        dataset_kwargs["indices_dir"] = args.indices_dir
    dataset = dataset_cls(args.data_path, args.label_path, **dataset_kwargs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    mistakes = []
    total = correct = correct_top5 = 0
    for features, labels, video_mask, video_ids in tqdm(loader, desc="Testing"):
        output = model(features.to(DEVICE), video_mask=video_mask.to(DEVICE))
        logits = output["logits"] if isinstance(output, dict) else output[0]
        if logits.shape[-1] != len(gloss2idx):
            raise ValueError("Model output class count does not match labels")
        probabilities = logits.softmax(dim=-1).cpu()
        confidence, predictions = probabilities.max(dim=-1)
        k = min(5, probabilities.shape[-1])
        top_probs, top_indices = probabilities.topk(k, dim=-1)
        total += len(labels)
        correct += (predictions == labels).sum().item()
        correct_top5 += (top_indices == labels[:, None]).any(dim=-1).sum().item()
        for i, video_id in enumerate(video_ids):
            if predictions[i] == labels[i]:
                continue
            mistakes.append({
                "video_id": str(video_id),
                "truth": idx2gloss[labels[i].item()],
                "prediction": idx2gloss[predictions[i].item()],
                "confidence": confidence[i].item(),
                "truth_probability": probabilities[i, labels[i]].item(),
                "top5": [(idx2gloss[j.item()], p.item())
                         for j, p in zip(top_indices[i], top_probs[i])],
            })

    if total == 0:
        raise RuntimeError("No test samples")
    print(f"\nSamples: {total}; wrong: {total - correct}")
    print(f"Top-1: {correct / total:.2%}; Top-{k}: {correct_top5 / total:.2%}")
    mistakes.sort(key=lambda row: (-row["confidence"], row["video_id"]))
    print(f"\nMost confident mistakes (up to {args.top_errors}):")
    for rank, row in enumerate(mistakes[:args.top_errors], 1):
        print(f"\n{rank:02d}. Video: {row['video_id']}")
        print(f"    Truth: {row['truth']} ({row['truth_probability']:.2%})")
        print(f"    Predicted: {row['prediction']} ({row['confidence']:.2%})")
        print("    Top predictions: " + ", ".join(f"{g} ({p:.2%})" for g, p in row["top5"]))
    if not mistakes:
        print("No incorrect predictions.")


if __name__ == "__main__":
    main(parse_args())
