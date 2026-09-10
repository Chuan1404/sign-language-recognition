"""
Test / evaluation script for the checkpoint produced by
train_long_sentences.py (SignLanguageTranslatorV2 on How2Sign).

Evaluates on the TEST split, not the val split. train_long_sentences.py's
val set is used during training to pick the best checkpoint and drive the
LR scheduler / early-stopping — it's implicitly "seen" through that
selection process (a checkpoint that happens to score well on val by chance
gets kept over one that doesn't), so it's not a clean set for final
reported numbers. The test split, by contrast, is carved out up front and
never touched anywhere in training — see shuffled_split() in
train_long_sentences.py.

Reuses shuffled_split(), collate_fn_infer(), compute_bleu(), compute_rouge()
directly from train_long_sentences.py instead of redefining them here —
duplicating that logic risks silently drifting out of sync (e.g. if the
split logic or the BLEU/ROUGE computation changes in the training script
but not here, this script would quietly evaluate on the wrong split or
report numbers computed differently). Same SEED as training -> the 3-way
split reproduces identically, so test_indices here is exactly the slice
train_long_sentences.py carved out and never used.

Uses FULL beam search (default num_beams=4) for the final reported numbers
— unlike train_long_sentences.py's validate(), which uses greedy decoding
(num_beams=1) during training just to keep epoch time reasonable. Beam
search generally gives better generation quality (see earlier discussion:
num_beams doesn't affect the trained weights at all, only how well a given
checkpoint's knowledge gets "extracted" at generation time) — so THIS
script is what should be used for the numbers you'd actually report.

Usage:
    python test_how2sign.py
    python test_how2sign.py --checkpoint outputs/models/how2sign_v2_best.pt
    python test_how2sign.py --errors-only        # only print sentences that don't exactly match GT
    python test_how2sign.py --num-beams 8 --max-length 64
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from tqdm import tqdm

from src.data.hand_landmarks import HandLandmarksDataset
from src.utils import FusionComponent
from src.models.SLT_model import SignLanguageTranslatorV2
from config import ROOT
from src.training.train_long_sentences import (
    shuffled_split,
    collate_fn_infer,
    compute_bleu,
    compute_rouge,
    FEATURE_DIR,
    BATCH_SIZE,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CHECKPOINT = os.path.join(ROOT, "outputs", "models", "how2sign_v2_best.pt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the How2Sign V2 checkpoint with BLEU-1..4 and ROUGE-1/2/L, printing GT vs predicted sentence per sample."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                         help="Path to the .pt checkpoint saved by train_long_sentences.py")
    parser.add_argument("--num-beams", type=int, default=4,
                         help="Beam search width for generation (higher = generally better "
                              "quality, slower — this only affects how the already-trained "
                              "model is decoded, not the model itself)")
    parser.add_argument("--max-length", type=int, default=64,
                         help="Max generated sequence length")
    parser.add_argument("--errors-only", action="store_true",
                         help="Only print samples where the prediction doesn't exactly match GT")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)

    model_kwargs = checkpoint["model_kwargs"]

    model = SignLanguageTranslatorV2(**model_kwargs).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"Checkpoint epoch    : {checkpoint.get('epoch')}")
    print(f"Checkpoint val_loss : {checkpoint.get('val_loss'):.4f}")
    if "val_bleu" in checkpoint:
        # Logged during training with greedy decoding (num_beams=1) — expect
        # this run's numbers (default num_beams=4) to be equal or higher,
        # since beam search generally extracts better output from the same
        # trained weights, not because the model itself changed.
        print(f"Checkpoint val_bleu  (greedy, logged at train time): {checkpoint['val_bleu']}")

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=False)

    dataset = HandLandmarksDataset(
        FEATURE_DIR, tokenizer, FusionComponent(), max_samples=None
    )

    # Same SEED + same 3-way split as train_long_sentences.py -> test_indices
    # here is EXACTLY the slice that was never used for training OR for
    # checkpoint selection during training (val was used for that, so val
    # isn't a clean held-out set anymore — see module docstring above).
    _, _, test_indices = shuffled_split(
        len(dataset), val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED
    )
    test_dataset = Subset(dataset, test_indices)

    print(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=partial(collate_fn_infer, tokenizer=tokenizer),
        num_workers=2, pin_memory=True
    )

    all_refs, all_hyps = [], []
    sample_counter = 0

    with torch.no_grad():
        for features, text_ids, video_mask in tqdm(test_loader, desc="Generating"):
            features   = features.to(DEVICE, non_blocking=True)
            video_mask = video_mask.to(DEVICE, non_blocking=True)

            generated_ids = model.generate(
                features,
                video_mask=video_mask,
                max_length=args.max_length,
                num_beams=args.num_beams
            )

            pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            gt_texts   = tokenizer.batch_decode(text_ids, skip_special_tokens=True)

            for gt, pred in zip(gt_texts, pred_texts):
                sample_counter += 1
                gt   = gt.strip()
                pred = pred.strip()

                all_refs.append(gt)
                all_hyps.append(pred)

                is_exact_match = (gt.lower() == pred.lower())
                if args.errors_only and is_exact_match:
                    continue

                marker = "✓" if is_exact_match else "✗"
                # print(f"[{sample_counter:04d}] {marker}")
                # print(f"  GT  : {gt}")
                # print(f"  Pred: {pred}")

    print("\n" + "=" * 60)
    print("METRICS")
    print("=" * 60)

    bleu_scores = compute_bleu(all_refs, all_hyps)
    for name, score in bleu_scores.items():
        print(f"{name} : {score * 100:.2f}")

    rouge_scores = compute_rouge(all_refs, all_hyps)
    for name, score in rouge_scores.items():
        print(f"{name}: {score * 100:.2f}")

    exact_match_acc = sum(
        r.lower() == h.lower() for r, h in zip(all_refs, all_hyps)
    ) / max(len(all_refs), 1)
    print(f"Exact-match accuracy: {exact_match_acc * 100:.2f}%")


if __name__ == "__main__":
    main()