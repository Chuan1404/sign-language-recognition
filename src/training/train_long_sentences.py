import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
from src.data.hand_landmarks import HandLandmarksDataset
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Subset
from functools import partial

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from src.utils import FusionComponent
from src.models.SLT_model import SignLanguageTranslatorV2
from config import ROOT


DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 8
EPOCHS        = 100
LR_NEW        = 1e-4
LR_MT5        = 2e-5
PATIENCE      = 15
WARMUP_EPOCHS = 5
VAL_RATIO     = 0.1
TEST_RATIO    = 0.1
SEED          = 42     # for reproducible shuffling/splitting
MAX_TEXT_LEN  = 128     # sentences are much longer than a single gloss

# --- Generation-based validation (BLEU/ROUGE) --------------------------------
# Computing BLEU/ROUGE requires actually generating (model.generate()),
# which is far slower than the teacher-forced forward pass used for loss
# (sequential decoding vs one parallel forward pass) — beam search=4 over
# the full val set was observed at ~20s/batch (~2h for the whole val set).
# Two knobs to keep this affordable during training:
#   - GENERATE_NUM_BEAMS=1 (greedy) instead of beam search: much faster,
#     good enough to track whether generation quality is improving epoch
#     to epoch (this isn't the final reported number, just a training-time
#     signal — re-run test_long_sentences.py with proper beam search for
#     final reported metrics).
#   - GENERATE_EVERY_N_EPOCHS: only run generation-based validation every
#     N epochs; val_loss (cheap) is still computed and used for
#     early-stopping/scheduler.step() every epoch regardless.
GENERATE_NUM_BEAMS      = 1
GENERATE_EVERY_N_EPOCHS = 1
GENERATE_MAX_LENGTH     = 64

print(DEVICE)


FEATURE_DIR     = os.path.join(ROOT, "datasets", "processed", "full_body_how2sign")
ANNOTATION_PATH = os.path.join(ROOT, "datasets", "annotations", "how2sign_flat.json")
SAVE_DIR        = os.path.join(ROOT, "outputs", "models")
os.makedirs(SAVE_DIR, exist_ok=True)

fusion_component = FusionComponent()


def shuffled_split(n, val_ratio=0.1, test_ratio=0.1, seed=42):
    """3-way split: train / val / test.

    val is used during training to pick the best checkpoint and drive the
    LR scheduler/early-stopping — it's NOT a clean held-out set for final
    evaluation, since checkpoint selection is implicitly fit to it (a model
    that happens to do well on val by chance gets kept over one that
    doesn't, even if the difference is noise). test is never touched during
    training at all; it's the set test_how2sign.py should evaluate on for
    numbers you'd actually report.

    Same shuffle-then-cut approach as before, just two cuts instead of one:
    val first val_ratio, then test_ratio, remainder is train.
    """
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_val  = int(n * val_ratio)
    n_test = int(n * test_ratio)

    val_indices   = indices[:n_val]
    test_indices  = indices[n_val:n_val + n_test]
    train_indices = indices[n_val + n_test:]

    return train_indices, val_indices, test_indices

def collate_fn_infer(batch, tokenizer):
    """Like collate_fn, but keeps text padded with pad_token_id (NOT -100)
    — needed for decoding ground-truth text back to strings for BLEU/ROUGE.
    tokenizer.decode() would raise on -100 (not a valid vocab id)."""
    features, texts = [], []

    for feature, text in batch:
        features.append(torch.as_tensor(feature, dtype=torch.float32))
        texts.append(text)

    real_lengths = [f.shape[0] for f in features]

    features = pad_sequence(features, batch_first=True)
    texts    = pad_sequence(
        texts, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    video_mask = (
        torch.arange(features.shape[1]).unsqueeze(0)
        < torch.tensor(real_lengths).unsqueeze(1)
    ).long()

    return features, texts, video_mask


def compute_bleu(all_refs, all_hyps):
    """Corpus-level BLEU-1/2/3/4 (cumulative n-gram precision, the standard
    way papers report BLEU-1..4), with smoothing so short/imperfect
    sentences don't collapse to a hard 0 just because one n-gram order has
    zero overlap."""
    smoothing = SmoothingFunction().method1

    references = [[ref.split()] for ref in all_refs]
    hypotheses = [hyp.split() for hyp in all_hyps]

    bleu_scores = {}
    for n in (1, 2, 3, 4):
        weights = tuple(1.0 / n for _ in range(n)) + tuple(0.0 for _ in range(4 - n))
        bleu_scores[f"bleu{n}"] = corpus_bleu(
            references, hypotheses, weights=weights, smoothing_function=smoothing
        )

    return bleu_scores


def compute_rouge(all_refs, all_hyps):
    """Average ROUGE-1/2/L F1 across samples (sentence-level F1, macro-averaged
    — the usual way ROUGE is reported for generation tasks)."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for ref, hyp in zip(all_refs, all_hyps):
        scores = scorer.score(ref, hyp)
        for key in totals:
            totals[key] += scores[key].fmeasure

    n = max(len(all_refs), 1)
    return {key: value / n for key, value in totals.items()}


def collate_fn(batch, tokenizer):
    features, texts = [], []

    for feature, text in batch:
        features.append(torch.as_tensor(feature, dtype=torch.float32))
        texts.append(text)

    real_lengths = [f.shape[0] for f in features]

    features = pad_sequence(features, batch_first=True)
    texts    = pad_sequence(
        texts, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    video_mask = (
        torch.arange(features.shape[1]).unsqueeze(0)
        < torch.tensor(real_lengths).unsqueeze(1)
    ).long()

    labels = texts.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return features, labels, video_mask


def get_lr_scale(epoch, warmup_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0


def train_one_epoch(model, loader, optimizer):

    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training")

    for features, text_ids, video_mask in pbar:

        features   = features.to(DEVICE, non_blocking=True)
        text_ids   = text_ids.to(DEVICE, non_blocking=True)
        video_mask = video_mask.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            features,
            text_ids=text_ids,
            video_mask=video_mask
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


def validate(model, loader, tokenizer, run_generation=True,
             num_beams=1, max_length=64):
    """
    loader must be built with collate_fn_infer (texts padded with
    pad_token_id, NOT -100) — this function derives the -100 version
    itself for the loss, so both loss and generation-based metrics come
    from the exact same batch instead of two separate passes over the data.

    run_generation=False skips BLEU/ROUGE for this call (still returns loss)
    — used to only pay the generate() cost every GENERATE_EVERY_N_EPOCHS
    epochs, since it's much slower than the teacher-forced forward pass.
    """

    model.eval()
    total_loss = 0

    all_refs, all_hyps = [], []

    with torch.no_grad():
        for features, text_ids, video_mask in loader:
            features   = features.to(DEVICE, non_blocking=True)
            text_ids   = text_ids.to(DEVICE, non_blocking=True)
            video_mask = video_mask.to(DEVICE, non_blocking=True)

            # -100 version for the loss — text_ids here is padded with
            # pad_token_id (from collate_fn_infer), same convention the
            # old collate_fn used for the label tensor fed to mT5.
            labels = text_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(
                features,
                text_ids=labels,
                video_mask=video_mask
            )
            total_loss += outputs.loss.item()

            if run_generation:
                generated_ids = model.generate(
                    features,
                    video_mask=video_mask,
                    max_length=max_length,
                    num_beams=num_beams
                )
                pred_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                gt_texts   = tokenizer.batch_decode(text_ids, skip_special_tokens=True)

                all_refs.extend(t.strip() for t in gt_texts)
                all_hyps.extend(t.strip() for t in pred_texts)

    avg_loss = total_loss / len(loader)

    if not run_generation:
        return avg_loss, None, None

    bleu_scores  = compute_bleu(all_refs, all_hyps)
    rouge_scores = compute_rouge(all_refs, all_hyps)

    return avg_loss, bleu_scores, rouge_scores


def main():

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=False)

    dataset = HandLandmarksDataset(
        FEATURE_DIR, tokenizer, fusion_component, max_samples=None)

    # test_indices are intentionally unused here — they're excluded from
    # both train and val so test_how2sign.py (which recomputes this same
    # split with the same SEED) evaluates on data this run never touched
    # at all, not even indirectly through checkpoint selection.
    train_indices, val_indices, test_indices = shuffled_split(
        len(dataset), val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=SEED
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset   = Subset(dataset, val_indices)

    print(f"Dataset — train: {len(train_dataset)}, val: {len(val_dataset)}, "
          f"test (held out, not used here): {len(test_indices)}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=partial(collate_fn_infer, tokenizer=tokenizer),
        num_workers=2, pin_memory=True
    )

    feature, _ = train_dataset[0]


    model_kwargs = dict(
        input_dim=feature.shape[-1],
        hidden_dim=256,
        num_encoder_layers=6,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.2,
        max_seq_len=5000,
        pretrained_model="google/mt5-small"
    )

    model = SignLanguageTranslatorV2(**model_kwargs).to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    param_groups = model.get_optimizer_param_groups(
        lr_new_modules=LR_NEW,
        lr_mt5=LR_MT5
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_loss  = float("inf")
    no_improve = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}")
        if epoch < WARMUP_EPOCHS:
            lr_scale = get_lr_scale(epoch, WARMUP_EPOCHS)
            for i, group in enumerate(optimizer.param_groups):
                group["lr"] = base_lrs[i] * lr_scale

        torch.cuda.empty_cache()

        train_loss = train_one_epoch(model, train_loader, optimizer)

        run_generation = (epoch % GENERATE_EVERY_N_EPOCHS == 0)
        val_loss, bleu_scores, rouge_scores = validate(
            model, val_loader, tokenizer,
            run_generation=run_generation,
            num_beams=GENERATE_NUM_BEAMS,
            max_length=GENERATE_MAX_LENGTH
        )

        if epoch >= WARMUP_EPOCHS:
            scheduler.step(val_loss)

        print(f"Train loss : {train_loss:.4f}")
        print(f"Val   loss : {val_loss:.4f}")

        if bleu_scores is not None:
            print(f"Val   BLEU-1/2/3/4 : "
                  f"{bleu_scores['bleu1']*100:.2f} / {bleu_scores['bleu2']*100:.2f} / "
                  f"{bleu_scores['bleu3']*100:.2f} / {bleu_scores['bleu4']*100:.2f}  "
                  f"(greedy, num_beams={GENERATE_NUM_BEAMS} — free-running, not teacher-forced)")
            print(f"Val   ROUGE-1/2/L  : "
                  f"{rouge_scores['rouge1']*100:.2f} / {rouge_scores['rouge2']*100:.2f} / "
                  f"{rouge_scores['rougeL']*100:.2f}")
        else:
            print(f"Val   BLEU/ROUGE   : skipped this epoch "
                  f"(GENERATE_EVERY_N_EPOCHS={GENERATE_EVERY_N_EPOCHS})")

        if val_loss < best_loss:
            best_loss  = val_loss
            no_improve = 0
            checkpoint = {
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "epoch":        epoch,
                "val_loss":     val_loss,
                "model_kwargs": model_kwargs
            }
            if bleu_scores is not None:
                checkpoint["val_bleu"]  = bleu_scores
                checkpoint["val_rouge"] = rouge_scores

            torch.save(checkpoint, os.path.join(SAVE_DIR, "how2sign_v2_best.pt"))
            print(f"✓ Saved best model  (val_loss={best_loss:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement (best={best_loss:.4f}, patience={no_improve}/{PATIENCE})")

            if no_improve >= PATIENCE:
                print(f"\n⚑ Early stopping at epoch {epoch+1}")
                break


if __name__ == "__main__":
    main()