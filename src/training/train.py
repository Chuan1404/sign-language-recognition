import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def collate_fn(batch):
    features, labels, video_ids = [], [], []
    for feature, label, video_id in batch:
        features.append(torch.as_tensor(feature, dtype=torch.float32))
        labels.append(label)
        video_ids.append(video_id)

    real_lengths = [f.shape[0] for f in features]

    features = pad_sequence(features, batch_first=True)

    video_mask = (
        torch.arange(features.shape[1]).unsqueeze(0)
        < torch.tensor(real_lengths).unsqueeze(1)
    ).long()

    labels = torch.tensor(labels, dtype=torch.long)

    return features, labels, video_mask, video_ids

def collate_fn_multimodal(batch):
    features, images, labels, video_ids = [], [], [], []
    for feature, image, label, video_id in batch:
        features.append(torch.as_tensor(feature, dtype=torch.float32))
        images.append(image) # tensor (5, 3, 224, 224)
        labels.append(label)
        video_ids.append(video_id)

    real_lengths = [f.shape[0] for f in features]
    features = pad_sequence(features, batch_first=True)
    images = torch.stack(images, dim=0) # (B, 5, 3, 224, 224)

    video_mask = (
        torch.arange(features.shape[1]).unsqueeze(0)
        < torch.tensor(real_lengths).unsqueeze(1)
    ).long()

    labels = torch.tensor(labels, dtype=torch.long)
    return features, images, labels, video_mask, video_ids

def train_one_epoch(model, loader, optimizer, device='cuda'):

    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training")

    for features, labels, video_mask, video_ids in pbar:

        features   = features.to(device, non_blocking=True)
        labels     = labels.to(device, non_blocking=True)
        video_mask = video_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits, loss = model(
            features,
            labels=labels,
            video_mask=video_mask
        )

        # loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)

def validate(model, loader, top_k=5, device='cuda'):

    model.eval()
    total_loss = 0

    total_correct_top1 = 0
    total_correct_topk = 0
    total_samples = 0

    with torch.no_grad():
        for features, labels, video_mask, video_ids in loader:
            features   = features.to(device, non_blocking=True)
            labels     = labels.to(device, non_blocking=True)
            video_mask = video_mask.to(device, non_blocking=True)

            logits, loss = model(
                features,
                labels=labels,
                video_mask=video_mask
            )
            total_loss += loss

            pred_top1 = logits.argmax(dim=-1)

            total_correct_top1 += (pred_top1 == labels).sum().item()

            k = min(top_k, logits.size(-1))
            pred_topk = logits.topk(k=k, dim=-1).indices

            in_topk = (pred_topk == labels.unsqueeze(-1)).any(dim=-1)
            total_correct_topk += in_topk.sum().item()

            total_samples += labels.size(0)

    avg_loss = total_loss / len(loader)
    top1_acc = total_correct_top1 / total_samples if total_samples > 0 else 0.0
    topk_acc = total_correct_topk / total_samples if total_samples > 0 else 0.0

    return avg_loss, top1_acc, topk_acc

def test(model, loader, idx2gloss, top_k=5, device='cuda'):
    model.eval()

    total_loss = 0.0

    total_correct_top1 = 0
    total_correct_topk = 0
    total_samples = 0

    # Đếm các cặp:
    # (true_label, predicted_label)
    wrong_predictions = Counter()

    with torch.no_grad():
        for features, labels, video_mask, video_ids in loader:

            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            video_mask = video_mask.to(device, non_blocking=True)

            logits, loss = model(
                features,
                labels=labels,
                video_mask=video_mask
            )

            total_loss += loss.item()

            # -------------------------
            # Top-1
            # -------------------------
            pred_top1 = logits.argmax(dim=-1)

            total_correct_top1 += (
                pred_top1 == labels
            ).sum().item()

            # -------------------------
            # Top-k
            # -------------------------
            k = min(top_k, logits.size(-1))

            pred_topk = logits.topk(
                k=k,
                dim=-1
            ).indices

            in_topk = (
                pred_topk == labels.unsqueeze(-1)
            ).any(dim=-1)

            total_correct_topk += in_topk.sum().item()

            total_samples += labels.size(0)

            # -------------------------
            # Collect wrong predictions
            # -------------------------
            wrong_mask = pred_top1 != labels

            wrong_true = labels[wrong_mask].cpu().tolist()
            wrong_pred = pred_top1[wrong_mask].cpu().tolist()

            for true_label, pred_label in zip(
                wrong_true,
                wrong_pred
            ):
                wrong_predictions[
                    (true_label, pred_label)
                ] += 1

    # -------------------------
    # Metrics
    # -------------------------

    avg_loss = total_loss / len(loader)

    top1_acc = (
        total_correct_top1 / total_samples
        if total_samples > 0 else 0.0
    )

    topk_acc = (
        total_correct_topk / total_samples
        if total_samples > 0 else 0.0
    )

    # -------------------------
    # Top 10 wrong pairs
    # -------------------------

    top10_wrong = wrong_predictions.most_common(10)

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(f"Loss      : {avg_loss:.4f}")
    print(f"Top-1 Acc : {top1_acc:.4f}")
    print(f"Top-{top_k} Acc : {topk_acc:.4f}")

    print("\nTop 10 most frequent wrong predictions:")
    print("-" * 70)

    for rank, ((true_id, pred_id), count) in enumerate(
        top10_wrong,
        start=1
    ):
        true_name = idx2gloss.get(
            true_id,
            str(true_id)
        )

        pred_name = idx2gloss.get(
            pred_id,
            str(pred_id)
        )

        print(
            f"{rank:2d}. "
            f"True: {true_name:<20} "
            f"Pred: {pred_name:<20} "
            f"Count: {count}"
        )

    print("=" * 70)

    return avg_loss, top1_acc, topk_acc