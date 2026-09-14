import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def collate_fn(batch):
    features, labels = [], []
    for feature, label in batch:
        features.append(torch.as_tensor(feature, dtype=torch.float32))
        labels.append(label)

    real_lengths = [f.shape[0] for f in features]

    features = pad_sequence(features, batch_first=True)

    video_mask = (
        torch.arange(features.shape[1]).unsqueeze(0)
        < torch.tensor(real_lengths).unsqueeze(1)
    ).long()

    labels = torch.tensor(labels, dtype=torch.long)

    return features, labels, video_mask

def train_one_epoch(model, loader, optimizer, device='cuda'):

    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training")

    for features, labels, video_mask in pbar:

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
        for features, labels, video_mask in loader:
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
