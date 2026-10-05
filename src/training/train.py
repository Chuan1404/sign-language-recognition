from collections import Counter

import torch
import torch.nn.functional as F
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

def masked_mean_pool(x, video_mask):
    mask = video_mask.unsqueeze(-1).float()  # (B, T, 1)
    summed = (x * mask).sum(dim=1)  # (B, D)
    counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1) — avoid /0
    return summed / counts


def collate_fn_rgb(batch):
    feats, rgbs, labels, video_id = zip(*batch)

    # Convert everything to tensors up front
    feats = [torch.as_tensor(f, dtype=torch.float32) for f in feats]
    rgbs = [torch.as_tensor(r, dtype=torch.float32) for r in rgbs]

    B = len(batch)

    feature_lengths = [f.shape[0] for f in feats]
    rgb_lengths = [r.shape[0] for r in rgbs]

    T_feat = max(feature_lengths)
    T_rgb = max(rgb_lengths)

    features = torch.zeros(B, T_feat, *feats[0].shape[1:])
    rgb = torch.zeros(B, T_rgb, *rgbs[0].shape[1:])

    feature_mask = torch.zeros(B, T_feat, dtype=torch.bool)
    rgb_mask = torch.zeros(B, T_rgb, dtype=torch.bool)

    for i, (f, r) in enumerate(zip(feats, rgbs)):
        L_feat = f.shape[0]
        L_rgb = r.shape[0]

        features[i, :L_feat] = f
        rgb[i, :L_rgb] = r

        feature_mask[i, :L_feat] = True
        rgb_mask[i, :L_rgb] = True

    return (
        features,
        rgb,
        feature_mask,
        rgb_mask,
        torch.tensor(labels, dtype=torch.long),
        video_id,
    )


def train_one_epoch(model, loader, optimizer, device="cuda"):

    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training")

    for features, labels, video_mask, _ in pbar:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        video_mask = video_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        _logits, loss = model(features, labels=labels, video_mask=video_mask)

        # loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, top_k=5, device="cuda"):

    model.eval()
    total_loss = 0

    total_correct_top1 = 0
    total_correct_topk = 0
    total_samples = 0

    with torch.no_grad():
        for features, labels, video_mask, _ in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            video_mask = video_mask.to(device, non_blocking=True)

            logits, loss = model(features, labels=labels, video_mask=video_mask)
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


def train_one_epoch_selector(model, loader, optimizer, device="cuda"):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training")

    for features, labels, video_mask, video_ids in pbar:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        video_mask = video_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        output = model(features, labels=labels, video_mask=video_mask)

        logits = output["logits"]
        loss = output["loss"]

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # -------------------------
        # Loss
        # -------------------------
        total_loss += loss.item()

        # -------------------------
        # Accuracy
        # -------------------------
        pred = logits.argmax(dim=-1)

        total_correct += (pred == labels).sum().item()

        total_samples += labels.size(0)

        acc = total_correct / total_samples

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc * 100:.2f}%")

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


@torch.no_grad()
def validate_selector(model, loader, device, top_k=2):
    model.eval()

    total_loss = 0.0

    total_samples = 0

    top1_correct = 0
    topk_correct = 0

    for features, labels, video_mask, video_ids in loader:
        features = features.to(device, non_blocking=True)

        labels = labels.to(device, non_blocking=True)

        video_mask = video_mask.to(device, non_blocking=True)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        output = model(features, labels=labels, video_mask=video_mask)

        logits = output["logits"]

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        total_samples += batch_size

        # ----------------------------------------------------
        # Top-1
        # ----------------------------------------------------

        predictions = logits.argmax(dim=1)

        top1_correct += (predictions == labels).sum().item()

        # ----------------------------------------------------
        # Top-K
        # ----------------------------------------------------

        _, topk_indices = torch.topk(logits, k=min(top_k, logits.size(1)), dim=1)

        topk_correct += (topk_indices == labels.unsqueeze(1)).any(dim=1).sum().item()

    val_loss = total_loss / max(total_samples, 1)

    top1_acc = top1_correct / max(total_samples, 1)

    topk_acc = topk_correct / max(total_samples, 1)

    return (val_loss, top1_acc, topk_acc)

def train_rgb_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc="Training")

    for batch in pbar:
        landmarks, rgb, feature_mask, rgb_mask, labels, _video_id = batch

        landmarks = landmarks.to(device, non_blocking=True)
        rgb = rgb.to(device, non_blocking=True)
        feature_mask = feature_mask.to(device, non_blocking=True)
        rgb_mask = rgb_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        out = model.forward(landmarks, rgb, labels=labels,
                            feature_mask=feature_mask, rgb_mask=rgb_mask)

        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = labels.size(0)
        total_loss += out["loss"].item() * bs
        total_correct += (out["logits"].argmax(dim=-1) == labels).sum().item()
        total_samples += bs

        acc = total_correct / max(total_samples, 1)
        pbar.set_postfix(loss=f"{out['loss'].item():.4f}", acc=f"{acc * 100:.2f}%")

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
    }



@torch.no_grad()
def validate_rgb(model, loader, device, top_k=5):
    model.eval()

    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_topk = 0
    total_samples = 0

    for batch in loader:
        landmarks, rgb, feature_mask, rgb_mask, labels, _video_id = batch

        landmarks = landmarks.to(device, non_blocking=True)
        rgb = rgb.to(device, non_blocking=True)
        feature_mask = feature_mask.to(device, non_blocking=True)
        rgb_mask = rgb_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out = model.forward(landmarks, rgb, labels=labels,
                            feature_mask=feature_mask, rgb_mask=rgb_mask)

        logits = out["logits"]
        bs = labels.size(0)
        total_loss += out["loss"].item() * bs
        total_samples += bs

        # Top-1
        total_correct_top1 += (logits.argmax(dim=1) == labels).sum().item()

        # Top-K
        k = min(top_k, logits.size(1))
        topk_idx = torch.topk(logits, k=k, dim=1).indices
        total_correct_topk += (topk_idx == labels.unsqueeze(1)).any(dim=1).sum().item()

    n = max(total_samples, 1)
    return {
        "loss": total_loss / n,
        "top1": total_correct_top1 / n,
        "topk": total_correct_topk / n,
    }


def pretrain_rgb_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss, n_batches = 0.0, 0

    pbar = tqdm(loader, desc="Pretrain")

    for batch in pbar:
        landmarks, rgb, feature_mask, rgb_mask, _, _video_id = batch


        landmarks = landmarks.to(device, non_blocking=True)
        rgb = rgb.to(device, non_blocking=True)
        feature_mask = feature_mask.to(device, non_blocking=True)
        rgb_mask = rgb_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        out = model.forward_pretrain(landmarks, rgb, feature_mask, rgb_mask)

        out["loss"].backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += out["loss"].item()
        n_batches += 1

        pbar.set_postfix(loss=f"{out['loss'].item():.4f}")

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def prevalidate_rgb(model, loader, device, top_k=5):
    model.eval()

    all_pose, all_rgb = [], []

    for batch in loader:
        landmarks, rgb, feature_mask, rgb_mask, _, _video_id = batch

        landmarks = landmarks.to(device, non_blocking=True)
        rgb = rgb.to(device, non_blocking=True)
        feature_mask = feature_mask.to(device, non_blocking=True)
        rgb_mask = rgb_mask.to(device, non_blocking=True)

        out = model.forward_pretrain(landmarks, rgb, feature_mask, rgb_mask)

        all_pose.append(out["p_g"].cpu())
        all_rgb.append(out["r_g"].cpu())

    pose = torch.cat(all_pose)
    rgb = torch.cat(all_rgb)

    scale = model.logit_scale.exp().clamp(max=100).item()

    loss = model._info_nce(pose, rgb, scale).item()

    sim = pose @ rgb.t()

    target = torch.arange(sim.size(0)).unsqueeze(1)

    def recall(s, k):
        k = min(k, s.size(1))

        return (s.topk(k, dim=1).indices == target).any(dim=1).float().mean().item()

    return {
        "loss": loss,
        "p2r_r1": recall(sim, 1),
        "p2r_rk": recall(sim, top_k),
        "r2p_r1": recall(sim.t(), 1),
        "n": sim.size(0),
    }


def test(model, loader, idx2gloss, top_k=5, device="cuda"):
    model.eval()

    total_loss = 0.0

    total_correct_top1 = 0
    total_correct_topk = 0
    total_samples = 0

    # Đếm các cặp:
    # (true_label, predicted_label)
    wrong_predictions = Counter()

    with torch.no_grad():
        for features, labels, video_mask in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            video_mask = video_mask.to(device, non_blocking=True)

            logits, loss = model(features, labels=labels, video_mask=video_mask)

            total_loss += loss.item()

            # -------------------------
            # Top-1
            # -------------------------
            pred_top1 = logits.argmax(dim=-1)

            total_correct_top1 += (pred_top1 == labels).sum().item()

            # -------------------------
            # Top-k
            # -------------------------
            k = min(top_k, logits.size(-1))

            pred_topk = logits.topk(k=k, dim=-1).indices

            in_topk = (pred_topk == labels.unsqueeze(-1)).any(dim=-1)

            total_correct_topk += in_topk.sum().item()

            total_samples += labels.size(0)

            # -------------------------
            # Collect wrong predictions
            # -------------------------
            wrong_mask = pred_top1 != labels

            wrong_true = labels[wrong_mask].cpu().tolist()
            wrong_pred = pred_top1[wrong_mask].cpu().tolist()

            for true_label, pred_label in zip(wrong_true, wrong_pred):
                wrong_predictions[(true_label, pred_label)] += 1

    # -------------------------
    # Metrics
    # -------------------------

    avg_loss = total_loss / len(loader)

    top1_acc = total_correct_top1 / total_samples if total_samples > 0 else 0.0

    topk_acc = total_correct_topk / total_samples if total_samples > 0 else 0.0

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

    for rank, ((true_id, pred_id), count) in enumerate(top10_wrong, start=1):
        true_name = idx2gloss.get(true_id, str(true_id))

        pred_name = idx2gloss.get(pred_id, str(pred_id))

        print(f"{rank:2d}. True: {true_name:<20} Pred: {pred_name:<20} Count: {count}")

    print("=" * 70)

    return avg_loss, top1_acc, topk_acc
