import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DEVICE, ROOT
from pipeline_config import DATA_PATH, LABEL_DIR, IMPORTANCE_DIR, importance_path
from src.data.WSASL_raw import WLASLLandmarksDataset
from src.models.SLT_model import ISLR_Transformer
from src.training.train import collate_fn
from src.utils import FusionComponent

CKPT_PATH = os.path.join(ROOT, "outputs", "models", "contest_100_v1.pt")

@torch.no_grad()
def export_importance_for_split(model, split):
    fusion_component = FusionComponent()
    dataset = WLASLLandmarksDataset(
        feature_dir=DATA_PATH,
        annotation_dir=LABEL_DIR,
        fusion_component=fusion_component,
        mode=split,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    results = {}
    model.eval()

    for features, labels, video_mask, video_ids in loader:
        features = features.to(DEVICE)
        video_mask = video_mask.to(DEVICE).bool()
        labels = labels.to(DEVICE)

        # 1. Trích xuất đặc trưng từng frame: x có shape (B, T, d_model)
        x = model.encode(features, video_mask)

        # 2. Chiếu từng frame qua classifier để tính logit đóng góp: (B, T, num_classes)
        frame_logits = model.classifier(x)

        # 3. Lấy logit ứng với nhãn đúng c (hoặc c = logits.argmax(-1))
        # shape: (B, T)
        target_logits = frame_logits.gather(dim=-1, index=labels.view(-1, 1, 1).expand(-1, x.size(1), 1)).squeeze(-1)

        # 4. Mask các frame padding và áp dụng Softmax theo thời gian
        target_logits = target_logits.masked_fill(~video_mask, -1e9)
        importance = F.softmax(target_logits, dim=1).cpu()

        for i, vid in enumerate(video_ids):
            n = int(video_mask[i].sum().item())
            results[str(vid)] = {
                "importance": importance[i, :n].tolist(),
                "length": n
            }

    out_path = importance_path(split)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"✓ Đã xuất {len(results)} video cho split '{split}' -> {out_path}")


def main():
    print(f"Đang load checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model_kwargs = ckpt.get("model_kwargs", {"num_classes": 100})
    model = ISLR_Transformer(**model_kwargs).to(DEVICE)
    model.load_state_dict(ckpt["model"])

    for split in ["train", "val", "test"]:
        export_importance_for_split(model, split)


if __name__ == "__main__":
    main()