import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import ROOT

# ---------------------------------------------------------------------------
# Defaults (giống prertrain_main.py)
# ---------------------------------------------------------------------------
DEFAULT_VIDEO_DIR = os.path.join(ROOT, "outputs", "extracted_frames")
DEFAULT_FEATURE_DIR = os.path.join(ROOT, "datasets", "processed", "wlasl_features_v2")
DEFAULT_IMAGE_SIZE = 112
OUTPUT_FILENAME = "rgb_frames.npy"


def build_transform(image_size: int) -> transforms.Compose:
    """Transform giống hệt WLASLLandmarksRGBDataset để đảm bảo nhất quán."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def preprocess_video(
    video_dir: str,
    save_path: str,
    transform: transforms.Compose,
) -> int:
    """
    Load tất cả ảnh trong `video_dir`, apply transform, lưu vào `save_path`.

    Returns:
        Số frame đã xử lý, hoặc 0 nếu không có ảnh.
    """
    image_files = sorted(
        f
        for f in os.listdir(video_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not image_files:
        return 0

    frames = []
    for filename in image_files:
        path = os.path.join(video_dir, filename)
        image = Image.open(path).convert("RGB")
        tensor = transform(image)  # (C, H, W), float32
        frames.append(tensor.numpy())

    # Stack → (T, C, H, W), lưu dưới dạng float32
    arr = np.stack(frames, axis=0).astype(np.float32)
    np.save(save_path, arr)
    return len(frames)


def main(args: argparse.Namespace) -> None:
    transform = build_transform(args.image_size)

    # Lấy danh sách tất cả video trong video_dir
    if not os.path.isdir(args.video_dir):
        raise FileNotFoundError(f"video_dir không tồn tại: {args.video_dir}")

    video_names = sorted(os.listdir(args.video_dir))
    print(f"Tìm thấy {len(video_names)} thư mục video trong: {args.video_dir}")
    print(f"Feature dir: {args.feature_dir}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Output file per video: {OUTPUT_FILENAME}\n")

    skipped = 0
    processed = 0
    total_frames = 0

    for video_name in tqdm(video_names, desc="Pre-processing RGB"):
        video_path = os.path.join(args.video_dir, video_name)

        if not os.path.isdir(video_path):
            skipped += 1
            continue

        feat_dir = os.path.join(args.feature_dir, video_name)
        if not os.path.isdir(feat_dir):
            # Feature chưa được extract → bỏ qua
            skipped += 1
            continue

        save_path = os.path.join(feat_dir, OUTPUT_FILENAME)

        if os.path.exists(save_path) and not args.overwrite:
            # Đã xử lý rồi → bỏ qua
            processed += 1
            continue

        n_frames = preprocess_video(video_path, save_path, transform)

        if n_frames == 0:
            skipped += 1
        else:
            processed += 1
            total_frames += n_frames

    print("\n✓ Hoàn thành!")
    print(f"  Đã xử lý : {processed} video ({total_frames} frames)")
    print(f"  Bỏ qua   : {skipped} video (không có ảnh hoặc feature dir)")
    print("\nBây giờ có thể train với WLASLLandmarksRGBDataset(use_cache=True).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-process RGB frames and cache as .npy files."
    )
    parser.add_argument(
        "--video_dir",
        default=DEFAULT_VIDEO_DIR,
        help="Thư mục chứa các thư mục video (extracted frames)",
    )
    parser.add_argument(
        "--feature_dir",
        default=DEFAULT_FEATURE_DIR,
        help="Thư mục chứa feature .npy của từng video (nơi lưu rgb_frames.npy)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Kích thước ảnh sau khi resize (mặc định: 112)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ghi đè rgb_frames.npy nếu đã tồn tại",
    )

    args = parser.parse_args()
    main(args)
