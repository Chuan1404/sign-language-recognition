import os
import json
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from config import _COORD_DIM

class WLASLMultimodalDataset(Dataset):
    def __init__(self, feature_dir, annotation_dir, extracted_frames_dir, fusion_component, mode="train"):
        self.feature_dir = feature_dir
        self.extracted_frames_dir = extracted_frames_dir
        self.fusion_component = fusion_component

        self.samples = []

        with open(os.path.join(annotation_dir, "gloss2idx.json"), "r") as f:
            self.gloss2idx = json.load(f)

        with open(os.path.join(annotation_dir, f"{mode}.json"), "r") as f:
            data = json.load(f)
            video_ids = [d["video_id"] for d in data]
            all_video_names = sorted(video_ids)

        for index, video_name in enumerate(all_video_names):
            video_dir = os.path.join(feature_dir, video_name)
            frames_dir = os.path.join(extracted_frames_dir, video_name)

            if not os.path.isdir(video_dir):
                continue
                
            # Kiểm tra xem có đủ 5 frame đã được trích xuất hay không
            if not os.path.exists(frames_dir) or len(os.listdir(frames_dir)) == 0:
                continue

            left_hand_path = os.path.join(video_dir, "left_hand.npy")
            right_hand_path = os.path.join(video_dir, "right_hand.npy")
            pose_path = os.path.join(video_dir, "pose.npy")
            text_path = os.path.join(video_dir, "gloss.txt")

            if (os.path.exists(left_hand_path) and os.path.exists(right_hand_path) and 
                os.path.exists(pose_path) and os.path.exists(text_path)):
                self.samples.append({
                    "video_name": video_name, 
                    "left_hand_path": left_hand_path, 
                    "right_hand_path": right_hand_path, 
                    "pose_path": pose_path,
                    "text_path": text_path, 
                    "frames_dir": frames_dir
                })
                
        # Image augmentations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 1. Load Landmarks
        left_features = np.load(item["left_hand_path"])
        right_features = np.load(item["right_hand_path"])
        pose_features = np.load(item["pose_path"])
        video_id = item["video_name"]

        T = left_features.shape[0]

        pose_features = pose_features.reshape(T, 33, 3)[:, :, :_COORD_DIM]
        left_features = left_features.reshape(T, 21, 3)[:, :, :_COORD_DIM]
        right_features = right_features.reshape(T, 21, 3)[:, :, :_COORD_DIM]

        with open(item["text_path"], "r", encoding="utf-8") as f:
            gloss = f.read().strip()
        label_id = self.gloss2idx[gloss]

        position_features = self.fusion_component.fuse_follow_position(pose_features, left_features, right_features)
        shape_features = self.fusion_component.fuse_follow_shape(pose_features, left_features, right_features)
        average_feature = self.fusion_component.fuse(pose_features, left_features, right_features)

        features = np.concatenate([position_features, shape_features, average_feature], axis=-1)
        
        # 2. Load RGB Images (5 frames)
        frames_dir = item["frames_dir"]
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        
        images = []
        for file in frame_files:
            img_path = os.path.join(frames_dir, file)
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)
            images.append(img_tensor)
            
        # Nếu ít hơn 5 frames do video ngắn, lặp lại frame cuối
        while len(images) < 5 and len(images) > 0:
            images.append(images[-1])
            
        if len(images) == 0:
            images = [torch.zeros(3, 224, 224) for _ in range(5)]
            
        # Stack thành shape: (5, 3, 224, 224)
        images_tensor = torch.stack(images[:5])

        return features, images_tensor, label_id, video_id
