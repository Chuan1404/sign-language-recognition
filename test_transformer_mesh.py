import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from main2 import run_epoch
from config import _N_POSE
from src.data.WSASL_mesh_raw import WLASLMeshDataset, collate_mesh
from src.extract_features import WLASL_hand_contour_extract as extractor
from src.models import ISLR_Transformer_Mesh
from src.utils.fuse_mesh_component import FuseMeshComponent, FEATURE_DIM, NUM_JOINTS
from src.utils.fusion_block import FusionComponent


class MeshTests(unittest.TestCase):
    def test_fusion_matches_original_keypoint_formulas(self):
        rng = np.random.default_rng(42)
        pose = rng.uniform(.1, .8, (4, 33, 2)).astype(np.float32)
        pose[:, [23, 24], 1] = .9
        left = rng.uniform(.1, .8, (4, 21, 2)).astype(np.float32)
        right = rng.uniform(.1, .8, (4, 21, 2)).astype(np.float32)
        left[0, 4] = 0
        right[1] = 0
        original, mesh_fusion = FusionComponent(), FuseMeshComponent(num_hand_nodes=21, wrist_index=0)
        for name in ('fuse', 'fuse_follow_shape', 'fuse_follow_position'):
            np.testing.assert_allclose(getattr(mesh_fusion, name)(pose, left, right),
                                       getattr(original, name)(pose, left, right), atol=1e-6)
        mesh = np.repeat(left[:, :1], 85, axis=1)
        mesh[:, 64:] = left
        before = mesh.copy()
        shaped = FuseMeshComponent().fuse_follow_shape(pose, mesh, mesh).reshape(4, NUM_JOINTS, 2)
        np.testing.assert_array_equal(shaped[:, _N_POSE + 64], 0)
        np.testing.assert_array_equal(mesh, before)

        # Lowering the hip cutoff must not remove or change any hand feature.
        lowered_hips = pose.copy()
        lowered_hips[:, [23, 24], 1] = .05
        fusion = FuseMeshComponent()
        for name in ('fuse', 'fuse_follow_shape', 'fuse_follow_position'):
            below = getattr(fusion, name)(lowered_hips, mesh, mesh).reshape(4, NUM_JOINTS, 2)
            if name != 'fuse':
                above = getattr(fusion, name)(pose, mesh, mesh).reshape(4, NUM_JOINTS, 2)
                np.testing.assert_array_equal(below[:, _N_POSE:], above[:, _N_POSE:])
            self.assertGreater(np.abs(below[:, _N_POSE:]).sum(), 0)

    def test_model_gradients_and_padding(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        model = ISLR_Transformer_Mesh(d_model=8, num_encoder_layers=1, nhead=2,
                                     dim_feedforward=16, dropout=0, num_classes=3, max_seq_len=10)
        features = torch.randn(2, 4, 3 * FEATURE_DIM)
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        model.eval()
        logits, loss = model(features, labels=torch.tensor([0, 2]), video_mask=mask)
        self.assertEqual(tuple(logits.shape), (2, 3))
        loss.backward()
        self.assertFalse(any('gcn' in name.lower() for name, _ in model.named_parameters()))
        self.assertFalse(hasattr(model, 'mesh_projection'))
        for projection in (model.shape_projection, model.average_projection, model.position_projection):
            gradient = projection[0].weight.grad
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        features[1, 2:] = 1e6
        torch.testing.assert_close(logits, model(features, video_mask=mask).logits)
        self.assertTrue(torch.isfinite(model(features).logits).all())
        with self.assertRaises(ValueError):
            model(features[..., :-1], video_mask=mask)
        with self.assertRaises(ValueError):
            model(features, video_mask=torch.zeros_like(mask))

    def test_extractor_preserves_landmarks_when_mesh_fails(self):
        from mediapipe.tasks.python import vision
        xyz = np.full((21, 3), .4, np.float32)
        xyz[:, 0] += np.arange(21) * .005
        xyz[:, 2] = np.linspace(-.1, .1, 21)
        detection = SimpleNamespace(
            hand_landmarks=[[SimpleNamespace(x=float(x), y=float(y), z=float(z)) for x,y,z in xyz]],
            handedness=[[SimpleNamespace(index=1, score=.9)]])
        empty = SimpleNamespace(hand_landmarks=[], handedness=[])
        frames = [np.zeros((80, 100, 3), np.uint8) for _ in range(3)]
        class Capture:
            def isOpened(self): return True
            def get(self, key): return 25.
            def read(self): return (True, frames.pop(0)) if frames else (False, None)
            def release(self): pass
        class Detector:
            def __init__(self): self.results = iter([detection, detection, empty])
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def detect_for_video(self, image, ms): return next(self.results)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            with patch.object(extractor.cv2, 'VideoCapture', return_value=Capture()), \
                 patch.object(vision.HandLandmarker, 'create_from_options', return_value=Detector()), \
                 patch.object(extractor, 'segment_hand', return_value=np.zeros((80,100), np.uint8)), \
                 patch.object(extractor, 'contour_nodes', side_effect=[np.ones((64,2), np.float32) * 20, None]):
                metadata = extractor.extract_video('fake.mp4', temp, 'unused.task')
            left = np.load(Path(temp) / 'left_hand.npy').reshape(3, 85, 2)
            mask = np.load(Path(temp) / 'keypoint_valid.npy')
            saved_xyz = np.load(Path(temp) / 'landmark_xyz.npy')
            self.assertEqual(metadata['nodes_per_hand'], 85)
            np.testing.assert_array_equal(saved_xyz[:2, 0], np.stack([xyz, xyz]))
            np.testing.assert_array_equal(left[1, 64:], xyz[:, :2])
            np.testing.assert_array_equal(left[1, :64], 0)
            self.assertEqual(mask[:,0].sum(axis=1).tolist(), [85, 21, 0])
            self.assertEqual(np.load(Path(temp) / 'valid.npy')[:,0].tolist(), [True, False, False])

    def test_mesh_dataset_and_training_without_old_hand_files(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            root = Path(temp)
            ann, poses, meshes = [root / name for name in ('annotations', 'poses', 'meshes')]
            ann.mkdir()
            (ann / 'gloss2idx.json').write_text(json.dumps({'hello': 0, 'bye': 1}))
            (ann / 'train.json').write_text(json.dumps([
                {'video_id': 'a', 'gloss': 'hello'}, {'video_id': 'b', 'gloss': 'bye'}]))
            for video, frames in [('a', 3), ('b', 5)]:
                directory = poses / video
                directory.mkdir(parents=True)
                pose = np.ones((frames, 33, 3), dtype=np.float32) * .5
                pose[:, 11, 0] = .3
                pose[:, 12, 0] = .7
                pose[:, [23,24], 1] = .9
                np.save(directory / 'pose.npy', pose)
                directory = meshes / video
                directory.mkdir(parents=True)
                for hand in ('left_hand', 'right_hand'):
                    np.save(directory / (hand + '.npy'), np.ones((frames, 85, 2), dtype=np.float32) * .4)
                np.save(directory / 'valid.npy', np.zeros((frames, 2), dtype=bool))
                mask = np.ones((frames, 2, 85), dtype=bool)
                mask[..., :64] = False
                np.save(directory / 'keypoint_valid.npy', mask)
            dataset = WLASLMeshDataset(str(meshes), str(ann), pose_dir=str(poses), cache_landmarks=False)
            features, _, _ = dataset[0]
            self.assertEqual(features.shape, (3, 3 * FEATURE_DIM))
            blocks = features.reshape(3, 3, NUM_JOINTS, 2)
            np.testing.assert_array_equal(blocks[:, :, _N_POSE:_N_POSE + 64], 0)
            self.assertGreater(np.abs(blocks[:, 2, _N_POSE + 64: _N_POSE + 85]).sum(), 0)
            loader = DataLoader(dataset, batch_size=2, collate_fn=collate_mesh)
            batch = next(iter(loader))
            self.assertEqual(batch[-1], ['a', 'b'])
            self.assertEqual(batch[2].sum(dim=1).tolist(), [3, 5])
            model = ISLR_Transformer_Mesh(d_model=8, num_encoder_layers=1, nhead=2,
                                         dim_feedforward=16, num_classes=2, max_seq_len=10)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            for result in (run_epoch(model, loader, 'cpu', optimizer=optimizer), run_epoch(model, loader, 'cpu')):
                self.assertTrue(np.isfinite(result).all())
            np.save(meshes / 'a' / 'left_hand.npy', np.zeros((3, 64, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, '85 points'):
                dataset[0]


if __name__ == '__main__':
    unittest.main()
