import numpy as np


class KalmanFilter1D:
    """Kalman filter mô hình 'vận tốc không đổi' (constant velocity), áp dụng
    ĐỘC LẬP cho từng chiều toạ độ trong 1 landmark array (T, D) — vector hoá
    bằng numpy để xử lý toàn bộ D chiều cùng lúc, không loop Python qua từng
    chiều.

    State mỗi chiều: [position, velocity]  (2,)
    Chỉ đo được position (H = [1, 0]) — velocity được ước lượng ẩn, giúp
    filter "đoán" hợp lý cả khi thiếu quan sát (frame bị occlusion).
    """

    def __init__(self, process_noise=1e-3, measurement_noise=1e-2, dt=1.0):
        self.dt = dt
        self.F = np.array([[1.0, dt], [0.0, 1.0]])            # state transition
        self.H = np.array([[1.0, 0.0]])                        # observation (chỉ đo position)

        q = process_noise
        self.Q = q * np.array([
            [dt ** 3 / 3, dt ** 2 / 2],
            [dt ** 2 / 2, dt],
        ])                                                      # process noise (white-noise-accel)
        self.R = np.array([[measurement_noise]])                 # measurement noise

    def smooth(self, series, visible=None):
        """
        series : (T, D) — chuỗi toạ độ theo thời gian, D = số chiều phẳng
                 (vd pose 33*3=99, 1 tay 21*3=63, lips 40*3=120, ...).
        visible: (T,) bool hoặc None — True nếu frame đó có đo đạc hợp lệ.
                 Frame KHÔNG hợp lệ -> chỉ PREDICT (không UPDATE) -> state
                 trôi theo vận tốc ước lượng gần nhất thay vì bị kéo về 0.
                 None = coi tất cả frame đều hợp lệ.

        Trả về: (T, D) đã làm mượt, cùng shape với input.
        """
        T, D = series.shape
        if visible is None:
            visible = np.ones(T, dtype=bool)

        # state: (D, 2) — [position, velocity] cho mỗi chiều D
        # cov  : (D, 2, 2) — hiệp phương sai ước lượng cho mỗi chiều D
        state = np.zeros((D, 2), dtype=np.float64)
        state[:, 0] = series[0]
        cov = np.tile(np.eye(2), (D, 1, 1))

        F, H, Q, R = self.F, self.H, self.Q, self.R
        out = np.zeros((T, D), dtype=np.float64)
        out[0] = series[0]

        for t in range(1, T):
            # ---- Predict (vector hoá qua D chiều cùng lúc) ----
            state = state @ F.T                                   # (D, 2)
            cov = F @ cov @ F.T + Q                                # (D, 2, 2)

            if visible[t]:
                # ---- Update ----
                z = series[t]                                        # (D,)
                y = z - (state @ H.T).squeeze(-1)                      # innovation (D,)
                S = (H @ cov @ H.T).reshape(D) + R.reshape(())          # (D,)
                K = (cov @ H.T).squeeze(-1) / S[:, None]                 # Kalman gain (D, 2)

                state = state + K * y[:, None]
                I_KH = np.eye(2)[None, :, :] - K[:, :, None] @ H[None, :, :]
                cov = I_KH @ cov

            out[t] = state[:, 0]

        return out.astype(np.float32)


def _first_visible_frame(features):
    visible = ~np.all(features == 0.0, axis=-1)
    if not visible.any():
        return None, visible
    return int(np.argmax(visible)), visible


def smooth_landmarks(features, process_noise=1e-3, measurement_noise=1e-2,
                      treat_all_zero_as_missing=True):

    features = np.asarray(features, dtype=np.float32)
    T, D = features.shape

    if not treat_all_zero_as_missing:
        kf = KalmanFilter1D(process_noise, measurement_noise)
        return kf.smooth(features)

    first_visible, visible = _first_visible_frame(features)
    if first_visible is None:
        # Cả clip không có frame nào hợp lệ -> không có gì để filter
        return features.copy()

    kf = KalmanFilter1D(process_noise, measurement_noise)

    # Chỉ filter từ frame visible ĐẦU TIÊN trở đi — các frame thiếu ở đầu
    # clip (trước khi landmark xuất hiện lần đầu) giữ nguyên = 0, không có
    # cơ sở nào để "đoán" chúng.
    sub = features[first_visible:]
    sub_visible = visible[first_visible:]
    smoothed_sub = kf.smooth(sub, visible=sub_visible)

    out = features.copy()
    out[first_visible:] = smoothed_sub
    return out


def smooth_for_fusion(pose_feature, hand_feature, lips_feature=None,
                       process_noise=1e-3, measurement_noise=1e-2):

    pose_smooth = smooth_landmarks(pose_feature, process_noise, measurement_noise)

    left = np.asarray(hand_feature)[:, :63]
    right = np.asarray(hand_feature)[:, 63:]
    left_smooth = smooth_landmarks(left, process_noise, measurement_noise)
    right_smooth = smooth_landmarks(right, process_noise, measurement_noise)
    hand_smooth = np.concatenate([left_smooth, right_smooth], axis=-1)

    lips_smooth = None
    if lips_feature is not None:
        lips_smooth = smooth_landmarks(lips_feature, process_noise, measurement_noise)

    return pose_smooth, hand_smooth, lips_smooth


def calculate_velocity(points, fps):
    dt = 1.0 / fps

    velocity = np.zeros_like(points, dtype=np.float32)

    velocity[1:] = (
        points[1:] - points[:-1]
    ) / dt

    return velocity

def interpolate_missing_points(features, mask):

    features = np.asarray(features, dtype=np.float32).copy()

    mask = np.asarray(mask, dtype=bool)

    T, N, D = features.shape

    if mask.ndim == 1:
        mask = np.tile(mask[:, None], (1, N))

    for n in range(N):

        valid_frames = np.where(mask[:, n] == True)[0]
        if len(valid_frames) == 0:
            continue

        for t in range(T):
            if mask[t, n]:
                continue

            previous = valid_frames[valid_frames < t]
            next_frames = valid_frames[valid_frames > t]

            if len(previous) > 0 and len(next_frames) > 0:

                t_prev = previous[-1]
                t_next = next_frames[0]

                alpha = (t - t_prev) / (t_next - t_prev)

                features[t, n] = (
                    features[t_prev, n]
                    + alpha
                    * (
                        features[t_next, n]
                        - features[t_prev, n]
                    )
                )

            # --------------------------------
            # Missing at the beginning
            # --------------------------------

            # elif len(next_frames) > 0:
            #
            #     t_next = next_frames[0]
            #
            #     features[t, n] = features[t_next, n]
            #
            # # --------------------------------
            # # Missing at the end
            # # --------------------------------
            #
            # elif len(previous) > 0:
            #
            #     t_prev = previous[-1]
            #
            #     features[t, n] = features[t_prev, n]

    return features

def restore_missing_points(
    pose_feature,
    left_feature,
    right_feature,
    fps=30,
    method='linear',
    model_path=None
):

    pose = np.asarray(
        pose_feature,
        dtype=np.float32
    ).reshape(-1, 5, 3)

    left = np.asarray(
        left_feature,
        dtype=np.float32
    ).reshape(-1, 21, 3)

    right = np.asarray(
        right_feature,
        dtype=np.float32
    ).reshape(-1, 21, 3)

    T = left.shape[0]

    left_mask = ~np.all(left.reshape(T, 21, 3) == 0, axis=-1)
    right_mask = ~np.all(right.reshape(T, 21, 3) == 0, axis=-1)
    pose_mask = ~np.all(pose.reshape(T, 5, 3) == 0, axis=-1)

    if method == 'linear':
        pose = interpolate_missing_points(pose, pose_mask)
        left = interpolate_missing_points(left, left_mask)
        right = interpolate_missing_points(right, right_mask)
    elif method == 'kalman':
        T = pose.shape[0]
        pose_flat = pose.reshape(T, -1)
        left_flat = left.reshape(T, -1)
        right_flat = right.reshape(T, -1)
        
        pose_smooth = smooth_landmarks(pose_flat)
        left_smooth = smooth_landmarks(left_flat)
        right_smooth = smooth_landmarks(right_flat)
        
        pose = pose_smooth.reshape(-1, 5, 3)
        left = left_smooth.reshape(-1, 21, 3)
        right = right_smooth.reshape(-1, 21, 3)
    elif method == 'model':
        import torch
        from src.models.motion_model import KeypointMotionTransformer

        if model_path is None:
            raise ValueError("model_path must be provided when method='model'")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(model_path, map_location=device)
        model_kwargs = ckpt["model_kwargs"]
        
        model = KeypointMotionTransformer(**model_kwargs).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        coord_dim = model_kwargs.get("pose_dim", 10) // 5
        
        pose_in = pose[:, :, :coord_dim].reshape(T, -1)
        left_in = left[:, :, :coord_dim].reshape(T, -1)
        right_in = right[:, :, :coord_dim].reshape(T, -1)

        coords_input = np.concatenate([pose_in, left_in, right_in], axis=-1)
        
        mask_flags = np.zeros((T, 3), dtype=np.float32)
        mask_flags[:, 0] = (~pose_mask).astype(np.float32)
        mask_flags[:, 1] = (~left_mask).astype(np.float32)
        mask_flags[:, 2] = (~right_mask).astype(np.float32)
        
        coords_input_t = torch.tensor(coords_input, dtype=torch.float32).unsqueeze(0).to(device)
        mask_flags_t = torch.tensor(mask_flags, dtype=torch.float32).unsqueeze(0).to(device)
        
        max_len = model.pos_encoder.pe.shape[1]
        
        with torch.no_grad():
            restored_chunks = []
            for i in range(0, T, max_len):
                c_chunk = coords_input_t[:, i:i+max_len]
                m_chunk = mask_flags_t[:, i:i+max_len]
                restored_chunk = model.restore(c_chunk, m_chunk)
                restored_chunks.append(restored_chunk)
            restored_coords_t = torch.cat(restored_chunks, dim=1)
            
        restored_coords = restored_coords_t.squeeze(0).cpu().numpy()
        
        pose_out = restored_coords[:, :model_kwargs["pose_dim"]].reshape(T, 5, coord_dim)
        left_out = restored_coords[:, model_kwargs["pose_dim"]:model_kwargs["pose_dim"]+model_kwargs["left_dim"]].reshape(T, 21, coord_dim)
        right_out = restored_coords[:, model_kwargs["pose_dim"]+model_kwargs["left_dim"]:].reshape(T, 21, coord_dim)
        
        pose[~pose_mask, :, :coord_dim] = pose_out[~pose_mask]
        left[~left_mask, :, :coord_dim] = left_out[~left_mask]
        right[~right_mask, :, :coord_dim] = right_out[~right_mask]
    else:
        raise ValueError("method must be either 'linear', 'kalman', or 'model'")

    return (
        pose,
        left,
        right
    )