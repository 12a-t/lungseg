# dataset.py

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage
from scipy.ndimage import affine_transform

# ================= 基本尺寸配置 =================
INPUT_SIZE = 96  # 使用完整的96x96x96 ROI
SAVED_SIZE = 96

RANDOM_SHIFT_PROB = 0.3
MAX_SHIFT_PIXELS = 10

RESIZE_PROB = 0.3
RESIZE_MIN_SCALE = 0.85
RESIZE_MAX_SCALE = 1.15

ROTATION_PROB = 0.5
MAX_ANGLE = 15  # degrees

SMALL_NODULE_THRESH = 10
LARGE_NODULE_THRESH = 30


class LIDCDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir, subset='train',
                 input_size=INPUT_SIZE, saved_size=SAVED_SIZE,
                 shift_prob=RANDOM_SHIFT_PROB, max_shift=MAX_SHIFT_PIXELS,
                 resize_prob=RESIZE_PROB, resize_min_scale=RESIZE_MIN_SCALE, resize_max_scale=RESIZE_MAX_SCALE,
                 rotation_prob=ROTATION_PROB, max_angle=MAX_ANGLE,
                 small_thr=SMALL_NODULE_THRESH, large_thr=LARGE_NODULE_THRESH):

        warnings.filterwarnings("ignore", category=UserWarning)

        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame[
            (self.data_frame['subset'] == subset) &
            (self.data_frame['class_target'] == 1)
        ].reset_index(drop=True)

        print(f"[LIDCDataset] subset={subset}, positive samples={len(self.data_frame)}")

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.saved_size = saved_size
        self.subset = subset
        self.shift_prob = shift_prob
        self.max_shift = max_shift
        self.resize_prob = resize_prob
        self.resize_min_scale = resize_min_scale
        self.resize_max_scale = resize_max_scale
        self.rotation_prob = rotation_prob
        self.max_angle = max_angle
        self.small_thr = small_thr
        self.large_thr = large_thr

    def __len__(self):
        return len(self.data_frame)

    @staticmethod
    def _get_mask_bbox(mask: np.ndarray):
        coords = np.where(mask > 0.5)
        if coords[0].size == 0:
            return None
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        return z_min, z_max, y_min, y_max, x_min, x_max

    def _smart_random_start_1d(self, center_start, total_buffer, max_shift, bb_min, bb_max, input_size):
        min_start_for_mask = max(0, bb_max - input_size + 1)
        max_start_for_mask = min(total_buffer, bb_min)
        if min_start_for_mask > max_start_for_mask:
            low = max(0, center_start - max_shift)
            high = min(total_buffer, center_start + max_shift)
            if low > high:
                return max(0, min(center_start, total_buffer))
            return random.randint(low, high)
        low = max(min_start_for_mask, center_start - max_shift)
        high = min(max_start_for_mask, center_start + max_shift)
        if low > high:
            low, high = min_start_for_mask, max_start_for_mask
        if low > high:
            return max(0, min(center_start, total_buffer))
        return random.randint(int(low), int(high))

    @staticmethod
    def _fit_to_shape(volume: np.ndarray, target_shape):
        Dz, Dy, Dx = target_shape
        z, y, x = volume.shape
        out = np.zeros((Dz, Dy, Dx), dtype=volume.dtype)

        def _compute_slice(src_len, dst_len):
            if src_len >= dst_len:
                start_src = (src_len - dst_len) // 2
                return start_src, start_src + dst_len, 0, dst_len
            else:
                start_dst = (dst_len - src_len) // 2
                return 0, src_len, start_dst, start_dst + src_len

        zs, ze, zds, zde = _compute_slice(z, Dz)
        ys, ye, yds, yde = _compute_slice(y, Dy)
        xs, xe, xds, xde = _compute_slice(x, Dx)
        out[zds:zde, yds:yde, xds:xde] = volume[zs:ze, ys:ye, xs:xe]
        return out

    def _scale_patch(self, img: np.ndarray, mask: np.ndarray, scale: float):
        D, H, W = img.shape
        new_D = max(1, int(round(D * scale)))
        new_H = max(1, int(round(H * scale)))
        new_W = max(1, int(round(W * scale)))
        zoom_factors = (new_D / D, new_H / H, new_W / W)
        img_zoom = scipy.ndimage.zoom(img, zoom_factors, order=1)
        mask_zoom = scipy.ndimage.zoom(mask, zoom_factors, order=0)
        return self._fit_to_shape(img_zoom, (D, H, W)), self._fit_to_shape(mask_zoom, (D, H, W))

    def _random_rotation_3d(self, img: np.ndarray, mask: np.ndarray, max_angle: float):
        """Random 3D rotation within ±max_angle degrees."""
        angle = random.uniform(-max_angle, max_angle)
        rad = np.radians(angle)

        # Rotation matrix around Z-axis
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot_matrix = np.array([[cos_a, -sin_a, 0],
                               [sin_a, cos_a, 0],
                               [0, 0, 1]])

        # Apply rotation to image (order=1 for interpolation)
        img_rot = affine_transform(
            img, rot_matrix, order=1, mode='nearest'
        )
        # Apply rotation to mask (order=0 for nearest-neighbor to preserve binary values)
        mask_rot = affine_transform(
            mask, rot_matrix, order=0, mode='nearest'
        )

        return img_rot, mask_rot

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        filename = row["FileName"]

        img_path = os.path.join(self.img_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)

        cz, cy, cx = np.where(mask > 0.5)
        if len(cz) > 0:
            center_z, center_y, center_x = int(np.mean(cz)), int(np.mean(cy)), int(np.mean(cx))
        else:
            center_z, center_y, center_x = img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2

        half_size = self.input_size // 2
        z_base = np.clip(center_z - half_size, 0, img.shape[0] - self.input_size)
        y_base = np.clip(center_y - half_size, 0, img.shape[1] - self.input_size)
        x_base = np.clip(center_x - half_size, 0, img.shape[2] - self.input_size)

        z_start, y_start, x_start = z_base, y_base, x_base

        if self.subset == 'train' and random.random() < self.shift_prob:
            bbox = self._get_mask_bbox(mask)
            if bbox is not None:
                z_min, z_max, y_min, y_max, x_min, x_max = bbox
                z_start = self._smart_random_start_1d(z_base, self.total_buffer, self.max_shift, z_min, z_max, self.input_size)
                y_start = self._smart_random_start_1d(y_base, self.total_buffer, self.max_shift, y_min, y_max, self.input_size)
                x_start = self._smart_random_start_1d(x_base, self.total_buffer, self.max_shift, x_min, x_max, self.input_size)

        z_end, y_end, x_end = z_start + self.input_size, y_start + self.input_size, x_start + self.input_size
        img_crop = img[z_start:z_end, y_start:y_end, x_start:x_end]
        mask_crop = mask[z_start:z_end, y_start:y_end, x_start:x_end]

        # ---- Data Augmentation (training only) ----
        if self.subset == 'train':
            # 1. Random scaling (0.85 to 1.15)
            if random.random() < self.resize_prob:
                bbox_local = self._get_mask_bbox(mask_crop)
                if bbox_local is not None:
                    z_min, z_max, y_min, y_max, x_min, x_max = bbox_local
                    max_extent = max(z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1)
                    if max_extent <= self.small_thr:
                        scale = random.uniform(1.0, self.resize_max_scale)
                    elif max_extent >= self.large_thr:
                        scale = random.uniform(self.resize_min_scale, 1.0)
                    else:
                        scale = random.uniform(self.resize_min_scale, self.resize_max_scale)
                    if abs(scale - 1.0) > 1e-6:
                        img_crop, mask_crop = self._scale_patch(img_crop, mask_crop, scale)

            # 2. Random flipping (horizontal and vertical)
            if random.random() < 0.5:
                img_crop = np.flip(img_crop, axis=2).copy()
                mask_crop = np.flip(mask_crop, axis=2).copy()
            if random.random() < 0.5:
                img_crop = np.flip(img_crop, axis=1).copy()
                mask_crop = np.flip(mask_crop, axis=1).copy()

            # 3. Random 3D rotation (±15°)
            if random.random() < self.rotation_prob:
                img_crop, mask_crop = self._random_rotation_3d(img_crop, mask_crop, self.max_angle)

        # Normalize: clip to [-1000, 400] HU and map to [0, 1]
        MIN_HU, MAX_HU = -1000.0, 400.0
        img_norm = np.clip(img_crop, MIN_HU, MAX_HU)
        img_norm = (img_norm - MIN_HU) / (MAX_HU - MIN_HU)

        # Random Gaussian noise (training only)
        if self.subset == 'train' and random.random() < 0.2:
            noise = np.random.normal(0.0, 0.02, img_norm.shape)
            img_norm = np.clip(img_norm + noise, 0.0, 1.0)

        # Add channel dimension
        img_norm = np.expand_dims(img_norm, axis=0)   # [1, D, H, W]
        mask_crop = np.expand_dims(mask_crop, axis=0)  # [1, D, H, W]

        return torch.from_numpy(img_norm), torch.from_numpy(mask_crop)


def load_data(csv_path, img_dir, mask_dir,
              input_size=INPUT_SIZE, saved_size=SAVED_SIZE,
              batch_size=2, shift_prob=RANDOM_SHIFT_PROB,
              num_workers=4):
    if not os.path.exists(csv_path):
        return None, None

    train_ds = LIDCDataset(
        csv_path, img_dir, mask_dir, subset='train',
        input_size=input_size, saved_size=saved_size, shift_prob=shift_prob
    )
    val_ds = LIDCDataset(
        csv_path, img_dir, mask_dir, subset='val',
        input_size=input_size, saved_size=saved_size, shift_prob=0.0
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False
    )

    print(f"Data loaded. Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader