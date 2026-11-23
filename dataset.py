# dataset.py

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage
from pathlib import Path

# ================= 基本尺寸配置 =================
INPUT_SIZE = 64  # 模型最终输入 64³
SAVED_SIZE = 96  # 预先裁好的大 ROI 96³

# 偏移相关
RANDOM_SHIFT_PROB = 0.3  # 在已有结节ROI内进行随机裁剪
MAX_SHIFT_PIXELS = 10  # 最大偏移量

# 尺度增强相关：30%训练Patch随机缩放0.85-1.15（论文未规定增强概率）
RESIZE_PROB = 0.3  # 有 30% 的训练样本做尺度增强



class LIDCDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir, subset='train',
                 input_size=INPUT_SIZE, saved_size=SAVED_SIZE,
                 shift_prob=RANDOM_SHIFT_PROB, max_shift=MAX_SHIFT_PIXELS,
                 resize_prob=RESIZE_PROB,
                 ):

        if not 0 < input_size <= saved_size:
            raise ValueError('input_size must be <= saved_size')

        # 1) 读取 CSV，只保留当前 subset 的正样本（class_target == 1）
        # 注意：如果你做了CSV清洗，这里会自动读取清洗后的数据，只要路径对
        self.data_frame = pd.read_csv(csv_file)
        self.data_frame = self.data_frame[
            (self.data_frame['subset'] == subset) &
            (self.data_frame['class_target'] == 1)
            ].reset_index(drop=True)

        print(f"[LIDCDataset] subset={subset}, 正样本数量={len(self.data_frame)}")

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.saved_size = saved_size

        self.total_buffer = saved_size - input_size
        # 下面这个变量在旧逻辑里用，新逻辑里主要靠动态计算
        self.center_start_idx = self.total_buffer // 2

        self.subset = subset
        self.shift_prob = shift_prob
        self.max_shift = max_shift

        # 尺度增强配置
        self.resize_prob = resize_prob

    def __len__(self):
        return len(self.data_frame)

    # ---------- 工具函数：3D mask 的包围盒 ----------
    @staticmethod
    def _get_mask_bbox(mask: np.ndarray):
        coords = np.where(mask > 0.5)
        if coords[0].size == 0:
            return None
        z_min, z_max = coords[0].min(), coords[0].max()
        y_min, y_max = coords[1].min(), coords[1].max()
        x_min, x_max = coords[2].min(), coords[2].max()
        return z_min, z_max, y_min, y_max, x_min, x_max

    # ---------- 工具函数：智能偏移（保证包围盒完整包含在 patch 内） ----------
    def _smart_random_start_1d(self, center_start, total_buffer, max_shift,
                               bb_min, bb_max, input_size):
        """
        1D 上选择裁剪起点：
        - 保证 [start, start+input_size) 完整覆盖 [bb_min, bb_max]
        - 同时尽量围绕 center_start ± max_shift 随机
        """
        # 完整覆盖 mask 的合法 start 范围
        min_start_for_mask = max(0, bb_max - input_size + 1)
        max_start_for_mask = min(total_buffer, bb_min)

        if min_start_for_mask > max_start_for_mask:
            # 极端情况 fallback：中心附近随机
            low = max(0, center_start - max_shift)
            high = min(total_buffer, center_start + max_shift)
            if low > high:
                return max(0, min(center_start, total_buffer))
            return random.randint(low, high)

        # 与 center±max_shift 取交集
        low = max(min_start_for_mask, center_start - max_shift)
        high = min(max_start_for_mask, center_start + max_shift)

        if low > high:
            # 没有交集，就在完全覆盖结节的范围里随机
            low, high = min_start_for_mask, max_start_for_mask

        if low > high:
            # 再极端 fallback
            return max(0, min(center_start, total_buffer))

        return random.randint(int(low), int(high))

    # ---------- 工具函数：把 zoom 后的体积居中裁剪/填充到 target_shape ----------
    @staticmethod
    def _fit_to_shape(volume: np.ndarray, target_shape):
        Dz, Dy, Dx = target_shape
        z, y, x = volume.shape

        out = np.zeros((Dz, Dy, Dx), dtype=volume.dtype)

        def _compute_slice(src_len, dst_len):
            if src_len >= dst_len:
                # 从 src 中心裁剪 dst_len
                start_src = (src_len - dst_len) // 2
                end_src = start_src + dst_len
                start_dst = 0
                end_dst = dst_len
            else:
                # 把 src 放到 dst 的中心
                start_src = 0
                end_src = src_len
                start_dst = (dst_len - src_len) // 2
                end_dst = start_dst + src_len
            return start_src, end_src, start_dst, end_dst

        zs, ze, zds, zde = _compute_slice(z, Dz)
        ys, ye, yds, yde = _compute_slice(y, Dy)
        xs, xe, xds, xde = _compute_slice(x, Dx)

        out[zds:zde, yds:yde, xds:xde] = volume[zs:ze, ys:ye, xs:xe]
        return out

    # ---------- 工具函数：训练Patch随机缩放（实际scale为0.85--1.15） ----------
    def _scale_patch(self, img: np.ndarray, mask: np.ndarray, scale: float):
        """
        img, mask: [D, H, W]
        scale: >1 放大，<1 缩小
        最后输出仍为 [D, H, W]，通过中心裁剪/填充实现
        """
        D, H, W = img.shape
        new_D = max(1, int(round(D * scale)))
        new_H = max(1, int(round(H * scale)))
        new_W = max(1, int(round(W * scale)))

        zoom_factors = (new_D / D, new_H / H, new_W / W)

        # img 用线性插值，mask 用最近邻插值
        img_zoom = scipy.ndimage.zoom(img, zoom_factors, order=1)
        mask_zoom = scipy.ndimage.zoom(mask, zoom_factors, order=0)

        img_out = self._fit_to_shape(img_zoom, (D, H, W))
        mask_out = self._fit_to_shape(mask_zoom, (D, H, W))

        return img_out, mask_out

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]

        # ⚠️ 请确认你的 CSV 列名是 "FileName" 还是 "filename"
        filename = row["FileName"]

        img_path = os.path.join(self.img_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        try:
            img = np.load(img_path).astype(np.float32)
            mask = np.load(mask_path).astype(np.float32)
        except Exception as e:
            print(f"[LIDCDataset] 加载失败：")
            print(f"  img_path : {img_path}")
            print(f"  mask_path: {mask_path}")
            print(f"  error    : {e}")
            raise e

        if img.shape != (self.saved_size,) * 3 or mask.shape != img.shape:
            raise ValueError(f"{filename}: expected 3D ROI {(self.saved_size,) * 3}, got {img.shape}/{mask.shape}")
        if not np.isfinite(img).all():
            raise ValueError(f'{filename}: invalid NaN/Inf in image')

        # =========================================================
        # 【关键修改 Step 1】计算 Mask 重心，作为初始裁剪中心
        # =========================================================
        cz, cy, cx = np.where(mask > 0.5)

        if len(cz) > 0:
            # 以结节重心为锚点
            center_z = int(np.mean(cz))
            center_y = int(np.mean(cy))
            center_x = int(np.mean(cx))
        else:
            # 负样本回退到图片中心
            center_z = img.shape[0] // 2
            center_y = img.shape[1] // 2
            center_x = img.shape[2] // 2

        # 计算"理想的"左上角起点 (不考虑偏移)
        # 这里的 z_base 就是"结节在中间"时的切片起点
        half_size = self.input_size // 2

        # 限制范围，防止越界
        z_base = np.clip(center_z - half_size, 0, img.shape[0] - self.input_size)
        y_base = np.clip(center_y - half_size, 0, img.shape[1] - self.input_size)
        x_base = np.clip(center_x - half_size, 0, img.shape[2] - self.input_size)

        # 验证/测试在已知结节ROI几何中心裁剪，和inference.py的center64对齐。
        # 不依赖真实Mask来决定测试时的空间位置。
        if self.subset != 'train':
            z_base = (img.shape[0] - self.input_size) // 2
            y_base = (img.shape[1] - self.input_size) // 2
            x_base = (img.shape[2] - self.input_size) // 2

        # 初始化最终起点为 base
        z_start, y_start, x_start = z_base, y_base, x_base

        # =========================================================
        # 🔥【关键修改 Step 2】让随机偏移围绕"z_base"(结节)进行，而不是图片中心
        # =========================================================
        if self.subset == 'train':
            if random.random() < self.shift_prob:
                bbox = self._get_mask_bbox(mask)
                if bbox is not None:
                    z_min, z_max, y_min, y_max, x_min, x_max = bbox

                    # ⚠️ 注意：这里传入的是 z_base, y_base, x_base 作为参考中心
                    z_start = self._smart_random_start_1d(
                        z_base, self.total_buffer, self.max_shift,
                        z_min, z_max, self.input_size
                    )
                    y_start = self._smart_random_start_1d(
                        y_base, self.total_buffer, self.max_shift,
                        y_min, y_max, self.input_size
                    )
                    x_start = self._smart_random_start_1d(
                        x_base, self.total_buffer, self.max_shift,
                        x_min, x_max, self.input_size
                    )

        # 3) 执行裁剪
        z_end = z_start + self.input_size
        y_end = y_start + self.input_size
        x_end = x_start + self.input_size

        img_crop = img[z_start:z_end, y_start:y_end, x_start:x_end]
        mask_crop = mask[z_start:z_end, y_start:y_end, x_start:x_end]

        # 4) Methods: random scaling 0.85--1.15.
        if self.subset == 'train' and random.random() < self.resize_prob:
            scale = random.uniform(0.85, 1.15)
            img_crop, mask_crop = self._scale_patch(img_crop, mask_crop, scale)

        # 5) 训练集：水平/垂直翻转与随机连续角度旋转（无额外噪声）
        if self.subset == 'train':
            # 随机翻转
            if random.random() < 0.5:
                if random.random() < 0.5:
                    img_crop = np.flip(img_crop, axis=2).copy()
                    mask_crop = np.flip(mask_crop, axis=2).copy()
                if random.random() < 0.5:
                    img_crop = np.flip(img_crop, axis=1).copy()
                    mask_crop = np.flip(mask_crop, axis=1).copy()

            if random.random() < 0.5:
                angle = random.uniform(-15.0, 15.0)
                axes = random.choice(((1, 2), (0, 2), (0, 1)))
                img_crop = scipy.ndimage.rotate(img_crop, angle, axes=axes,
                                                reshape=False, order=1,
                                                mode='constant', cval=0.0)
                mask_crop = scipy.ndimage.rotate(mask_crop, angle, axes=axes,
                                                 reshape=False, order=0,
                                                 mode='constant', cval=0.0)

        # [D, H, W] -> [1, D, H, W]
        img_crop = np.ascontiguousarray(img_crop[None], dtype=np.float32)
        mask_crop = np.ascontiguousarray(mask_crop[None] > 0.5, dtype=np.float32)
        return torch.from_numpy(img_crop), torch.from_numpy(mask_crop)


# ================= DataLoader 封装 =================
def load_data(csv_path, img_dir, mask_dir,
              input_size=INPUT_SIZE, saved_size=SAVED_SIZE,
              batch_size=2, shift_prob=RANDOM_SHIFT_PROB,
              num_workers=4):
    if not os.path.exists(csv_path):
        print(f"[load_data] CSV 不存在: {csv_path}")
        return None, None

    train_ds = LIDCDataset(csv_path, img_dir, mask_dir, subset='train',
                           input_size=input_size, saved_size=saved_size,
                           shift_prob=shift_prob)
    # 验证集不开启随机偏移，按ROI几何中心取64³，与默认推理协议一致
    val_ds = LIDCDataset(csv_path, img_dir, mask_dir, subset='val',
                         input_size=input_size, saved_size=saved_size,
                         shift_prob=0.0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,  # Keep ALL annotated training nodules; see sample-weighted accumulation in train.py.
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"✅ 数据加载成功。Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader