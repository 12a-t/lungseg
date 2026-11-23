# -*- coding: utf-8 -*-
"""DSSW-Net known-location ROI inference for the unchanged original model.py.

Preprocessing saves a 96^3 nodule ROI, while DSSW-Net consumes/outputs 64^3.
center64 evaluates a central 64^3 patch (default); sliding64 stitches 64^3
predictions over the complete 96^3 ROI. Both are GT-free and distinguishable
protocols. The submitted paper does not uniquely specify which inference crop
was used; these outputs must not be claimed as original published numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi

# ================= PyCharm 配置（只需在这里改路径）=================
PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT_DEFAULT = Path(__import__('os').getenv('DSSWNET_DATA_ROOT', str(PROJECT_DIR / 'LIDC')))
CHECKPOINT_CANDIDATES = [
    PROJECT_DIR / "best_model.pth",
    PROJECT_DIR / "checkpoints" / "best_model.pth",
]
OUTPUT_DIR_DEFAULT = PROJECT_DIR / "results_inference"
SAVED_ROI_SIZE = 96        # 预处理生成的原始结节ROI
MODEL_PATCH_SIZE = 64      # 原train.py初始化的Swin输入，也是预测Patch尺寸
FEATURE_SIZE = 96         # 网络通道宽度！不是空间尺寸
INFERENCE_MODE = "center64"  # center64 / sliding64
OVERLAP = 0.5            # 仅sliding64使用
THRESHOLD = 0.5
MIN_COMPONENT_SIZE = 50  # 论文称50；原train.py验证阶段没有该后处理，结果可能不同
# ===============================================================


def parse_args():
    p = argparse.ArgumentParser(description="DSSW-Net 96 ROI -> 64 patch prediction")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--img-dir", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR_DEFAULT)
    p.add_argument("--subset", choices=("auto", "train", "val", "test"), default="auto")
    p.add_argument("--file-column", default="FileName")
    p.add_argument("--subset-column", default="subset")
    p.add_argument("--class-column", default="class_target")
    p.add_argument("--positive-class", default="1")
    p.add_argument("--roi-size", type=int, default=SAVED_ROI_SIZE,
                   help="Expected size of saved ROI (96); not model spatial input")
    p.add_argument("--patch-size", type=int, default=MODEL_PATCH_SIZE)
    p.add_argument("--feature-size", type=int, default=FEATURE_SIZE)
    p.add_argument("--mode", choices=("center64", "sliding64"), default=INFERENCE_MODE)
    p.add_argument("--overlap", type=float, default=OVERLAP)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--min-component-size", type=int, default=MIN_COMPONENT_SIZE)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--save-prob", action="store_true")
    p.add_argument("--allow-unsafe-pickle", action="store_true")
    return p.parse_args()


def find_checkpoint():
    for candidate in CHECKPOINT_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("未找到 best_model.pth；请放在项目根目录，或使用 --checkpoint 指定真实权重。")


def remove_small_components(mask, min_size):
    mask = np.asarray(mask, dtype=bool)
    if min_size <= 1:
        return mask
    labels, n = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def choose_center_start(shape, patch_size):
    """Use known ROI center, never GT information, when choosing inference patch."""
    if len(shape) != 3 or any(dim < patch_size for dim in shape):
        raise ValueError(f"ROI {shape} must be at least {patch_size} voxels per axis")
    return tuple((int(dim) - patch_size) // 2 for dim in shape)


def crop_at(array, start, patch_size):
    return array[tuple(slice(s, s + patch_size) for s in start)]


def window_positions(length, patch_size, overlap):
    if length < patch_size:
        raise ValueError(f"ROI length {length} < model patch {patch_size}")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    step = max(1, round(patch_size * (1 - overlap)))
    starts = list(range(0, length - patch_size + 1, step))
    if starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def sliding_probability(volume, patch_size, overlap, predict_patch):
    """Blend 64^3 probabilities into the *original* ROI size, without GT access."""
    if volume.ndim != 3 or any(s < patch_size for s in volume.shape):
        raise ValueError(f"Expected ROI >= {patch_size} in all axes, got {volume.shape}")
    axis = np.linspace(-1, 1, patch_size, dtype=np.float32)
    w1 = np.exp(-0.5 * (axis / 0.5) ** 2).astype(np.float32)
    weight = np.maximum(w1[:, None, None] * w1[None, :, None] * w1[None, None, :], 1e-4)
    acc = np.zeros(volume.shape, dtype=np.float32)
    norm = np.zeros(volume.shape, dtype=np.float32)
    for z in window_positions(volume.shape[0], patch_size, overlap):
        for y in window_positions(volume.shape[1], patch_size, overlap):
            for x in window_positions(volume.shape[2], patch_size, overlap):
                start = (z, y, x)
                block = crop_at(volume, start, patch_size)
                prob = np.asarray(predict_patch(block), dtype=np.float32)
                if prob.shape != block.shape or not np.isfinite(prob).all():
                    raise ValueError(f"Invalid model output: {prob.shape}, expected {block.shape}")
                slices = tuple(slice(s, s + patch_size) for s in start)
                acc[slices] += prob * weight
                norm[slices] += weight
    if np.any(norm <= 0):
        raise RuntimeError("sliding window failed to cover the complete ROI")
    return acc / norm


def load_trained_model(path, device, patch_size=MODEL_PATCH_SIZE, feature_size=FEATURE_SIZE,
                       allow_unsafe_pickle=False):
    from model import SwinUNETR_Optimized
    model = SwinUNETR_Optimized(
        img_size=(patch_size,) * 3,
        in_channels=1, out_channels=1,
        feature_size=feature_size, use_checkpoint=False,
    )
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_unsafe_pickle:
            raise RuntimeError("安全加载权重失败；仅对可信的自有权重使用 --allow-unsafe-pickle。") from exc
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(state.get(key), dict):
                state = state[key]
                break
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint没有有效state_dict")
    state = {str(k).removeprefix("module."): v for k, v in state.items()}
    config_file = Path(path).parent / 'training_config.json'
    if config_file.is_file():
        recorded = json.loads(config_file.read_text(encoding='utf-8'))
        if recorded.get('architecture') != 'original_model_unchanged':
            raise ValueError('The checkpoint training_config.json indicates a different architecture. '
                             'Use a checkpoint trained from the unchanged model.py.')
        recorded_sha = recorded.get('model_py_sha256')
        current_sha = hashlib.sha256((PROJECT_DIR / 'model.py').read_bytes()).hexdigest()
        if recorded_sha and recorded_sha != current_sha:
            raise ValueError('The checkpoint was trained with a different model.py SHA256. '
                             'Strict loading may fail or silently change model semantics.')
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def select_rows(csv_path, args):
    df = pd.read_csv(csv_path, dtype=str)
    for col in (args.file_column, args.subset_column):
        if col not in df.columns:
            raise KeyError(f"CSV缺少{col!r}列；不能将未划分数据冒充测试集。")
    choices = set(df[args.subset_column].dropna().str.lower())
    subset = args.subset
    if subset == "auto":
        if "test" in choices:
            subset = "test"
        elif "val" in choices:
            subset = "val"
            print("[警告] 未找到test分组，回退val；不能作为论文独立测试结果。")
        else:
            raise ValueError("CSV没有test或val分组")
    df = df[df[args.subset_column].astype(str).str.lower() == subset].copy()
    if args.class_column in df:
        numeric = pd.to_numeric(df[args.class_column], errors="coerce")
        try:
            df = df[numeric == float(args.positive_class)].copy()
        except ValueError:
            df = df[df[args.class_column].astype(str) == args.positive_class].copy()
    if df.empty:
        raise ValueError(f"subset={subset}没有符合要求的结节")
    fnames = df[args.file_column]
    if fnames.isna().any() or fnames.duplicated().any() or any(Path(x).name != x for x in fnames):
        raise ValueError("FileName需要非空、不重复且仅含文件名")
    return subset, df.reset_index(drop=True)


def main():
    args = parse_args()
    if args.roi_size < args.patch_size or args.patch_size <= 0 or args.patch_size % 32:
        raise ValueError("模型patch需要是32的倍数（原训练=64），并且不能大于保存ROI")
    if not 0 < args.threshold < 1 or args.min_component_size < 0:
        raise ValueError("threshold/min-component-size配置错误")
    if not 0 <= args.overlap < 1:
        raise ValueError("overlap必须在[0,1)范围内")
    csv_path = args.csv or args.data_root / "detailed_dataset_metadata.csv"
    img_dir = args.img_dir or args.data_root / "lung"
    checkpoint = args.checkpoint or find_checkpoint()
    if not csv_path.is_file() or not img_dir.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError(f"检查CSV/肺CT目录/权重文件：{csv_path} / {img_dir} / {checkpoint}")
    subset, rows = select_rows(csv_path, args)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = load_trained_model(checkpoint, device, args.patch_size,
                               args.feature_size, args.allow_unsafe_pickle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    prob_dir = args.output_dir / "probabilities"
    if args.save_prob:
        prob_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []
    print(f"[Inference] subset={subset}, N={len(rows)}, {args.mode}: ROI={args.roi_size}^3, "
          f"model patch={args.patch_size}^3, feature_size={args.feature_size}")
    print("[提醒] center64不使用GT，和dataset.py验证阶段的裁剪须核对；"
          "sliding64与center64为不同推理协议。不能直接宣称复现论文结果。")
    with torch.inference_mode():
        for index, (_, row) in enumerate(rows.iterrows(), start=1):
            fname = str(row[args.file_column])
            path = img_dir / fname
            if path.suffix.lower() != ".npy" or not path.is_file():
                raise FileNotFoundError(path)
            volume = np.load(path, allow_pickle=False)
            if volume.ndim != 3 or volume.shape != (args.roi_size,) * 3:
                raise ValueError(f"{fname}: 预处理ROI应为{args.roi_size}^3，实际{volume.shape}")
            if not np.isfinite(volume).all():
                raise ValueError(f"{fname}含NaN或Inf")
            volume = np.ascontiguousarray(volume, dtype=np.float32)
            if volume.min() < -0.01 or volume.max() > 1.01:
                print(f"[警告] {fname}: 归一化结果超出[0,1]，请核查预处理")

            def predict_patch(block):
                x = torch.from_numpy(np.ascontiguousarray(block)[None, None]).to(device, dtype=torch.float32)
                logits = model(x)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                expected = (1, 1, args.patch_size, args.patch_size, args.patch_size)
                if tuple(logits.shape) != expected:
                    raise RuntimeError(f"模型输出{tuple(logits.shape)}；预期{expected}")
                return torch.sigmoid(logits[0, 0]).float().cpu().numpy()

            if args.mode == 'center64':
                start = choose_center_start(volume.shape, args.patch_size)
                probability = predict_patch(crop_at(volume, start, args.patch_size))
            else:
                start = (0, 0, 0)
                probability = sliding_probability(volume, args.patch_size, args.overlap, predict_patch)
            pred = remove_small_components(probability >= args.threshold,
                                           args.min_component_size).astype(np.uint8)
            pred_path = pred_dir / fname
            np.save(pred_path, pred)
            if args.save_prob:
                np.save(prob_dir / fname, probability.astype(np.float32))
            record = row.to_dict()
            record.update({
                "FileName": fname, "pred_path": str(pred_path.resolve()),
                "crop_mode": args.mode, "crop_start_z": int(start[0]),
                "crop_start_y": int(start[1]), "crop_start_x": int(start[2]),
                "crop_size": args.patch_size if args.mode != "sliding64" else args.roi_size,
                "original_depth": volume.shape[0], "original_height": volume.shape[1],
                "original_width": volume.shape[2],
                "predicted_voxels": int(pred.sum()),
            })
            rows_out.append(record)
            if index == 1 or index % 25 == 0 or index == len(rows):
                print(f"[{index}/{len(rows)}] {fname}: output={pred.shape}, foreground={int(pred.sum())}")
    manifest_path = args.output_dir / "inference_manifest.csv"
    pd.DataFrame(rows_out).to_csv(manifest_path, index=False, encoding="utf-8-sig")
    config = {
        "checkpoint": str(checkpoint.resolve()), "subset": subset,
        "n_nodules": len(rows_out), "saved_roi_size": args.roi_size,
        "model_patch_size": args.patch_size, "feature_size": args.feature_size,
        "architecture": "original_model_unchanged",
        "model_py_sha256": hashlib.sha256((PROJECT_DIR / "model.py").read_bytes()).hexdigest(),
        "crop_mode": args.mode, "overlap": args.overlap if args.mode == "sliding64" else None,
        "threshold": args.threshold, "min_component_size": args.min_component_size,
        "uses_gt_in_inference": False,
        "protocol_note": "Paper specifies 96 ROI and 64 training patch, not exact testing crop; center64/sliding64 are distinct and cannot automatically reproduce reported figures.",
    }
    (args.output_dir / "inference_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[完成] 预测文件：", pred_dir)
    print("[完成] 推理清单：", manifest_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"INFERENCE FAILED: {exc}", file=sys.stderr)
        raise
