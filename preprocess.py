# -*- coding: utf-8 -*-
"""DSSW-Net / LIDC-IDRI 数据预处理（Windows + PyCharm 直接运行）。

原始 DICOM + pylidc 标注 -> 1 mm CT和共识Mask -> 已知结节位置96^3 ROI -> NPY
训练时dataset.py从96^3 ROI中进行结节约束随机裁剪得到64^3 patch；验证使用固定中心64^3。
模型输入和输出均为64^3。本预处理不会把96^3直接缩放到64^3。
筛选: CT层厚<=3 mm，至少3名医生，50%体素共识且有效Mask。
此脚本只进行筛选及预处理，不进行 train/val/test 划分。
划分是独立的实验设计步骤，应由实际使用的患者级划分记录或后续脚本完成。

只输出:
  OUTPUT_ROOT/lung/*.npy
  OUTPUT_ROOT/mask/*.npy
  OUTPUT_ROOT/preprocessed_nodules.csv  （原始预处理清单）
  OUTPUT_ROOT/detailed_dataset_metadata.csv（仅在SPLIT_MODE=reuse且提供原划分时输出）

注意：此版本不会人为固定最终结节数；新生成的Mask可能与历史训练数据不同。
"""

import os
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from tqdm import tqdm

# ==================== 配置区域：PyCharm可直接运行 ====================
PROJECT_DIR = Path(__file__).resolve().parent
DICOM_ROOT = Path(os.environ.get('LIDC_DICOM_ROOT', str(PROJECT_DIR / 'raw_LIDC-IDRI')))
# 输出文件夹名称与 train.py 对齐：LIDC/lung/ 与 LIDC/mask/。
# 独立保存在 DSSWNET 项目中，避免覆盖 C:\Users\dell\Desktop\th\3\LIDC 的历史训练数据。
OUTPUT_ROOT = Path(os.environ.get('LIDC_OUTPUT_ROOT', str(PROJECT_DIR / 'LIDC')))

MIN_ANNOTATORS = 3           # 至少3位放射科医生
CONSENSUS_LEVEL = 0.50      # 50%体素共识
MAX_SLICE_THICKNESS = 3.0  # mm
TARGET_SPACING = (1.0, 1.0, 1.0)  # (Z,Y,X), mm
ROI_SIZE = 96                # 保存的结节区域空间尺寸
TRAIN_PATCH_SIZE = 64        # 训练时dataset.py从96^3提取64^3；不在本脚本裁掉边缘
# 划分方式：none(仅预处理) / reuse(复用真实的患者级分层划分)
SPLIT_MODE = "none"
REFERENCE_SPLIT_CSV = None   # reuse时设置原detailed_dataset_metadata.csv路径
HU_MIN, HU_MAX = -1000.0, 400.0
LUNG_CLOSING_ITER = 8
LUNG_DILATION_ITER = 8
# 论文原文排除“直径<=3 mm且缺少体素轮廓”的样本。
# 本程序要求能生成非空共识Mask；不额外排除所有<=3 mm的有效标注结节。
# ================================================================
if TRAIN_PATCH_SIZE > ROI_SIZE or TRAIN_PATCH_SIZE % 32:
    raise ValueError("TRAIN_PATCH_SIZE必须不大于ROI_SIZE且适配原Swin模型(64^3)")

IMG_DIR = OUTPUT_ROOT / "lung"
MASK_DIR = OUTPUT_ROOT / "mask"


def setup_pylidc():
    if not DICOM_ROOT.is_dir():
        raise FileNotFoundError(f"未找到原始DICOM数据目录：{DICOM_ROOT}")
    # pylidc导入时会读取用户目录中的配置；须先写配置再导入。
    config = Path.home() / ("pylidc.conf" if os.name == "nt" else ".pylidcrc")
    expected = f"[dicom]\npath = {DICOM_ROOT}\nwarn = True\n"
    if not config.exists() or config.read_text(encoding="utf-8") != expected:
        if config.exists():
            backup = config.with_name(config.name + ".backup")
            if not backup.exists():
                backup.write_bytes(config.read_bytes())
        config.write_text(expected, encoding="utf-8")

    # 兼容部分较旧的pylidc版本使用的NumPy旧别名。
    for name, val in (("int", int), ("float", float), ("bool", np.bool_)):
        if name not in np.__dict__:
            setattr(np, name, val)

    import pylidc as pl
    from pylidc.utils import consensus
    from sqlalchemy.orm import object_session
    import SimpleITK as sitk
    from lungmask import mask as lungmask_lib
    return pl, consensus, object_session, sitk, lungmask_lib


def load_scan_in_hu(scan):
    """pylidc原始像素需要RescaleSlope/Intercept转换为HU。

    pylidc体数据顺序为[row, column, slice]，返回同一顺序；
    后续转换为SimpleITK的[Z, Y, X]。
    """
    images = scan.load_all_dicom_images(verbose=False)
    if len(images) < 2:
        raise ValueError("有效DICOM切片不足2张")
    rows, cols = images[0].pixel_array.shape
    volume = np.empty((rows, cols, len(images)), dtype=np.float32)
    for k, ds in enumerate(images):
        pixels = ds.pixel_array
        if pixels.shape != (rows, cols):
            raise ValueError("同一CT序列中图像尺寸不一致")
        if not hasattr(ds, "RescaleSlope") or not hasattr(ds, "RescaleIntercept"):
            raise ValueError("DICOM缺少RescaleSlope/RescaleIntercept，无法可靠换算HU")
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)
        if not np.isfinite(slope) or not np.isfinite(intercept):
            raise ValueError("无效的RescaleSlope/RescaleIntercept")
        volume[:, :, k] = pixels.astype(np.float32) * slope + intercept
    return volume


def scan_spacing_zyx(scan):
    ps = scan.pixel_spacing
    if isinstance(ps, str):
        values = [float(x) for x in ps.replace("\\", ",").split(",") if x.strip()]
    elif isinstance(ps, (list, tuple, np.ndarray)):
        values = list(np.asarray(ps, dtype=float).ravel())
    else:
        values = [float(ps)]
    row_spacing = float(values[0])
    col_spacing = float(values[1] if len(values) > 1 else values[0])
    z = np.asarray(scan.slice_zvals, dtype=float)
    delta = np.diff(z)
    if delta.size == 0 or np.any(delta <= 0):
        raise ValueError("切片坐标间距无效")
    z_spacing = float(np.median(delta))
    # 非均匀层间距不应当使用一个常量进行空间重采样。
    if np.max(np.abs(delta - z_spacing)) > 0.25:
        raise ValueError("CT切片间距不均匀，跳过该序列")
    spacing = (z_spacing, row_spacing, col_spacing)
    if not all(np.isfinite(x) and x > 0 for x in spacing):
        raise ValueError(f"体素间距无效：{spacing}")
    return spacing


def make_sitk_image(volume_ijk, spacing_zyx, sitk):
    image = sitk.GetImageFromArray(
        np.ascontiguousarray(np.transpose(volume_ijk, (2, 0, 1)), dtype=np.float32)
    )
    image.SetSpacing(tuple(spacing_zyx[::-1]))  # SITK使用(X,Y,Z)
    return image


def resample_ct(image, sitk):
    new_spacing_xyz = tuple(TARGET_SPACING[::-1])
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    new_size = [
        max(1, int(round((old_size[i] - 1) * old_spacing[i] / new_spacing_xyz[i])) + 1)
        for i in range(3)
    ]
    return sitk.Resample(
        image,
        new_size,
        sitk.Transform(),
        sitk.sitkLinear,
        image.GetOrigin(),
        new_spacing_xyz,
        image.GetDirection(),
        HU_MIN,
        sitk.sitkFloat32,
    )


def resample_mask(mask_ijk, original_img, reference_img, sitk):
    mask_img = sitk.GetImageFromArray(
        np.ascontiguousarray(np.transpose(mask_ijk, (2, 0, 1)), dtype=np.uint8)
    )
    mask_img.CopyInformation(original_img)
    aligned = sitk.Resample(
        mask_img, reference_img, sitk.Transform(),
        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
    )
    return sitk.GetArrayFromImage(aligned) > 0


def lung_mask_on_resampled_image(image, sitk, lungmask_lib):
    result = lungmask_lib.apply(image, batch_size=1, noip=True)
    binary = np.asarray(result) > 0
    if binary.shape != sitk.GetArrayFromImage(image).shape or not binary.any():
        raise ValueError("lungmask结果无效")
    structure = ndi.generate_binary_structure(3, 1)
    if LUNG_CLOSING_ITER:
        binary = ndi.binary_closing(binary, structure=structure,
                                    iterations=LUNG_CLOSING_ITER)
    if LUNG_DILATION_ITER:
        binary = ndi.binary_dilation(binary, structure=structure,
                                     iterations=LUNG_DILATION_ITER)
    return binary


def crop_center(arr, center_zyx, size=ROI_SIZE, fill_value=0):
    result = np.full((size, size, size), fill_value, dtype=arr.dtype)
    start = np.rint(center_zyx).astype(int) - size // 2
    from_slices = []
    into_slices = []
    for axis in range(3):
        a = max(0, int(start[axis]))
        b = min(arr.shape[axis], int(start[axis]) + size)
        if b <= a:
            raise ValueError("结节中心超出体数据范围")
        from_slices.append(slice(a, b))
        into_slices.append(slice(a - int(start[axis]), b - int(start[axis])))
    result[tuple(into_slices)] = arr[tuple(from_slices)]
    return result


def center_from_annotations(cluster, original_img, new_img):
    # pylidc Annotation.centroid为[row, column, slice]。
    centroid = np.mean([a.centroid for a in cluster], axis=0)
    point = original_img.TransformContinuousIndexToPhysicalPoint(
        (float(centroid[1]), float(centroid[0]), float(centroid[2]))
    )
    xyz = new_img.TransformPhysicalPointToContinuousIndex(point)
    return np.asarray(xyz[::-1], dtype=float)  # Z,Y,X


def equivalent_sphere_diameter(mask):
    volume_mm3 = int(np.count_nonzero(mask)) * float(np.prod(TARGET_SPACING))
    if volume_mm3 == 0:
        raise ValueError("空Mask无法计算等效球体直径")
    return float((6.0 * volume_mm3 / math.pi) ** (1.0 / 3.0))


def prepare_training_manifest(df: pd.DataFrame):
    """Explicitly handle patient-level split; never silently pretend to reproduce paper splits.

    reuse: match exact existing FileName/subset; fail on unmatched cases or patient leakage.
    none: only emit preprocessed_nodules.csv, since the historical split is unknown.
    """
    mode = str(SPLIT_MODE).strip().lower()
    if mode == "none":
        print("[数据划分] 未提供论文原始划分：只生成preprocessed_nodules.csv。")
        return
    out_path = OUTPUT_ROOT / "detailed_dataset_metadata.csv"
    if out_path.exists():
        raise FileExistsError(f"拒绝覆盖已有患者划分记录：{out_path}")
    manifest = df.copy()
    if mode == "reuse":
        if REFERENCE_SPLIT_CSV is None:
            raise ValueError("SPLIT_MODE=reuse必须设置REFERENCE_SPLIT_CSV")
        ref_file = Path(REFERENCE_SPLIT_CSV)
        if not ref_file.is_file():
            raise FileNotFoundError(ref_file)
        ref = pd.read_csv(ref_file, dtype=str)
        if "FileName" not in ref or "subset" not in ref:
            raise KeyError("参考CSV必须有FileName与subset列")
        if ref["FileName"].duplicated().any() or ref["FileName"].isna().any():
            raise ValueError("参考CSV的FileName不能为空或重复")
        # Preserve trusted labels for subgroup evaluation if they were present
        # in the original experiment; do not infer morphology from CT intensity.
        optional_labels = [c for c in ("morphology", "NoduleType", "nodule_type", "texture")
                           if c in ref.columns and c not in manifest.columns]
        ref = ref[["FileName", "subset", *optional_labels]].copy()
        manifest = manifest.merge(ref, on="FileName", how="left", validate="one_to_one")
        if manifest["subset"].isna().any():
            raise ValueError("部分新生成FileName不在旧划分里，不能凭空映射为论文测试集")
    else:
        raise ValueError(f"未知SPLIT_MODE: {SPLIT_MODE!r}")
    if not set(manifest["subset"]).issubset({"train", "val", "test"}):
        raise ValueError("subset只允许train/val/test")
    patient_groups = manifest.groupby("PatientID")["subset"].nunique()
    if (patient_groups > 1).any():
        raise ValueError("发现同一患者被分到多个子集！")
    manifest.to_csv(out_path, index=False, encoding="utf-8-sig")
    counts = manifest.groupby("subset").agg(n_nodules=("FileName", "size"), n_patients=("PatientID", "nunique"))
    print("[数据划分] 已写入：", out_path)
    print(counts.to_string())


def main():
    csv_file = OUTPUT_ROOT / "preprocessed_nodules.csv"
    if csv_file.exists():
        raise FileExistsError(
            f"预处理输出已存在：{csv_file}。为避免覆盖旧结果，请先另存备份或设置新输出目录。"
        )
    pl, consensus, object_session, sitk, lungmask_lib = setup_pylidc()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    scans = pl.query(pl.Scan).all()
    print(f"[开始] CT数={len(scans)}，标注医生>={MIN_ANNOTATORS}，共识={CONSENSUS_LEVEL}")
    print(f"[原始数据] {DICOM_ROOT}\n[输出目录] {OUTPUT_ROOT}")
    records = []
    passed_scans = 0
    skipped_nodules = 0
    skipped_scans = 0
    failure_records = []

    for scan in tqdm(scans, desc="Preprocessing LIDC-IDRI"):
        session = object_session(scan)
        try:
            if session is not None:
                session.autoflush = False
            thickness = scan.slice_thickness
            if thickness is None or not np.isfinite(float(thickness)) or float(thickness) > MAX_SLICE_THICKNESS:
                skipped_scans += 1
                continue
            clusters = scan.cluster_annotations(verbose=False)
            candidates = [(idx, annotations) for idx, annotations in enumerate(clusters)
                          if len(annotations) >= MIN_ANNOTATORS]
            if not candidates:
                skipped_scans += 1
                continue

            spacing = scan_spacing_zyx(scan)
            vol_ijk = load_scan_in_hu(scan)
            orig_img = make_sitk_image(vol_ijk, spacing, sitk)
            res_img = resample_ct(orig_img, sitk)
            res_array = sitk.GetArrayFromImage(res_img)
            lung_array = lung_mask_on_resampled_image(res_img, sitk, lungmask_lib)
            passed_scans += 1

            for idx, annotations in candidates:
                try:
                    cmask, bbox = consensus(
                        annotations, clevel=CONSENSUS_LEVEL,
                        pad=None, ret_masks=False, verbose=False
                    )
                    full_mask = np.zeros(vol_ijk.shape, dtype=np.uint8)
                    if full_mask[bbox].shape != cmask.shape:
                        raise ValueError("共识Mask的bbox尺寸不匹配")
                    full_mask[bbox] = cmask.astype(np.uint8)
                    mask_res = resample_mask(full_mask, orig_img, res_img, sitk)
                    if not mask_res.any():
                        raise ValueError("重采样后共识Mask为空")

                    center = center_from_annotations(annotations, orig_img, res_img)
                    img_crop = crop_center(res_array, center, fill_value=HU_MIN)
                    mask_crop = crop_center(mask_res, center, fill_value=False)
                    lung_crop = crop_center(lung_array, center, fill_value=False)
                    if img_crop.shape != (ROI_SIZE,) * 3 or mask_crop.shape != img_crop.shape:
                        raise ValueError("预处理ROI尺寸异常")
                    if not mask_crop.any() or np.count_nonzero(mask_crop) != np.count_nonzero(mask_res):
                        raise ValueError("96³ ROI未完整包含结节")

                    image = (np.clip(img_crop, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)
                    image *= np.logical_or(lung_crop, mask_crop)
                    diameter_mm = equivalent_sphere_diameter(mask_res)

                    file_stem = f"LIDC_{scan.patient_id}_Scan{scan.id}_Nodule{idx:03d}"
                    file_name = file_stem + ".npy"
                    # 不覆盖同名历史数据；避免意外改变已经训练的样本。
                    image_file = IMG_DIR / file_name
                    mask_file = MASK_DIR / file_name
                    if image_file.exists() or mask_file.exists():
                        raise FileExistsError(f"输出文件已存在：{file_name}；请使用空输出目录")
                    np.save(image_file, image.astype(np.float32))
                    np.save(mask_file, mask_crop.astype(np.uint8))
                    records.append({
                        "FileName": file_name,
                        "PatientID": str(scan.patient_id),
                        "ScanID": str(scan.id),
                        "class_target": 1,
                        "Diameter_mm": round(diameter_mm, 5),
                        "Annotators": len(annotations),
                        "ROI_size": ROI_SIZE,
                        "Train_patch_size": TRAIN_PATCH_SIZE,
                        "Spacing_mm": TARGET_SPACING[0],
                    })
                except Exception as exc:
                    skipped_nodules += 1
                    failure_records.append({"ScanID": str(scan.id),
                                            "PatientID": str(scan.patient_id),
                                            "NoduleIndex": int(idx), "Reason": str(exc)})
                    if skipped_nodules <= 3:
                        print(f"[提示] 有结节无法完成处理：{type(exc).__name__}: {exc}")
        except Exception as exc:
            skipped_scans += 1
            failure_records.append({"ScanID": str(scan.id),
                                    "PatientID": str(scan.patient_id),
                                    "NoduleIndex": None, "Reason": str(exc)})
            if skipped_scans <= 3:
                print(f"[提示] 有CT未完成处理：{type(exc).__name__}: {exc}")
        finally:
            if session is not None:
                session.rollback()

    if not records:
        raise RuntimeError("没有生成有效结节，请检查原始数据路径、pylidc数据库及依赖环境")

    df = pd.DataFrame(records)
    df = df.sort_values(["PatientID", "ScanID", "FileName"]).reset_index(drop=True)
    csv_file = OUTPUT_ROOT / "preprocessed_nodules.csv"
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    if failure_records:
        pd.DataFrame(failure_records).to_csv(
            OUTPUT_ROOT / "preprocess_failures.csv", index=False, encoding="utf-8-sig")
    prepare_training_manifest(df)

    print("\n========== 预处理完成 ==========")
    print(f"有效CT序列：{passed_scans}；最终患者：{df['PatientID'].nunique()}；结节：{len(df)}")
    print(f"跳过CT数：{skipped_scans}；处理失败结节：{skipped_nodules}")
    print(f"图像目录：{IMG_DIR}；Mask目录：{MASK_DIR}")
    print(f"未划分样本CSV：{csv_file}")
    print(f"[空间尺寸] 已保存结节ROI: {ROI_SIZE}^3；训练时由dataset.py裁成{TRAIN_PATCH_SIZE}^3输入，模型输出同尺寸。")
    print("注意：当前SPLIT_MODE=none时不创建train/val/test；训练需要原始划分CSV。")
    print("注意：重做预处理不保证样本与旧论文数据完全相同，不得覆盖历史数据或声称自动复现论文。")


if __name__ == "__main__":
    main()
