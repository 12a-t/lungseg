# -*- coding: utf-8 -*-
"""DSSW-Net training using the ORIGINAL, UNMODIFIED model.py.

Data/training configuration follows the submitted manuscript where supported:
96^3 saved ROIs -> 64^3 training patches; random rotations within +/-15 degrees,
scaling 0.85-1.15, flips, Dice+Focal+Boundary warm-up, AdamW, cosine LR.

IMPORTANT: Original model.py uses trainable strided Conv3d rather than fixed Haar,
and standard 3D conv rather than depth-wise SAPM. It has NOT been changed.
This implementation MUST NOT be claimed as exact paper-operator reproduction.
"""
from __future__ import annotations
import json
import hashlib
import math
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from dataset import load_data
from loss import DiceLoss, FocalLoss, BoundaryLoss
from model import SwinUNETR_Optimized, load_pretrained_weights

# ================== EDIT CONFIG HERE IN PYCHARM ==================
PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv('DSSWNET_DATA_ROOT', str(PROJECT_DIR / 'LIDC')))
CSV_PATH = DATA_ROOT / 'detailed_dataset_metadata.csv'
IMG_DIR = DATA_ROOT / 'lung'
MASK_DIR = DATA_ROOT / 'mask'
CHECKPOINT_DIR = Path(os.getenv('DSSWNET_CHECKPOINT_DIR', str(PROJECT_DIR / 'checkpoints')))
PRETRAINED_PATH = os.getenv('DSSWNET_PRETRAINED', '')  # optional trusted Swin weights

INPUT_SIZE = 64           # Input and output tensor spatial size.
SAVED_SIZE = 96           # Preprocessed ROI spatial size.
FEATURE_SIZE = 96         # Encoder feature channel width, NOT spatial size.
SEED = 42
NUM_EPOCHS = 100
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 2
RANDOM_SHIFT_PROB = 0.3
GRAD_CLIP_NORM = 1.0
# ================================================================


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_dice_per_nodule(probabilities, targets, threshold=0.5):
    """Returns mean of per-nodule Dice, not Dice over a flattened whole batch."""
    a = (probabilities >= threshold).float().reshape(probabilities.shape[0], -1)
    b = (targets > 0.5).float().reshape(targets.shape[0], -1)
    intersection = (a * b).sum(dim=1)
    denom = a.sum(dim=1) + b.sum(dim=1)
    dice = torch.where(denom > 0, 2.0 * intersection / denom.clamp_min(1),
                       torch.ones_like(denom))
    return dice


def initialize_model(device):
    model = SwinUNETR_Optimized(
        img_size=(INPUT_SIZE,) * 3,
        in_channels=1,
        out_channels=1,
        feature_size=FEATURE_SIZE,
        use_checkpoint=True,
    ).to(device)
    if PRETRAINED_PATH:
        if not Path(PRETRAINED_PATH).is_file():
            raise FileNotFoundError(f'Pretrained weights not found: {PRETRAINED_PATH}')
        print('[Warning] Loading pretrained weights via original model.py loader: use only a trusted self-owned checkpoint.')
        model = load_pretrained_weights(model, PRETRAINED_PATH)
    return model


def _loss_all(outputs, masks, dice_fn, focal_fn, boundary_fn, bd_weight):
    if isinstance(outputs, (list, tuple)):
        logits = outputs[0]
        ds1 = outputs[1] if len(outputs) > 1 else None
        ds2 = outputs[2] if len(outputs) > 2 else None
    else:
        logits, ds1, ds2 = outputs, None, None
    prob = torch.sigmoid(logits)
    loss = dice_fn(prob, masks) + focal_fn(prob, masks)
    if bd_weight > 0:
        loss = loss + bd_weight * boundary_fn(prob, masks)
    for auxiliary, coefficient in ((ds1, 0.5), (ds2, 0.25)):
        if auxiliary is not None:
            small_target = F.interpolate(masks, size=auxiliary.shape[2:], mode='nearest')
            small_prob = torch.sigmoid(auxiliary)
            aux_loss = dice_fn(small_prob, small_target) + 0.5 * focal_fn(small_prob, small_target)
            loss = loss + coefficient * aux_loss
    return loss, prob


def normalize_accumulated_gradients(model, n_samples):
    """Normalize accumulated sum-of-per-sample gradients before optimizer step.

    Correctly handles an incomplete last minibatch (e.g. 950 nodules / batch 4).
    Use this only with loss functions reduced by batch mean, as below.
    """
    if n_samples < 1:
        raise ValueError('Cannot normalize gradient with no samples')
    for param in model.parameters():
        if param.grad is not None:
            param.grad.div_(n_samples)


def train_model(model, train_loader, val_loader, device, epochs=NUM_EPOCHS):
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError('Empty train/validation loader')
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Previous code used floor division and silently missed the final accumulated update.
    steps_per_epoch = math.ceil(len(train_loader) / ACCUMULATION_STEPS)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch, eta_min=MIN_LR)
    dice_fn, focal_fn, boundary_fn = DiceLoss(), FocalLoss(alpha=0.75, gamma=2.0), BoundaryLoss()
    dice_fn, focal_fn, boundary_fn = (m.to(device) for m in (dice_fn, focal_fn, boundary_fn))
    best = -1.0
    history = []
    for epoch in range(epochs):
        model.train()
        bd_weight = 0.0 if epoch < 20 else 0.01
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        dice_sum = 0.0
        sample_count = 0
        accumulated_samples = 0
        bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Train {epoch+1}/{epochs}')
        for i, (images, masks) in bar:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            masks = masks.to(device, dtype=torch.float32, non_blocking=True)
            if images.shape[2:] != (INPUT_SIZE,)*3 or masks.shape != images.shape:
                raise ValueError(f'Unexpected patch shapes: {images.shape}, {masks.shape}')
            output = model(images)
            loss, prob = _loss_all(output, masks, dice_fn, focal_fn, boundary_fn, bd_weight)
            batch_n = images.shape[0]
            # Convert per-batch mean loss to sum, normalize by actual samples at optimizer.step.
            # This treats a short last minibatch correctly instead of overweighting it.
            (loss * batch_n).backward()
            accumulated_samples += batch_n
            loss_sum += float(loss.detach()) * batch_n
            dice_sum += float(calculate_dice_per_nodule(prob.detach(), masks).sum())
            sample_count += batch_n
            if (i + 1) % ACCUMULATION_STEPS == 0 or i + 1 == len(train_loader):
                normalize_accumulated_gradients(model, accumulated_samples)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accumulated_samples = 0
            bar.set_postfix(loss=f'{float(loss.detach()):.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.2e}')
        model.eval()
        val_dice_sum, val_valid = 0.0, 0
        with torch.inference_mode():
            for images, masks in tqdm(val_loader, desc=f'Val {epoch+1}/{epochs}'):
                images = images.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32)
                logits = model(images)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                scores = calculate_dice_per_nodule(torch.sigmoid(logits), masks)
                present = masks.reshape(masks.shape[0], -1).sum(dim=1) > 0
                val_dice_sum += float(scores[present].sum())
                val_valid += int(present.sum())
        val_dice = val_dice_sum / val_valid if val_valid else float('nan')
        row = {'epoch': epoch + 1, 'train_loss': loss_sum/sample_count,
               'train_dice': dice_sum/sample_count, 'val_dice': val_dice,
               'lr': optimizer.param_groups[0]['lr'], 'boundary_weight': bd_weight}
        history.append(row)
        print(f'[E{epoch+1}] train_dice={row["train_dice"]:.4f} '
              f'val_dice={val_dice:.4f} loss={row["train_loss"]:.4f}')
        if math.isfinite(val_dice) and val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), CHECKPOINT_DIR/'best_model.pth')
        torch.save(model.state_dict(), CHECKPOINT_DIR/'last_model.pth')
        import pandas as pd
        pd.DataFrame(history).to_csv(CHECKPOINT_DIR/'history.csv', index=False)
    print(f'[Training complete] best validation Dice={best:.5f}')
    return history


def main():
    set_seed(SEED)
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f'Missing patient-level split CSV: {CSV_PATH}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        'architecture': 'original_model_unchanged',
        'model_py_sha256': hashlib.sha256((PROJECT_DIR / 'model.py').read_bytes()).hexdigest(),
        'paper_operator_mismatch': 'Original model: learned stride2 Conv3d, standard 3D SAPM; differs from fixed Haar/depthwise manuscript text',
        'model_patch_size': INPUT_SIZE,
        'saved_roi_size': SAVED_SIZE,
        'feature_size': FEATURE_SIZE,
        'batch_size': BATCH_SIZE,
        'accumulation_steps': ACCUMULATION_STEPS,
        'num_epochs': NUM_EPOCHS,
        'lr': LEARNING_RATE,
        'min_lr': MIN_LR,
        'weight_decay': WEIGHT_DECAY,
        'seed': SEED,
        'source_pretrained_path': PRETRAINED_PATH or None,
        'evaluation_note': 'val Dice is GT-free ROI-center-cropped patch Dice; not an independent test score',
    }
    (CHECKPOINT_DIR/'training_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    train_loader, val_loader = load_data(
        str(CSV_PATH), str(IMG_DIR), str(MASK_DIR),
        input_size=INPUT_SIZE, saved_size=SAVED_SIZE,
        batch_size=BATCH_SIZE, shift_prob=RANDOM_SHIFT_PROB,
        num_workers=NUM_WORKERS,
    )
    if train_loader is None or val_loader is None:
        raise RuntimeError('Failed to create DataLoaders')
    print(f'[device] {device}; [architecture] original model.py kept UNCHANGED')
    model = initialize_model(device)
    train_model(model, train_loader, val_loader, device)


if __name__ == '__main__':
    main()
