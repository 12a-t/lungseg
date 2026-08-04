import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from dataset import load_data
from loss import DiceLoss, FocalLoss, BoundaryLoss
from model import SwinUNETR_Optimized, load_pretrained_weights


DATA_ROOT = "./data"
CSV_PATH = os.path.join(DATA_ROOT, "clean_dataset_metadata.csv")
IMG_DIR = os.path.join(DATA_ROOT, "lungs")
MASK_DIR = os.path.join(DATA_ROOT, "masks")
CHECKPOINT_DIR = "./checkpoints"
PRETRAINED_PATH = "./pretrained/model_swinvit.pt"

INPUT_SIZE = 64
SAVED_SIZE = 96
NUM_EPOCHS = 100

BATCH_SIZE = 4
ACCUMULATION_STEPS = 4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
RANDOM_SHIFT_PROB = 0.3

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def calculate_dice_score(preds, targets, threshold=0.5, smooth=1e-6):
    preds = (preds > threshold).float().view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()


def plot_metrics(train_scores, val_scores, metric_name="Dice Score"):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_scores) + 1), train_scores, 'b-', label=f'Train {metric_name}')
    plt.plot(range(1, len(val_scores) + 1), val_scores, 'r-', label=f'Val {metric_name}')
    plt.title(f'{metric_name} vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(CHECKPOINT_DIR, "training_curve.png"))
    plt.close('all')


def initialize_model(device=DEVICE):
    model = SwinUNETR_Optimized(
        img_size=(64, 64, 64),
        in_channels=1,
        out_channels=1,
        feature_size=96,
        use_checkpoint=True,
    ).to(device)

    if os.path.exists(PRETRAINED_PATH):
        model = load_pretrained_weights(model, PRETRAINED_PATH)

    return model


def train_model(model, train_loader, val_loader,
                num_epochs=NUM_EPOCHS,
                lr=LEARNING_RATE,
                device=DEVICE):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    updates_per_epoch = max(1, len(train_loader) // ACCUMULATION_STEPS)
    total_updates = num_epochs * updates_per_epoch
    scheduler = CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)

    criterion_dice = DiceLoss().to(device)
    criterion_focal = FocalLoss(alpha=0.75, gamma=2.0).to(device)
    criterion_boundary = BoundaryLoss().to(device)

    train_dice_history = []
    val_structural_dice_history = []
    best_val_dice = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_train_dice = 0.0

        if epoch < 20:
            lambda_bd = 0.0
        else:
            lambda_bd = 0.01

        optimizer.zero_grad()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Train) [λ_bd:{lambda_bd}]")

        step_count = 0

        for i, (imgs_norm, masks) in enumerate(loop):
            if imgs_norm.size(0) == 1:
                continue

            imgs_norm = imgs_norm.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            outputs = model(imgs_norm)

            if isinstance(outputs, (list, tuple)):
                logits_main = outputs[0]
                logits_ds1 = outputs[1] if len(outputs) > 1 else None
                logits_ds2 = outputs[2] if len(outputs) > 2 else None
            else:
                logits_main = outputs
                logits_ds1, logits_ds2 = None, None

            prob_main = torch.sigmoid(logits_main)
            prob_ds1 = torch.sigmoid(logits_ds1) if logits_ds1 is not None else None
            prob_ds2 = torch.sigmoid(logits_ds2) if logits_ds2 is not None else None

            loss_dice = criterion_dice(prob_main, masks)
            loss_focal = criterion_focal(prob_main, masks)

            loss_bd = torch.tensor(0.0, device=device)
            if lambda_bd > 0:
                loss_bd = criterion_boundary(prob_main, masks)

            loss_main = loss_dice + loss_focal + lambda_bd * loss_bd

            loss_ds1 = 0.0
            if prob_ds1 is not None:
                mask_ds1 = F.interpolate(masks, size=prob_ds1.shape[2:], mode='nearest')
                loss_ds1 = criterion_dice(prob_ds1, mask_ds1) + 0.5 * criterion_focal(prob_ds1, mask_ds1)

            loss_ds2 = 0.0
            if prob_ds2 is not None:
                mask_ds2 = F.interpolate(masks, size=prob_ds2.shape[2:], mode='nearest')
                loss_ds2 = criterion_dice(prob_ds2, mask_ds2) + 0.5 * criterion_focal(prob_ds2, mask_ds2)

            loss = loss_main + 0.5 * loss_ds1 + 0.25 * loss_ds2

            loss = loss / ACCUMULATION_STEPS
            loss.backward()

            step_count += 1

            current_loss = loss.item() * ACCUMULATION_STEPS
            total_loss += current_loss
            total_train_dice += calculate_dice_score(prob_main.detach(), masks.detach())

            if step_count % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']
            loop.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr:.2e}")

        if step_count % ACCUMULATION_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        effective_batches = max(1, len(train_loader))
        avg_train_loss = total_loss / effective_batches
        avg_train_dice = total_train_dice / effective_batches
        train_dice_history.append(avg_train_dice)

        model.eval()
        total_structural_dice = 0.0
        valid_count = 0

        with torch.no_grad():
            for imgs_norm, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} (Val)"):
                imgs_norm = imgs_norm.to(device, dtype=torch.float32)
                masks = masks.to(device, dtype=torch.float32)

                is_padded = False
                if imgs_norm.size(0) == 1:
                    imgs_norm = torch.cat([imgs_norm, imgs_norm], dim=0)
                    masks = torch.cat([masks, masks], dim=0)
                    is_padded = True

                outputs_val = model(imgs_norm)

                if isinstance(outputs_val, (list, tuple)):
                    logits_val = outputs_val[0]
                else:
                    logits_val = outputs_val

                prob_val = torch.sigmoid(logits_val)

                if is_padded:
                    prob_val = prob_val[:1]
                    masks = masks[:1]

                for k in range(masks.size(0)):
                    if masks[k].sum() > 0:
                        dice_val = calculate_dice_score(prob_val[k], masks[k])
                        total_structural_dice += dice_val
                        valid_count += 1

        avg_val_dice = total_structural_dice / valid_count if valid_count > 0 else 0.0
        val_structural_dice_history.append(avg_val_dice)

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))

        print(f"\nEpoch {epoch + 1}: "
              f"TrainLoss={avg_train_loss:.4f} | TrainDice={avg_train_dice:.4f} | ValDice={avg_val_dice:.4f}")

        plot_metrics(train_dice_history, val_structural_dice_history, "Single-Input Dice")
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "last_model.pth"))


if __name__ == '__main__':
    train_loader, val_loader = load_data(
        CSV_PATH, IMG_DIR, MASK_DIR,
        INPUT_SIZE, SAVED_SIZE,
        BATCH_SIZE,
        RANDOM_SHIFT_PROB, NUM_WORKERS
    )

    if train_loader:
        model = initialize_model(DEVICE)
        train_model(model, train_loader, val_loader)
