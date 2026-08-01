import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt as distance


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if input.shape != target.shape:
            raise ValueError(f"DiceLoss shapes mismatch: {input.shape} vs {target.shape}")

        B = input.size(0)
        input_flat = input.view(B, -1)
        target_flat = target.view(B, -1)

        intersection = (input_flat * target_flat).sum(dim=1)
        denom = input_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        loss = 1.0 - dice
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, eps: float = 1e-6):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.clamp(inputs, self.eps, 1.0 - self.eps)
        pt = inputs * targets + (1.0 - inputs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = -alpha_t * torch.pow(1.0 - pt, self.gamma) * torch.log(pt)
        return loss.mean()


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def _one_hot2dist(self, seg: np.ndarray):
        res = np.zeros_like(seg, dtype=np.float32)
        for i in range(len(seg)):
            posmask = seg[i].astype(bool)

            if posmask.any():
                negmask = ~posmask
                raw_dist = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
                res[i] = np.clip(raw_dist, -5.0, 5.0)
        return res

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            gt_np = target.cpu().numpy().squeeze(1)
            dist_map_np = self._one_hot2dist(gt_np)
            dist_map = torch.from_numpy(dist_map_np).float().to(prediction.device).unsqueeze(1)

        loss = prediction * dist_map
        return loss.mean()