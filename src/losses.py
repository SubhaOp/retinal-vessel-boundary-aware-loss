
"""
Loss functions for retinal vessel segmentation.

Implemented:
1. BCE Loss
2. Dice Loss
3. BCE + Dice Loss
4. Focal Loss
5. Jaccard Loss
6. Boundary-aware BCE + Dice Loss

All losses support FOV masking.
Model predictions must be raw logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_inputs(logits, targets, fov=None):
    """Validate segmentation tensors."""

    if logits.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: logits={logits.shape}, "
            f"targets={targets.shape}"
        )

    if fov is not None and fov.shape != targets.shape:
        raise ValueError(
            f"FOV shape mismatch: {fov.shape}"
        )

    targets = targets.float()

    if fov is not None:
        fov = fov.float()

    return targets, fov


def get_valid_mask(targets, fov=None):
    """Return valid pixels inside the FOV."""

    if fov is None:
        return torch.ones_like(targets)

    return (fov > 0.5).float()


class BCELoss(nn.Module):
    """Binary Cross Entropy with FOV masking."""

    def forward(self, logits, targets, fov=None):

        targets, fov = validate_inputs(logits, targets, fov)
        valid = get_valid_mask(targets, fov)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        return (loss * valid).sum() / valid.sum().clamp_min(1.0)


class DiceLoss(nn.Module):
    """Soft Dice loss."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, fov=None):

        targets, fov = validate_inputs(logits, targets, fov)

        probabilities = torch.sigmoid(logits)
        valid = get_valid_mask(targets, fov)

        probabilities = probabilities * valid
        targets = targets * valid

        probabilities = probabilities.flatten(1)
        targets = targets.flatten(1)

        intersection = (probabilities * targets).sum(dim=1)

        denominator = (
            probabilities.sum(dim=1) + targets.sum(dim=1)
        )

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice.mean()


class BCEPlusDiceLoss(nn.Module):
    """Combined BCE and Dice loss."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()

        self.bce = BCELoss()
        self.dice = DiceLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets, fov=None):

        bce = self.bce(logits, targets, fov)
        dice = self.dice(logits, targets, fov)

        return (
            self.bce_weight * bce
            + self.dice_weight * dice
        )


class FocalLoss(nn.Module):
    """Binary Focal loss for difficult pixels."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, fov=None):

        targets, fov = validate_inputs(logits, targets, fov)
        valid = get_valid_mask(targets, fov)

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probabilities = torch.sigmoid(logits)

        pt = (
            probabilities * targets
            + (1.0 - probabilities) * (1.0 - targets)
        )

        alpha_factor = (
            self.alpha * targets
            + (1.0 - self.alpha) * (1.0 - targets)
        )

        focal = alpha_factor * (1.0 - pt).pow(self.gamma) * bce

        return (focal * valid).sum() / valid.sum().clamp_min(1.0)


class JaccardLoss(nn.Module):
    """Soft Jaccard / Intersection over Union loss."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, fov=None):

        targets, fov = validate_inputs(logits, targets, fov)

        probabilities = torch.sigmoid(logits)
        valid = get_valid_mask(targets, fov)

        probabilities = (probabilities * valid).flatten(1)
        targets = (targets * valid).flatten(1)

        intersection = (probabilities * targets).sum(dim=1)

        union = (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            - intersection
        )

        jaccard = (
            intersection + self.smooth
        ) / (
            union + self.smooth
        )

        return 1.0 - jaccard.mean()


class BoundaryWeightedBCELoss(nn.Module):
    """
    Boundary-weighted BCE.

    Gives additional importance to vessel boundary pixels.
    """

    def __init__(self, boundary_weight=3.0):
        super().__init__()

        self.boundary_weight = boundary_weight

    def forward(self, logits, targets, fov=None):

        targets, fov = validate_inputs(logits, targets, fov)
        valid = get_valid_mask(targets, fov)

        # Calculate approximate vessel boundaries
        dilated = F.max_pool2d(
            targets, kernel_size=3, stride=1, padding=1
        )

        eroded = -F.max_pool2d(
            -targets, kernel_size=3, stride=1, padding=1
        )

        boundary = ((dilated - eroded) > 0.5).float()

        # Give more weight to boundary pixels
        weights = 1.0 + self.boundary_weight * boundary

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        weighted_loss = bce * weights * valid

        return (
            weighted_loss.sum()
            / (weights * valid).sum().clamp_min(1.0)
        )


class BoundaryAwareLoss(nn.Module):
    """
    Boundary-aware segmentation objective.

    Total = BCE + Dice + lambda * Boundary BCE
    """

    def __init__(
        self,
        bce_weight=0.4,
        dice_weight=0.4,
        boundary_weight=0.2
    ):
        super().__init__()

        self.bce = BCELoss()
        self.dice = DiceLoss()

        self.boundary = BoundaryWeightedBCELoss(
            boundary_weight=3.0
        )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def forward(self, logits, targets, fov=None):

        bce = self.bce(logits, targets, fov)
        dice = self.dice(logits, targets, fov)
        boundary = self.boundary(logits, targets, fov)

        return (
            self.bce_weight * bce
            + self.dice_weight * dice
            + self.boundary_weight * boundary
        )
