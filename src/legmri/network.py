"""Six-level nnU-Net backbone and the manuscript's IMAT-weighted objective."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class IMATDiceCELoss(nn.Module):
    def __init__(
        self, num_classes=5, imat_label=1, imat_weight=1.5, include_background=False, smooth=1e-5
    ):
        super().__init__()
        weights = torch.ones(num_classes)
        weights[imat_label] = imat_weight
        self.register_buffer("weights", weights)
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits, target):
        if target.ndim == logits.ndim:
            target = target[:, 0]
        target = target.long()
        # reduction='mean' in weighted torch CE divides by SUM OF WEIGHTS.
        # S1 specifies 1/N, so compute unreduced weighted CE then the voxel mean.
        ce = F.cross_entropy(logits.float(), target, weight=self.weights, reduction="none").mean()
        probability = logits.float().softmax(1)
        spatial = tuple(range(1, target.ndim))
        dice = []
        for label in range(0 if self.include_background else 1, logits.shape[1]):
            truth, pred = target == label, probability[:, label]
            overlap = (pred * truth).sum(spatial)
            denom = pred.sum(spatial) + truth.sum(spatial)
            dice.append((2 * overlap + self.smooth) / (denom + self.smooth))
        return ce + 1 - torch.stack(dice).mean()


class PaddedPlainConvUNet(nn.Module):
    """Keep the requested patch shape, pad internally to legal U-Net strides.

    The padding is removed from logits before the loss or inference export.
    Deep supervision is disabled because S1 specifies one composite objective.
    """

    def __init__(
        self, input_channels=3, num_classes=5, features=(32, 64, 128, 256, 320, 320), strides=None
    ):
        super().__init__()
        from dynamic_network_architectures.architectures.unet import PlainConvUNet

        strides = strides or [[1, 1, 1]] + [[2, 2, 2]] * (len(features) - 1)
        self.divisibility = np.prod(strides, axis=0).astype(int).tolist()
        self.backbone = PlainConvUNet(
            input_channels=input_channels,
            n_stages=len(features),
            features_per_stage=list(features),
            conv_op=nn.Conv3d,
            kernel_sizes=[[3, 3, 3]] * len(features),
            strides=strides,
            n_conv_per_stage=[2] * len(features),
            num_classes=num_classes,
            n_conv_per_stage_decoder=[2] * (len(features) - 1),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"negative_slope": 0.01, "inplace": True},
            deep_supervision=False,
        )
        self.backbone.apply(self.backbone.initialize)

    def forward(self, x):
        original = x.shape[2:]
        pads = [(-n) % d for n, d in zip(original, self.divisibility)]
        x = F.pad(x, [v for p in reversed(pads) for v in (0, p)])
        output = self.backbone(x)
        return output[(slice(None), slice(None)) + tuple(slice(0, n) for n in original)]
