"""Minimal nnU-Net extension; no copied framework or site-packages modification."""

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler

from .network import IMATDiceCELoss, PaddedPlainConvUNet


class LegMRITrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        cfg = plans["legmri"]["segmentation"]
        self.initial_lr, self.weight_decay = cfg["learning_rate"], cfg["weight_decay"]
        self.num_epochs = cfg["epochs"]
        self.num_iterations_per_epoch = cfg["iterations_per_epoch"]
        self.num_val_iterations_per_epoch = cfg["validation_iterations"]
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        return PaddedPlainConvUNet(
            num_input_channels,
            num_output_channels,
            arch_init_kwargs["features_per_stage"],
            arch_init_kwargs["strides"],
        )

    def _build_loss(self):
        cfg = self.plans_manager.plans["legmri"]
        return IMATDiceCELoss(
            self.label_manager.num_segmentation_heads,
            cfg["labels"]["IMAT"],
            cfg["segmentation"]["imat_weight"],
            cfg["segmentation"]["dice_background"],
        ).to(self.device)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay
        )
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def set_deep_supervision_enabled(self, enabled):
        # All paths use the one full-resolution output described in Supplementary S1.
        return None
