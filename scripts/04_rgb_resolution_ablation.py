#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RGB spatial-resolution ablation for the final cross-sensor RGB+802 Gram pipeline.

Usage:
    python scripts/04_rgb_resolution_ablation.py --factor 2
    python scripts/04_rgb_resolution_ablation.py --factor 4
    python scripts/04_rgb_resolution_ablation.py --factor 8

Only Sony RGB spatial resolution is changed. The HR Mjolnir 802-nm anchor,
LR-HSI, 40-band target representation, mapper/loss, and unsupervised
checkpoint-selection protocol remain fixed.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda:0"
SEED = 0
DETERMINISTIC = False

ROOT = Path.home() / "RGB2msi"

MJOLNIR_H5 = (
    ROOT
    / "hyspex_mjolnir1024"
    / "mjolnir_prepared"
    / "mjolnir_field_40band.h5"
)

GUIDE_H5 = (
    ROOT
    / "hyspex_mjolnir1024"
    / "sony_cross_sensor"
    / "cross_sensor_hr_guide_1000x1000.h5"
)

ENCODER_CHECKPOINT_NAME = (
    "unsupervised_CRF_learning_"
    "encoder_uav_mjolnir_CE_9_endmember_sam_"
    "4spc40_visNir_g54_01_sam1p09.pth"
)

DECODER_CHECKPOINT_NAME = (
    "unsupervised_CRF_learning_"
    "decoder_uav_mjolnir_CE_9_endmember_sam_"
    "4spc40_visNir_g54_01_sam1p09.pth"
)

parser = argparse.ArgumentParser(description="RGB spatial-resolution ablation")
parser.add_argument("--factor", type=int, choices=[2, 4, 8], required=True)
args = parser.parse_args()
RGB_DOWNSAMPLE_FACTOR = int(args.factor)

OUTPUT_ROOT = (
    ROOT
    / "hyspex_mjolnir1024"
    / "rgb_resolution_ablation"
    / f"sony_bgr_plus_802_rgb_x{RGB_DOWNSAMPLE_FACTOR}_seed0"
)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MAX_EPOCHS = 1_000_001
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EVAL_EVERY = 200
SAVE_LATEST_EVERY = 10_000
GRAD_CLIP = 0.0
SAM_PATIENCE_EPOCHS = 5_000
SAM_MIN_DECREASE = 1e-3
RECON_WINDOW_EPOCHS = 2_000
RECON_REL_CHANGE_THRESHOLD = 1e-3
RECON_STABLE_WINDOWS = 2
RECON_EARLY_STOP_MIN_EPOCH = 5_000
SKIP_COMPLETED = True
RESUME_INCOMPLETE = True
SAVE_RECONSTRUCTION = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if DETERMINISTIC:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.benchmark = True

device = torch.device(DEVICE)

if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but unavailable.")


class EncoderLRHSI(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Linear(40, 5, bias=False)
        self.conv12 = nn.Linear(45, 5, bias=False)
        self.conv13 = nn.Linear(50, 40, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        layer_11 = self.relu(self.conv11(x))
        stack_layer_11 = torch.cat((x, layer_11), dim=1)
        layer_12 = self.relu(self.conv12(stack_layer_11))
        stack_layer_12 = torch.cat((stack_layer_11, layer_12), dim=1)
        return self.relu(self.conv13(stack_layer_12))


class DecoderHSI(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv31 = nn.Linear(40, 40, bias=False)

    def forward(self, x):
        return self.conv31(x)


class InputToHSIMapper(nn.Module):
    """
    Stored guide order [R,G,B,NIR]; mapper order [B,G,R,NIR].
    """

    def __init__(self, n_inputs=4, n_outputs=40):
        super().__init__()

        if n_inputs != 4 or n_outputs != 40:
            raise ValueError("Expected 4 inputs and 40 HSI outputs.")

        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)

        self.conv11 = nn.Linear(
            self.n_inputs,
            self.n_outputs,
            bias=False,
        )

        self.register_buffer(
            "repeat_counts",
            torch.tensor([10, 10, 10, 10], dtype=torch.long),
        )

        self.register_buffer(
            "channel_order",
            torch.tensor([2, 1, 0, 3], dtype=torch.long),
        )

    def forward(self, x):
        x = torch.index_select(
            x,
            dim=1,
            index=self.channel_order,
        )

        learned = self.conv11(x)
        residual = torch.repeat_interleave(
            x,
            self.repeat_counts,
            dim=1,
        )
        return learned + residual


def get_stick_segments(v):
    one_minus = 1.0 - v
    cumulative = torch.cumprod(one_minus, dim=1)
    previous = torch.cat(
        [
            torch.ones_like(cumulative[:, :1]),
            cumulative[:, :-1],
        ],
        dim=1,
    )
    return v * previous


def l_2_1_loss(x_true, x_pred):
    return (((x_pred - x_true).pow(2)).sum(1)).pow(0.5).sum(0)


def sam_ev_gen(x_true, x_pred):
    N = x_true.size()[0]
    nom = torch.sum(x_true * x_pred, dim=1)
    denom1 = torch.sqrt(torch.sum(x_true ** 2, dim=1))
    denom2 = torch.sqrt(torch.sum(x_pred ** 2, dim=1))
    sam = torch.acos(
        (nom / (denom1 * denom2 + 1e-8)).clamp(
            -1.0 + 1e-8,
            1.0 - 1e-8,
        )
    )
    sam = sam / np.pi * 180.0
    return sam.sum() / N


def psnr_ev_gen(x_true, x_pred):
    msr = ((x_true - x_pred) ** 2).mean(1)
    max2 = (x_true ** 2).max()
    return (10.0 * torch.log10(max2 / (msr + 1e-8))).mean()


def ergas_loss_gen(x_true, x_pred, scale=1.0, eps=1e-8):
    means_real = x_true.mean(dim=0)
    mses = ((x_true - x_pred) ** 2).mean(dim=0)
    return 100.0 / scale * torch.sqrt(
        (mses / (means_real ** 2 + eps)).mean()
    )


def centered_gram(abundance):
    gram = torch.mm(
        abundance.transpose(0, 1),
        abundance,
    ) / abundance.size(0)

    gram_centered = gram - gram.mean(0, keepdim=True)
    gram_diagonal = torch.diag(gram).unsqueeze(0)
    return gram_centered, gram_diagonal


def compute_original_composite_loss(
    mapped_hsi,
    reconstructed_hsi,
    hr_gram_centered,
    hr_gram_diagonal,
    lr_gram_centered,
    lr_gram_diagonal,
):
    loss_de_msi = l_2_1_loss(
        mapped_hsi + 1e-8,
        reconstructed_hsi,
    )
    style_abundance_loss = l_2_1_loss(
        hr_gram_centered,
        lr_gram_centered,
    )
    style_abundance_sam_loss = sam_ev_gen(
        hr_gram_centered,
        lr_gram_centered,
    )
    style_abundance_sam_loss_t = sam_ev_gen(
        hr_gram_centered.transpose(0, 1),
        lr_gram_centered.transpose(0, 1),
    )
    style_ergas_loss = ergas_loss_gen(
        hr_gram_centered,
        lr_gram_centered,
        1,
        1e-8,
    )
    style_abundance_loss_abs = l_2_1_loss(
        hr_gram_diagonal,
        lr_gram_diagonal,
    )
    style_abundance_loss_sam = sam_ev_gen(
        hr_gram_diagonal,
        lr_gram_diagonal,
    )

    loss_de_final_multi = (
        style_abundance_loss
        + style_ergas_loss
        + 10.0 * style_abundance_sam_loss
        + style_abundance_sam_loss_t
    )
    loss_de_final_new = (
        style_abundance_loss_abs
        + 10.0 * style_abundance_loss_sam
    )
    loss_de_final = (
        0.1 * loss_de_msi
        + 10.0 * loss_de_final_new
        + 100.0 * loss_de_final_multi
    )

    parts = {
        "reconstruction_l21": loss_de_msi,
        "gram_l21": style_abundance_loss,
        "gram_ergas": style_ergas_loss,
        "gram_sam": style_abundance_sam_loss,
        "gram_sam_t": style_abundance_sam_loss_t,
        "diag_l21": style_abundance_loss_abs,
        "diag_sam": style_abundance_loss_sam,
    }
    return loss_de_final, parts


def evaluate_numpy(gt_hwc, pred_hwc, mask=None):
    gt = np.asarray(gt_hwc, dtype=np.float64).reshape(-1, gt_hwc.shape[-1])
    pred = np.asarray(pred_hwc, dtype=np.float64).reshape(-1, pred_hwc.shape[-1])

    if mask is not None:
        mask_flat = np.asarray(mask, dtype=bool).reshape(-1)
        gt = gt[mask_flat]
        pred = pred[mask_flat]

    eps = 1e-8
    numerator = (gt * pred).sum(1)
    denominator = (
        np.sqrt((gt ** 2).sum(1))
        * np.sqrt((pred ** 2).sum(1))
        + eps
    )

    sam = float(
        np.degrees(
            np.arccos(
                np.clip(numerator / denominator, -1.0, 1.0)
            )
        ).mean()
    )

    mse_per_pixel = ((gt - pred) ** 2).mean(1)
    max2 = (gt ** 2).max()
    psnr = float(
        (10.0 * np.log10(max2 / (mse_per_pixel + eps))).mean()
    )
    rmse_band = np.sqrt(((gt - pred) ** 2).mean(0))
    mean_band = np.abs(gt.mean(0)) + eps
    ergas = float(
        100.0 * np.sqrt(np.mean((rmse_band / mean_band) ** 2))
    )

    return {
        "SAM": sam,
        "PSNR": psnr,
        "ERGAS": ergas,
        "RMSE": float(np.sqrt(((gt - pred) ** 2).mean())),
        "MAE": float(np.abs(gt - pred).mean()),
        "rmse_band": rmse_band,
    }
