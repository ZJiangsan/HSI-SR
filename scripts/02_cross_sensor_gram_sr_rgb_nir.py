#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real cross-sensor abundance-Gram HSI super-resolution.

Observed data:
    HR guide:
        Sony RGB              1000 x 1000 x 3
        Mjolnir NIR (~801 nm)1000 x 1000 x 1

    LR-HSI:
        Mjolnir 40-band       125 x 125 x 40

Reconstruction:
        1000 x 1000 x 40

Key rules:
- Process the COMPLETE field at once; do NOT train plot-by-plot.
- Use one global HR abundance Gram statistic over all 1,000,000 HR pixels.
- Use one global LR abundance Gram statistic over all 15,625 LR pixels.
- Reuse the frozen pretrained 40-band encoder/decoder.
- Optimize only the 4->40 HR-guide mapper.
- Official checkpoint selection is fully unsupervised:
      minimum original composite loss.
- HR-HSI GT metrics are diagnostic only.
- Official final reconstructed cube is recentered with the observed LR-HSI mean.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =============================================================================
# SETTINGS
# =============================================================================

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

OUTPUT_ROOT = (
    ROOT
    / "hyspex_mjolnir1024"
    / "sony_rgb_plus_801_cross_sensor_gram_seed0_v2"
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
    Original residual mapper semantics.

    n=4:
        learned 4->40 linear correction
        +
        deterministic repeat residual [10,10,10,10]
    """

    def __init__(self, n_inputs=4, n_outputs=40):
        super().__init__()

        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)

        self.conv11 = nn.Linear(
            self.n_inputs,
            self.n_outputs,
            bias=False,
        )

        base = self.n_outputs // self.n_inputs
        counts = [base] * self.n_inputs
        counts[-1] += self.n_outputs - sum(counts)

        self.register_buffer(
            "repeat_counts",
            torch.tensor(
                counts,
                dtype=torch.long,
            ),
        )

    def forward(self, x):
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
    return (
        (
            ((x_pred - x_true).pow(2)).sum(1)
        ).pow(0.5)
    ).sum(0)


def sam_ev_gen(x_true, x_pred):
    N = x_true.size()[0]

    nom = torch.sum(
        x_true * x_pred,
        dim=1,
    )

    denom1 = torch.sqrt(
        torch.sum(
            x_true ** 2,
            dim=1,
        )
    )

    denom2 = torch.sqrt(
        torch.sum(
            x_pred ** 2,
            dim=1,
        )
    )

    sam = torch.acos(
        (
            nom
            / (
                denom1
                * denom2
                + 1e-8
            )
        ).clamp(
            -1.0 + 1e-8,
            1.0 - 1e-8,
        )
    )

    sam = sam / np.pi * 180.0

    return sam.sum() / N


def psnr_ev_gen(x_true, x_pred):
    msr = (
        (x_true - x_pred) ** 2
    ).mean(1)

    max2 = (
        x_true ** 2
    ).max()

    return (
        10.0
        * torch.log10(
            max2
            / (
                msr
                + 1e-8
            )
        )
    ).mean()


def ergas_loss_gen(
    x_true,
    x_pred,
    scale=1.0,
    eps=1e-8,
):
    means_real = x_true.mean(dim=0)

    mses = (
        (x_true - x_pred) ** 2
    ).mean(dim=0)

    return (
        100.0
        / scale
        * torch.sqrt(
            (
                mses
                / (
                    means_real ** 2
                    + eps
                )
            ).mean()
        )
    )


def centered_gram(abundance):
    gram = (
        torch.mm(
            abundance.transpose(0, 1),
            abundance,
        )
        / abundance.size(0)
    )

    gram_centered = (
        gram
        - gram.mean(
            0,
            keepdim=True,
        )
    )

    gram_diagonal = (
        torch.diag(gram)
        .unsqueeze(0)
    )

    return (
        gram_centered,
        gram_diagonal,
    )


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
