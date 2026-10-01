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


def evaluate_numpy(
    gt_hwc,
    pred_hwc,
    mask=None,
):
    gt = np.asarray(
        gt_hwc,
        dtype=np.float64,
    ).reshape(
        -1,
        gt_hwc.shape[-1],
    )

    pred = np.asarray(
        pred_hwc,
        dtype=np.float64,
    ).reshape(
        -1,
        pred_hwc.shape[-1],
    )

    if mask is not None:
        mask_flat = np.asarray(
            mask,
            dtype=bool,
        ).reshape(-1)

        gt = gt[mask_flat]
        pred = pred[mask_flat]

    eps = 1e-8

    numerator = (
        gt * pred
    ).sum(1)

    denominator = (
        np.sqrt(
            (gt ** 2).sum(1)
        )
        * np.sqrt(
            (pred ** 2).sum(1)
        )
        + eps
    )

    sam = float(
        np.degrees(
            np.arccos(
                np.clip(
                    numerator / denominator,
                    -1.0,
                    1.0,
                )
            )
        ).mean()
    )

    mse_per_pixel = (
        (gt - pred) ** 2
    ).mean(1)

    max2 = (
        gt ** 2
    ).max()

    psnr = float(
        (
            10.0
            * np.log10(
                max2
                / (
                    mse_per_pixel
                    + eps
                )
            )
        ).mean()
    )

    rmse_band = np.sqrt(
        (
            (gt - pred) ** 2
        ).mean(0)
    )

    mean_band = (
        np.abs(
            gt.mean(0)
        )
        + eps
    )

    ergas = float(
        100.0
        * np.sqrt(
            np.mean(
                (
                    rmse_band
                    / mean_band
                )
                ** 2
            )
        )
    )

    return {
        "SAM": sam,
        "PSNR": psnr,
        "ERGAS": ergas,
        "RMSE": float(
            np.sqrt(
                (
                    (gt - pred) ** 2
                ).mean()
            )
        ),
        "MAE": float(
            np.abs(
                gt - pred
            ).mean()
        ),
        "rmse_band": rmse_band,
    }


def resolve_checkpoint(checkpoint_name):
    search_dirs = [
        Path.cwd(),
        Path.home(),
        ROOT,
        ROOT / "hyspex_mjolnir1024",
    ]

    exact_candidates = []

    for folder in search_dirs:
        p = folder / checkpoint_name

        if p.exists():
            exact_candidates.append(
                p.resolve()
            )

    exact_candidates = list(
        dict.fromkeys(
            exact_candidates
        )
    )

    if len(exact_candidates) == 1:
        return exact_candidates[0]

    if len(exact_candidates) > 1:
        print(
            "[checkpoint] multiple exact copies; using:",
            exact_candidates[0],
        )
        return exact_candidates[0]

    found = []

    for folder in search_dirs:
        if folder.exists():
            found.extend(
                folder.glob(
                    "**/" + checkpoint_name
                )
            )

    found = sorted(
        set(
            p.resolve()
            for p in found
        )
    )

    if len(found) == 1:
        return found[0]

    raise FileNotFoundError(
        "\nCould not uniquely resolve checkpoint:\n"
        + checkpoint_name
        + "\nCandidates:\n"
        + "\n".join(
            str(x)
            for x in found
        )
    )


def load_module_checkpoint(
    module,
    path,
    device,
):
    checkpoint = torch.load(
        path,
        map_location=device,
    )

    state = checkpoint.get(
        "state_dict",
        checkpoint,
    )

    module.load_state_dict(state)


def atomic_torch_save(payload, path):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temp,
    )

    os.replace(
        temp,
        path,
    )


if not MJOLNIR_H5.exists():
    raise FileNotFoundError(MJOLNIR_H5)

with h5py.File(MJOLNIR_H5, "r") as f:
    img_hr_hsi = np.float32(
        f["hsi_hr"][:1000, :1000, :]
    )

    img_lr_hsi = np.float32(
        f["hsi_lr"][:]
    )

    wavelengths = np.float32(
        f["selected_wavelengths_nm"][:]
    )

    plot_mask_hr = np.uint8(
        f["plot_mask_hr"][:1000, :1000]
    )

    valid_mask_hr = np.uint8(
        f["valid_mask_hr"][:1000, :1000]
    )

if img_hr_hsi.shape != (1000, 1000, 40):
    raise ValueError(
        f"Expected HR HSI=(1000,1000,40), got {img_hr_hsi.shape}"
    )

if img_lr_hsi.shape != (125, 125, 40):
    raise ValueError(
        f"Expected LR HSI=(125,125,40), got {img_lr_hsi.shape}"
    )


if not GUIDE_H5.exists():
    raise FileNotFoundError(
        "Run 01_prepare_cross_sensor_hr_guide.py first:\n"
        + str(GUIDE_H5)
    )

with h5py.File(GUIDE_H5, "r") as f:
    img_hr_guide = np.float32(
        f["hr_guide"][:]
    )

    guide_nir_actual_nm = float(
        f.attrs["nir_actual_nm"]
    )

if img_hr_guide.shape != (1000, 1000, 4):
    raise ValueError(
        f"Expected HR guide=(1000,1000,4), got {img_hr_guide.shape}"
    )

print("=" * 100)
print("REAL CROSS-SENSOR WHOLE-FIELD GRAM HSI-SR")
print("=" * 100)
print("HR guide:", img_hr_guide.shape)
print("LR HSI  :", img_lr_hsi.shape)
print("HR HSI  :", img_hr_hsi.shape)
print("NIR guide wavelength:", guide_nir_actual_nm)
print("HR pixels used in ONE global mapper/Gram:", 1000 * 1000)
print("LR pixels used in ONE global Gram:", 125 * 125)
print("Plot-by-plot training: NO")


encoder_checkpoint = resolve_checkpoint(
    ENCODER_CHECKPOINT_NAME
)

decoder_checkpoint = resolve_checkpoint(
    DECODER_CHECKPOINT_NAME
)

encoder = (
    EncoderLRHSI()
    .to(device)
    .float()
)

decoder = (
    DecoderHSI()
    .to(device)
    .float()
)

load_module_checkpoint(
    encoder,
    encoder_checkpoint,
    device,
)

load_module_checkpoint(
    decoder,
    decoder_checkpoint,
    device,
)

encoder.eval()
decoder.eval()

for parameter in encoder.parameters():
    parameter.requires_grad_(False)

for parameter in decoder.parameters():
    parameter.requires_grad_(False)

print("Frozen encoder/decoder loaded.")


input_lr_hsi_raw = torch.tensor(
    img_lr_hsi.reshape(
        -1,
        40,
    ),
    dtype=torch.float32,
)

input_lr_hsi_raw_mean = (
    input_lr_hsi_raw
    .mean(
        0,
        keepdim=True,
    )
)

input_lr_hsi_centered = (
    input_lr_hsi_raw
    - input_lr_hsi_raw_mean
)

input_lr_hsi_var = (
    input_lr_hsi_centered
    .to(device)
)

with torch.no_grad():
    out_LR_hsi_v = encoder(
        input_lr_hsi_var
    )

    out_LR_img_s = get_stick_segments(
        out_LR_hsi_v
    ).clamp(
        1e-8,
        1.0 - 1e-8,
    )

    (
        lr_gram_centered,
        lr_gram_diagonal,
    ) = centered_gram(
        out_LR_img_s
    )

print(
    "Global LR abundance shape:",
    tuple(out_LR_img_s.shape),
)


input_hr_guide_raw = torch.tensor(
    img_hr_guide.reshape(
        -1,
        4,
    ),
    dtype=torch.float32,
)

input_hr_guide_raw_mean = (
    input_hr_guide_raw.mean(
        0,
        keepdim=True,
    )
)

input_hr_guide_centered = (
    input_hr_guide_raw
    - input_hr_guide_raw_mean
)

input_hr_guide_var = (
    input_hr_guide_centered
    .to(device)
)

print(
    "Global HR guide matrix:",
    tuple(input_hr_guide_var.shape),
)

print(
    "HR guide channel means:",
    input_hr_guide_raw_mean.squeeze().numpy(),
)


input_hr_hsi_raw = torch.tensor(
    img_hr_hsi.reshape(
        -1,
        40,
    ),
    dtype=torch.float32,
)

input_hr_hsi_raw_mean = (
    input_hr_hsi_raw.mean(
        0,
        keepdim=True,
    )
)

gt_mean_gpu = (
    input_hr_hsi_raw_mean
    .to(device)
)


mapper = (
    InputToHSIMapper(
        n_inputs=4,
        n_outputs=40,
    )
    .to(device)
    .float()
)

print(
    "Mapper repeat counts:",
    mapper.repeat_counts
    .detach()
    .cpu()
    .tolist(),
)

optimizer = torch.optim.Adam(
    mapper.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


best_path = (
    OUTPUT_ROOT
    / "best_mapper_unsupervised.pth"
)

latest_path = (
    OUTPUT_ROOT
    / "latest_mapper.pth"
)

history_path = (
    OUTPUT_ROOT
    / "history.csv"
)

result_path = (
    OUTPUT_ROOT
    / "result.json"
)

diagnostic_path = (
    OUTPUT_ROOT
    / "diagnostic_best_hr_metrics.json"
)

reconstruction_path = (
    OUTPUT_ROOT
    / "cross_sensor_reconstruction_40band.h5"
)


if (
    SKIP_COMPLETED
    and result_path.exists()
):
    previous = json.loads(
        result_path.read_text(
            encoding="utf-8"
        )
    )

    if previous.get("status") == "complete":
        print("Already complete:")
        print(result_path)
        raise SystemExit


start_epoch = 0

best_loss = np.inf
saved_gram_sam = np.inf
saved_gram_psnr = -np.inf
saved_epoch = -1

plateau_gram_sam = np.inf
plateau_epoch = 0

recon_window_reference = None
recon_window_reference_epoch = None
recon_relative_change = float("nan")
recon_stable_windows = 0

diagnostic_best_full_sam = np.inf
diagnostic_best_plot_sam = np.inf
diagnostic_best_full_sam_epoch = -1
diagnostic_best_plot_sam_epoch = -1


if (
    RESUME_INCOMPLETE
    and latest_path.exists()
    and not result_path.exists()
):
    print("[resume] loading:", latest_path)

    payload = torch.load(
        latest_path,
        map_location=device,
    )

    mapper.load_state_dict(
        payload["state_dict"]
    )

    optimizer.load_state_dict(
        payload["optimizer"]
    )

    start_epoch = (
        int(payload["epoch"])
        + 1
    )

    best_loss = float(
        payload.get(
            "best_loss",
            np.inf,
        )
    )

    saved_gram_sam = float(
        payload.get(
            "saved_gram_sam",
            np.inf,
        )
    )

    saved_gram_psnr = float(
        payload.get(
            "saved_gram_psnr",
            -np.inf,
        )
    )

    saved_epoch = int(
        payload.get(
            "saved_epoch",
            -1,
        )
    )

    plateau_gram_sam = float(
        payload.get(
            "plateau_gram_sam",
            np.inf,
        )
    )

    plateau_epoch = int(
        payload.get(
            "plateau_epoch",
            start_epoch,
        )
    )

    diagnostic_best_full_sam = float(
        payload.get(
            "diagnostic_best_full_sam",
            np.inf,
        )
    )

    diagnostic_best_plot_sam = float(
        payload.get(
            "diagnostic_best_plot_sam",
            np.inf,
        )
    )

    diagnostic_best_full_sam_epoch = int(
        payload.get(
            "diagnostic_best_full_sam_epoch",
            -1,
        )
    )

    diagnostic_best_plot_sam_epoch = int(
        payload.get(
            "diagnostic_best_plot_sam_epoch",
            -1,
        )
    )

    print("[resume] starting epoch:", start_epoch)


history_fields = [
    "epoch",
    "total_loss",
    "gram_sam",
    "gram_psnr",
    "best_loss",
    "best_epoch",
    "recon_SAM_full",
    "recon_PSNR_full",
    "recon_ERGAS_full",
    "recon_RMSE_full",
    "recon_MAE_full",
    "recon_SAM_plot",
    "recon_PSNR_plot",
    "recon_ERGAS_plot",
    "recon_RMSE_plot",
    "recon_MAE_plot",
    "diagnostic_best_full_sam",
    "diagnostic_best_full_sam_epoch",
    "diagnostic_best_plot_sam",
    "diagnostic_best_plot_sam_epoch",
    "recon_relative_change",
    "recon_stable_windows",
    "reconstruction_l21",
    "gram_l21",
    "gram_ergas",
    "gram_sam_loss",
    "gram_sam_t",
    "diag_l21",
    "diag_sam",
    "seconds",
]


run_start_time = time.time()
last_epoch = start_epoch - 1

for epoch in range(
    start_epoch,
    MAX_EPOCHS,
):
    last_epoch = epoch

    optimizer.zero_grad()

    mapped_hsi = mapper(
        input_hr_guide_var
    ).clamp(
        -1.0 + 1e-8,
        1.0 - 1e-8,
    )

    out_HR_v = encoder(
        mapped_hsi
    )
