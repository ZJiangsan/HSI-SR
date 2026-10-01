#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ResTrans21 re-training on the current 40-band Mjolnir dataset.

Architecture
------------
This implements the ResTrans21 model from the uploaded historical script as
closely as possible, with ONLY the required spectral input adaptation:

    SimpleViT channels: 200 -> 40

All downstream dimensions remain unchanged because the SimpleViT output stays
800-dimensional.

Historical architecture retained:
- SimpleViT image_size=(15,100), patch_size=(3,10)
- num_classes=800, dim=800, depth=6, heads=6, mlp_dim=800
- dense concatenation layers conv9 ... conv15
- custom one-token TransformerBlock modules
- feature chain conv23_0 ... conv23_5
- output-head dropout:
      DM = 0.01
      NC = 0.50
      NU = 0.01
  exactly matching the uploaded version.

Training protocol
-----------------
The historical architecture is retained, but the CURRENT clean evaluation
protocol is used:
- train only on genuine Mjolnir HSI
- standard 5-fold GroupKFold by parent plot
- four folds used entirely for training; one fold held out for evaluation
- no inner validation split
- fixed 150-epoch training schedule
- input normalization fitted only on genuine training HSI
- target min-max normalization fitted only on training labels
- held-out fold never used for tuning/checkpoint selection
- frozen model evaluated on:
      genuine
      bgr_only
      bgr_plus_802
- correct r2_score(y_true, y_pred)
- one Adam optimizer is preserved across epochs
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold

try:
    from vit_pytorch import SimpleViT
except Exception as e:
    raise ImportError(
        "This script requires vit-pytorch.\n"
        "Install it in the same environment, e.g.:\n"
        "    pip install vit-pytorch\n"
        f"Original import error: {e}"
    )


ROOT = Path.home() / "RGB2msi"

DATA_DIR = (
    ROOT
    / "hyspex_mjolnir1024"
    / "cross_sensor_downstream_sections"
)

MATCHED_H5 = DATA_DIR / "matched_sections_40band.h5"
MANIFEST_CSV = DATA_DIR / "matched_sections_manifest.csv"
LABEL_CSV = ROOT / "groundtruth" / "groundtruth_sections_n.csv"

OUT_DIR = (
    ROOT
    / "hyspex_mjolnir1024"
    / "restrans21_uploaded_model_40band_old5fold_minmax"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


TARGET_COLUMNS = [
    "dm_gpermsq",
    "n-content_perc",
    "n-uptake_gpermsq",
]

TARGET_SHORT = ["DM", "NC", "NU"]

N_FOLDS = 5
SEED = 42


INPUT_CHANNELS = 40
IMAGE_SIZE = (15, 100)
PATCH_SIZE = (3, 10)

VIT_OUT = 200 * 4
VIT_DIM = 200 * 4
VIT_DEPTH = 6
VIT_HEADS = 6
VIT_MLP_DIM = 200 * 4

DROPOUT_COMMON = 0.01
DROPOUT_FEATURE = 0.01
DROPOUT_DM = 0.01
DROPOUT_NC = 0.50
DROPOUT_NU = 0.01


BATCH_SIZE = 16
LR = 5e-5
WEIGHT_DECAY = 1e-4

FIXED_EPOCHS = 150
GRAD_CLIP = 5.0

NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def fit_x_scaler(x):
    mean = x.mean(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std = x.std(axis=(0, 1, 2), dtype=np.float64).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def fit_y_scaler(y):
    """
    Historical-style 0-1 target normalization, fitted on TRAINING FOLD ONLY.

    This reproduces the target representation used by the old ResTrans21
    pipeline without leaking outer validation/test target ranges.
    """
    ymin = y.min(axis=0).astype(np.float32)
    ymax = y.max(axis=0).astype(np.float32)
    yrange = (ymax - ymin).astype(np.float32)
    yrange[yrange < 1e-8] = 1.0
    return ymin, yrange


def compute_metrics(y_true, y_pred):
    out = {}

    for i, name in enumerate(TARGET_SHORT):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        out[name] = {
            "R2": float(r2_score(yt, yp)),
            "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
            "MAE": float(mean_absolute_error(yt, yp)),
        }

    return out


class HSIDataset(Dataset):
    def __init__(self, x, y, idx, xmean, xstd, ymin, yrange):
        self.x = x
        self.y = y
        self.idx = np.asarray(idx, dtype=np.int64)
        self.xmean = np.asarray(xmean, dtype=np.float32)
        self.xstd = np.asarray(xstd, dtype=np.float32)
        self.ymin = np.asarray(ymin, dtype=np.float32)
        self.yrange = np.asarray(yrange, dtype=np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])

        cube = np.asarray(self.x[i], dtype=np.float32)

        cube = (
            cube - self.xmean[None, None, :]
        ) / self.xstd[None, None, :]

        cube = np.moveaxis(cube, -1, 0).copy()

        target = (
            (self.y[i] - self.ymin) / self.yrange
        ).astype(np.float32)

        return (
            torch.from_numpy(cube),
            torch.from_numpy(target),
            i,
        )


def make_loader(x, y, idx, xmean, xstd, ymin, yrange, shuffle):
    ds = HSIDataset(
        x=x,
        y=y,
        idx=idx,
        xmean=xmean,
        xstd=xstd,
        ymin=ymin,
        yrange=yrange,
    )

    return DataLoader(
        ds,
        batch_size=min(BATCH_SIZE, len(ds)),
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        drop_last=False,
    )


class SelfAttention(nn.Module):
    def __init__(self, emb, heads=1, mask=False, kqnorm=False):
        super().__init__()

        assert emb % heads == 0, (
            f"Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})"
        )

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads

        self.tokeys = nn.Linear(emb, emb, bias=False)
        self.toqueries = nn.Linear(emb, emb, bias=False)
        self.tovalues = nn.Linear(emb, emb, bias=False)

        self.unifyheads = nn.Linear(emb, emb)

        self.kqnorm = kqnorm

        if kqnorm:
            self.kln = nn.LayerNorm([s])
            self.qln = nn.LayerNorm([s])

    def forward(self, x):
        b, t, e = x.size()
        h = self.heads
        s = e // h

        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)

        keys = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values = values.view(b, t, h, s)

        if self.kqnorm:
            keys = self.kln(keys)
            queries = self.qln(queries)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1 / 4))
        keys = keys / (e ** (1 / 4))

        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention1 = SelfAttention(
            k,
            heads=heads,
            kqnorm=True,
        )

        self.norm1 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k),
        )

        self.norm_ff = nn.LayerNorm(k)

        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        attended_1 = self.attention1(x)
        x = self.norm1(attended_1 + x)

        fedforward = self.ff(x)
        x = self.norm_ff(fedforward + x)

        x = self.dropout(x)

        return x


class ResTrans21(nn.Module):
    def __init__(self, pool_size=2):
        super().__init__()

        self.vit = SimpleViT(
            image_size=IMAGE_SIZE,
            patch_size=PATCH_SIZE,
            num_classes=VIT_OUT,
            dim=VIT_DIM,
            depth=VIT_DEPTH,
            heads=VIT_HEADS,
            mlp_dim=VIT_MLP_DIM,
            channels=INPUT_CHANNELS,
        )

        self.conv9 = nn.Linear(100 * 8, 200, bias=False)
        self.bn9 = nn.BatchNorm1d(200)
        self.relu9 = nn.ReLU()
        self.pool9 = nn.AvgPool1d(2)
        self.transformer9 = TransformerBlock(100, 1)

        self.conv10 = nn.Linear(100 * 9, 200, bias=False)
        self.bn10 = nn.BatchNorm1d(200)
        self.relu10 = nn.ReLU()
        self.pool10 = nn.AvgPool1d(2)
        self.transformer10 = TransformerBlock(100, 1)

        self.conv11 = nn.Linear(100 * 10, 200, bias=False)
        self.bn11 = nn.BatchNorm1d(200)
        self.relu11 = nn.ReLU()
        self.pool11 = nn.AvgPool1d(2)
        self.transformer11 = TransformerBlock(100, 1)

        self.conv12 = nn.Linear(100 * 11, 200, bias=False)
        self.bn12 = nn.BatchNorm1d(200)
        self.relu12 = nn.ReLU()
        self.pool12 = nn.AvgPool1d(2)
        self.transformer12 = TransformerBlock(100, 1)

        self.conv13 = nn.Linear(100 * 12, 200, bias=False)
        self.bn13 = nn.BatchNorm1d(200)
        self.relu13 = nn.ReLU()
        self.pool13 = nn.AvgPool1d(2)
        self.transformer13 = TransformerBlock(100, 1)

        self.conv14 = nn.Linear(100 * 13, 200, bias=False)
        self.bn14 = nn.BatchNorm1d(200)
        self.relu14 = nn.ReLU()
        self.pool14 = nn.AvgPool1d(2)
        self.transformer14 = TransformerBlock(100, 1)

        self.conv15 = nn.Linear(100 * 6, 400, bias=False)
        self.bn15 = nn.BatchNorm1d(400)
        self.relu15 = nn.ReLU()
        self.pool15 = nn.AvgPool1d(2)
        self.transformer15 = TransformerBlock(200, 1)

        self.dropout = nn.Dropout(DROPOUT_COMMON)
        self.dropout_f = nn.Dropout(DROPOUT_FEATURE)

        self.dropout_dm = nn.Dropout(DROPOUT_DM)
        self.dropout_nc = nn.Dropout(DROPOUT_NC)
        self.dropout_nu = nn.Dropout(DROPOUT_NU)

        self.conv22 = nn.Linear(100, 1, bias=False)

        self.conv23_0 = nn.Linear(100 * 2, 200, bias=False)
        self.bn23_0 = nn.BatchNorm1d(200)
        self.relu23_0 = nn.ReLU()
        self.pool23_0 = nn.AvgPool1d(2)
        self.transformer23_0 = TransformerBlock(100, 1)

        self.conv23_1 = nn.Linear(100 * 3, 200, bias=False)
        self.bn23_1 = nn.BatchNorm1d(200)
        self.relu23_1 = nn.ReLU()
        self.pool23_1 = nn.AvgPool1d(2)
        self.transformer23_1 = TransformerBlock(100, 1)

        self.conv23_2 = nn.Linear(100 * 4, 200, bias=False)
        self.bn23_2 = nn.BatchNorm1d(200)
        self.relu23_2 = nn.ReLU()
        self.pool23_2 = nn.AvgPool1d(2)
        self.transformer23_2 = TransformerBlock(100, 1)

        self.conv23_3 = nn.Linear(100 * 5, 200, bias=False)
        self.bn23_3 = nn.BatchNorm1d(200)
        self.relu23_3 = nn.ReLU()
        self.pool23_3 = nn.AvgPool1d(2)
        self.transformer23_3 = TransformerBlock(100, 1)

        self.conv23_4 = nn.Linear(100 * 6, 200, bias=False)
        self.bn23_4 = nn.BatchNorm1d(200)
        self.relu23_4 = nn.ReLU()
        self.pool23_4 = nn.AvgPool1d(2)
        self.transformer23_4 = TransformerBlock(100, 1)

        self.conv23_5 = nn.Linear(100 * 7, 200, bias=False)
        self.bn23_5 = nn.BatchNorm1d(200)
        self.relu23_5 = nn.ReLU()
        self.pool23_5 = nn.AvgPool1d(2)
        self.transformer23_5 = TransformerBlock(100, 1)

        self.conv23 = nn.Linear(100, 1, bias=False)
        self.conv24 = nn.Linear(100, 1, bias=False)

        self.relu = nn.ReLU()

    @staticmethod
    def token_block(block, x):
        return block(x.unsqueeze(1)).squeeze(1)

    def forward(self, x):
        layer_1 = self.vit(x)

        layer_9 = self.relu9(self.bn9(self.conv9(layer_1)))
        layer_9 = self.pool9(layer_9)
        layer_9 = self.token_block(self.transformer9, layer_9)
        layer_9 = self.dropout(layer_9)
        cat_9 = torch.cat((layer_1, layer_9), dim=1)

        layer_10 = self.relu10(self.bn10(self.conv10(cat_9)))
        layer_10 = self.pool10(layer_10)
        layer_10 = self.token_block(self.transformer10, layer_10)
        layer_10 = self.dropout(layer_10)
        cat_10 = torch.cat((cat_9, layer_10), dim=1)

        layer_11 = self.relu11(self.bn11(self.conv11(cat_10)))
        layer_11 = self.pool11(layer_11)
        layer_11 = self.token_block(self.transformer11, layer_11)
        layer_11 = self.dropout(layer_11)
        cat_11 = torch.cat((cat_10, layer_11), dim=1)

        layer_12 = self.relu12(self.bn12(self.conv12(cat_11)))
        layer_12 = self.pool12(layer_12)
        layer_12 = self.token_block(self.transformer12, layer_12)
        layer_12 = self.dropout(layer_12)
        cat_12 = torch.cat((cat_11, layer_12), dim=1)

        layer_13 = self.relu13(self.bn13(self.conv13(cat_12)))
        layer_13 = self.pool13(layer_13)
        layer_13 = self.token_block(self.transformer13, layer_13)
        layer_13 = self.dropout(layer_13)
        cat_13 = torch.cat((cat_12, layer_13), dim=1)

        layer_14 = self.relu14(self.bn14(self.conv14(cat_13)))
        layer_14 = self.pool14(layer_14)
        layer_14 = self.token_block(self.transformer14, layer_14)
        layer_14 = self.dropout(layer_14)

        cat_1_x = torch.cat(
            (
                layer_9,
                layer_10,
                layer_11,
                layer_12,
                layer_13,
                layer_14,
            ),
            dim=1,
        )

        layer_15 = self.relu15(self.bn15(self.conv15(cat_1_x)))
        layer_15 = self.pool15(layer_15)
        layer_15 = self.token_block(self.transformer15, layer_15)
        layer_15 = self.dropout(layer_15)

        layer_23_0 = self.relu23_0(
            self.bn23_0(
                self.conv23_0(layer_15)
            )
        )
        layer_23_0 = self.dropout_f(layer_23_0)
        layer_23_0 = self.pool23_0(layer_23_0)
        layer_23_0 = self.token_block(self.transformer23_0, layer_23_0)
        cat_23_0 = torch.cat((layer_15, layer_23_0), dim=1)

        layer_23_1 = self.relu23_1(
            self.bn23_1(
                self.conv23_1(cat_23_0)
            )
        )
        layer_23_1 = self.dropout_f(layer_23_1)
        layer_23_1 = self.pool23_1(layer_23_1)
        layer_23_1 = self.token_block(self.transformer23_1, layer_23_1)
        cat_23_1 = torch.cat((cat_23_0, layer_23_1), dim=1)

        layer_23_2 = self.relu23_2(
            self.bn23_2(
                self.conv23_2(cat_23_1)
            )
        )
        layer_23_2 = self.dropout_f(layer_23_2)
        layer_23_2 = self.pool23_2(layer_23_2)
        layer_23_2 = self.token_block(self.transformer23_2, layer_23_2)
        cat_23_2 = torch.cat((cat_23_1, layer_23_2), dim=1)

        layer_23_3 = self.relu23_3(
            self.bn23_3(
                self.conv23_3(cat_23_2)
            )
        )
        layer_23_3 = self.dropout_f(layer_23_3)
        layer_23_3 = self.pool23_3(layer_23_3)
        layer_23_3 = self.token_block(self.transformer23_3, layer_23_3)
        cat_23_3 = torch.cat((cat_23_2, layer_23_3), dim=1)

        layer_23_4 = self.relu23_4(
            self.bn23_4(
                self.conv23_4(cat_23_3)
            )
        )
        layer_23_4 = self.dropout_f(layer_23_4)
        layer_23_4 = self.pool23_4(layer_23_4)
        layer_23_4 = self.token_block(self.transformer23_4, layer_23_4)

        cat_23_x = torch.cat(
            (
                layer_15,
                layer_23_0,
                layer_23_1,
                layer_23_2,
                layer_23_3,
                layer_23_4,
            ),
            dim=1,
        )

        layer_23_5 = self.relu23_5(
            self.bn23_5(
                self.conv23_5(cat_23_x)
            )
        )
        layer_23_5 = self.dropout_f(layer_23_5)
        layer_23_5 = self.pool23_5(layer_23_5)
        layer_23_5 = self.token_block(self.transformer23_5, layer_23_5)

        dm = self.conv22(self.dropout_dm(layer_23_5))
        nc = self.conv23(self.dropout_nc(layer_23_5))
        nu = self.conv24(self.dropout_nu(layer_23_5))

        return torch.cat((dm, nc, nu), dim=1)


def eval_loader(model, loader, ymin, yrange):
    model.eval()

    pred_z = []
    true_z = []
    ids = []

    with torch.no_grad():
        for xb, yb, ib in loader:
            xb = xb.to(DEVICE, non_blocking=True)

            pred = model(xb)

            pred_z.append(pred.cpu().numpy())
            true_z.append(yb.numpy())
            ids.append(ib.numpy())

    pred_z = np.concatenate(pred_z, axis=0)
    true_z = np.concatenate(true_z, axis=0)
    ids = np.concatenate(ids, axis=0)

    pred = pred_z * yrange[None, :] + ymin[None, :]
    true = true_z * yrange[None, :] + ymin[None, :]

    return pred, true, ids


def train_one_fold(
    genuine,
    Y,
    train_idx,
    xmean,
    xstd,
    ymin,
    yrange,
    fold_seed,
    fold_dir,
):
    seed_everything(fold_seed)

    train_loader = make_loader(
        genuine, Y, train_idx,
        xmean, xstd, ymin, yrange,
        True,
    )

    train_eval_loader = make_loader(
        genuine, Y, train_idx,
        xmean, xstd, ymin, yrange,
        False,
    )

    model = ResTrans21().to(DEVICE)

    n_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
    print("Trainable parameters:", n_params)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    hist = []

    for epoch in range(FIXED_EPOCHS):
        model.train()
        train_sum = 0.0
        train_n = 0

        for xb, yb, _ in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            bs = xb.size(0)
            train_sum += loss.item() * bs
            train_n += bs

        train_loss = train_sum / train_n

        hist.append({
            "epoch": epoch,
            "train_loss_norm": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if epoch % 10 == 0 or epoch == FIXED_EPOCHS - 1:
            print(
                "E{:04d}/{:04d} | train={:.6f} | lr={:.3e}".format(
                    epoch,
                    FIXED_EPOCHS - 1,
                    train_loss,
                    optimizer.param_groups[0]["lr"],
                )
            )

    pred_tr, true_tr, _ = eval_loader(
        model,
        train_eval_loader,
        ymin,
        yrange,
    )

    train_metrics = compute_metrics(true_tr, pred_tr)

    pd.DataFrame(hist).to_csv(
        fold_dir / "history.csv",
        index=False,
    )

    return {
        "model": model,
        "final_epoch": int(FIXED_EPOCHS - 1),
        "train_metrics": train_metrics,
    }


for p in [MATCHED_H5, MANIFEST_CSV, LABEL_CSV]:
    if not p.exists():
        raise FileNotFoundError(p)


with h5py.File(MATCHED_H5, "r") as f:
    genuine = np.asarray(
        f["genuine_mjolnir"][:],
        dtype=np.float32,
    )

    bgr_only = np.asarray(
        f["sony_bgr_only"][:],
        dtype=np.float32,
    )

    bgr_plus_802 = np.asarray(
        f["sony_bgr_plus_802"][:],
        dtype=np.float32,
    )

    wavelengths = np.asarray(
        f["wavelengths_nm"][:],
        dtype=np.float32,
    )


manifest = pd.read_csv(MANIFEST_CSV)
label_df = pd.read_csv(LABEL_CSV)


missing = [
    c for c in TARGET_COLUMNS
    if c not in label_df.columns
]

if missing:
    raise KeyError(
        f"Missing target columns: {missing}"
    )


id_cols = list(label_df.columns[:3])

lookup = {}

for _, row in label_df.iterrows():
    key = tuple(
        int(float(row[c]))
        for c in id_cols
    )

    lookup[key] = np.asarray(
        [
            float(row[c])
            for c in TARGET_COLUMNS
        ],
        dtype=np.float32,
    )


Y = []
keep = []

for _, row in manifest.iterrows():
    key = (
        int(row["label_id_1"]),
        int(row["label_id_2"]),
        int(row["label_id_3"]),
    )

    if key not in lookup:
        print(
            "[missing label]",
            key,
            row["section_file"],
        )
        keep.append(False)
        Y.append([np.nan, np.nan, np.nan])
    else:
        keep.append(True)
        Y.append(lookup[key])


keep = np.asarray(keep, dtype=bool)
Y = np.asarray(Y, dtype=np.float32)

manifest = manifest.loc[keep].reset_index(drop=True)

genuine = genuine[keep]
bgr_only = bgr_only[keep]
bgr_plus_802 = bgr_plus_802[keep]
Y = Y[keep]

finite = np.all(np.isfinite(Y), axis=1)

manifest = manifest.loc[finite].reset_index(drop=True)

genuine = genuine[finite]
bgr_only = bgr_only[finite]
bgr_plus_802 = bgr_plus_802[finite]
Y = Y[finite]


expected = (15, 100, 40)

if genuine.shape[1:] != expected:
    raise ValueError(
        f"Expected [N,15,100,40], got {genuine.shape}"
    )


groups = (
    manifest["parent_plot"]
    .astype(str)
    .to_numpy()
)


sources = {
    "genuine": genuine,
    "bgr_only": bgr_only,
    "bgr_plus_802": bgr_plus_802,
}


print("\n" + "=" * 100)
print("RESTRANS21 — 40-BAND INPUT + HISTORICAL MIN-MAX TARGET NORMALIZATION")
print("=" * 100)

print("Sections:", len(Y))
print("Tensor:", genuine.shape)
print("Parent plots:", len(np.unique(groups)))
print("Wavelengths:", float(wavelengths.min()), "to", float(wavelengths.max()), "nm")
print("Device:", DEVICE)
print("Input channels:", INPUT_CHANNELS)
print(
    "Head dropout:",
    {
        "DM": DROPOUT_DM,
        "NC": DROPOUT_NC,
        "NU": DROPOUT_NU,
    },
)
print("Held-out fold used for training/checkpoint selection: NO")
print("Training schedule: fixed", FIXED_EPOCHS, "epochs")


all_metrics = []
all_predictions = []
selected_rows = []

gkf = GroupKFold(n_splits=N_FOLDS)


for fold, (train_idx, test_idx) in enumerate(
    gkf.split(
        genuine,
        Y,
        groups=groups,
    )
):
    print("\n" + "#" * 100)
    print(f"OUTER FOLD {fold + 1}/{N_FOLDS}")
    print("#" * 100)

    fold_seed = SEED + fold
    seed_everything(fold_seed)

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])

    if train_groups & test_groups:
        raise RuntimeError("Group leakage detected.")

    print(
        "Samples train/test:",
        len(train_idx),
        len(test_idx),
    )

    print(
        "Plots train/test:",
        len(train_groups),
        len(test_groups),
    )

    xmean, xstd = fit_x_scaler(
        genuine[train_idx]
    )

    ymin, yrange = fit_y_scaler(
        Y[train_idx]
    )

    fold_dir = OUT_DIR / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    result = train_one_fold(
        genuine=genuine,
        Y=Y,
        train_idx=train_idx,
        xmean=xmean,
        xstd=xstd,
        ymin=ymin,
        yrange=yrange,
        fold_seed=fold_seed,
        fold_dir=fold_dir,
    )

    model = result["model"]
    tm = result["train_metrics"]

    print("\nFinal fixed-epoch model:")
    print(" epoch:", result["final_epoch"])
    print(" train metrics:", tm)

    selected_rows.append({
        "fold": fold,
        "final_epoch": result["final_epoch"],
        "train_R2_DM": tm["DM"]["R2"],
        "train_R2_NC": tm["NC"]["R2"],
        "train_R2_NU": tm["NU"]["R2"],
    })

    torch.save(
        {
            "fold": fold,
            "architecture": "ResTrans21_uploaded_model_40band",
            "input_channels": INPUT_CHANNELS,
            "dropout_DM": DROPOUT_DM,
            "dropout_NC": DROPOUT_NC,
            "dropout_NU": DROPOUT_NU,
            "final_epoch": result["final_epoch"],
            "train_metrics": tm,
            "model_state_dict": model.state_dict(),
            "xmean": xmean,
            "xstd": xstd,
            "ymin": ymin,
            "yrange": yrange,
            "train_idx": train_idx,
            "test_idx": test_idx,
        },
        fold_dir / "best_model.pth",
    )

    for source_name, source_array in sources.items():
        test_loader = make_loader(
            source_array,
            Y,
            test_idx,
            xmean,
            xstd,
            ymin,
            yrange,
            False,
        )

        pred, true, ids = eval_loader(
            model,
            test_loader,
            ymin,
            yrange,
        )

        m = compute_metrics(true, pred)

        print(
            f"\nFold {fold} | source={source_name}"
        )

        for target in TARGET_SHORT:
            print(
                "  {}: R2={:.4f} RMSE={:.4f} MAE={:.4f}".format(
                    target,
                    m[target]["R2"],
                    m[target]["RMSE"],
                    m[target]["MAE"],
                )
            )

            all_metrics.append({
                "fold": fold,
                "source": source_name,
                "target": target,
                "R2": m[target]["R2"],
                "RMSE": m[target]["RMSE"],
                "MAE": m[target]["MAE"],
                "final_epoch": result["final_epoch"],
            })

        for j, sample_idx in enumerate(ids):
            sample_idx = int(sample_idx)

            row = {
                "fold": fold,
                "source": source_name,
                "sample_index": sample_idx,
                "section_file": manifest.iloc[sample_idx]["section_file"],
                "parent_plot": manifest.iloc[sample_idx]["parent_plot"],
            }

            for k, target in enumerate(TARGET_SHORT):
                row[f"{target}_true"] = float(true[j, k])
                row[f"{target}_pred"] = float(pred[j, k])

            all_predictions.append(row)


metrics_df = pd.DataFrame(all_metrics)
predictions_df = pd.DataFrame(all_predictions)
selected_df = pd.DataFrame(selected_rows)

metrics_df.to_csv(
    OUT_DIR / "fold_metrics.csv",
    index=False,
)

predictions_df.to_csv(
    OUT_DIR / "test_predictions.csv",
    index=False,
)

selected_df.to_csv(
    OUT_DIR / "selected_checkpoint_by_fold.csv",
    index=False,
)


summary = (
    metrics_df
    .groupby(["source", "target"])[["R2", "RMSE", "MAE"]]
    .agg(["mean", "std"])
)

summary.to_csv(
    OUT_DIR / "cross_fold_summary.csv"
)


metadata = {
    "model": "ResTrans21_uploaded_model_40band",

    "source_model": (
        "uploaded historical ResTrans21; "
        "SimpleViT input channels adapted from 200 to 40"
    ),

    "input_shape_HWC": [15, 100, 40],

    "simple_vit": {
        "image_size": list(IMAGE_SIZE),
        "patch_size": list(PATCH_SIZE),
        "num_classes": VIT_OUT,
        "dim": VIT_DIM,
        "depth": VIT_DEPTH,
        "heads": VIT_HEADS,
        "mlp_dim": VIT_MLP_DIM,
        "channels": INPUT_CHANNELS,
    },

    "dropout": {
        "common": DROPOUT_COMMON,
        "feature": DROPOUT_FEATURE,
        "DM": DROPOUT_DM,
        "NC": DROPOUT_NC,
        "NU": DROPOUT_NU,
    },

    "training_source": "genuine_mjolnir_only",

    "checkpoint_selection": "none; fixed epoch count",
    "fixed_epochs": FIXED_EPOCHS,

    "heldout_fold_used_for_tuning": False,
    "heldout_fold_used_for_checkpoint_selection": False,

    "split": (
        "standard 5-fold GroupKFold by parent plot; "
        "four folds used entirely for training and one fold held out for evaluation"
    ),

    "optimizer": "Adam",
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,

    "normalization": (
        "per-band input z-score fitted on genuine training HSI only; "
        "target min-max normalization fitted on training labels only; "
        "same genuine-training input statistics applied to reconstructed domains"
    ),

    "target_normalization_formula": (
        "(y - train_min) / (train_max - train_min), independently for DM/NC/NU"
    ),

    "target_inverse_formula": (
        "y = y_norm * (train_max - train_min) + train_min"
    ),

    "n_sections": int(len(Y)),
    "n_parent_plots": int(len(np.unique(groups))),
    "wavelength_min_nm": float(wavelengths.min()),
    "wavelength_max_nm": float(wavelengths.max()),
}


with open(
    OUT_DIR / "experiment_metadata.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(metadata, f, indent=2)


print("\n" + "=" * 100)
print("RESTRANS21 STANDARD 5-FOLD TRAINING COMPLETE")
print("=" * 100)

print(summary)

print("\nPer-fold final training states:")
print(selected_df.to_string(index=False))

print("\nSaved:")
print(OUT_DIR / "cross_fold_summary.csv")
print(OUT_DIR / "fold_metrics.csv")
print(OUT_DIR / "selected_checkpoint_by_fold.csv")
print(OUT_DIR / "test_predictions.csv")
print(OUT_DIR / "experiment_metadata.json")
