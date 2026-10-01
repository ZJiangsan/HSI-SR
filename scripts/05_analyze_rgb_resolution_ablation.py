#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize RGB-resolution ablation after the x2/x4/x8 runs finish.

Important:
- x1 uses the already-finished final BGR+802 experiment.
- x2/x4/x8 use the parameterized ablation output folders.
- LR-HSI and HR NIR are fixed in every experiment.
"""

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path.home() / "RGB2msi" / "hyspex_mjolnir1024"

RUNS = {
    1: ROOT / "sony_rgb_plus_801_cross_sensor_gram_seed0" / "result.json",
    2: ROOT / "rgb_resolution_ablation" / "sony_bgr_plus_802_rgb_x2_seed0" / "result.json",
    4: ROOT / "rgb_resolution_ablation" / "sony_bgr_plus_802_rgb_x4_seed0" / "result.json",
    8: ROOT / "rgb_resolution_ablation" / "sony_bgr_plus_802_rgb_x8_seed0" / "result.json",
}

OUT_DIR = ROOT / "rgb_resolution_ablation" / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for factor, result_path in RUNS.items():
    if not result_path.exists():
        print("[missing]", result_path)
        continue

    with open(result_path, "r", encoding="utf-8") as f:
        r = json.load(f)

    full = r["official_checkpoint_diagnostic_full"]
    plot = r["official_checkpoint_diagnostic_plot"]

    rows.append(
        {
            "rgb_downsample_factor": factor,
            "rgb_native_pixels": 1000 // factor,
            "FULL_SAM": full["SAM"],
            "FULL_PSNR": full["PSNR"],
            "FULL_ERGAS": full["ERGAS"],
            "FULL_RMSE": full["RMSE"],
            "FULL_MAE": full["MAE"],
            "PLOT_SAM": plot["SAM"],
            "PLOT_PSNR": plot["PSNR"],
            "PLOT_ERGAS": plot["ERGAS"],
            "PLOT_RMSE": plot["RMSE"],
            "PLOT_MAE": plot["MAE"],
            "best_epoch_unsupervised": r["best_epoch_unsupervised"],
        }
    )

df = pd.DataFrame(rows).sort_values("rgb_downsample_factor")

csv_path = OUT_DIR / "rgb_resolution_ablation_summary.csv"
df.to_csv(csv_path, index=False)

print("\n" + "=" * 100)
print("RGB RESOLUTION ABLATION")
print("=" * 100)
print(df.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.plot(
    df["rgb_native_pixels"],
    df["FULL_SAM"],
    marker="o",
    label="Full field",
)
plt.plot(
    df["rgb_native_pixels"],
    df["PLOT_SAM"],
    marker="o",
    label="Plot mask",
)
plt.gca().invert_xaxis()
plt.xlabel("Sony RGB native image size (pixels per side)")
plt.ylabel("SAM (degrees)")
plt.title("Effect of Sony RGB spatial resolution")
plt.legend()
plt.tight_layout()

sam_path = OUT_DIR / "sam_vs_rgb_resolution.png"
plt.savefig(sam_path, dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(
    df["rgb_native_pixels"],
    df["FULL_PSNR"],
    marker="o",
    label="Full field",
)
plt.plot(
    df["rgb_native_pixels"],
    df["PLOT_PSNR"],
    marker="o",
    label="Plot mask",
)
plt.gca().invert_xaxis()
plt.xlabel("Sony RGB native image size (pixels per side)")
plt.ylabel("PSNR (dB)")
plt.title("Effect of Sony RGB spatial resolution")
plt.legend()
plt.tight_layout()

psnr_path = OUT_DIR / "psnr_vs_rgb_resolution.png"
plt.savefig(psnr_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved:")
print(csv_path)
print(sam_path)
print(psnr_path)
