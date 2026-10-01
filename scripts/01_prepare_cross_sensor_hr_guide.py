#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare the full-field cross-sensor HR guide for Gram HSI super-resolution.

Output:
    Sony RGB            : 1000 x 1000 x 3
    Mjolnir NIR (~801)  : 1000 x 1000 x 1
    HR guide            : 1000 x 1000 x 4

Design principles:
- Use the 100 Mjolnir-covered plots as the master set.
- Match each Mjolnir plot to the Sony plot with the same geographic center.
- Stack all 100 matched plots once into one deterministic 10 x 10 field composite.
- Save the manifest and ALWAYS reuse it in downstream scripts.
- Do not train/decompose plot-by-plot.
- Keep Sony RGB in [0,1] by /255 for 8-bit input.
- Keep the observed Mjolnir NIR band on the existing prepared-HSI normalization.
"""

from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
import matplotlib.pyplot as plt


# =============================================================================
# SETTINGS
# =============================================================================

ROOT = Path.home() / "RGB2msi"

RGB_PLOT_DIR = ROOT / "sony_a5100" / "plots"
MJOLNIR_PLOT_DIR = ROOT / "hyspex_mjolnir1024" / "plots"

MJOLNIR_H5 = (
    ROOT
    / "hyspex_mjolnir1024"
    / "mjolnir_prepared"
    / "mjolnir_field_40band.h5"
)

OUT_DIR = (
    ROOT
    / "hyspex_mjolnir1024"
    / "sony_cross_sensor"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_H5 = OUT_DIR / "cross_sensor_hr_guide_1000x1000.h5"
OUT_TIF = OUT_DIR / "sony_hr_rgb_1000x1000.tif"
MANIFEST_CSV = OUT_DIR / "sony_mjolnir_plot_stack_manifest.csv"
SETTINGS_JSON = OUT_DIR / "cross_sensor_hr_guide_settings.json"

TARGET_NIR_NM = 801.0
TARGET_CRS = "EPSG:25832"

# If True and a manifest already exists, reuse its exact row/column order.
REUSE_EXISTING_MANIFEST = True

# Save a visual sanity check.
SAVE_PREVIEW = True


# =============================================================================
# HELPERS
# =============================================================================

def list_tiffs(folder):
    return sorted(
        list(folder.glob("*.tif"))
        + list(folder.glob("*.tiff"))
    )


def raster_info(path, target_crs=TARGET_CRS):
    with rasterio.open(path) as src:
        bounds = transform_bounds(
            src.crs,
            target_crs,
            *src.bounds,
            densify_pts=21,
        )

        cx = (bounds[0] + bounds[2]) / 2.0
        cy = (bounds[1] + bounds[3]) / 2.0

        return {
            "path": path,
            "name": path.name,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs),
            "res_x": abs(src.transform.a),
            "res_y": abs(src.transform.e),
            "bounds_common": bounds,
            "cx": cx,
            "cy": cy,
        }


def read_rgb(path):
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f"RGB file has <3 bands: {path}")

        arr = src.read([1, 2, 3])

    return np.moveaxis(arr, 0, -1)


def percentile_stretch_rgb(rgb, p_low=2, p_high=98):
    out = np.zeros_like(rgb, dtype=np.float32)

    for c in range(3):
        x = rgb[:, :, c].astype(np.float32)
        valid = np.isfinite(x)

        if not np.any(valid):
            continue

        lo = np.percentile(x[valid], p_low)
        hi = np.percentile(x[valid], p_high)

        if hi <= lo:
            continue

        out[:, :, c] = np.clip(
            (x - lo) / (hi - lo),
            0,
            1,
        )

    return out


# =============================================================================
# LOAD PREPARED MJOLNIR HSI AND FIND THE OBSERVED NIR BAND
# =============================================================================

if not MJOLNIR_H5.exists():
    raise FileNotFoundError(f"Missing prepared Mjolnir HDF5:\n{MJOLNIR_H5}")

with h5py.File(MJOLNIR_H5, "r") as f:
    hsi_hr = np.float32(f["hsi_hr"][:1000, :1000, :])
    wavelengths = np.float32(f["selected_wavelengths_nm"][:])

if hsi_hr.shape != (1000, 1000, 40):
    raise ValueError(f"Expected Mjolnir HR=(1000,1000,40), got {hsi_hr.shape}")

if wavelengths.shape != (40,):
    raise ValueError(f"Expected 40 wavelengths, got {wavelengths.shape}")

nir_index = int(np.argmin(np.abs(wavelengths - TARGET_NIR_NM)))
nir_actual_nm = float(wavelengths[nir_index])

img_hr_nir = hsi_hr[:, :, nir_index:nir_index + 1].astype(
    np.float32,
    copy=True,
)

print("=" * 90)
print("PREPARE CROSS-SENSOR HR GUIDE")
print("=" * 90)
print("Mjolnir HR-HSI:", hsi_hr.shape)
print(
    "NIR requested {:.3f} nm -> 40-band index {} -> actual {:.3f} nm".format(
        TARGET_NIR_NM,
        nir_index,
        nir_actual_nm,
    )
)


# =============================================================================
# INVENTORY
# =============================================================================

rgb_files = list_tiffs(RGB_PLOT_DIR)
mjo_files = list_tiffs(MJOLNIR_PLOT_DIR)

print("\nSony plots:", len(rgb_files))
print("Mjolnir plots:", len(mjo_files))

rgb_info = [raster_info(x) for x in rgb_files]
mjo_info = [raster_info(x) for x in mjo_files]

for item in rgb_info:
    if (
        item["width"] != 100
        or item["height"] != 100
        or item["bands"] < 3
    ):
        raise ValueError(
            "Unexpected Sony plot: "
            f"{item['name']} "
            f"{item['width']}x{item['height']} "
            f"bands={item['bands']}"
        )

for item in mjo_info:
    if (
        item["width"] != 100
        or item["height"] != 100
    ):
        raise ValueError(
            "Unexpected Mjolnir plot: "
            f"{item['name']} "
            f"{item['width']}x{item['height']}"
        )

if len(mjo_info) != 100:
    raise ValueError(
        f"Expected 100 Mjolnir-covered plots, found {len(mjo_info)}."
    )


# =============================================================================
# BUILD OR REUSE THE EXACT 10 x 10 MANIFEST
# =============================================================================

rgb_by_name = {x["name"]: x for x in rgb_info}
mjo_by_name = {x["name"]: x for x in mjo_info}

if REUSE_EXISTING_MANIFEST and MANIFEST_CSV.exists():
    print("\nReusing existing manifest:")
    print(MANIFEST_CSV)

    manifest_df = pd.read_csv(MANIFEST_CSV)

    required_columns = {
        "row",
        "column",
        "rgb_file",
        "mjolnir_file",
        "center_distance_m",
    }

    missing = required_columns - set(manifest_df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    if len(manifest_df) != 100:
        raise ValueError(
            f"Expected 100 rows in manifest, found {len(manifest_df)}"
        )

    for _, row in manifest_df.iterrows():
        if row["rgb_file"] not in rgb_by_name:
            raise FileNotFoundError(
                f"RGB file from manifest not found: {row['rgb_file']}"
            )
        if row["mjolnir_file"] not in mjo_by_name:
            raise FileNotFoundError(
                f"Mjolnir file from manifest not found: {row['mjolnir_file']}"
            )

else:
    print("\nCreating manifest from geographic centers...")

    pairs = []
    used_rgb = set()

    for m in mjo_info:
        candidates = []

        for j, r in enumerate(rgb_info):
            if j in used_rgb:
                continue

            distance = np.hypot(
                m["cx"] - r["cx"],
                m["cy"] - r["cy"],
            )

            candidates.append((distance, j, r))

        if not candidates:
            raise RuntimeError("No unused Sony candidate available.")

        candidates.sort(key=lambda x: x[0])
        distance, j, r = candidates[0]

        pairs.append(
            {
                "mjolnir": m,
                "rgb": r,
                "distance_m": float(distance),
            }
        )
        used_rgb.add(j)

    distances = np.array(
        [x["distance_m"] for x in pairs],
        dtype=np.float64,
    )

    print("\nMatching center distance:")
    print(" min   :", distances.min())
    print(" mean  :", distances.mean())
    print(" median:", np.median(distances))
    print(" max   :", distances.max())

    if distances.max() > 2.0:
        raise ValueError(
            "At least one RGB/Mjolnir plot-center distance exceeds 2 m. "
            "Inspect matching before continuing."
        )

    pairs_sorted_y = sorted(
        pairs,
        key=lambda p: -p["mjolnir"]["cy"],
    )

    rows = []

    for row_idx in range(10):
        row = pairs_sorted_y[
            row_idx * 10:
            (row_idx + 1) * 10
        ]

        if len(row) != 10:
            raise ValueError(
                f"Row {row_idx} contains {len(row)} plots, expected 10."
            )

        row = sorted(
            row,
            key=lambda p: p["mjolnir"]["cx"],
        )
        rows.append(row)

    manifest = []

    for row_idx, row in enumerate(rows):
        for col_idx, pair in enumerate(row):
            manifest.append(
                {
                    "row": row_idx,
                    "column": col_idx,
                    "rgb_file": pair["rgb"]["name"],
                    "mjolnir_file": pair["mjolnir"]["name"],
                    "center_distance_m": pair["distance_m"],
                    "mjolnir_x": pair["mjolnir"]["cx"],
                    "mjolnir_y": pair["mjolnir"]["cy"],
                }
            )

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(MANIFEST_CSV, index=False)

    print("Saved manifest:")
    print(MANIFEST_CSV)


# =============================================================================
# VALIDATE MANIFEST GRID
# =============================================================================

if sorted(manifest_df["row"].unique().tolist()) != list(range(10)):
    raise ValueError("Manifest rows are not exactly 0..9.")

if sorted(manifest_df["column"].unique().tolist()) != list(range(10)):
    raise ValueError("Manifest columns are not exactly 0..9.")

if manifest_df.duplicated(["row", "column"]).any():
    raise ValueError("Duplicate row/column positions in manifest.")

if manifest_df["rgb_file"].duplicated().any():
    raise ValueError("Duplicate Sony RGB file in manifest.")

if manifest_df["mjolnir_file"].duplicated().any():
    raise ValueError("Duplicate Mjolnir file in manifest.")


# =============================================================================
# BUILD ONE FULL 1000 x 1000 SONY RGB IMAGE
# =============================================================================

rgb_hr_raw = np.zeros(
    (1000, 1000, 3),
    dtype=np.float32,
)

for _, row in manifest_df.iterrows():
    rr = int(row["row"])
    cc = int(row["column"])

    rgb = read_rgb(
        rgb_by_name[row["rgb_file"]]["path"]
    ).astype(np.float32)

    if rgb.shape != (100, 100, 3):
        raise ValueError(
            f"Unexpected RGB array shape for {row['rgb_file']}: {rgb.shape}"
        )

    y0 = rr * 100
    x0 = cc * 100

    rgb_hr_raw[
        y0:y0 + 100,
        x0:x0 + 100,
        :
    ] = rgb


# =============================================================================
# SONY RGB NORMALIZATION
# =============================================================================

rgb_max = float(np.nanmax(rgb_hr_raw))
rgb_min = float(np.nanmin(rgb_hr_raw))

print("\nRaw Sony RGB range:", rgb_min, "to", rgb_max)

if rgb_max <= 1.5:
    rgb_01 = rgb_hr_raw.copy()

elif rgb_max <= 255.5:
    rgb_01 = rgb_hr_raw / 255.0

else:
    raise ValueError(
        "Unexpected Sony RGB range. "
        f"Observed max={rgb_max}; expected <=255 for the current experiment."
    )

rgb_01 = np.clip(
    rgb_01,
    0,
    1,
).astype(np.float32)


# =============================================================================
# FORM THE 4-CHANNEL HR GUIDE
# =============================================================================

hr_guide = np.concatenate(
    [
        rgb_01,
        img_hr_nir,
    ],
    axis=2,
).astype(np.float32)

if hr_guide.shape != (1000, 1000, 4):
    raise ValueError(f"Unexpected HR guide shape: {hr_guide.shape}")

print("\nFull HR guide:", hr_guide.shape)
print("  channel 0 = Sony R")
print("  channel 1 = Sony G")
print("  channel 2 = Sony B")
print("  channel 3 = Mjolnir {:.3f} nm".format(nir_actual_nm))


# =============================================================================
# SAVE HDF5
# =============================================================================

with h5py.File(OUT_H5, "w") as f:
    f.create_dataset(
        "sony_rgb_raw",
        data=rgb_hr_raw,
        compression="gzip",
        compression_opts=4,
    )

    f.create_dataset(
        "sony_rgb_01",
        data=rgb_01,
        compression="gzip",
        compression_opts=4,
    )

    f.create_dataset(
        "observed_nir",
        data=img_hr_nir,
        compression="gzip",
        compression_opts=4,
    )

    f.create_dataset(
        "hr_guide",
        data=hr_guide,
        compression="gzip",
        compression_opts=4,
    )

    f.create_dataset(
        "wavelengths_nm",
        data=wavelengths,
    )

    f.attrs["nir_requested_nm"] = TARGET_NIR_NM
    f.attrs["nir_actual_nm"] = nir_actual_nm
    f.attrs["nir_40band_index"] = nir_index
    f.attrs["sony_normalization"] = "divide_by_255"
    f.attrs["nir_normalization"] = "unchanged_prepared_mjolnir_HSI_scale"
    f.attrs["plot_layout"] = "10x10 artificial plot composite"
    f.attrs["n_plots"] = 100


# =============================================================================
# SAVE SIMPLE RGB TIFF
# =============================================================================

profile = {
    "driver": "GTiff",
    "height": 1000,
    "width": 1000,
    "count": 3,
    "dtype": "float32",
    "compress": "deflate",
}

with rasterio.open(
    OUT_TIF,
    "w",
    **profile,
) as dst:
    for b in range(3):
        dst.write(
            rgb_01[:, :, b],
            b + 1,
        )


# =============================================================================
# SETTINGS / PROVENANCE
# =============================================================================

settings = {
    "rgb_plot_dir": str(RGB_PLOT_DIR),
    "mjolnir_plot_dir": str(MJOLNIR_PLOT_DIR),
    "mjolnir_h5": str(MJOLNIR_H5),
    "output_h5": str(OUT_H5),
    "manifest_csv": str(MANIFEST_CSV),
    "n_sony_plots_available": len(rgb_files),
    "n_mjolnir_plots_used": len(mjo_files),
    "n_matched_plots": len(manifest_df),
    "target_nir_nm": TARGET_NIR_NM,
    "actual_nir_nm": nir_actual_nm,
    "nir_40band_index": nir_index,
    "sony_normalization": "divide_by_255",
    "nir_normalization": "unchanged_prepared_mjolnir_HSI_scale",
    "global_whole_image_processing": True,
    "plot_by_plot_training": False,
}

SETTINGS_JSON.write_text(
    json.dumps(settings, indent=2),
    encoding="utf-8",
)


# =============================================================================
# PREVIEW
# =============================================================================

if SAVE_PREVIEW:
    preview = percentile_stretch_rgb(rgb_01)

    plt.figure(figsize=(10, 10))
    plt.imshow(preview)
    plt.title(
        "Sony HR-RGB — 100 matched Mjolnir plots\n"
        "full 1000x1000 composite"
    )
    plt.axis("off")
    plt.tight_layout()

    preview_path = OUT_DIR / "sony_hr_rgb_preview.png"
    plt.savefig(
        preview_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()

    print("Preview:", preview_path)


print("\nDONE")
print("Sony RGB:", rgb_01.shape)
print("Observed NIR:", img_hr_nir.shape)
print("HR guide:", hr_guide.shape)
print("Saved:", OUT_H5)
print("Manifest:", MANIFEST_CSV)
print("Settings:", SETTINGS_JSON)
