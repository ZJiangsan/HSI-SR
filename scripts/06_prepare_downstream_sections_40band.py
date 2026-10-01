#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare matched 40-band Mjolnir sections for downstream functional-transfer analysis.

Creates ONE matched dataset containing, for every original Mjolnir section:

    genuine_mjolnir   : original section TIFF -> exact same 40 native bands
                        used by the Gram/SR framework -> same global normalization

    sony_bgr_only     : corresponding 15x100x40 crop from the final whole-field
                        Sony-BGR-only reconstruction

    sony_bgr_plus_802 : corresponding 15x100x40 crop from the final whole-field
                        Sony-BGR + Mjolnir ~802-nm reconstruction

Important:
- The original Mjolnir section TIFFs are 200-band (or full sensor-band) products.
- We DO NOT interpolate spectra.
- We select the exact source sensor band indices stored in mjolnir_field_40band.h5.
- Genuine sections use the same reflectance quantification value and the same
  field-level normalization maximum used to create mjolnir_field_40band.h5.
- Native section orientation is preserved as H x W = 15 x 100.
- Reconstructed sections are extracted from the already-finished whole-field
  reconstructions; the Gram mapper is NOT rerun section-by-section.
"""

from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path.home() / "RGB2msi"
MJ_ROOT = ROOT / "hyspex_mjolnir1024"

SECTION_DIR = MJ_ROOT / "sections"
PLOT_DIR = MJ_ROOT / "plots"

FIELD_40_H5 = (
    MJ_ROOT
    / "mjolnir_prepared"
    / "mjolnir_field_40band.h5"
)

STACK_MANIFEST = (
    MJ_ROOT
    / "sony_cross_sensor"
    / "sony_mjolnir_plot_stack_manifest.csv"
)

BGR_ONLY_H5 = (
    MJ_ROOT
    / "sony_bgr_only_cross_sensor_gram_seed0"
    / "sony_bgr_only_reconstruction_40band.h5"
)

BGR_NIR_H5 = (
    MJ_ROOT
    / "sony_rgb_plus_801_cross_sensor_gram_seed0"
    / "cross_sensor_reconstruction_40band.h5"
)

OUT_DIR = (
    MJ_ROOT
    / "cross_sensor_downstream_sections"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_H5 = OUT_DIR / "matched_sections_40band.h5"
OUT_CSV = OUT_DIR / "matched_sections_manifest.csv"

EXPECTED_H = 15
EXPECTED_W = 100
EXPECTED_BANDS = 40


# =============================================================================
# HELPERS
# =============================================================================

def list_tiffs(folder: Path):
    return sorted(
        list(folder.glob("*.tif"))
        + list(folder.glob("*.tiff"))
    )


def read_wavelengths(ds):
    vals = []

    for b in range(1, ds.count + 1):
        tags = ds.tags(b)

        if "WAVELENGTH" in tags:
            vals.append(
                float(tags["WAVELENGTH"])
            )

        elif ds.descriptions and ds.descriptions[b - 1]:
            vals.append(
                float(ds.descriptions[b - 1])
            )

        else:
            raise ValueError(
                f"{ds.name}: missing wavelength metadata for band {b}"
            )

    return np.asarray(
        vals,
        dtype=np.float64,
    )


def read_quantification(ds):
    values = []

    probe_bands = sorted(
        set(
            [
                1,
                max(1, ds.count // 2),
                ds.count,
            ]
        )
    )

    for b in probe_bands:
        q = ds.tags(b).get(
            "SIGNAL_QUANTIFICATION_VALUE"
        )

        if q is not None:
            values.append(
                float(q)
            )

    if not values:
        raise ValueError(
            f"{ds.name}: SIGNAL_QUANTIFICATION_VALUE is missing."
        )

    if max(values) - min(values) > 1e-9:
        raise ValueError(
            f"{ds.name}: inconsistent quantification values: {values}"
        )

    return float(
        values[0]
    )


def parse_section_identity(path: Path):
    m = re.search(
        r"_(\d+)-(\d+)-(\d+)\.(?:tif|tiff)$",
        path.name,
        flags=re.IGNORECASE,
    )

    if not m:
        raise ValueError(
            f"Cannot parse section identity from: {path.name}"
        )

    a, b, c = (
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
    )

    prefix = path.name[
        :
        m.start()
    ]

    suffix = path.suffix

    parent_name = (
        f"{prefix}_{a}-{b}{suffix}"
    )

    return (
        (a, b, c),
        parent_name,
        c,
    )


def exact_section_window_inside_plot(
    section_path: Path,
    plot_path: Path,
):
    with rasterio.open(
        plot_path
    ) as plot_ds, rasterio.open(
        section_path
    ) as sec_ds:

        if sec_ds.crs != plot_ds.crs:
            raise ValueError(
                f"CRS mismatch:\n{section_path}\n{plot_path}"
            )

        win = from_bounds(
            *sec_ds.bounds,
            transform=plot_ds.transform,
        )

        y0 = int(
            round(
                win.row_off
            )
        )

        x0 = int(
            round(
                win.col_off
            )
        )

        h = int(
            sec_ds.height
        )

        w = int(
            sec_ds.width
        )

        if (
            h != EXPECTED_H
            or w != EXPECTED_W
        ):
            raise ValueError(
                f"Unexpected section raster size for {section_path.name}: "
                f"{h}x{w}; expected {EXPECTED_H}x{EXPECTED_W}"
            )

        y0 = min(
            max(
                y0,
                0,
            ),
            plot_ds.height - h,
        )

        x0 = min(
            max(
                x0,
                0,
            ),
            plot_ds.width - w,
        )

        y1 = y0 + h
        x1 = x0 + w

        return (
            y0,
            y1,
            x0,
            x1,
        )


required_files = [
    FIELD_40_H5,
    STACK_MANIFEST,
    BGR_ONLY_H5,
    BGR_NIR_H5,
]

for p in required_files:
    if not p.exists():
        raise FileNotFoundError(
            f"Missing required file:\n{p}"
        )

if not SECTION_DIR.exists():
    raise FileNotFoundError(
        f"Missing section folder:\n{SECTION_DIR}"
    )

if not PLOT_DIR.exists():
    raise FileNotFoundError(
        f"Missing plot folder:\n{PLOT_DIR}"
    )


with h5py.File(
    FIELD_40_H5,
    "r",
) as f:

    source_indices_0based = np.asarray(
        f[
            "source_sensor_band_indices_0based"
        ][:],
        dtype=np.int64,
    )

    selected_wavelengths = np.asarray(
        f[
            "selected_wavelengths_nm"
        ][:],
        dtype=np.float32,
    )

    reference_quantification = float(
        f.attrs[
            "reflectance_quantification_value"
        ]
    )

    common_max = float(
        f.attrs[
            "normalization_max_reflectance"
        ]
    )


if len(
    source_indices_0based
) != EXPECTED_BANDS:
    raise ValueError(
        f"Expected {EXPECTED_BANDS} source indices, "
        f"got {len(source_indices_0based)}"
    )

if selected_wavelengths.shape != (
    EXPECTED_BANDS,
):
    raise ValueError(
        f"Unexpected wavelength shape: {selected_wavelengths.shape}"
    )


print("=" * 100)
print("PREPARE MATCHED DOWNSTREAM SECTIONS")
print("=" * 100)

print(
    "Selected source-band indices:",
    source_indices_0based.tolist(),
)

print(
    "Selected wavelength range:",
    float(
        selected_wavelengths.min()
    ),
    "to",
    float(
        selected_wavelengths.max()
    ),
    "nm",
)

print(
    "Reference quantification:",
    reference_quantification,
)

print(
    "Global normalization maximum:",
    common_max,
)


with h5py.File(
    BGR_ONLY_H5,
    "r",
) as f:

    bgr_only_field = np.asarray(
        f[
            "reconstruction_official_lr_mean"
        ][:],
        dtype=np.float32,
    )


with h5py.File(
    BGR_NIR_H5,
    "r",
) as f:

    bgr_nir_field = np.asarray(
        f[
            "reconstruction_official_lr_mean"
        ][:],
        dtype=np.float32,
    )


expected_field_shape = (
    1000,
    1000,
    EXPECTED_BANDS,
)

if bgr_only_field.shape != expected_field_shape:
    raise ValueError(
        f"BGR-only reconstruction has shape {bgr_only_field.shape}; "
        f"expected {expected_field_shape}"
    )

if bgr_nir_field.shape != expected_field_shape:
    raise ValueError(
        f"BGR+802 reconstruction has shape {bgr_nir_field.shape}; "
        f"expected {expected_field_shape}"
    )


stack_df = pd.read_csv(
    STACK_MANIFEST
)

required_manifest_cols = {
    "row",
    "column",
    "mjolnir_file",
}

missing_cols = (
    required_manifest_cols
    - set(
        stack_df.columns
    )
)

if missing_cols:
    raise KeyError(
        f"Stack manifest is missing: {sorted(missing_cols)}"
    )


stack_lookup = {}

for _, row in stack_df.iterrows():

    name = str(
        row[
            "mjolnir_file"
        ]
    )

    if name in stack_lookup:
        raise ValueError(
            f"Duplicate plot in stack manifest: {name}"
        )

    stack_lookup[
        name
    ] = (
        int(
            row[
                "row"
            ]
        ),
        int(
            row[
                "column"
            ]
        ),
    )


print(
    "Stacked Mjolnir plots:",
    len(
        stack_lookup
    ),
)


section_files = list_tiffs(
    SECTION_DIR
)

print(
    "Original Mjolnir section TIFFs:",
    len(
        section_files
    ),
)


genuine_sections = []
bgr_only_sections = []
bgr_nir_sections = []

records = []


for i, section_path in enumerate(
    section_files,
    start=1,
):

    (
        section_key,
        parent_plot_name,
        section_number,
    ) = parse_section_identity(
        section_path
    )

    if parent_plot_name not in stack_lookup:
        print(
            "[skip outside cross-sensor field]",
            section_path.name,
        )
        continue

    plot_path = (
        PLOT_DIR
        / parent_plot_name
    )

    if not plot_path.exists():
        raise FileNotFoundError(
            f"Parent plot TIFF not found:\n{plot_path}"
        )

    with rasterio.open(
        section_path
    ) as ds:

        quant = read_quantification(
            ds
        )

        if abs(
            quant
            - reference_quantification
        ) > 1e-9:
            raise ValueError(
                f"Quantification mismatch in {section_path.name}: "
                f"{quant} vs reference {reference_quantification}"
            )

        sensor_wavelengths = read_wavelengths(
            ds
        )

        if (
            source_indices_0based.max()
            >= ds.count
        ):
            raise ValueError(
                f"{section_path.name}: source index exceeds band count "
                f"({ds.count})"
            )

        selected_here = sensor_wavelengths[
            source_indices_0based
        ]

        if not np.allclose(
            selected_here,
            selected_wavelengths,
            atol=1e-3,
        ):
            raise ValueError(
                f"Wavelength mismatch in {section_path.name}"
            )

        raw = ds.read(
            indexes=(
                source_indices_0based
                + 1
            ).tolist()
        )

        genuine = np.moveaxis(
            raw,
            0,
            -1,
        ).astype(
            np.float32
        )

        genuine = (
            genuine
            / float(
                quant
            )
        )

        genuine = (
            genuine
            / float(
                common_max
            )
        ).astype(
            np.float32
        )

    if genuine.shape != (
        EXPECTED_H,
        EXPECTED_W,
        EXPECTED_BANDS,
    ):
        raise ValueError(
            f"Unexpected genuine section shape for {section_path.name}: "
            f"{genuine.shape}"
        )

    (
        plot_y0,
        plot_y1,
        plot_x0,
        plot_x1,
    ) = exact_section_window_inside_plot(
        section_path,
        plot_path,
    )

    stack_row, stack_col = stack_lookup[
        parent_plot_name
    ]

    base_y = (
        stack_row
        * 100
    )

    base_x = (
        stack_col
        * 100
    )

    global_y0 = (
        base_y
        + plot_y0
    )

    global_y1 = (
        base_y
        + plot_y1
    )

    global_x0 = (
        base_x
        + plot_x0
    )

    global_x1 = (
        base_x
        + plot_x1
    )

    bgr_only = bgr_only_field[
        global_y0:global_y1,
        global_x0:global_x1,
        :,
    ]

    bgr_nir = bgr_nir_field[
        global_y0:global_y1,
        global_x0:global_x1,
        :,
    ]

    expected_section_shape = (
        EXPECTED_H,
        EXPECTED_W,
        EXPECTED_BANDS,
    )

    if bgr_only.shape != expected_section_shape:
        raise RuntimeError(
            f"BGR-only extraction for {section_path.name}: "
            f"{bgr_only.shape}"
        )

    if bgr_nir.shape != expected_section_shape:
        raise RuntimeError(
            f"BGR+802 extraction for {section_path.name}: "
            f"{bgr_nir.shape}"
        )

    genuine_sections.append(
        genuine
    )

    bgr_only_sections.append(
        np.asarray(
            bgr_only,
            dtype=np.float32,
        )
    )

    bgr_nir_sections.append(
        np.asarray(
            bgr_nir,
            dtype=np.float32,
        )
    )

    records.append(
        {
            "section_index":
                len(records),

            "section_file":
                section_path.name,

            "label_id_1":
                section_key[0],

            "label_id_2":
                section_key[1],

            "label_id_3":
                section_key[2],

            "parent_plot":
                parent_plot_name,

            "section_number":
                section_number,

            "stack_row":
                stack_row,

            "stack_col":
                stack_col,

            "plot_y0":
                plot_y0,

            "plot_y1":
                plot_y1,

            "plot_x0":
                plot_x0,

            "plot_x1":
                plot_x1,

            "global_y0":
                global_y0,

            "global_y1":
                global_y1,

            "global_x0":
                global_x0,

            "global_x1":
                global_x1,
        }
    )

    if (
        i <= 3
        or i == len(
            section_files
        )
    ):
        print(
            f"[{i}/{len(section_files)}] "
            f"{section_path.name} -> "
            f"plot window y={plot_y0}:{plot_y1}, x={plot_x0}:{plot_x1}"
        )


if not records:
    raise RuntimeError(
        "No matched sections were created."
    )


genuine_sections = np.stack(
    genuine_sections,
    axis=0,
).astype(
    np.float32
)

bgr_only_sections = np.stack(
    bgr_only_sections,
    axis=0,
).astype(
    np.float32
)

bgr_nir_sections = np.stack(
    bgr_nir_sections,
    axis=0,
).astype(
    np.float32
)


print("\n" + "=" * 100)
print("MATCHED SECTION DATASET")
print("=" * 100)

print(
    "Matched sections:",
    len(
        records
    ),
)

print(
    "Genuine:",
    genuine_sections.shape,
)

print(
    "BGR only:",
    bgr_only_sections.shape,
)

print(
    "BGR + 802:",
    bgr_nir_sections.shape,
)


with h5py.File(
    OUT_H5,
    "w",
) as f:

    f.create_dataset(
        "genuine_mjolnir",
        data=genuine_sections,
        compression="gzip",
        compression_opts=4,
    )

    f.create_dataset(
        "sony_bgr_only",
        data=bgr_only_sections,
        compression="gzip",
        compression_opts=4,
    )

    f.create_dataset(
        "sony_bgr_plus_802",
        data=bgr_nir_sections,
        compression="gzip",
        compression_opts=4,
    )

    f.create_dataset(
        "wavelengths_nm",
        data=selected_wavelengths,
    )

    f.create_dataset(
        "source_sensor_band_indices_0based",
        data=source_indices_0based.astype(
            np.int32
        ),
    )

    f.attrs[
        "section_height"
    ] = EXPECTED_H

    f.attrs[
        "section_width"
    ] = EXPECTED_W

    f.attrs[
        "n_bands"
    ] = EXPECTED_BANDS

    f.attrs[
        "reflectance_quantification_value"
    ] = reference_quantification

    f.attrs[
        "normalization_max_reflectance"
    ] = common_max

    f.attrs[
        "genuine_source"
    ] = (
        "original Mjolnir section TIFFs"
    )

    f.attrs[
        "reconstruction_source"
    ] = (
        "official LR-mean whole-field reconstructions"
    )


records_df = pd.DataFrame(
    records
)

records_df.to_csv(
    OUT_CSV,
    index=False,
)


print("\nSaved:")
print(OUT_H5)
print(OUT_CSV)
