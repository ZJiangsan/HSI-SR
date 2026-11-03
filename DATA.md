# Inputs and data availability

Only research scripts are supplied in this code integration package. **Do not upload raw field imagery, reconstructed image cubes, large trained model checkpoints or project-private data** without the relevant permissions.

## Source data

The field and agronomic measurements are described by Żelazny et al. (2022), *Remote Sensing*, 14, 5977, https://doi.org/10.3390/rs14235977. The independent Sony RGB and Mjolnir hyperspectral products must cover comparable plots/field areas; dense pixelwise cross-sensor registration is not required by the Gram objective.

## Prepared reconstruction inputs

The NIR-only reconstruction expects `mjolnir_field_40band.h5` containing:

- `hsi_hr`: prepared full-resolution native-band field cube, approximately `1000 × 1000 × 40`.
- `hsi_lr`: `125 × 125 × 40` low-resolution HSI.
- `selected_wavelengths_nm`: 40 native Mjolnir wavelengths, including the 802.320-nm guide (zero-based index 26).

You also need the corresponding pretrained 40-band Mjolnir encoder and decoder checkpoint files (`.pth`). These weights are **not** distributed in this scripts-only package.

## Prediction inputs

The final prediction scripts expect a local data directory with:

- `matched_sections_40band.h5` containing genuine and reconstructed 40-band sections;
- `matched_sections_manifest.csv` with section IDs, fertilizer strata and, for NIR-only section extraction, exact `global_y0`, `global_y1`, `global_x0`, `global_x1` columns;
- `groundtruth_sections_n.csv` with spring-wheat DM, NC, and NU labels.

The matched-label selection uses 148 of 150 extracted section entries. Each section has shape `15 × 100 × 40`. Do not independently resize, reorder or rematch the 802-only sections.

## Interpretation

The NIR-only and RGB+802 inputs reuse a high-resolution 802.320-nm band from the Mjolnir reference. This is a **reference-derived NIR anchor**, not a third independent field sensor. Dataset-wide scaling in the historical prediction procedure includes observations from validation sections; unsupervised spectral reconstruction uses imagery covering the study field and is transductive.
