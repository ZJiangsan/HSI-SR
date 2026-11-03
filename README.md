# Registration-Free Cross-Sensor Hyperspectral Reconstruction for Wheat Trait Prediction

Research scripts for registration-free cross-sensor hyperspectral reconstruction from independently acquired Sony RGB and HySpex Mjolnir HSI, with functional validation on spring-wheat dry matter (DM), nitrogen content (NC), and nitrogen uptake (NU).

The underlying abundance-Gram method is **previous work**; this repository documents its application to independent agricultural sensor products and four guide/input representations. The 802.320-nm high-resolution observation was extracted from the Mjolnir reference, **not collected with a third camera**. Reconstruction uses comparable field/plot support but does not impose pixelwise Sony–Mjolnir registration.

## Scripts

| Directory | Task |
|---|---|
| `scripts/00_...` through `scripts/06_...` | Original data preparation, pretrained spectral decomposition, RGB+802 and RGB-only reconstruction, RGB resolution ablation, and matched-section preparation |
| `scripts/nir802_reconstruction/` | Reference-derived native 802.320-nm guide: 40-band Gram reconstruction and spectral evaluation |
| `scripts/restrans_dual15/` | Genuine-HSI, RGB-only and RGB+802 ResTrans21 prediction: 3 sources × 5 folds, dual R² checkpoint tracks |
| `scripts/nir802_prediction/` | Exact manifest-based extraction, five additional ResTrans21 folds for 802-only, and frozen genuine-HSI checkpoint transfer |
| `scripts/08_plot_manuscript_figures.py` | Existing plotting utilities; inspect expected input summaries before running |

The legacy `scripts/07_restrans21_40band_5fold.py` belongs to an earlier development protocol and **must not** be used to reproduce the final submitted manuscript's 20,000-epoch experiments. Use the new `restrans_dual15/` and `nir802_prediction/` scripts instead.

## Final prediction protocol

- Inputs: genuine 40-band HSI, RGB-only reconstruction, RGB+802 reconstruction and 802-only reconstruction.
- Matched labels: 148 wheat sections, each `15 × 100 × 40`.
- Five fertilizer-stratified, section-level folds: `StratifiedKFold(n_splits=5, shuffle=True, random_state=3)`; identical folds across representations.
- 20,000 epochs per independently initialized model (20 training runs across four representations), initial learning rate `5e-5`, weight decay `1e-4`, seed 42, Adam recreated each epoch.
- Training records a historical reversed-argument R² track and a conventional R² checkpoint-selection track. Historical R² also governs learning-rate scheduling. **For reported conventional selection, use `standard_best_*.pth` and `standard_selected_fold_metrics.csv`.** Both selection tracks share the same training trajectory.
- Historical normalization uses each representation's *full-dataset* maximum and full-dataset target min–max scaling. Validation sections contribute to these scalers; report these as validation-selected, field-specific results, not independent external test results.

## Requirements and data

Python, PyTorch (CUDA recommended), `vit-pytorch`, `numpy`, `pandas`, `h5py`, `scikit-learn`, and other dependencies from [`requirements.txt`](requirements.txt). Raw image cubes, large intermediate reconstructions and model weights are not included. See [`DATA.md`](DATA.md) and [`REPRODUCE.md`](REPRODUCE.md) for expected input files and commands.

This code is adapted to the original Trinity experimental environment. Python script syntax and Slurm launcher syntax were checked during release preparation. **The package was not end-to-end rerun on the original HPC dataset.**

## Citation

Cite the associated manuscript once published, and the foundational abundance-Gram work where appropriate. The study's agricultural field and trait measurement provenance is documented in Żelazny et al. (2022), *Remote Sensing*, 14, 5977, doi:10.3390/rs14235977.
