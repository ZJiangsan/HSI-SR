# HSI_SR

## Registration-free cross-sensor hyperspectral super-resolution with downstream functional validation

This repository contains the code and compact result summaries for a field-scale hyperspectral super-resolution study using **independently acquired Sony RGB and HySpex Mjolnir imagery**.

The central idea is to remove the need for pixelwise cross-sensor registration by matching the low- and high-resolution observations through a **permutation-invariant abundance Gram matrix**. The field experiment reconstructs a 40-band HySpex Mjolnir HSI spanning approximately 411–992 nm from a spectrally complete LR-HSI and high-resolution guidance.

The main comparison is:

- **Sony RGB only**
- **Sony RGB + one Mjolnir NIR anchor at 802.320 nm**

The Sony RGB observations are acquired independently from the hyperspectral sensor. The 802-nm anchor is a native Mjolnir band, so the RGB+802 guide is a heterogeneous two-sensor guide rather than a fully independent four-channel RGB-NIR camera.

## Main results

| Guide | SAM ↓ | PSNR ↑ | RMSE ↓ |
|---|---:|---:|---:|
| Sony RGB | 4.997° | 26.29 dB | 0.0701 |
| Sony RGB + 802 nm | **2.301°** | **35.65 dB** | **0.0227** |

The benefit is not limited to image-reconstruction metrics. A ResTrans21 predictor trained **only on genuine Mjolnir HSI** was frozen and applied unchanged to the reconstructed HSI.

| Input to genuine-HSI model | DM R² | NC R² | NU R² |
|---|---:|---:|---:|
| Genuine HSI | 0.360 | 0.421 | 0.592 |
| RGB + 802 reconstruction | 0.273 | 0.179 | **0.585** |
| RGB-only reconstruction | -3.060 | -1.096 | -3.439 |

Thus, the NIR-assisted Gram reconstruction not only reconstructs the spectrum accurately but also preserves substantial downstream predictive information, with nitrogen-uptake performance nearly unchanged.

## Repository structure

```text
HSI_SR/
├── README.md
├── CITATION.cff
├── DATA.md
├── REPRODUCE.md
├── requirements.txt
├── scripts/
│   ├── 01_prepare_cross_sensor_hr_guide.py
│   ├── 02_cross_sensor_gram_sr_rgb_nir.py
│   ├── 03_cross_sensor_gram_sr_rgb_only.py
│   ├── 04_rgb_resolution_ablation.py
│   ├── 04_run_all_rgb_resolution_ablations.py
│   ├── 05_analyze_rgb_resolution_ablation.py
│   ├── 06_prepare_downstream_sections_40band.py
│   ├── 07_restrans21_40band_5fold.py
│   └── 08_plot_manuscript_figures.py
├── results/
│   ├── reconstruction/
│   ├── downstream/
│   └── per_band/
└── .gitignore
```

Large raw imagery, reconstructed HDF5 cubes, model checkpoints, manuscript files, and rendered figures are intentionally excluded from the code repository. The compact result figures can be regenerated from the included CSV/JSON summaries using `scripts/08_plot_manuscript_figures.py`. The observed-vs-predicted scatterplots require `test_predictions.csv`, which is generated when the final ResTrans21 five-fold experiment is rerun.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is recommended for reconstruction and ResTrans21 training.

## Data

Large raw imagery, reconstructed HDF5 cubes, and model checkpoint files are intentionally excluded from Git. See [`DATA.md`](DATA.md) for the expected local data layout.

## Reproduction

See [`REPRODUCE.md`](REPRODUCE.md) for the experiment order and the exact final downstream protocol used in the manuscript.

## Methodological note

“Registration-free” means that the abundance-Gram optimization does **not require pixelwise RGB–HSI correspondence**. The observations must still represent the same field / scene support so that their global material distributions are comparable.

## Citation

A formal citation entry will be added after manuscript publication.
