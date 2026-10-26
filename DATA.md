# Data and checkpoints

Raw hyperspectral/RGB imagery and large reconstructed HDF5 cubes are **not included in this repository**.

The scripts currently expect the project root at:

```text
~/RGB2msi/
```

Key local inputs used by the experiments include:

```text
~/RGB2msi/
├── sony_a5100/
│   └── plots/
└── hyspex_mjolnir1024/
    ├── plots/
    ├── sections/
    ├── mjolnir_prepared/
    │   └── mjolnir_field_40band.h5
    └── ...
```

The reconstruction scripts also reuse the pretrained Mjolnir 40-band encoder/decoder checkpoints referenced by filename inside the scripts. These model files are not included in the repository package because they were not part of the files available for packaging.

## Important experimental representation

- HR field: `1000 x 1000 x 40`
- LR-HSI: `125 x 125 x 40`
- scale factor: `8`
- wavelengths: `410.731–992.269 nm`
- RGB guide: independent Sony RGB acquisition
- NIR anchor: Mjolnir native band near `802.320 nm`

## Functional-validation dataset

The final crop-trait prediction experiment used:

- 148 sections
- 50 parent plots
- section shape: `15 x 100 x 40`
- sources: genuine Mjolnir, RGB-only reconstruction, RGB+802 reconstruction

The small final result tables needed to reproduce the manuscript plots are included under `results/`.

## Encoder–decoder pretraining source

The historical training script is archived as [`scripts/00_pretrain_mjolnir_encoder_decoder_legacy.py`](scripts/00_pretrain_mjolnir_encoder_decoder_legacy.py). It trains the 40-band encoder and decoder on **unlabeled, mean-centered low-resolution Mjolnir HSI from the study field** using L2,1 spectral reconstruction, spectral information divergence, and entropy regularization. This pretraining is **field-specific and transductive** with respect to the later reconstruction evaluation; it must not be interpreted as an external independent-field pretraining set.

The training script is preserved as historical research code, with the original notebook magic removed, source author identifier removed, and the input location made configurable using the `HSI_SR_MJOLNIR_H5` environment variable. It has not been re-run or refactored into a fully standalone training program. The filename suffix `_sam1p09` on the selected checkpoints was added **after** training to identify the achieved SAM result; the historical training script saves checkpoints without that suffix. The checkpoint save helper stores a fixed epoch field of `10000`, which is not reliable evidence of the true optimization step.

**Missing artifacts:** The original pretrained weight files and raw field data remain excluded from the public repository. Consequently, the project cannot be reproduced end-to-end from GitHub alone without the corresponding data and checkpoint files and any legacy setup dependencies.
