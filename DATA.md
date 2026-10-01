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

## Downstream dataset

The final crop-trait prediction experiment used:

- 148 sections
- 50 parent plots
- section shape: `15 x 100 x 40`
- sources: genuine Mjolnir, RGB-only reconstruction, RGB+802 reconstruction

The small final result tables needed to reproduce the manuscript plots are included under `results/`.
