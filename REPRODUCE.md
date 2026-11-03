# Reproduction instructions — final manuscript protocol

**Important:** Do not follow the earlier 150-epoch plot-GroupKFold instructions. The final manuscript uses 20,000 epochs and five fertilizer-stratified section-level folds across four representations. This repository provides scripts; original field images, pretrained spectral encoder/decoder weights and other large artifacts must be supplied separately.

## 1. Reconstruct the two main guide configurations

Run the original repository's preparation and reconstruction scripts:

```bash
python scripts/01_prepare_cross_sensor_hr_guide.py
python scripts/02_cross_sensor_gram_sr_rgb_nir.py
python scripts/03_cross_sensor_gram_sr_rgb_only.py
```

The original reconstruction scripts include project-specific default paths; inspect their command-line options and your input directory layout. For the spatial-resolution ablation use `scripts/04_run_all_rgb_resolution_ablations.py` and `scripts/05_analyze_rgb_resolution_ablation.py`.

## 2. Reconstruct the reference-derived 802-nm-only guide

The 802-only guide is the native Mjolnir 802.320-nm band and **not** an independently acquired NIR image. Supply the matching pretrained Mjolnir 40-band encoder and decoder weights.

```bash
cd /trinity/home/jizh/HSI-SR        # adjust to your repository checkout
source /trinity/home/jizh/venvs/nerfstudio/bin/activate
export HSI_SR_REPO_ROOT="$PWD"
export HSI_SR_VENV=/trinity/home/jizh/venvs/nerfstudio
export HSI_SR_PREPARED_H5=/trinity/home/jizh/RGB2msi/hyspex_mjolnir1024/mjolnir_prepared/mjolnir_field_40band.h5
export HSI_SR_ENCODER=/path/to/original_40band_encoder.pth
export HSI_SR_DECODER=/path/to/original_40band_decoder.pth
export HSI_SR_NIR802_OUT=/trinity/home/jizh/nir802_reconstruction/gram_802_seed0
sbatch scripts/nir802_reconstruction/run_reconstruction.sbatch
```

Or invoke `reconstruct_nir802_gram.py` directly with `--prepared-h5`, `--encoder`, `--decoder`, and `--out-dir`. It writes `nir802_reconstructed_full.h5` with dataset name `reconstructed_hsi` and a corresponding `.npy` cube. Use `evaluate_reconstruction.py` for basic NIR-only reconstruction metrics.

## 3. Prepare matched prediction sections

Prepare 40-band genuine and RGB-derived sections with the existing repository script `scripts/06_prepare_trait_prediction_sections_40band.py`. For the NIR-only cube, **use the original manifest's exact global section coordinates**:

```bash
export HSI_SR_SECTIONS_DIR=/trinity/home/jizh/RGB2msi/hyspex_mjolnir1024/cross_sensor_functional_validation_sections
export HSI_SR_NIR802_SECTIONS_DIR=/trinity/home/jizh/nir802_reconstruction/sections_data
python scripts/nir802_prediction/extract_verified_sections.py \
  --sections-dir "$HSI_SR_SECTIONS_DIR" \
  --reconstructed "$HSI_SR_NIR802_OUT/nir802_reconstructed_full.h5" \
  --output-dir "$HSI_SR_NIR802_SECTIONS_DIR"
```

The extractor retains original 150 section positions in the HDF5 and uses the historical matched-label filtering to select 148 examples. Check `extraction_verified.json` before training.

## 4. Train 15 dual-selection models (genuine, RGB-only, RGB+802)

```bash
cd /trinity/home/jizh/HSI-SR
export HSI_SR_REPO_ROOT="$PWD"
export HSI_SR_SECTIONS_DIR=/trinity/home/jizh/RGB2msi/hyspex_mjolnir1024/cross_sensor_functional_validation_sections
sbatch scripts/restrans_dual15/run_15_folds.sbatch
```

The Slurm array is `1-15%5`, allowing up to five GPUs concurrently. The output root defaults to `/trinity/home/jizh/HSI_SR_prediction/results_dual_checkpoint_5fold_fresh`. Change it via `HSI_SR_DUAL_RESULTS_DIR`. The training scripts deliberately preserve historical learning-rate scheduling while saving both `historical_best_{DM,NC,NU}.pth` and `standard_best_{DM,NC,NU}.pth`.

To combine the conventional-selected results:

```bash
python scripts/restrans_dual15/summarize_dual15.py \
  --base /trinity/home/jizh/HSI_SR_prediction/results_dual_checkpoint_5fold_fresh
```

## 5. Train the remaining five 802-only models

```bash
export HSI_SR_NIR802_SECTIONS_DIR=/trinity/home/jizh/nir802_reconstruction/sections_data
sbatch scripts/nir802_prediction/run_5_folds.sbatch
```

Array `1-5%5`; output defaults to `/trinity/home/jizh/HSI_SR_prediction/nir802_reconstructed/fold_1` through `fold_5`. The NIR-only training code is an exact duplicate of the source implementation supplied with the separate NIR prediction package, not a new or simplified model.

## 6. Frozen-model transfer

Reuse the *standard-selected* genuine-HSI trait-specific checkpoint from each fold for evaluation without retraining:

```bash
export HSI_SR_GENUINE_FOLDS_ROOT=/trinity/home/jizh/HSI_SR_prediction/results_dual_checkpoint_5fold_fresh/genuine
sbatch scripts/nir802_prediction/run_frozen_transfer.sbatch
```

The other two reconstructed representation transfers are evaluated within the genuine-HSI training script's source-matched evaluation outputs. **Do not substitute a single fold's weights for all five folds.**

## Reporting caveats

These scripts use field-specific unsupervised reconstruction, historical dataset-wide scaling and validation-based checkpoint selection. They are not an evaluation on an independent unseen field or a fully separate holdout test set. The NIR anchor was extracted from the Mjolnir HR reference, not acquired from a third camera.
