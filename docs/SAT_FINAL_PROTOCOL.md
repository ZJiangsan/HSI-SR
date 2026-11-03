# Final Smart Agricultural Technology prediction protocol (November 2026)

This supplement documents the current manuscript's **20,000-epoch** functional-validation experiment. The older `scripts/07_restrans21_40band_5fold.py` and original `REPRODUCE.md` describe a **150-epoch plot-grouped development experiment** and must not be mistaken for the final evaluation.

## Data and splits

- 40 sensor-native HySpex Mjolnir bands (410.731–992.269 nm); reconstruction scale 8.
- 150 original 15 × 100 × 40 section samples, of which **148** have labels (excluded original zero-based indices **65** and **74**).
- 50 parent wheat plots. Each section is the prediction unit.
- Fertilizer-stratified section-level `StratifiedKFold(n_splits=5,shuffle=True,random_state=3)`, 29–30 validation sections per fold. Some parent plots occur in both training and validation. This is **not** plot-grouped validation.
- Trait targets: dry matter (DM, g m⁻²), nitrogen content (NC, %), and nitrogen uptake (NU, g m⁻²).

## Representations and model

1. Genuine HSI.
2. Sony RGB-only guided reconstructed 40-band HSI.
3. Sony RGB plus reference-derived, nonregistered Mjolnir 802.320 nm guide reconstructed 40-band HSI.
4. Mjolnir 802.320 nm-only guided reconstructed 40-band HSI.

All ResTrans21 inputs contain **40 bands**; NIR802-only is **not** a one-band prediction model. Separate model training for every representation and fold uses the original corrected ResTrans21 architecture (SimpleViT with 40 input channels), 20,000 epochs, seed 42, learning rate 5 × 10⁻⁵, weight decay 10⁻⁴ and Adam optimizer recreated per epoch. Individual DM, NC, and NU checkpoints are selected by **conventional validation R²**; final metrics are fold mean ± sample SD.

Normalization uses each representation's own **full labeled dataset maximum**; target scaling uses **full labeled dataset min–max**. Both include validation observations. Therefore all values are **validation-selected, transductive cross-validation results**, not independent test-set estimates.

## Result summary: standard validation R² (five-fold means)

| Input | DM | NC | NU |
|---|---:|---:|---:|
| Genuine HSI | 0.717 | 0.583 | 0.816 |
| RGB-only reconstruction | 0.622 | 0.561 | 0.672 |
| RGB + 802 nm reconstruction | 0.756 | 0.533 | 0.807 |
| 802 nm-only reconstruction | 0.573 | 0.340 | 0.581 |

Frozen genuine-HSI-model transfer R² (DM, NC, NU): RGB-only **−1.411, −0.710, −1.504**; RGB+802 **0.589, 0.453, 0.686**; NIR802-only **−0.521, −0.613, −0.699**. For frozen transfer, use **genuine-HSI-trained, fold- and trait-specific checkpoints** and their genuine-HSI normalization; never use the representation-trained model weights.

## Available final scripts

- `scripts/final_prediction/train_genuine_5fold.py`
- `scripts/final_prediction/train_rgb_5fold.py`
- `scripts/final_prediction/train_rgb802_5fold.py`

These scripts are archived for traceability. The final manuscript results use a dual-checkpoint training/selection implementation; the 5-fold scripts should not be silently treated as interchangeable with those final dual checkpoint scripts.

NIR802-only cube export, verified-section extraction, genuine-checkpoint frozen transfer, and final dual-checkpoint code are in the separately supplied **HSI_SR_SAT_GitHub_Reproducibility_Update.zip** package, not yet contained in this Git branch. Local Trinity-specific paths and missing raw imagery/checkpoints prevent out-of-the-box reproduction.

## Important scope notes

The **Sony RGB** observations were independently acquired from the Mjolnir HSI. The 802.320-nm high-resolution guidance was extracted from the Mjolnir reference and **was not acquired by an independent NIR camera**. The unsupervised reconstruction is field-specific and transductive. The abundance-Gram optimization does not require dense RGB–HSI registration, but its global statistics do not guarantee accurate local geometry.

Do not commit large image cubes, raw field data, private agronomic labels or pretrained weights without explicit authorization.
