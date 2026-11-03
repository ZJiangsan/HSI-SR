#!/bin/bash
#SBATCH --job-name=nir802_frozen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
set -euo pipefail
REPO_ROOT="${HSI_SR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${HSI_SR_VENV:-/trinity/home/jizh/venvs/nerfstudio}"
DATA_DIR="${HSI_SR_NIR802_SECTIONS_DIR:?Set HSI_SR_NIR802_SECTIONS_DIR}"
GENUINE_FOLDS_ROOT="${HSI_SR_GENUINE_FOLDS_ROOT:?Set HSI_SR_GENUINE_FOLDS_ROOT with fold_1..fold_5 standard-selected checkpoints}"
OUT_DIR="${HSI_SR_NIR802_TRANSFER_OUT:-/trinity/home/jizh/HSI_SR_prediction/nir802_frozen_transfer}"
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
python -u "$REPO_ROOT/scripts/nir802_prediction/evaluate_frozen_transfer.py" \
 --data-dir "$DATA_DIR" --genuine-folds-root "$GENUINE_FOLDS_ROOT" --out-dir "$OUT_DIR"
