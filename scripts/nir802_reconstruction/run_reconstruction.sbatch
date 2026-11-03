#!/bin/bash
#SBATCH --job-name=nir802_gram
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=168:00:00
set -euo pipefail
REPO_ROOT="${HSI_SR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${HSI_SR_VENV:-/trinity/home/jizh/venvs/nerfstudio}"
PREPARED_H5="${HSI_SR_PREPARED_H5:-/trinity/home/jizh/RGB2msi/hyspex_mjolnir1024/mjolnir_prepared/mjolnir_field_40band.h5}"
ENCODER="${HSI_SR_ENCODER:?Set HSI_SR_ENCODER to pretrained encoder .pth path}"
DECODER="${HSI_SR_DECODER:?Set HSI_SR_DECODER to pretrained decoder .pth path}"
OUT="${HSI_SR_NIR802_OUT:-/trinity/home/jizh/nir802_reconstruction/gram_802_seed0}"
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
python -u "$REPO_ROOT/scripts/nir802_reconstruction/reconstruct_nir802_gram.py" \
 --prepared-h5 "$PREPARED_H5" --encoder "$ENCODER" --decoder "$DECODER" \
 --out-dir "$OUT" --epochs 20000 --seed 0
