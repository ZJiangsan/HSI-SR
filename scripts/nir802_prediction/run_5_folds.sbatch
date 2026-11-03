#!/bin/bash
#SBATCH --job-name=nir802_5fold
#SBATCH --array=1-5%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=168:00:00
set -euo pipefail
REPO_ROOT="${HSI_SR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${HSI_SR_VENV:-/trinity/home/jizh/venvs/nerfstudio}"
DATA_DIR="${HSI_SR_NIR802_SECTIONS_DIR:?Set HSI_SR_NIR802_SECTIONS_DIR to reconstructed section dataset directory}"
RESULTS_DIR="${HSI_SR_NIR802_RESULTS_DIR:-/trinity/home/jizh/HSI_SR_prediction/nir802_reconstructed}"
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
export TMPDIR="${HSI_SR_SCRATCH_DIR:-${TMPDIR:-/tmp}}/nir802_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID}"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"
python -u "$REPO_ROOT/scripts/nir802_prediction/train_nir802_reconstructed_5fold.py" \
 --data-dir "$DATA_DIR" --output-dir "$RESULTS_DIR/fold_${SLURM_ARRAY_TASK_ID}" \
 --fold "$SLURM_ARRAY_TASK_ID" --max-epochs 20000 --eval-every 10 \
 --lr 5e-5 --weight-decay 1e-4 --seed 42
