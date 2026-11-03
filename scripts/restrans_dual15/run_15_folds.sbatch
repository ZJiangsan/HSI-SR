#!/bin/bash
#SBATCH --job-name=restrans_dual15
#SBATCH --array=1-15%5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=168:00:00
# Set --output/--error on the sbatch command if desired; Slurm's default is slurm-%j.out.
set -euo pipefail
REPO_ROOT="${HSI_SR_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${HSI_SR_VENV:-/trinity/home/jizh/venvs/nerfstudio}"
DATA_DIR="${HSI_SR_SECTIONS_DIR:-/trinity/home/jizh/RGB2msi/hyspex_mjolnir1024/cross_sensor_functional_validation_sections}"
RESULTS_DIR="${HSI_SR_DUAL_RESULTS_DIR:-/trinity/home/jizh/HSI_SR_prediction/results_dual_checkpoint_5fold_fresh}"
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
task="${SLURM_ARRAY_TASK_ID:?missing Slurm array ID}"
(( task >= 1 && task <= 15 )) || exit 2
sources=(genuine rgb rgb802)
source_idx=$(((task - 1) / 5))
fold=$(((task - 1) % 5 + 1))
source="${sources[$source_idx]}"
mkdir -p "${HSI_SR_SCRATCH_DIR:-${TMPDIR:-/tmp}}/hsi_sr_${SLURM_JOB_ID:-local}_${task}"
export TMPDIR="${HSI_SR_SCRATCH_DIR:-${TMPDIR:-/tmp}}/hsi_sr_${SLURM_JOB_ID:-local}_${task}"
export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"
python -u "$REPO_ROOT/scripts/restrans_dual15/train_${source}_dual.py" \
  --data-dir "$DATA_DIR" --output-dir "$RESULTS_DIR/$source/fold_$fold" \
  --fold "$fold" --max-epochs 20000 --eval-every 10 \
  --lr 5e-5 --weight-decay 1e-4 --seed 42
