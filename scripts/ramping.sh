#!/bin/bash
#SBATCH -J mhd_pino_pde_ramp
#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40:3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=21
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=errors/%x_%j.out
#SBATCH --error=errors/%x_%j.err
#SBATCH --tmp=400G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-here>


set -euo pipefail

# Run from repo root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

# Example: stage data to local node scratch.
# cp -r /path/to/BHAC_output/orszagTang/<run-name> /tmp/

source "$HOME/thesis/venv/bin/activate"

# Snapshot config to checkpoint dir so plotters use the trained run's config.
# Only model_params/dataset_params matter to plotters, and those don't change
# across phases, so a single snapshot at the start of the run is enough.
CKPT_PATH=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('config/mhd_bhac.yaml').train_params.ckpt_path)")
mkdir -p "$CKPT_PATH"
cp config/mhd_bhac.yaml "$CKPT_PATH/config.yaml"
echo "[config] saved snapshot to $CKPT_PATH/config.yaml"

# ============================================================
# Phase 0: epochs 0-100  — data loss only, no physics
# ============================================================
echo "=== Phase 0: epochs 0-100 (data loss only) ==="
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_bhac.py \
    train_params.epochs=100 \
    loss_params.pde_weight=0 \
    loss_params.constraint_weight=0 \
    loss_params.pde_weight_ramp_epoch=0 \
    loss_params.pde_weight_ramp_increment=0 \
    loss_params.constraint_weight_ramp_epoch=0 \
    loss_params.constraint_weight_ramp_increment=0

# ============================================================
# Phase 1: epochs 100-500
# ============================================================
echo "=== Phase 1: epochs 100-500 ==="
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_bhac.py \
    train_params.epochs=500 \
    loss_params.pde_weight=5e-6 \
    loss_params.constraint_weight=0.0004 \
    loss_params.pde_weight_ramp_epoch=12 \
    loss_params.pde_weight_ramp_increment=1e-5 \
    loss_params.constraint_weight_ramp_epoch=12 \
    loss_params.constraint_weight_ramp_increment=0.0008
 
# ============================================================
# Phase 2: epochs 500-700
# ============================================================
echo "=== Phase 2: epochs 500-700 ==="
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_bhac.py \
    train_params.epochs=700 \
    loss_params.pde_weight=0.000335 \
    loss_params.constraint_weight=0.0268 \
    loss_params.pde_weight_ramp_epoch=4 \
    loss_params.pde_weight_ramp_increment=1e-5 \
    loss_params.constraint_weight_ramp_epoch=7 \
    loss_params.constraint_weight_ramp_increment=0.0008
 
# ============================================================
# Phase 3: epochs 700-1100
# ============================================================
echo "=== Phase 3: epochs 700-1100 ==="
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_bhac.py \
    train_params.epochs=1100 \
    loss_params.pde_weight=0.000835 \
    loss_params.constraint_weight=0.0492 \
    loss_params.pde_weight_ramp_epoch=4 \
    loss_params.pde_weight_ramp_increment=2e-5 \
    loss_params.constraint_weight_ramp_epoch=7 \
    loss_params.constraint_weight_ramp_increment=0.001
 
# ============================================================
# Phase 4: epochs 1100-2000
# ============================================================
echo "=== Phase 4: epochs 1100-2000 ==="
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_bhac.py \
    train_params.epochs=2000 \
    loss_params.pde_weight=0.002835 \
    loss_params.constraint_weight=0.1062 \
    loss_params.pde_weight_ramp_epoch=4 \
    loss_params.pde_weight_ramp_increment=2.2e-5 \
    loss_params.constraint_weight_ramp_epoch=7 \
    loss_params.constraint_weight_ramp_increment=0.0005
