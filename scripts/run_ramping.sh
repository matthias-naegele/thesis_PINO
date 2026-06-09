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

# Stage data to local node scratch.
# cp -r /path/to/BHAC_output/orszagTang/<run-name> /tmp/

source "$HOME/thesis/venv/bin/activate"

# Snapshot config to checkpoint dir so plotters use the trained run's config.
# Only model_params/dataset_params matter to plotters, and those don't change
# across phases, so a single snapshot at the start of the run is enough.
CKPT_PATH=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('config/mhd_bhac.yaml').train_params.ckpt_path)")
mkdir -p "$CKPT_PATH"
cp config/mhd_bhac.yaml "$CKPT_PATH/config.yaml"
echo "[config] saved snapshot to $CKPT_PATH/config.yaml"

# Helper: compute end weight = start + ramps_in_phase * increment
# ramps_in_phase = floor(phase_length / ramp_every)
calc_end() {
    python3 -c "print(${1} + (${2}//${3}) * ${4})"
}

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

PHASE_LEN=500

# ============================================================
# Phase 1: epochs 100-600
# ============================================================
PDE_W1=5e-6
CON_W1=0.0004
PDE_RAMP_EPOCH1=12;  PDE_RAMP_INC1=1e-5
CON_RAMP_EPOCH1=12;  CON_RAMP_INC1=0.0008

echo "=== Phase 1: epochs 100-600 ==="
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_bhac.py \
    train_params.epochs=600 \
    loss_params.pde_weight=$PDE_W1 \
    loss_params.constraint_weight=$CON_W1 \
    loss_params.pde_weight_ramp_epoch=$PDE_RAMP_EPOCH1 \
    loss_params.pde_weight_ramp_increment=$PDE_RAMP_INC1 \
    loss_params.constraint_weight_ramp_epoch=$CON_RAMP_EPOCH1 \
    loss_params.constraint_weight_ramp_increment=$CON_RAMP_INC1

# ============================================================
# Phase 2: epochs 600-1100
# ============================================================
PDE_W2=$(calc_end $PDE_W1 $PHASE_LEN $PDE_RAMP_EPOCH1 $PDE_RAMP_INC1)
CON_W2=$(calc_end $CON_W1 $PHASE_LEN $CON_RAMP_EPOCH1 $CON_RAMP_INC1)
PDE_RAMP_EPOCH2=12;  PDE_RAMP_INC2=7e-5
CON_RAMP_EPOCH2=12;  CON_RAMP_INC2=0.0022

echo "=== Phase 2: epochs 600-1100 (start pde=$PDE_W2 con=$CON_W2) ==="
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_bhac.py \
    train_params.epochs=1100 \
    loss_params.pde_weight=$PDE_W2 \
    loss_params.constraint_weight=$CON_W2 \
    loss_params.pde_weight_ramp_epoch=$PDE_RAMP_EPOCH2 \
    loss_params.pde_weight_ramp_increment=$PDE_RAMP_INC2 \
    loss_params.constraint_weight_ramp_epoch=$CON_RAMP_EPOCH2 \
    loss_params.constraint_weight_ramp_increment=$CON_RAMP_INC2

# ============================================================
# Phase 3: epochs 1100-1600
# ============================================================
PDE_W3=$(calc_end $PDE_W2 $PHASE_LEN $PDE_RAMP_EPOCH2 $PDE_RAMP_INC2)
CON_W3=$(calc_end $CON_W2 $PHASE_LEN $CON_RAMP_EPOCH2 $CON_RAMP_INC2)
PDE_RAMP_EPOCH3=10;  PDE_RAMP_INC3=50e-5
CON_RAMP_EPOCH3=10;  CON_RAMP_INC3=0.008

echo "=== Phase 3: epochs 1100-1600 (start pde=$PDE_W3 con=$CON_W3) ==="
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_bhac.py \
    train_params.epochs=1600 \
    loss_params.pde_weight=$PDE_W3 \
    loss_params.constraint_weight=$CON_W3 \
    loss_params.pde_weight_ramp_epoch=$PDE_RAMP_EPOCH3 \
    loss_params.pde_weight_ramp_increment=$PDE_RAMP_INC3 \
    loss_params.constraint_weight_ramp_epoch=$CON_RAMP_EPOCH3 \
    loss_params.constraint_weight_ramp_increment=$CON_RAMP_INC3

# ============================================================
# Phase 4: epochs 1600-2100
# ============================================================
PDE_W4=$(calc_end $PDE_W3 $PHASE_LEN $PDE_RAMP_EPOCH3 $PDE_RAMP_INC3)
CON_W4=$(calc_end $CON_W3 $PHASE_LEN $CON_RAMP_EPOCH3 $CON_RAMP_INC3)
PDE_RAMP_EPOCH4=10;  PDE_RAMP_INC4=40e-4
CON_RAMP_EPOCH4=10;  CON_RAMP_INC4=0.08

echo "=== Phase 4: epochs 1600-2100 (start pde=$PDE_W4 con=$CON_W4) ==="
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_bhac.py \
    train_params.epochs=2100 \
    loss_params.pde_weight=$PDE_W4 \
    loss_params.constraint_weight=$CON_W4 \
    loss_params.pde_weight_ramp_epoch=$PDE_RAMP_EPOCH4 \
    loss_params.pde_weight_ramp_increment=$PDE_RAMP_INC4 \
    loss_params.constraint_weight_ramp_epoch=$CON_RAMP_EPOCH4 \
    loss_params.constraint_weight_ramp_increment=$CON_RAMP_INC4
