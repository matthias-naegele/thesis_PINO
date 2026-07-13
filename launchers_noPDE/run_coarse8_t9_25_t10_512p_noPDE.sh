#!/bin/bash
#SBATCH -J mhd_noPDE_coarse8_t9_25_t10_512p
#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40:3
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=21
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=errors/%x_%j.out
#SBATCH --error=errors/%x_%j.err
#SBATCH --tmp=400G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=matthias.naegele@proton.me

# ============================================================================
# noPDE control of run_ramping_coarse8_t9_25_t10_512p.sh: the same
# run, but the physics losses are NEVER switched on — pure data loss for all
# 600 epochs (the parent's data-only warm-up, continued to the end).
#
# All run-specific settings live in config/${CONFIG_NAME}.yaml, a small
# overlay that inherits the PARENT config via its Hydra defaults list and
# overrides only the control-defining knobs (one ramp-free 600-epoch
# weight_schedule phase, use_pde/constraint_loss off, epochs, ckpt dir, wandb
# group). Because of that composition, the CKPT_PATH lookup and the resolved
# snapshot below merge paths.yaml + parent + overlay — the same composition
# Hydra performs at train time.
#
# Unlike the parent launcher, NO plot jobs are submitted: the noPDE controls
# are evaluated against their parents via wandb / the eval scripts, not their
# own figures.
#
# Submit from the repo root:
#   sbatch launchers_noPDE/run_coarse8_t9_25_t10_512p_noPDE.sh
# ============================================================================

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

source "$HOME/thesis/venv/bin/activate"

CONFIG_NAME=coarse8_t9_25_t10_512p_noPDE
PARENT_NAME=coarse8_t9_25_t10_512p
CONFIG_FILE="config/${CONFIG_NAME}.yaml"
PARENT_FILE="config/${PARENT_NAME}.yaml"

# Pull the checkpoint dir straight from the composed config (paths + parent +
# overlay) so there is a single source of truth.
CKPT_PATH=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.merge(OmegaConf.load('config/paths.yaml'), OmegaConf.load('${PARENT_FILE}'), OmegaConf.load('${CONFIG_FILE}')).train_params.ckpt_path)")

# ---------------------------------------------------------------------------
# Safety: never clobber an existing run. Refuse if the ckpt dir already holds
# trained checkpoints, unless RESUME=1 is set (to continue an interrupted run,
# since load_ckpt=True resumes from the latest checkpoint).
# ---------------------------------------------------------------------------
mkdir -p "$CKPT_PATH"
if ls "$CKPT_PATH"/*.0.*.pt >/dev/null 2>&1 || ls "$CKPT_PATH"/*.0.*.mdlus >/dev/null 2>&1; then
    if [ "${RESUME:-0}" != "1" ]; then
        echo "ERROR: $CKPT_PATH already contains checkpoints." >&2
        echo "       Set RESUME=1 to continue this run, or pick another ckpt dir." >&2
        exit 1
    fi
    echo "[resume] RESUME=1 set; continuing run in $CKPT_PATH"
fi

# Snapshot a fully RESOLVED, self-contained config to the checkpoint dir so
# any later evaluation uses this run's exact dataset_params / model_params
# (the plotters read <ckpt>/config.yaml directly, should figures ever be
# wanted after all). paths.yaml AND the parent config are composed in
# (matching the Hydra defaults list of the overlay) and every ${...} is
# expanded to a literal path, so the snapshot is self-contained. Params are
# fixed for the run, so one snapshot at the start is enough.
python -c "from omegaconf import OmegaConf; d=OmegaConf.to_container(OmegaConf.merge(OmegaConf.load('config/paths.yaml'), OmegaConf.load('${PARENT_FILE}'), OmegaConf.load('${CONFIG_FILE}')), resolve=True); d.pop('defaults', None); OmegaConf.save(OmegaConf.create(d), '${CKPT_PATH}/config.yaml')"
echo "[config] saved resolved snapshot to $CKPT_PATH/config.yaml"

# Snapshot this launcher too, so the exact recipe that produced the run is
# reproducible from the checkpoint dir (same spirit as the config copy).
cp "${BASH_SOURCE[0]}" "$CKPT_PATH/$(basename "${BASH_SOURCE[0]}")"
echo "[script] saved launcher snapshot to $CKPT_PATH/$(basename "${BASH_SOURCE[0]}")"

mkdir -p errors

# ============================================================================
# Train: a single run, data loss only for the whole 600 epochs (the overlay's
# weight_schedule is one ramp-free phase; the physics losses are never
# computed). Resumable via RESUME=1 (load_ckpt restores the model and the loss
# replays the schedule to the resumed epoch). No plot jobs afterwards.
# ============================================================================
echo "=== Training: single data-only run, no physics losses (see config overlay) ==="
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_bhac.py \
    --config-name "$CONFIG_NAME"

echo "=== Training done. ==="
