#!/bin/bash
#SBATCH -J mhd_ramp_stride8_t9_25_t10_512p
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
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=matthias.naegele@proton.me

# ============================================================================
# noAC / t370 variant of run_ramping_512p_stride8_t363_t400_LR150.sh.
#
# Motivation / spirit: re-run the stride8/LR150 setup on the t370..401 window
# the other runs use, so it is directly comparable to them; and turn activation
# checkpointing OFF (it was found inefficient for this setup). If this run
# performs well on the shared window, the stride8/LR150 approach can be carried
# forward without the AC compute overhead.
#
# stride8, full-res 512p, LR milestones to 150 and the same physics-loss ramp
# schedule as the t363_t400 run; only the dataset window (t370..401) and the
# ckpt/wandb names differ. (Activation checkpointing, used by the t363_t400 run
# and found inefficient here, has since been removed from the codebase.)
#
# The physics-loss ramp (formerly scripted here as a chain of per-phase
# training runs) now lives entirely in the config's
# `loss_params.weight_schedule`, so this launcher just runs ONE training job.
# See config/stride8_t9_25_t10_512p.yaml for the phase table;
# train_params.epochs there is the full run length (2000).
# ============================================================================

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

source "$HOME/thesis/venv/bin/activate"

CONFIG_NAME=stride8_t9_25_t10_512p
CONFIG_FILE="config/${CONFIG_NAME}.yaml"

CKPT_PATH=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.merge(OmegaConf.load('config/paths.yaml'), OmegaConf.load('${CONFIG_FILE}')).train_params.ckpt_path)")

# ---------------------------------------------------------------------------
# Safety: never clobber an existing run.
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

# Snapshot a fully RESOLVED, self-contained config to the checkpoint dir so the
# plotters use this run's exact dataset_params / model_params: plot_index.py &
# plot_B2_onValidation.py read <ckpt>/config.yaml directly. paths.yaml is
# composed in and every ${...} is expanded to a literal path, so the snapshot
# carries no paths.yaml dependency (the resolved roots stay as
# data_root/output_root keys for the record). Params are fixed for the run, so
# one snapshot at the start is enough.
python -c "from omegaconf import OmegaConf; d=OmegaConf.to_container(OmegaConf.merge(OmegaConf.load('config/paths.yaml'), OmegaConf.load('${CONFIG_FILE}')), resolve=True); d.pop('defaults', None); OmegaConf.save(OmegaConf.create(d), '${CKPT_PATH}/config.yaml')"
echo "[config] saved resolved snapshot to $CKPT_PATH/config.yaml"

# Snapshot this launcher too, so the exact ramp/phase recipe that produced the
# run is reproducible from the checkpoint dir (same spirit as the config copy).
cp "${BASH_SOURCE[0]}" "$CKPT_PATH/$(basename "${BASH_SOURCE[0]}")"
echo "[script] saved launcher snapshot to $CKPT_PATH/$(basename "${BASH_SOURCE[0]}")"

mkdir -p errors errorsPlot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
submit_index_plots() {
    local EPOCH=$1
    local PLOT_ROOT="$CKPT_PATH/plots/epoch_${EPOCH}"
    for v in index_global index_local; do
        local MAIL_FLAGS=(--mail-type=NONE)
        if [ "$EPOCH" = "2000" ] && [ "$v" = "index_global" ]; then
            MAIL_FLAGS=(--mail-type=END,FAIL --mail-user=matthias.naegele@proton.me)
        fi
        jid=$(sbatch --parsable \
            "${MAIL_FLAGS[@]}" \
            --job-name="plot_${EPOCH}_${v}" \
            --export=ALL,CKPT="$CKPT_PATH",EPOCH="$EPOCH",VARIANT="$v",PLOT_ROOT="$PLOT_ROOT" \
            "$REPO/plot_at_epoch.sh")
        echo "[plot] submitted $v @ epoch $EPOCH : job $jid"
    done
}

submit_b2_plot() {
    local jid
    jid=$(sbatch --parsable \
        --mail-type=NONE \
        --job-name="plot_b2_all" \
        --export=ALL,CKPT="$CKPT_PATH",EPOCH=0,VARIANT="b2",PLOT_ROOT="$CKPT_PATH/plots" \
        "$REPO/plot_at_epoch.sh")
    echo "[plot] submitted b2 (all checkpoints) : job $jid"
}

# ============================================================================
# Train: a single run over the full schedule. The per-epoch pde/constraint
# ramp lives in loss_params.weight_schedule in the config; train_params.epochs
# is the total length. Resumable via RESUME=1 (load_ckpt restores the model and
# the loss replays the schedule to the resumed epoch).
# ============================================================================
echo "=== Training: single run over weight_schedule (see config) ==="
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_bhac.py \
    --config-name "$CONFIG_NAME"

# ============================================================================
# Plots: submitted once training has written all checkpoints. The plotted
# epochs (100 / 800 / 2000) are multiples of ckpt_freq so the checkpoints
# exist; epoch 2000 is the final global-index plot that carries the run email.
# ============================================================================
submit_index_plots 100
submit_index_plots 800
submit_index_plots 2000
submit_b2_plot

echo "=== Training done; plot jobs submitted. ==="
