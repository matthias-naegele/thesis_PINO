#!/bin/bash
#SBATCH -J mhd_ramp_coarse8_t0_t2_5_256p
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
# NOTE: this run had a weird LR in its sweep -- the coarse8 sweep WINNER landed
# on the non-standard milestones [35,75,100,200,300,400]. LR150 (the standard
# schedule [25,50,75,100,125,150]) worked just as well, so we use it here.
# ----------------------------------------------------------------------------
# Replay the coarse8_betterSweep (fromWINNER) winning ramp sequence on the
# 256p level1 "finished" data, t=0..101 window, coarse8 (enforced data loss
# spatially downsampled by factor 8), WITHOUT activation checkpointing, into a
# new project and checkpoint dir.
#
# LR150 variant: identical to run_ramping_coarse8_t0_t2_5_256p.sh except the
# config uses the STANDARD LR milestones [25,50,75,100,125,150] (the "general
# trend" the other runs follow) and a separate ckpt/wandb name.
#
# The sibling coarse8_t0_t2_5_256p instead keeps the milestones
# [35,75,100,200,300,400] that came out of the coarse8 sweep WINNER -- that
# sweep also tuned the LR schedule, so its best run landed on that
# non-standard set. This variant compares the standard schedule against the
# sweep-selected one on an otherwise identical run.
#
# The fromWINNER ramp (pde/constraint ramp up from 0, data_weight decays, every
# epoch) was formerly scripted here as a chain of per-phase training runs. It
# now lives entirely in the config's `loss_params.weight_schedule`, so this
# launcher just runs ONE training job. See
# config/coarse8_t0_t2_5_256p.yaml for the phase table; train_params.epochs
# there is the full run length (2800).
#
# Same ramp mechanism as run_ramping_coarse8_t9_25_t10_512p.sh; only
# the dataset window/data (256p level1, t=0..101) and the ckpt/wandb names
# differ.
#
# All run-specific settings (dataset window, data, coarse8 factor, ckpt
# path, wandb project) live in CONFIG so the
# snapshot copied into the ckpt dir drives the plotters too.
#
# Plot jobs (plot_index global + local at the requested epochs, and the B^2
# plotter once at the end) are submitted as separate small_cpu sbatch jobs,
# the same way the sweep fires them off.
# ============================================================================

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"

source "$HOME/thesis/venv/bin/activate"

CONFIG_NAME=coarse8_t0_t2_5_256p
CONFIG_FILE="config/${CONFIG_NAME}.yaml"

# Pull the checkpoint dir straight from the config so there is a single source
# of truth.
CKPT_PATH=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.merge(OmegaConf.load('config/paths.yaml'), OmegaConf.load('${CONFIG_FILE}')).train_params.ckpt_path)")

# ---------------------------------------------------------------------------
# Safety: never clobber an existing run. Refuse if the ckpt dir already holds
# trained checkpoints, unless RESUME=1 is set (to continue an interrupted run,
# since every phase resumes from the previous via load_ckpt=True).
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
# Fire the index plotters (global + local) for one epoch.
#
# Email policy: plot_at_epoch.sh carries #SBATCH --mail-type=END,FAIL, which
# would mail on EVERY plot job. We override it per-submission so only ONE plot
# email is sent: when the FINAL global index plot (epoch 2800) finishes (or
# fails). All other plot jobs are submitted with --mail-type=NONE.
submit_index_plots() {
    local EPOCH=$1
    local PLOT_ROOT="$CKPT_PATH/plots/epoch_${EPOCH}"
    for v in index_global index_local; do
        local MAIL_FLAGS=(--mail-type=NONE)
        if [ "$EPOCH" = "2800" ] && [ "$v" = "index_global" ]; then
            MAIL_FLAGS=(--mail-type=END,FAIL --mail-user=matthias.naegele@proton.me)
        fi
        # 256p grid: halve the 512-tuned slice positions (254->127, 240->120).
        jid=$(sbatch --parsable \
            "${MAIL_FLAGS[@]}" \
            --job-name="plot_${EPOCH}_${v}" \
            --export=ALL,CKPT="$CKPT_PATH",EPOCH="$EPOCH",VARIANT="$v",PLOT_ROOT="$PLOT_ROOT",SLICE_X1=127,SLICE_X2=120,SLICE_Y1=127,SLICE_Y2=120 \
            "$REPO/plot_at_epoch.sh")
        echo "[plot] submitted $v @ epoch $EPOCH : job $jid"
    done
}

# Fire the B^2 plotter once (plots all checkpoints in the ckpt dir).
# No email for this one (--mail-type=NONE overrides the script default).
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
# Train: a single run over the full schedule. The per-epoch data/pde/constraint
# ramp lives in loss_params.weight_schedule in the config; train_params.epochs
# is the total length. Resumable via RESUME=1 (load_ckpt restores the model and
# the loss replays the schedule to the resumed epoch).
# ============================================================================
echo "=== Training: single run over weight_schedule (see config) ==="
torchrun --standalone --nnodes=1 --nproc_per_node=3 train_bhac.py \
    --config-name "$CONFIG_NAME"

# ============================================================================
# Plots: submitted once training has written all checkpoints. The plotted
# epochs (100 / 800 / 1800 / 2800) are multiples of ckpt_freq so the
# checkpoints exist; epoch 2800 is the final global-index plot that carries the
# run email.
# ============================================================================
submit_index_plots 100
submit_index_plots 800
submit_index_plots 1800
submit_index_plots 2800
submit_b2_plot

echo "=== Training done; plot jobs submitted. ==="
