#!/bin/bash
#SBATCH -J mhd_ramp_coarse8_t9_25_t10_512p_zusatz
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
# Same run as run_ramping_coarse8_t9_25_t10_512p.sh (which replays
# the fromWINNER sweep winners, not the noSweep sequence) but on the ENLARGED
# dataset finished_512_zusatz (38 train / 11 val sims instead of 23 / 6; the
# old 6 val sims were renamed so they still sort into the val tail).
#
# To keep the run compute-matched with the 23-sim original (same total number
# of training samples seen, since one epoch = one pass over the train set),
# the schedule is rescaled by 0.6 (~= 23/38, total samples seen 1680*38 = 63840
# vs 2800*23 = 64400, within 1%): phase lengths x0.6 and per-epoch increments
# x5/3, plus LR milestones / ckpt_freq rescaled in the config.
#
# That rescaled fromWINNER ramp (formerly scripted here as a chain of per-phase
# training runs) now lives entirely in the config's
# `loss_params.weight_schedule`, so this launcher just runs ONE training job.
# See config/coarse8_t9_25_t10_512p_zusatz.yaml for the phase table;
# train_params.epochs there is the full run length (1680).
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

CONFIG_NAME=coarse8_t9_25_t10_512p_zusatz
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
# email is sent: when the FINAL global index plot (epoch 1680) finishes (or
# fails). All other plot jobs are submitted with --mail-type=NONE.
submit_index_plots() {
    local EPOCH=$1
    local PLOT_ROOT="$CKPT_PATH/plots/epoch_${EPOCH}"
    for v in index_global index_local; do
        local MAIL_FLAGS=(--mail-type=NONE)
        if [ "$EPOCH" = "1680" ] && [ "$v" = "index_global" ]; then
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
# epochs (60 / 480 / 1080 / 1680) are multiples of ckpt_freq so the checkpoints
# exist; epoch 1680 is the final global-index plot that carries the run email.
# ============================================================================
submit_index_plots 60
submit_index_plots 480
submit_index_plots 1080
submit_index_plots 1680
submit_b2_plot

echo "=== Training done; plot jobs submitted. ==="
