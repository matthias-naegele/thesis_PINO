#!/bin/bash
#SBATCH -J plotFigs
#SBATCH -c 26
#SBATCH --mem=64G
#SBATCH -p small_cpu
#SBATCH --tmp=250G
#SBATCH --time=40:00:00
#SBATCH --output=errorsPlot/%x_%j.out
#SBATCH --error=errorsPlot/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matthias.naegele@proton.me

# CPU plotter for one (epoch, variant) combination. Variants:
#   index_global   -> plot_index.py with --scale global + slices (one epoch)
#   index_local    -> plot_index.py with --scale local  + slices (one epoch)
#   b2             -> plot_B2_onValidation.py (this branch's version has no
#                     --epoch flag, so it plots ALL checkpoints in the ckpt
#                     dir; EPOCH is ignored for this variant)
#
# plot_index.py / plot_B2_onValidation.py load the config SNAPSHOT from
# <ckpt>/config.yaml, so passing train_params.ckpt_path is enough for them to
# pick up the right dataset window / data / coarse factor for this run.
#
# Required env vars (sbatch --export):
#   CKPT       checkpoint dir (must contain checkpoint.0.${EPOCH}.pt + config.yaml)
#   EPOCH      epoch to plot
#   VARIANT    one of {index_global, index_local, b2}
#   PLOT_ROOT  output root for plots (a /<VARIANT>/ subdir is created under it;
#              index variants also share a /index_data/ subdir for the
#              scale-independent .h5 fields + .txt slice tables)
#
# Optional env vars (index_global/index_local slice positions; defaults are
# tuned for the 512 grid, halve them for 256p runs):
#   SLICE_X1 SLICE_X2   x-slice indices (default 254 240)
#   SLICE_Y1 SLICE_Y2   y-slice indices (default 254 240)

set -euo pipefail
source ~/thesis/venv/bin/activate
unset SLURM_JOB_ID SLURM_NTASKS SLURM_NPROCS SLURM_PROCID SLURM_LOCALID

: "${CKPT:?CKPT is required}"
: "${EPOCH:?EPOCH is required}"
: "${VARIANT:?VARIANT is required}"
: "${PLOT_ROOT:?PLOT_ROOT is required}"

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=""

OUT_DIR="$PLOT_ROOT/$VARIANT"
mkdir -p "$OUT_DIR"

# PLOT_ROOT already encodes the epoch (".../plots/epoch_${EPOCH}"), so tell
# plot_index.py NOT to add its own epoch_NNNN/ layer (avoids the redundant
# double-nested epoch dir). The scale-independent raw data (.h5 + slice .txt)
# goes to a shared index_data/ dir so the global and local colour-scale jobs
# don't each write their own copy; index_global writes it, index_local skips it.
DATA_DIR="$PLOT_ROOT/index_data"

echo "=== plot_at_epoch: $VARIANT @ epoch $EPOCH ==="
echo "    ckpt=$CKPT"
echo "    out =$OUT_DIR"
echo "    data=$DATA_DIR"

case "$VARIANT" in
    index_global)
        python plot_index.py \
            --field u1 u2 b1 b2 p e3 rho \
            --time_index -1 \
            --modes true pred error \
            --plot_jz \
            --epoch "$EPOCH" \
            --scale global \
            --slice_y "${SLICE_Y1:-254}" "${SLICE_Y2:-240}" --slice_x "${SLICE_X1:-254}" "${SLICE_X2:-240}" \
            --lowres_style gray \
            --output_dir "$OUT_DIR" \
            --data_dir "$DATA_DIR" \
            --no_epoch_subdir \
            train_params.ckpt_path="'$CKPT'"
        ;;
    index_local)
        python plot_index.py \
            --field u1 u2 b1 b2 p e3 rho \
            --time_index -1 \
            --modes true pred error \
            --plot_jz \
            --epoch "$EPOCH" \
            --scale local \
            --slice_y "${SLICE_Y1:-254}" "${SLICE_Y2:-240}" --slice_x "${SLICE_X1:-254}" "${SLICE_X2:-240}" \
            --lowres_style gray \
            --output_dir "$OUT_DIR" \
            --data_dir "$DATA_DIR" \
            --no_epoch_subdir \
            --skip_data_files \
            train_params.ckpt_path="'$CKPT'"
        ;;
    b2)
        # No --epoch on this branch's plotter -> plots all checkpoints found.
        python plot_B2_onValidation.py \
            --output_dir "$OUT_DIR" \
            train_params.ckpt_path="'$CKPT'"
        ;;
    *)
        echo "ERROR: unknown VARIANT=$VARIANT" >&2
        exit 1
        ;;
esac

echo "=== plot_at_epoch done ==="
