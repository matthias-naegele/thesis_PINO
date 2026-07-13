#!/bin/bash
#SBATCH -J collageFigs
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -p small_cpu
#SBATCH --time=4:00:00
#SBATCH --output=errorsPlot/%x_%j.out
#SBATCH --error=errorsPlot/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matthias.naegele@proton.me

# Fire the four collage plotters (BHAC truth | FNO prediction | error, per field
# and timestep, early data-only vs final physics-ramped checkpoint) on the HPC.
#
# They read the fields.h5 dumps plot_index.py wrote under each run's checkpoint
# dir (<ckpt>/plots/epoch_<E>/index_data/...; see plotters/collage_h5.py), so
# they need those dumps -- i.e. run on the HPC where the checkpoints live. No
# GPU / PhysicsNeMo needed: this is pure h5 read + matplotlib, so it runs on the
# small_cpu partition. The wall time is dominated by the high-DPI PNG rendering.
#
# Submit from the repo root on the HPC:
#
#   sbatch plotters/run_collages.sh                 # all four
#   PLOTTERS="plot_coarse8_t9_25_t10_512p.py" sbatch --export=ALL plotters/run_collages.sh
#
# or run directly in a shell (e.g. HPC login node):
#
#   bash plotters/run_collages.sh
#   python plotters/plot_coarse8_t9_25_t10_512p.py  # just one

set -euo pipefail
source ~/thesis/venv/bin/activate
unset SLURM_JOB_ID SLURM_NTASKS SLURM_NPROCS SLURM_PROCID SLURM_LOCALID

# Repo root: where sbatch was invoked, or this script's grandparent dir.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
mkdir -p errorsPlot

# The four collage plotters, or a subset via the PLOTTERS env var.
PLOTTERS="${PLOTTERS:-plot_coarse8_t0_t2_5_256p.py plot_stride8_t0_t2_5_256p.py plot_coarse8_t9_25_t10_512p.py plot_stride8_t9_25_t10_512p.py}"

for p in $PLOTTERS; do
    echo "=== collage plotter: $p ==="
    python "plotters/$p"
done

echo "=== run_collages: done; outputs under figs/collages/ ==="
