#!/bin/bash
#SBATCH -J pdeResEval
#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=21
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=errorsPlot/%x_%j.out
#SBATCH --error=errorsPlot/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=matthias.naegele@proton.me

# Run the full pde_res_eval suite (metrics table + residual maps) for the
# 2x2 runs. Submit from the repo root on the HPC:
#
#   sbatch plotters/pde_res_eval/run_all.sh                # all four runs
#   RUNS=coarse8_512p sbatch --export=ALL plotters/pde_res_eval/run_all.sh   # one run
#
# or run directly in an interactive shell:
#
#   bash plotters/pde_res_eval/run_all.sh
#
# One GPU (same partition/gres as the training launchers) — the workload is
# just 2 epochs x <=11 val-sample forward passes per run plus FFT residuals,
# so a single L40 finishes in minutes; on CPU the 512p forwards take hours.
# The scripts fall back to CPU automatically if no GPU is visible.

set -euo pipefail
source ~/thesis/venv/bin/activate
unset SLURM_JOB_ID SLURM_NTASKS SLURM_NPROCS SLURM_PROCID SLURM_LOCALID

# Repo root: where sbatch was invoked, or this script's grandparent dir.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"

RUNS="${RUNS:-all}"

echo "=== pde_res_eval: metrics (data vs FNO, full val set) — runs: $RUNS ==="
python plotters/pde_res_eval/compute_metrics.py --run "$RUNS"

echo "=== pde_res_eval: residual maps (sample 001) — runs: $RUNS ==="
python plotters/pde_res_eval/plot_residual_maps.py --run "$RUNS"

echo "=== pde_res_eval: done; outputs under figs/pde_res_eval_out/ ==="
