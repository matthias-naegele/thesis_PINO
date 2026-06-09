#!/bin/bash
#SBATCH -J plotFigs
#SBATCH -c 26
#SBATCH --mem=64G
#SBATCH -p small_cpu
#SBATCH --tmp=250G
#SBATCH --output=errorsPlot/%x_%j.out
#SBATCH --error=errorsPlot/%x_%j.err


# Run from repo root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

source ~/thesis/venv/bin/activate
unset SLURM_JOB_ID SLURM_NTASKS SLURM_NPROCS SLURM_PROCID SLURM_LOCALID

# Edit OUTPUT_DIR and CKPT_PATH to point at your run.
OUTPUT_DIR="/path/to/FNO_output/checkpoints/plotsFields"
CKPT_PATH="/path/to/FNO_output/checkpoints/<run-name>"

python plot_index.py \
    --field u1 u2 b1 b2 p e3 rho \
    --time_index -1 \
    --modes true pred error \
    --plot_jz \
    --epoch 1700 \
    --scale global \
    --slice_y 127 120 --slice_x 127 120 \
    --output_dir "$OUTPUT_DIR" \
    train_params.ckpt_path="$CKPT_PATH"
