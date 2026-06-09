#!/bin/bash
#SBATCH -J mhd_pino_train
#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=21
#SBATCH --mem=512G
#SBATCH --time=10:00:00
#SBATCH --output=errors/%x_%j.out
#SBATCH --error=errors/%x_%j.err
#SBATCH --tmp=400G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-here>


set -euo pipefail

# Run from repo root regardless of where the script is invoked from.
cd "$(dirname "$0")/.."

# Example: stage data to local node scratch before training.
# cp -r /path/to/BHAC_output/orszagTang/<run-name> /tmp/

source "$HOME/thesis/venv/bin/activate"

# Snapshot config to checkpoint dir so plotters use the trained run's config
CKPT_PATH=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('config/mhd_bhac.yaml').train_params.ckpt_path)")
mkdir -p "$CKPT_PATH"
cp config/mhd_bhac.yaml "$CKPT_PATH/config.yaml"
echo "[config] saved snapshot to $CKPT_PATH/config.yaml"

torchrun --standalone --nnodes=1 --nproc_per_node=8 train_bhac.py
