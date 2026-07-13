# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the pde_res_eval scripts.

Everything here deliberately reuses the training/eval code paths (same
dataloader, same FNO construction as plot_index.py, same
LossMHD_PhysicsNeMo residual operators) so that the numbers produced are in
exactly the metric logged to wandb during training (mean squared residual,
same spectral derivatives, same one-sided/centered time stencil, same
dt = tend/(nt-1) from the run's snapshot config).
"""

import os
import sys
from pathlib import Path

# Anchor to the repo root so the scripts can be run from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
from omegaconf import OmegaConf

from physicsnemo.models.fno import FNO
from physicsnemo.launch.utils import load_checkpoint
from dataloaders import BHACDataloader, BHACUniformDataset
from losses import LossMHD_PhysicsNeMo

# The individual residual quantities, in the order used everywhere below.
# These are the strong-form PDE residuals (Faraday induction x2, momentum/
# Ohm, equation-of-state x3, continuity) plus the div(B) constraint.
QUANTITIES = ["FI1", "FI2", "MO", "ES0", "ES1", "ES2", "C1", "divB"]


def load_run_config(config_name):
    """Load a run's config, preferring the resolved snapshot in its ckpt dir.

    Mirrors plot_index.py: the launcher snapshots a fully resolved config to
    <ckpt_path>/config.yaml; when that exists it is the authoritative source
    (it captures the run exactly as trained). Otherwise fall back to the live
    config/<name>.yaml merged with config/paths.yaml.

    Returns (cfg, ckpt_path, source_description).
    """
    live = REPO_ROOT / "config" / f"{config_name}.yaml"
    cfg = OmegaConf.load(live)
    paths_yaml = REPO_ROOT / "config" / "paths.yaml"
    if paths_yaml.is_file():
        cfg = OmegaConf.merge(OmegaConf.load(paths_yaml), cfg)
    ckpt_path = str(cfg.train_params.ckpt_path)

    snap = os.path.join(ckpt_path, "config.yaml")
    if os.path.isfile(snap):
        cfg = OmegaConf.load(snap)
        source = f"snapshot {snap}"
        ckpt_path = str(cfg.train_params.ckpt_path)
    else:
        source = f"live {live} (no snapshot at {snap})"
    OmegaConf.set_struct(cfg, False)
    return cfg, ckpt_path, source


def build_val_dataloader(cfg, num_workers=None):
    """Validation dataloader exactly as in training/plot_index.py.

    Returns (dataloader, t_real) with batch_size=1 and shuffle=False, so
    batch index == validation sample index (sample_001 = second batch).
    """
    dp = cfg.dataset_params
    dataset_val = BHACUniformDataset(
        dp.data_dir,
        output_names=dp.output_names,
        file_name=dp.file_name,
        num_train=dp.num_train,
        num_test=dp.num_test,
        use_train=False,
    )
    bhac_loader = BHACDataloader(
        dataset_val,
        sub_x=dp.sub_x,
        sub_t=dp.sub_t,
        ind_x=dp.ind_x,
        ind_t=dp.ind_t,
        ind_t_start=dp.ind_t_start,
    )
    dataloader, _ = bhac_loader.create_dataloader(
        batch_size=1,
        shuffle=False,
        num_workers=(
            num_workers if num_workers is not None
            else cfg.val_loader_params.num_workers
        ),
        pin_memory=False,
        distributed=False,
    )
    return dataloader, bhac_loader.t_real


def build_model(cfg, device):
    """FNO construction identical to train_bhac.py / plot_index.py."""
    mp = cfg.model_params
    model = FNO(
        in_channels=mp.in_dim,
        out_channels=mp.out_dim,
        decoder_layers=mp.decoder_layers,
        decoder_layer_size=mp.fc_dim,
        dimension=mp.dimension,
        latent_channels=mp.layers,
        num_fno_layers=mp.num_fno_layers,
        num_fno_modes=mp.modes,
        padding=[mp.pad_z, mp.pad_y, mp.pad_x],
    )
    model = model.to(device)
    input_norm = torch.tensor(mp.input_norm).to(device)
    output_norm = torch.tensor(mp.output_norm).to(device)
    return model, input_norm, output_norm


def load_model_epoch(model, ckpt_path, epoch, device):
    """Load checkpoint weights; epoch=None loads the latest.

    Returns the actually loaded epoch — callers must use it for labels and
    file names (and should warn if it differs from the requested one).
    """
    loaded = load_checkpoint(
        ckpt_path, model, optimizer=None, scheduler=None,
        epoch=epoch, device=device,
    )
    model.eval()
    return loaded


@torch.no_grad()
def predict(model, inputs, input_norm, output_norm):
    """Normalized forward pass, as in training. inputs: (B,nt,nx,ny,11)."""
    return (
        model((inputs / input_norm).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        * output_norm
    )


@torch.no_grad()
def residual_fields(loss_fn, fields, eta):
    """Pointwise residual fields for one batch.

    fields: (B, nt, nx, ny, 7) — [u1, u2, b1, b2, p, e3, rho], either the
    ground-truth outputs or a model prediction.
    eta: (B, nt, nx, ny) — the eta input channel (inputs[..., -1]).

    Returns {name: (B, nt, nx, ny) residual tensor} for QUANTITIES. The
    training losses are exactly residual.pow(2).mean() of these.
    """
    u1 = fields[..., 0]
    u2 = fields[..., 1]
    b1 = fields[..., 2]
    b2 = fields[..., 3]
    p = fields[..., 4]
    e3 = fields[..., 5]
    rho = fields[..., 6]
    FI1, FI2, MO, ES0, ES1, ES2, C1 = loss_fn.mhd_pde(u1, u2, b1, b2, p, e3, rho, eta)
    divB = loss_fn.mhd_constraint(b1, b2)
    return dict(zip(QUANTITIES, [FI1, FI2, MO, ES0, ES1, ES2, C1, divB]))


def mean_square(residual):
    """The logged loss metric: mean squared residual."""
    return residual.pow(2).mean().item()


@torch.no_grad()
def get_val_sample(dataloader, sample_idx, device):
    """Fetch one validation batch by index (batch_size=1, unshuffled)."""
    for i, (inputs, outputs) in enumerate(dataloader):
        if i == sample_idx:
            return (
                inputs.to(device, dtype=torch.float32),
                outputs.to(device, dtype=torch.float32),
            )
    raise SystemExit(
        f"Validation sample {sample_idx} not found (val set has {i + 1} samples)"
    )
