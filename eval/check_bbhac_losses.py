# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute BHAC data/PDE/divB losses directly on ground-truth data.

This script does not run a model. It loads BHAC batches from the configured dataloader,
then evaluates:
  - data loss on (outputs vs outputs), expected ~0
  - PDE residual losses on the same outputs (treated as prediction)
  - divB constraint loss on the same outputs
  - eta value summary for each batch when batch_size == 1

Example:
  python check_bbhac_losses.py --split train --num-batches 5
"""

import argparse
import os
from typing import Optional

import torch
from omegaconf import OmegaConf

from dataloaders import BHACDataloader, BHACUniformDataset
from losses import LossMHD_PhysicsNeMo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/mhd_bhac.yaml",
        help="Path to YAML config (relative to the repository root).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=3,
        help="Number of batches to process.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional dataloader batch-size override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cpu/cuda).",
    )
    return parser.parse_args()


def build_dataloader(cfg, split: str, batch_size_override: Optional[int]):
    dataset_params = cfg.dataset_params
    train_loader_params = cfg.train_loader_params
    val_loader_params = cfg.val_loader_params

    dataset = BHACUniformDataset(
        dataset_params.data_dir,
        output_names=dataset_params.output_names,
        file_name=dataset_params.file_name,
        num_train=dataset_params.num_train,
        num_test=dataset_params.num_test,
        use_train=(split == "train"),
    )

    loader_cfg = train_loader_params if split == "train" else val_loader_params
    batch_size = (
        batch_size_override if batch_size_override is not None else loader_cfg.batch_size
    )

    dataloader, _ = BHACDataloader(
        dataset,
        sub_x=dataset_params.sub_x,
        sub_t=dataset_params.sub_t,
        ind_x=dataset_params.ind_x,
        ind_t=dataset_params.ind_t,
        ind_t_start=dataset_params.ind_t_start,
    ).create_dataloader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_cfg.num_workers,
        pin_memory=loader_cfg.pin_memory,
        distributed=False,
    )
    return dataloader, batch_size


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    # Run configs derive their paths from a sibling paths.yaml via Hydra's
    # `defaults` list; OmegaConf.load doesn't process that, so compose the
    # sibling paths.yaml (config/ or a checkpoint snapshot dir) when present so
    # ${data_root}/${output_root} resolve. Harmless for already-literal configs.
    paths_yaml = os.path.join(os.path.dirname(args.config), "paths.yaml")
    if os.path.isfile(paths_yaml):
        cfg = OmegaConf.merge(OmegaConf.load(paths_yaml), cfg)

    dataloader, batch_size = build_dataloader(cfg, args.split, args.batch_size)
    loss_fn = LossMHD_PhysicsNeMo(**cfg.loss_params)

    keys = ["data", "pde", "FI1", "FI2", "MO", "ES0", "ES1", "ES2", "C1", "divB"]
    running = {k: 0.0 for k in keys}

    print(f"Using device: {args.device}")
    print(f"Split: {args.split}, batch_size={batch_size}, num_batches={args.num_batches}")
    print("Computing losses on raw data (pred = true = dataloader outputs)")

    n_seen = 0
    for batch_idx, (inputs, outputs) in enumerate(dataloader):
        if batch_idx >= args.num_batches:
            break

        inputs = inputs.to(args.device, dtype=torch.float32)
        outputs = outputs.to(args.device, dtype=torch.float32)

        with torch.no_grad():
            # Pure data-loss check: should be ~0 because pred == true
            loss_data = loss_fn.data_loss(outputs, outputs)

            # PDE residual check directly on dataset fields
            u1 = outputs[..., 0]
            u2 = outputs[..., 1]
            b1 = outputs[..., 2]
            b2 = outputs[..., 3]
            p = outputs[..., 4]
            e3 = outputs[..., 5]
            rho = outputs[..., 6]
            eta = inputs[..., -1]

            FI1, FI2, MO, ES0, ES1, ES2, C1 = loss_fn.mhd_pde(
                u1, u2, b1, b2, p, e3, rho, eta
            )
            (
                loss_pde,
                loss_FI1,
                loss_FI2,
                loss_MO,
                loss_ES0,
                loss_ES1,
                loss_ES2,
                loss_C1,
            ) = loss_fn.mhd_pde_loss(FI1, FI2, MO, ES0, ES1, ES2, C1, return_all_losses=True)

            # divB check directly on dataset fields
            div_B = loss_fn.mhd_constraint(b1, b2)
            _, loss_div_B = loss_fn.mhd_constraint_loss(div_B, return_all_losses=True)

        batch_values = {
            "data": loss_data.item(),
            "pde": loss_pde.item(),
            "FI1": loss_FI1.item(),
            "FI2": loss_FI2.item(),
            "MO": loss_MO.item(),
            "ES0": loss_ES0.item(),
            "ES1": loss_ES1.item(),
            "ES2": loss_ES2.item(),
            "C1": loss_C1.item(),
            "divB": loss_div_B.item(),
        }

        print(f"\nBatch {batch_idx}:")
        for key in keys:
            print(f"  {key:4s}: {batch_values[key]:.6e}")
            running[key] += batch_values[key]

        if inputs.shape[0] == 1:
            eta_batch = inputs[0, ..., -1]
            eta_min = torch.min(eta_batch).item()
            eta_max = torch.max(eta_batch).item()
            if abs(eta_min - eta_max) < 1e-12:
                print(f"  eta : {eta_min:.6e}")
            else:
                eta_mean = torch.mean(eta_batch).item()
                print(
                    "  eta : non-uniform over grid/time "
                    f"(min={eta_min:.6e}, max={eta_max:.6e}, mean={eta_mean:.6e})"
                )

        n_seen += 1

    if n_seen == 0:
        print("\nNo batches processed.")
        return

    print("\n=== Mean over evaluated batches ===")
    for key in keys:
        print(f"{key:4s}: {running[key] / n_seen:.6e}")


if __name__ == "__main__":
    main()
