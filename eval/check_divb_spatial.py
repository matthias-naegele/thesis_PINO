# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualise the spatial structure of divB for a single BHAC sample at one or
more time frames.

The magnetic-field components b1, b2 are read from a BHAC sample (output
channels 2 and 3, as in the loss code) and divB = d(b1)/dx + d(b2)/dy is
computed exactly as in the loss's `mhd_constraint`: the derivative operator is
selected by `--diff-type` (`fourier` for spectral derivatives, `fd` for finite
differences on the periodic domain). Both operators share the same I/O
contract — input (1, nt, nx, ny), output (1, 2*nt, nx, ny) with d/dx in
channels [0:nt] and d/dy in channels [nt:2*nt] — so the slicing below is
identical for either choice. The full time sequence (nt frames) is passed in
one call, matching how the loss uses it.

`--time-idx` accepts a comma-separated list of frames; each requested frame is
plotted as one row of three columns: b1 | b2 | |divB|.

Example:
  python check_divb_spatial.py --diff-type fourier --time-idx 0,5,10 --out divB_spatial.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from losses.fourier_derivatives import fourier_derivatives
from losses.finite_diff import fd_derivatives_periodic


from dataloaders import BHACDataloader, BHACUniformDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/mhd_bhac.yaml",
    )
    parser.add_argument(
        "--Lx",
        type=float,
        default=None,
        help="Physical domain length in x. Defaults to loss_params.Lx in config.",
    )
    parser.add_argument(
        "--Ly",
        type=float,
        default=None,
        help="Physical domain length in y. Defaults to loss_params.Ly in config.",
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Which sample (simulation run) in the training dataset to plot.",
    )
    parser.add_argument(
        "--time-idx",
        type=str,
        default="0",
        help="Comma-separated list of time-frame indices to plot, e.g. '0,5,10'.",
    )
    parser.add_argument(
        "--diff-type",
        choices=["fourier", "fd"],
        default=None,
        help="Derivative operator for divB. Defaults to loss_params.diff_type "
        "in the config (matching the loss).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="divB_spatial.png",
    )
    return parser.parse_args()


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
    dp = cfg.dataset_params
    lp = cfg.loss_params

    # Fall back to config values if not given on the command line
    Lx = args.Lx if args.Lx is not None else float(lp.Lx)
    Ly = args.Ly if args.Ly is not None else float(lp.Ly)
    diff_type = args.diff_type if args.diff_type is not None else str(lp.diff_type)

    # Parse requested time indices
    time_indices = [int(t) for t in args.time_idx.split(",")]

    # ------------------------------------------------------------------ #
    # Build dataset / dataloader (same as training code)
    # ------------------------------------------------------------------ #
    dataset = BHACUniformDataset(
        dp.data_dir,
        output_names=dp.output_names,
        file_name=dp.file_name,
        num_train=dp.num_train,
        num_test=dp.num_test,
        use_train=True,
    )

    dl_obj = BHACDataloader(
        dataset,
        sub_x=dp.sub_x,
        sub_t=dp.sub_t,
        ind_x=dp.ind_x,
        ind_t=dp.ind_t,
        ind_t_start=dp.ind_t_start,
    )

    # We only need one specific sample; skip to it via the dataset index
    # rather than iterating through a shuffled dataloader.
    inputs, outputs = dl_obj[args.sample_idx]
    # Add batch dimension: (1, nt, nx, ny, C)
    inputs = inputs.unsqueeze(0)
    outputs = outputs.unsqueeze(0)

    # ------------------------------------------------------------------ #
    # Grid info
    # ------------------------------------------------------------------ #
    t_vals = inputs[0, :, 0, 0, 0]
    x_vals = inputs[0, 0, :, 0, 1]
    y_vals = inputs[0, 0, 0, :, 2]
    nt = outputs.shape[1]
    nx = outputs.shape[2]
    ny = outputs.shape[3]

    print(f"Sample index : {args.sample_idx}")
    print(f"Grid         : nt={nt}, nx={nx}, ny={ny}")
    print(f"t range      : [{t_vals[0]:.4f}, {t_vals[-1]:.4f}]")
    print(f"x range      : [{x_vals[0]:.4f}, {x_vals[-1]:.4f}]  dx={x_vals[1]-x_vals[0]:.6f}")
    print(f"y range      : [{y_vals[0]:.4f}, {y_vals[-1]:.4f}]  dy={y_vals[1]-y_vals[0]:.6f}")
    print(f"Lx={Lx:.6f}  Ly={Ly:.6f}")
    print(f"diff_type    : {diff_type}")
    print()

    # ------------------------------------------------------------------ #
    # Compute divB over the FULL time sequence.
    # This exactly mirrors mhd_constraint() in the loss code:
    #   - select the derivative operator from diff_type (fourier / fd)
    #   - pass the full (1, nt, nx, ny) tensor
    #   - x-deriv lives in channels [0 : nt]
    #   - y-deriv lives in channels [nt : 2*nt]
    # ------------------------------------------------------------------ #
    b1_full = outputs[0, :, :, :, 2].unsqueeze(0).float()  # (1, nt, nx, ny)
    b2_full = outputs[0, :, :, :, 3].unsqueeze(0).float()

    deriv = fourier_derivatives if diff_type == "fourier" else fd_derivatives_periodic
    f_db1 = deriv(b1_full, [Lx, Ly])  # (1, 2*nt, nx, ny)
    f_db2 = deriv(b2_full, [Lx, Ly])

    # Sanity-check that the output has the expected layout
    assert f_db1.shape == (1, 2 * nt, nx, ny), (
        f"Unexpected {diff_type} derivative output shape: {f_db1.shape}, "
        f"expected (1, {2*nt}, {nx}, {ny})"
    )

    # ------------------------------------------------------------------ #
    # Plot: one ROW per requested time frame, three columns: b1 | b2 | |divB|
    # ------------------------------------------------------------------ #
    n_frames = len(time_indices)
    fig, axes = plt.subplots(
        n_frames, 3,                    # ← rows=frames, cols=fields (b1, b2, |divB|)
        figsize=(15, 4 * n_frames),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.45)   # small extra vertical breathing room between rows

    for row, t_idx in enumerate(time_indices):
        if t_idx >= nt:
            raise ValueError(
                f"--time-idx {t_idx} is out of range for nt={nt}. "
                f"Valid range: 0 .. {nt - 1}."
            )

        # Select derivatives for this specific time frame using nt-aware offsets
        b1_x = f_db1[0, t_idx,        :nx, :ny]  # x-deriv at t_idx
        b2_y = f_db2[0, nt + t_idx,   :nx, :ny]  # y-deriv at t_idx  ← key fix

        divB = (b1_x + b2_y).detach().cpu().numpy()

        b1_np = outputs[0, t_idx, :, :, 2].cpu().numpy()
        b2_np = outputs[0, t_idx, :, :, 3].cpu().numpy()

        mse  = float((divB ** 2).mean())
        mabs = float(np.abs(divB).mean())
        mmax = float(np.abs(divB).max())

        print(
            f"t_idx={t_idx:3d}  t={t_vals[t_idx]:.4f}  "
            f"MSE={mse:.4e}  mean|divB|={mabs:.4e}  max|divB|={mmax:.4e}"
        )

        extent = [0, Lx, 0, Ly]

        # Col 0: b1
        im0 = axes[row, 0].imshow(b1_np.T, origin="lower", extent=extent)
        axes[row, 0].set_title(f"b1  (t_idx={t_idx}, t={t_vals[t_idx]:.3f})")
        plt.colorbar(im0, ax=axes[row, 0])

        # Col 1: b2
        im1 = axes[row, 1].imshow(b2_np.T, origin="lower", extent=extent)
        axes[row, 1].set_title(f"b2  (t_idx={t_idx}, t={t_vals[t_idx]:.3f})")
        plt.colorbar(im1, ax=axes[row, 1])

        # Col 2: |divB|
        im2 = axes[row, 2].imshow(
            np.abs(divB).T, origin="lower", extent=extent, cmap="hot"
        )
        axes[row, 2].set_title(
            f"|divB|  MSE={mse:.2e}\nmean={mabs:.2e}  max={mmax:.2e}"
        )
        plt.colorbar(im2, ax=axes[row, 2])

    fig.suptitle(
        f"sample_idx={args.sample_idx}  diff_type={diff_type}  "
        f"Lx={Lx:.4f}  Ly={Ly:.4f}  nt={nt}  nx={nx}  ny={ny}",
        fontsize=11,
    )
    plt.savefig(args.out, dpi=1600, bbox_inches="tight")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
