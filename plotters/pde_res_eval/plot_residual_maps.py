# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Residual-map figures: signed residual, data vs FNO (data-only) vs FNO (+physics).

For each run and each requested quantity (default FI1, C1, divB) this renders
ONE three-panel figure on validation sample 001 at the run's default frame
(see runs.py — for the stride8 runs that is the middle of a data-unsupervised
range):

  left    BHAC ground truth passed through the training residual operator,
  middle  FNO at the end of the data-only warm-up (epoch 100 / 60 zusatz),
  right   FNO after the physics ramp (latest checkpoint by default),

all three showing the SIGNED residual (its sign structure is physically
meaningful; |residual| maps lose it) on ONE shared symmetric linear scale:
diverging RdBu_r, clipped at
+-3x the RMS of the data panel (--vmax-rms). Each panel is annotated with its
RMS over the shown frame; the quantitative claim in the logged metric (mean
squared residual, full val set) is the companion table from
compute_metrics.py.

A "residual error" panel would be meaningless — the residual already is the
error-like field — so the honest presentation is this side-by-side.

Run on the HPC (venv active), e.g.:
  python plotters/pde_res_eval/plot_residual_maps.py --run all
  python plotters/pde_res_eval/plot_residual_maps.py --run stride8_512p --quantities FI1 divB

Outputs under figs/pde_res_eval_out/<run_key>/figs/: one PNG per quantity plus a
matching .h5 with the raw 2D arrays (so the PNG can be re-rendered at any
color scale), both named with run/quantity/sample/frame/epochs.
"""

import argparse
import os

import h5py
import numpy as np
import torch
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from common import (  # noqa: E402 (sys.path set up in common)
    QUANTITIES,
    REPO_ROOT,
    build_model,
    build_val_dataloader,
    get_val_sample,
    load_model_epoch,
    load_run_config,
    predict,
    residual_fields,
)
from losses import LossMHD_PhysicsNeMo
from runs import RUNS, resolve_run_keys

# Same physics-paper style as plot_index.py.
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "axes.formatter.use_mathtext": True,
    "axes.labelsize": 12,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

QUANTITY_LABELS = {
    "FI1": r"Faraday $\mathrm{FI}_1$",
    "FI2": r"Faraday $\mathrm{FI}_2$",
    "MO": r"Ohm/Maxwell $\mathrm{MO}$",
    "ES0": r"EoS $\mathrm{ES}_0$",
    "ES1": r"EoS $\mathrm{ES}_1$",
    "ES2": r"EoS $\mathrm{ES}_2$",
    "C1": r"continuity $\mathrm{C}_1$",
    "divB": r"$\nabla\cdot\vec{B}$",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default="all",
                        help=f"all or one of: {', '.join(RUNS)}")
    parser.add_argument("--quantities", nargs="+", default=["FI1", "C1", "divB"],
                        choices=QUANTITIES,
                        help="Residual quantities to map (default: FI1 C1 divB)")
    parser.add_argument("--sample", type=int, default=1,
                        help="Validation sample index (default 1 = sample_001)")
    parser.add_argument("--time-index", type=int, default=None,
                        help="Frame within the loaded window (default: the "
                             "run's registry default, see runs.py)")
    parser.add_argument("--cmap", default="RdBu_r",
                        help="Diverging colormap for the signed residual "
                             "(default RdBu_r)")
    parser.add_argument("--vmax-rms", type=float, default=3.0,
                        help="Symmetric color range: vmax = N x RMS of the "
                             "data panel (default 3)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup-epoch", type=int, default=None,
                        help="Override the run's end-of-warm-up epoch")
    parser.add_argument("--final-epoch", type=int, default=None,
                        help="Checkpoint epoch for the +physics panel "
                             "(default: latest checkpoint)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figs" / "pde_res_eval_out"))
    return parser.parse_args()


def frame_supervision_note(ti, stride, coarse):
    """Human-readable note on how the data loss supervises the shown frame."""
    if stride > 1:
        kind = ("unsupervised mid-gap frame" if ti % stride
                else "supervised anchor frame")
        return f"{kind} (stride {stride})"
    return f"every frame supervised, spatially coarse $\\times${coarse}"


def plot_three_panel(panels, x, y, quantity, spec, sample, ti, t_val, eta_val,
                     sup_note, cmap, vmax_rms, save_path):
    """panels: ordered dict-like [(panel_title, 2D signed-residual array), ...]."""
    # Shared symmetric scale anchored to the DATA panel's RMS,
    # so the FNO panels are read against the data's own residual level.
    vmax = vmax_rms * float(np.sqrt(np.mean(panels[0][1] ** 2)))
    if vmax == 0.0:  # degenerate all-zero data residual
        vmax = max(float(np.abs(arr).max()) for _, arr in panels) or 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)

    X, Y = np.meshgrid(x, y, indexing="ij")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), sharey=True)
    for ax, (title, arr) in zip(axes, panels):
        pcm = ax.pcolormesh(X, Y, arr, cmap=cmap, norm=norm, shading="auto")
        rms = float(np.sqrt(np.mean(arr ** 2)))
        ax.set_title(f"{title}\nRMS $= {rms:.3g}$")
        ax.set_xlabel(r"$x$")
        ax.set_aspect("equal")
    axes[0].set_ylabel(r"$y$")
    cbar = fig.colorbar(pcm, ax=axes, shrink=0.9, pad=0.015)
    cbar.set_label(rf"{QUANTITY_LABELS[quantity]} residual")
    # DejaVu Serif for the suptitle: cmr10 lacks the underscore/dash glyphs
    # that the config name needs.
    fig.suptitle(
        f"{spec.config_name} — {QUANTITY_LABELS[quantity]} residual, "
        f"val sample {sample:03d}, $\\eta={eta_val:.2e}$\n"
        f"$t={t_val:.3f}$ (frame {ti}, {sup_note}), "
        f"linear scale, vmax $= {vmax_rms:g}\\,$RMS$_\\mathrm{{data}}$",
        y=1.04, fontfamily="DejaVu Serif",
    )
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[out] {save_path}")


def dump_h5(save_path, panels, x, y, attrs):
    with h5py.File(save_path, "w") as f:
        f.create_dataset("x", data=np.asarray(x))
        f.create_dataset("y", data=np.asarray(y))
        for name, arr in panels:
            f.create_dataset(name, data=arr)
        for k, v in attrs.items():
            f.attrs[k] = v


def main():
    for key in resolve_run_keys(ARGS.run):
        spec = RUNS[key]
        print(f"\n===== {spec.key}: {spec.config_name} =====")
        cfg, ckpt_path, source = load_run_config(spec.config_name)
        print(f"[config] {source}")
        loss_fn = LossMHD_PhysicsNeMo(**cfg.loss_params)
        dataloader, t_real = build_val_dataloader(cfg, num_workers=ARGS.num_workers)
        inputs, outputs = get_val_sample(dataloader, ARGS.sample, ARGS.device)
        eta = inputs[..., -1]
        eta_val = float(eta[0, 0, 0, 0])

        ti = ARGS.time_index if ARGS.time_index is not None else spec.fig_time_index
        nt = outputs.shape[1]
        if not 0 <= ti < nt:
            raise SystemExit(f"time index {ti} outside the loaded window [0, {nt - 1}]")
        t_val = float(t_real[ti])
        stride = int(cfg.loss_params.data_loss_stride)
        coarse = int(cfg.loss_params.data_loss_coarse_factor)
        sup_note = frame_supervision_note(ti, stride, coarse)
        print(f"[frame] {ti} (t={t_val:.3f}): {sup_note} — {spec.fig_time_note}")

        # Residuals of the data itself, then of the FNO at both epochs. The
        # residual is computed on the FULL window (the time stencil needs the
        # neighboring frames); only frame ti is rendered.
        res_by_panel = {"true": residual_fields(loss_fn, outputs, eta)}
        model, input_norm, output_norm = build_model(cfg, ARGS.device)
        warmup_req = ARGS.warmup_epoch if ARGS.warmup_epoch is not None else spec.warmup_epoch
        epochs = {}
        for tag, epoch_req in [("warmup", warmup_req), ("final", ARGS.final_epoch)]:
            loaded = load_model_epoch(model, ckpt_path, epoch_req, ARGS.device)
            if epoch_req is not None and loaded != epoch_req:
                print(f"[WARN] requested epoch {epoch_req} but loaded {loaded}")
            epochs[tag] = loaded
            pred = predict(model, inputs, input_norm, output_norm)
            res_by_panel[tag] = residual_fields(loss_fn, pred, eta)
        if epochs["final"] <= epochs["warmup"]:
            print(f"[WARN] final epoch {epochs['final']} is not after the "
                  f"warm-up epoch {epochs['warmup']} — panels 2 and 3 are "
                  f"not a data-only vs +physics comparison")

        # Grid coordinates from the input channels (inputs: B,nt,nx,ny,11
        # with channels [t, x, y, ic fields..., eta]).
        x = inputs[0, 0, :, 0, 1].cpu().numpy()
        y = inputs[0, 0, 0, :, 2].cpu().numpy()
        fig_dir = os.path.join(ARGS.out_dir, spec.key, "figs")
        os.makedirs(fig_dir, exist_ok=True)

        for q in ARGS.quantities:
            arrs = {tag: res_by_panel[tag][q][0, ti].cpu().numpy()
                    for tag in ("true", "warmup", "final")}
            panels = [
                ("BHAC data", arrs["true"]),
                (f"FNO epoch {epochs['warmup']} (data-only)", arrs["warmup"]),
                (f"FNO epoch {epochs['final']} (+physics)", arrs["final"]),
            ]
            base = (f"resmap_{q}_{spec.config_name}_sample{ARGS.sample:03d}"
                    f"_t{ti:03d}_ep{epochs['warmup']:04d}_vs_ep{epochs['final']:04d}")
            plot_three_panel(
                panels, x, y, q, spec, ARGS.sample, ti, t_val, eta_val,
                sup_note, ARGS.cmap, ARGS.vmax_rms,
                os.path.join(fig_dir, base + ".png"),
            )
            # The .h5 stores the same signed arrays, so the figure can be
            # re-rendered at any scale/colormap without checkpoints or data.
            dump_h5(
                os.path.join(fig_dir, base + ".h5"),
                [("true", arrs["true"]),
                 ("fno_warmup", arrs["warmup"]),
                 ("fno_final", arrs["final"])],
                x, y,
                attrs={
                    "run": spec.key, "config": spec.config_name,
                    "quantity": q, "sample": ARGS.sample,
                    "time_index": ti, "t": t_val, "eta": eta_val,
                    "epoch_warmup": epochs["warmup"],
                    "epoch_final": epochs["final"],
                    "data_loss_stride": stride,
                    "data_loss_coarse_factor": coarse,
                    "diff_type": str(cfg.loss_params.diff_type),
                    "tend": float(cfg.loss_params.tend),
                    "note": "arrays are the SIGNED residual at the given frame",
                },
            )


if __name__ == "__main__":
    ARGS = parse_args()
    main()
