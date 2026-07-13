"""
Plot the coarse8 / 256p / small-t (t = 0 .. 2.5) run for the fields (b1, rho, e3):
how the FNO prediction sharpens between an early checkpoint (epoch 100, end of
the data-only warm-up) and the converged model (epoch 2800).

This run is *spatially* sparse: the data loss only sees a 32x32 coarse grid
(stride 8 on the 256x256 grid, `coarse_factor = 8`), while every frame in time
is supervised.  The 1024 supervised pixels are marked with "+" markers -- white
on value panels, gray on error panels.  (The temporally sparse sibling is
plot_stride8_t0_t2_5_256p.py.)

Per field, per-epoch 3-panel figures (BHAC truth | FNO prediction | error) for
t=0 and a selection of later timesteps, plus per-timestep epoch-comparison
figures.

Output layout (one subfolder per field):
  figs/collages/coarse8_t0_t2_5_256p_figs/{field}/
    epoch{N}_s{SS}_t{TT}_{field}.png              (per epoch, per timestep)
    pred_comparison_ep{A}_ep{B}_t{TT}_{field}.png
    error_comparison_ep{A}_ep{B}_t{TT}_{field}.png

Value normalization (true/pred panels): per-timestep, from the true field at
that step.  Error normalization: per-timestep symmetric and SHARED across the
two epochs, so the shrinking error from epoch 100 -> 2800 is directly
comparable and early small errors are not invisible next to large late ones.

Data source: the fields.h5 dumps plot_index.py wrote under the run's checkpoint
dir (see plotters/collage_h5.py); the script reads them there and runs on the
HPC -- no local pre-copied .h5 under figs/data/ is needed.
"""

import os
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from collage_h5 import fields_h5_path

# This script lives in plotters/, one level below the repo root. Anchor the
# figs/ paths to the repo root so it runs the same from any working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "axes.formatter.use_mathtext": True,
    "axes.labelsize": 13,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "savefig.dpi": 1200,
    "savefig.bbox": "tight",
})

RUN = "coarse8_t0_t2_5_256p"
FIELDS = [
    {"name": "b1",  "latex": r"B_1"},   # in-plane magnetic field component
    {"name": "rho", "latex": r"\rho"},  # density
    {"name": "e3",  "latex": r"E_3"},   # out-of-plane electric field
]

# Epoch checkpoints to compare: end of data-only warm-up vs converged model.
EPOCHS = [100, 2800]

# Timestep indices to plot: t=0 and roughly t=0.25, 0.5, 1.0, 1.5, 2.0, 2.5.
PLOT_INDICES = [0, 10, 20, 40, 60, 80, 100]


def output_dir(field):
    return os.path.join(_REPO_ROOT, "figs", "collages", "coarse8_t0_t2_5_256p_figs", field)


def load_all(field, epoch):
    path = fields_h5_path(RUN, epoch, field)
    with h5py.File(path, "r") as f:
        return {
            "epoch":   epoch,
            "label":   f"epoch {epoch}",
            "sample":  int(f.attrs["sample"]),
            "coarse":  int(f.attrs["coarse_factor"]),
            "eta":     float(f.attrs["eta"]),
            "x":       f["x"][()],
            "y":       f["y"][()],
            "t":       f["t"][()],
            "true":    f["true"][()],
            "pred":    f["pred"][()],
            "error":   f["error"][()],
            "mask":    f["data_mask_spatial"][()].astype(bool),
        }


def add_mask_markers(ax, x, y, mask, color="white"):
    rows, cols = np.where(mask)
    ax.scatter(x[cols], y[rows], marker="+", s=8, linewidths=0.5,
               color=color, alpha=0.6, zorder=5)


def _add_mask_legend(fig, coarse, n_sup):
    side = int(round(np.sqrt(n_sup)))
    handle = mpl.lines.Line2D(
        [], [], marker="+", color="white", linestyle="",
        markeredgewidth=0.7, markersize=6,
        label=f"data-loss pixels (stride {coarse}, {side}x{side})",
    )
    fig.legend(handles=[handle], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), framealpha=0.8)


# ----------------------------------------------------------------------------
# Figure builders
# ----------------------------------------------------------------------------
def make_single_figure(d, ti, val_norm, err_norm, field_latex):
    """BHAC truth | FNO pred | error for one epoch at one timestep."""
    X, Y = np.meshgrid(d["x"], d["y"], indexing="ij")
    t_val = float(d["t"][ti])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    panels = [
        (axes[0], d["true"][ti],  val_norm, "jet",    "BHAC (truth)",            "white"),
        (axes[1], d["pred"][ti],  val_norm, "jet",    "FNO (prediction)",        "white"),
        (axes[2], d["error"][ti], err_norm, "RdBu_r", r"Error (truth $-$ pred)", "0.5"),
    ]
    for ax, data, norm, cmap, subtitle, mc in panels:
        pcm = ax.pcolormesh(X, Y, data, cmap=cmap, shading="auto", norm=norm)
        plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        add_mask_markers(ax, d["x"], d["y"], d["mask"], color=mc)
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
        ax.set_aspect("equal"); ax.set_title(subtitle)
    fig.suptitle(
        rf"${field_latex}$,  $t = {t_val:.2f}$,  {d['label']},  sample {d['sample']}",
        y=1.02,
    )
    _add_mask_legend(fig, d["coarse"], int(d["mask"].sum()))
    return fig


def make_pred_figure(data_list, ti, val_norm, field_latex):
    """BHAC truth (once) + FNO pred panels for each epoch, shared value scale."""
    n = len(data_list)
    fig, axes = plt.subplots(1, 1 + n, figsize=(4.5 * (1 + n), 4.8))
    d0 = data_list[0]
    X, Y = np.meshgrid(d0["x"], d0["y"], indexing="ij")
    t_val = float(d0["t"][ti])

    ax = axes[0]
    pcm = ax.pcolormesh(X, Y, d0["true"][ti], cmap="jet",
                        shading="auto", norm=val_norm)
    plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    add_mask_markers(ax, d0["x"], d0["y"], d0["mask"])
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
    ax.set_aspect("equal"); ax.set_title("BHAC (truth)")

    for i, d in enumerate(data_list):
        ax = axes[1 + i]
        pcm = ax.pcolormesh(X, Y, d["pred"][ti], cmap="jet",
                            shading="auto", norm=val_norm)
        plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        add_mask_markers(ax, d["x"], d["y"], d["mask"])
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
        ax.set_aspect("equal"); ax.set_title(f"FNO pred\n{d['label']}")

    eps = " vs ".join(str(d["epoch"]) for d in data_list)
    fig.suptitle(
        rf"${field_latex}$,  $t = {t_val:.2f}$,  $\eta = {d0['eta']:.2e}$,  "
        rf"FNO predictions (epochs {eps}),  sample {d0['sample']}", y=1.02)
    _add_mask_legend(fig, d0["coarse"], int(d0["mask"].sum()))
    return fig


def make_error_figure(data_list, ti, err_norm, field_latex):
    """Error panels for each epoch, shared symmetric scale + RMSE in title."""
    n = len(data_list)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8))
    d0 = data_list[0]
    X, Y = np.meshgrid(d0["x"], d0["y"], indexing="ij")
    t_val = float(d0["t"][ti])

    for i, d in enumerate(data_list):
        ax = axes[i]
        e = d["error"][ti]
        rmse = float(np.sqrt((e ** 2).mean()))
        pcm = ax.pcolormesh(X, Y, e, cmap="RdBu_r", shading="auto", norm=err_norm)
        plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        add_mask_markers(ax, d["x"], d["y"], d["mask"], color="0.5")
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
        ax.set_aspect("equal")
        ax.set_title(f"Error (truth - pred)\n{d['label']}, RMSE = {rmse:.3f}")

    eps = " vs ".join(str(d["epoch"]) for d in data_list)
    fig.suptitle(
        rf"${field_latex}$,  $t = {t_val:.2f}$,  prediction errors (epochs {eps}),  "
        rf"sample {d0['sample']}", y=1.02)
    _add_mask_legend(fig, d0["coarse"], int(d0["mask"].sum()))
    return fig


# ----------------------------------------------------------------------------
def plot_field(field, field_latex):
    out_dir = output_dir(field)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{field}] -> {out_dir}/")

    data_list = []
    for epoch in EPOCHS:
        d = load_all(field, epoch)
        data_list.append(d)
        print(f"  loaded epoch {epoch}")

    sample = data_list[0]["sample"]
    ep_tag = "_".join(f"ep{d['epoch']}" for d in data_list)

    def val_norm_at(ti):
        true_ti = data_list[0]["true"][ti]
        return Normalize(vmin=float(true_ti.min()), vmax=float(true_ti.max()))

    def err_norm_at(ti):
        err_abs = max(
            max(abs(float(d["error"][ti].min())),
                abs(float(d["error"][ti].max())))
            for d in data_list
        )
        return Normalize(vmin=-err_abs, vmax=err_abs)

    for ti in PLOT_INDICES:
        vn = val_norm_at(ti)
        en = err_norm_at(ti)

        for d in data_list:
            fig = make_single_figure(d, ti, vn, en, field_latex)
            out = os.path.join(
                out_dir, f"epoch{d['epoch']:04d}_s{sample:03d}_t{ti:03d}_{field}.png")
            fig.savefig(out); plt.close(fig); print(f"  saved {out}")

        fig = make_pred_figure(data_list, ti, vn, field_latex)
        out = os.path.join(out_dir, f"pred_comparison_{ep_tag}_t{ti:03d}_{field}.png")
        fig.savefig(out); plt.close(fig); print(f"  saved {out}")

        fig = make_error_figure(data_list, ti, en, field_latex)
        out = os.path.join(out_dir, f"error_comparison_{ep_tag}_t{ti:03d}_{field}.png")
        fig.savefig(out); plt.close(fig); print(f"  saved {out}")


def main():
    for spec in FIELDS:
        plot_field(spec["name"], spec["latex"])
    print("Done.")


if __name__ == "__main__":
    main()
