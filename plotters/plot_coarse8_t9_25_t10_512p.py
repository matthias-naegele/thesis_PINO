"""
Thesis figures for the coarse8 512p (zusatz) run: how the PDE +
div-B physics losses help the FNO learn beyond the sparse data supervision.
Generated per field (rho, Jz, e3); add more by extending FIELDS below.

The model is trained with a *data loss enforced only on a coarse 64x64 subset*
of the 512x512 grid (data_loss_coarse_factor = 8, i.e. stride 8 -> 4096
supervised pixels).  Two checkpoints are compared:

  - epoch 60   : end of phase 0, DATA LOSS ONLY (no physics yet).
  - epoch 1680 : final model, with PDE + div-B constraint losses ramped in.

Figures (style matched to plot_coarse8_t0_t2_5_256p.py), one subfolder per field:
  figs/collages/coarse8_t9_25_t10_512p_figs/{field}/
    coarse_vs_true_t{TT}.png             — full truth vs the coarse data the
                                           loss actually sees (motivation panel)
    single/epoch{N}_t{TT}.png            — truth | pred | error for one epoch
    compare_pred_ep{A}_ep{B}_t{TT}.png   — truth + pred(ep60) + pred(ep1680)
    compare_error_ep{A}_ep{B}_t{TT}.png  — error(ep60) vs error(ep1680)

Value normalization (truth/pred panels): per-timestep, from the true field at
that step.  Error normalization: per-timestep symmetric and SHARED across the
two epochs so the shrinking error from epoch 60 -> 1680 is directly comparable.

The coarse data-loss pixels (stride 8, 64x64) are marked with "+" markers
(white on value panels, gray on error panels).

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
    "savefig.dpi": 1000,
    "savefig.bbox": "tight",
})

# Fields to plot.  Both come from the same run; the h5 folder and per-field
# file names follow a fixed pattern, so each field is fully described by its
# name plus a LaTeX label for the titles.
RUN = "coarse8_t9_25_t10_512p_zusatz"
FIELDS = [
    {"name": "rho", "latex": r"\rho"},  # density
    {"name": "Jz",  "latex": r"J_z"},   # out-of-plane current density
    {"name": "e3",  "latex": r"E_3"},   # out-of-plane electric field
]

# (epoch, short label for titles/filenames)
EPOCHS = [
    (60,   "epoch 60\n(data only)"),
    (1680, "epoch 1680\n(+ PDE & div-B)"),
]

# Timestep indices to plot (31 steps, t = 9.25 .. 10.0). t=0 plus later ones.
PLOT_INDICES = [0, 10, 20, 30]


def output_dir(field):
    return os.path.join(_REPO_ROOT, "figs", "collages", "coarse8_t9_25_t10_512p_figs", field)


def load_all(field, epoch):
    path = fields_h5_path(RUN, epoch, field)
    with h5py.File(path, "r") as f:
        return {
            "epoch":  epoch,
            "eta":    float(f.attrs["eta"]),
            "x":      f["x"][()],
            "y":      f["y"][()],
            "t":      f["t"][()],
            "true":   f["true"][()],
            "pred":   f["pred"][()],
            "error":  f["error"][()],
            "mask":   f["data_mask_spatial"][()].astype(bool),
        }


def add_mask_markers(ax, x, y, mask, color="white"):
    rows, cols = np.where(mask)
    ax.scatter(x[cols], y[rows], marker="+", s=4, linewidths=0.35,
               color=color, alpha=0.5, zorder=5)


def add_colorbar(fig, ax, pcm):
    plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)


def _add_mask_legend(fig):
    handle = mpl.lines.Line2D(
        [], [], marker="+", color="white", linestyle="",
        markeredgewidth=0.7, markersize=6,
        label="data-loss pixels (stride 8, 64x64)",
    )
    fig.legend(handles=[handle], loc="upper center",
               bbox_to_anchor=(0.5, 1.0), framealpha=0.8)


def epoch_tag(label):
    """'epoch 60\\n(data only)' -> 'epoch0060'."""
    epoch = int(label.split("\n")[0].split()[-1])
    return f"epoch{epoch:04d}"


def coarse_from_true(true_f, mask):
    """Reduce a 512x512 true field to the 64x64 grid of supervised pixels."""
    rows = np.unique(np.where(mask)[0])
    cols = np.unique(np.where(mask)[1])
    return true_f[np.ix_(rows, cols)], rows, cols


# ----------------------------------------------------------------------------
# Figure builders
# ----------------------------------------------------------------------------
def make_single_figure(d, ti, val_norm, err_norm, field_latex):
    """truth | pred | error for one epoch at one timestep."""
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
        add_colorbar(fig, ax, pcm)
        add_mask_markers(ax, d["x"], d["y"], d["mask"], color=mc)
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
        ax.set_aspect("equal"); ax.set_title(subtitle)
    fig.suptitle(
        rf"${field_latex}$,  $t = {t_val:.2f}$,  {d['label'].replace(chr(10), ',  ')}",
        y=1.02,
    )
    _add_mask_legend(fig)
    return fig


def make_compare_pred_figure(data_list, ti, val_norm, field_latex):
    """truth (once) + pred panels for each epoch, one shared scale on the right."""
    d0 = data_list[0]
    X, Y = np.meshgrid(d0["x"], d0["y"], indexing="ij")
    t_val = float(d0["t"][ti])
    # truth, then pred(without PDE), pred(with PDE)
    panels = [("BHAC (truth)", d0["true"][ti])]
    panels += [(title, d["pred"][ti]) for title, d in
               zip(("FNO, without PDE", "FNO, with PDE"), data_list)]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.8))
    for ax, (title, field) in zip(axes, panels):
        pcm = ax.pcolormesh(X, Y, field, cmap="jet",
                            shading="auto", norm=val_norm)
        add_mask_markers(ax, d0["x"], d0["y"], d0["mask"])
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
        ax.set_aspect("equal"); ax.set_title(title)

    # single shared colorbar on the right of the row (thin; shrink so it
    # matches the square panels instead of overshooting above/below them)
    fig.colorbar(pcm, ax=list(axes), fraction=0.015, pad=0.02, aspect=30,
                 shrink=0.8)

    eps = " vs ".join(str(d["epoch"]) for d in data_list)
    fig.suptitle(
        rf"${field_latex}$,  $t = {t_val:.2f}$,  $\eta = {d0['eta']:.2e}$,  "
        rf"FNO predictions (epochs {eps})",
        y=1.02,
    )
    _add_mask_legend(fig)
    return fig


def make_compare_error_figure(data_list, ti, err_norm, field_latex):
    """error panels for each epoch, shared symmetric scale + RMSE in title."""
    n = len(data_list)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8))
    d0 = data_list[0]
    X, Y = np.meshgrid(d0["x"], d0["y"], indexing="ij")
    t_val = float(d0["t"][ti])

    for i, d in enumerate(data_list):
        ax = axes[i]
        e = d["error"][ti]
        rmse = float(np.sqrt((e ** 2).mean()))
        pcm = ax.pcolormesh(X, Y, e, cmap="RdBu_r",
                            shading="auto", norm=err_norm)
        add_colorbar(fig, ax, pcm)
        add_mask_markers(ax, d["x"], d["y"], d["mask"], color="0.5")
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
        ax.set_aspect("equal")
        ax.set_title(f"Error, {d['label']}\nRMSE = {rmse:.3f}")

    eps = " vs ".join(str(d["epoch"]) for d in data_list)
    fig.suptitle(
        rf"${field_latex}$,  $t = {t_val:.2f}$,  prediction errors (epochs {eps})",
        y=1.02)
    _add_mask_legend(fig)
    return fig


def make_coarse_vs_true_figure(d, ti, val_norm, field_latex):
    """Full truth vs the coarse 64x64 data the data loss actually sees."""
    X, Y = np.meshgrid(d["x"], d["y"], indexing="ij")
    t_val = float(d["t"][ti])
    coarse, rows, cols = coarse_from_true(d["true"][ti], d["mask"])
    xc, yc = d["x"][cols], d["y"][rows]
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))

    ax = axes[0]
    pcm = ax.pcolormesh(X, Y, d["true"][ti], cmap="jet",
                        shading="auto", norm=val_norm)
    add_colorbar(fig, ax, pcm)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    ax.set_title(r"BHAC truth (512$\times$512)")

    ax = axes[1]
    # flat shading -> blocky cells, so the coarseness of the supervision is
    # immediately visible against the smooth full-resolution truth.
    pcm = ax.pcolormesh(Xc, Yc, coarse, cmap="jet",
                        shading="nearest", norm=val_norm)
    add_colorbar(fig, ax, pcm)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    ax.set_title(r"Data loss sees (64$\times$64)")

    fig.suptitle(
        rf"${field_latex}$,  $t = {t_val:.2f}$,  sparse data supervision", y=1.02)
    return fig


# ----------------------------------------------------------------------------
def plot_field(field, field_latex):
    out_dir = output_dir(field)
    single_dir = os.path.join(out_dir, "single")
    os.makedirs(single_dir, exist_ok=True)
    print(f"[{field}] -> {out_dir}/")

    data_list = []
    for epoch, label in EPOCHS:
        d = load_all(field, epoch)
        d["label"] = label
        data_list.append(d)
        print(f"  loaded epoch {epoch}")

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

        # coarse-vs-true motivation panel (truth is epoch-independent)
        fig = make_coarse_vs_true_figure(data_list[0], ti, vn, field_latex)
        out = os.path.join(out_dir, f"coarse_vs_true_t{ti:03d}.png")
        fig.savefig(out); plt.close(fig); print(f"  saved {out}")

        # per-epoch single figures
        for d in data_list:
            fig = make_single_figure(d, ti, vn, err_norm_at(ti), field_latex)
            out = os.path.join(
                single_dir, f"{epoch_tag(d['label'])}_t{ti:03d}.png")
            fig.savefig(out); plt.close(fig); print(f"  saved {out}")

        # epoch-comparison figures (both epochs in the filename)
        ep_tag = "_".join(f"ep{d['epoch']}" for d in data_list)
        fig = make_compare_pred_figure(data_list, ti, vn, field_latex)
        out = os.path.join(out_dir, f"compare_pred_{ep_tag}_t{ti:03d}.png")
        fig.savefig(out); plt.close(fig); print(f"  saved {out}")

        fig = make_compare_error_figure(data_list, ti, en, field_latex)
        out = os.path.join(out_dir, f"compare_error_{ep_tag}_t{ti:03d}.png")
        fig.savefig(out); plt.close(fig); print(f"  saved {out}")


def main():
    for spec in FIELDS:
        plot_field(spec["name"], spec["latex"])
    print("Done.")


if __name__ == "__main__":
    main()
