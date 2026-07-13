"""
Thesis figures for the stride8_512p noAC (zusatz) run: how the PDE + div-B
physics losses help the FNO between *temporally* sparse data supervision.

Unlike the coarse8 run (see plot_coarse8_t9_25_t10_512p.py), here the data loss is
NOT spatially sparse: every pixel of the 512x512 grid is supervised.  Instead
the sparsity is in TIME -- the data loss is enforced only on every 8th frame
(`data_loss_stride = 8`), i.e. timesteps 0, 8, 16, 24 out of the 31 frames
(t = 9.25 .. 10.0).  Between those frames the FNO has no data anchor and must
interpolate in time; that is exactly where the PDE + div-B losses earn their
keep.

Because the *spatial* mask is fully dense here, the per-pixel "+" markers used
in the coarse8 figures carry no information (they would cover every pixel).  We
replace them with a TEMPORAL SUPERVISION STRIP beneath each figure: the full
sequence of 31 frames as a timeline, with the 4 data-supervised frames marked
and the frame currently shown highlighted (green = data-supervised, orange =
physics-only / interpolated).

Two checkpoints are compared, per field (rho, Jz, e3):
  - epoch 60  : end of phase 0, DATA LOSS ONLY (no physics yet).
  - epoch 480 : final model, with PDE + div-B constraint losses ramped in.

Figures (style matched to plot_coarse8_t9_25_t10_512p.py), one subfolder per field:
  figs/collages/stride8_t9_25_t10_512p_figs/{field}/
    single/epoch{N}_t{TT}.png            — truth | pred | error for one epoch
    compare_pred_ep{A}_ep{B}_t{TT}.png   — truth + pred(ep60) + pred(ep480)
    compare_error_ep{A}_ep{B}_t{TT}.png  — error(ep60) vs error(ep480)

Value normalization (truth/pred panels): per-timestep, from the true field at
that step.  Error normalization: per-timestep symmetric and SHARED across the
two epochs so the shrinking error from epoch 60 -> 480 is directly comparable.

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

# colours for the temporal strip
C_SUP = "#2ca02c"   # data-supervised frame (green)
C_PHYS = "#ff7f0e"  # physics-only / interpolated frame (orange)
C_BASE = "0.75"     # unsupervised frames in the strip (light gray)

RUN = "stride8_t9_25_t10_512p_zusatz"
FIELDS = [
    {"name": "rho", "latex": r"\rho"},  # density
    {"name": "Jz",  "latex": r"J_z"},   # out-of-plane current density
    {"name": "e3",  "latex": r"E_3"},   # out-of-plane electric field
]

# (epoch, short label for titles/filenames)
EPOCHS = [
    (60,  "epoch 60\n(data only)"),
    (480, "epoch 480\n(+ PDE & div-B)"),
]

# Timestep indices to plot. Mix of data-supervised (0, 8, 16, 24) and
# physics-only frames (4, 12, 20, 30) so the strip's highlight -- and the
# supervised/interpolated story -- varies across the figures.
PLOT_INDICES = [0, 4, 8, 12, 16, 20, 24, 30]


def output_dir(field):
    return os.path.join(_REPO_ROOT, "figs", "collages", "stride8_t9_25_t10_512p_figs", field)


def load_all(field, epoch):
    path = fields_h5_path(RUN, epoch, field)
    with h5py.File(path, "r") as f:
        return {
            "epoch":      epoch,
            "eta":        float(f.attrs["eta"]),
            "stride":     int(f.attrs["data_loss_stride"]),
            "x":          f["x"][()],
            "y":          f["y"][()],
            "t":          f["t"][()],
            "true":       f["true"][()],
            "pred":       f["pred"][()],
            "error":      f["error"][()],
            "mask_time":  f["data_mask_time"][()].astype(bool),
        }


# ----------------------------------------------------------------------------
# Temporal supervision strip (replaces the spatial "+" markers)
# ----------------------------------------------------------------------------
def add_time_strip(fig, t, mask_time, ti, stride):
    """Slim timeline under the panels showing which frames the data loss sees.

    All 31 frames are drawn as ticks along t; the data-supervised frames
    (every `stride`-th) are tall green marks, the rest short gray marks.  The
    frame currently shown is circled and colour-coded (green if supervised,
    orange if physics-only).
    """
    fig.subplots_adjust(bottom=0.26)
    ax = fig.add_axes([0.27, 0.05, 0.46, 0.055])

    sup = np.asarray(mask_time, dtype=bool)
    here_sup = bool(sup[ti])

    # baseline
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)
    # unsupervised frames: short light ticks
    ax.vlines(t[~sup], 0.0, 0.45, color=C_BASE, lw=1.0, zorder=2)
    # supervised frames: tall green ticks + dot
    ax.vlines(t[sup], 0.0, 1.0, color=C_SUP, lw=1.6, zorder=3)
    ax.scatter(t[sup], np.ones(sup.sum()), s=14, color=C_SUP, zorder=4)
    # current frame highlight
    cur_c = C_SUP if here_sup else C_PHYS
    ax.scatter([t[ti]], [0.5], s=70, facecolors="none",
               edgecolors=cur_c, linewidths=1.8, zorder=6)
    ax.scatter([t[ti]], [0.5], s=10, color=cur_c, zorder=6)

    ax.set_xlim(t.min() - 0.01, t.max() + 0.01)
    ax.set_ylim(-0.3, 1.5)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=9, direction="out", top=False)
    ax.minorticks_off()
    ax.set_xlabel(r"$t$", labelpad=1)

    n_sup = int(sup.sum())
    state = ("data-supervised frame" if here_sup
             else "physics-only frame (no data loss)")
    ax.set_title(
        rf"data-loss frames (stride {stride} in time): "
        rf"{n_sup} of {len(t)} supervised  $\rightarrow$  "
        rf"this frame: {state}",
        fontsize=9, color=cur_c, pad=4,
    )
    return ax


# ----------------------------------------------------------------------------
# Figure builders
# ----------------------------------------------------------------------------
def make_single_figure(d, ti, val_norm, err_norm, field_latex):
    """truth | pred | error for one epoch at one timestep."""
    X, Y = np.meshgrid(d["x"], d["y"], indexing="ij")
    t_val = float(d["t"][ti])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    panels = [
        (axes[0], d["true"][ti],  val_norm, "jet",    "BHAC (truth)"),
        (axes[1], d["pred"][ti],  val_norm, "jet",    "FNO (prediction)"),
        (axes[2], d["error"][ti], err_norm, "RdBu_r", r"Error (truth $-$ pred)"),
    ]
    for ax, data, norm, cmap, subtitle in panels:
        pcm = ax.pcolormesh(X, Y, data, cmap=cmap, shading="auto", norm=norm)
        plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
        ax.set_aspect("equal"); ax.set_title(subtitle)
    fig.suptitle(
        rf"${field_latex}$,  $t = {t_val:.2f}$,  {d['label'].replace(chr(10), ',  ')}",
        y=1.02,
    )
    add_time_strip(fig, d["t"], d["mask_time"], ti, d["stride"])
    return fig


def make_compare_pred_figure(data_list, ti, val_norm, field_latex):
    """truth (once) + pred panels for each epoch, one shared scale on the right."""
    d0 = data_list[0]
    X, Y = np.meshgrid(d0["x"], d0["y"], indexing="ij")
    t_val = float(d0["t"][ti])
    panels = [("BHAC (truth)", d0["true"][ti])]
    panels += [(title, d["pred"][ti]) for title, d in
               zip(("FNO, without PDE", "FNO, with PDE"), data_list)]

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.8))
    for ax, (title, field) in zip(axes, panels):
        pcm = ax.pcolormesh(X, Y, field, cmap="jet",
                            shading="auto", norm=val_norm)
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
        ax.set_aspect("equal"); ax.set_title(title)

    eps = " vs ".join(str(d["epoch"]) for d in data_list)
    fig.suptitle(
        rf"${field_latex}$,  $t = {t_val:.2f}$,  $\eta = {d0['eta']:.2e}$,  "
        rf"FNO predictions (epochs {eps})",
        y=1.02,
    )
    # Add the time strip first: it calls subplots_adjust, which lifts the
    # panels.  Creating the shared colorbar afterwards keeps it aligned to the
    # final panel height (otherwise it stretches down past them to the strip).
    add_time_strip(fig, d0["t"], d0["mask_time"], ti, d0["stride"])
    fig.colorbar(pcm, ax=list(axes), fraction=0.015, pad=0.02, aspect=30)
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
        plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
        ax.set_aspect("equal")
        ax.set_title(f"Error, {d['label']}\nRMSE = {rmse:.3f}")

    eps = " vs ".join(str(d["epoch"]) for d in data_list)
    fig.suptitle(
        rf"${field_latex}$,  $t = {t_val:.2f}$,  prediction errors (epochs {eps})",
        y=1.02)
    add_time_strip(fig, d0["t"], d0["mask_time"], ti, d0["stride"])
    return fig


def epoch_tag(label):
    """'epoch 60\\n(data only)' -> 'epoch0060'."""
    epoch = int(label.split("\n")[0].split()[-1])
    return f"epoch{epoch:04d}"


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

        for d in data_list:
            fig = make_single_figure(d, ti, vn, en, field_latex)
            out = os.path.join(
                single_dir, f"{epoch_tag(d['label'])}_t{ti:03d}.png")
            fig.savefig(out); plt.close(fig); print(f"  saved {out}")

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
