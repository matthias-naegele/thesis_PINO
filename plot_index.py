"""
Plot fields (e.g. b2, e3, rho) on validation data at specified timesteps.

Features:
  - Choose which field(s) to plot via --field (default: b2); accepts multiple
  - Choose which quantities via --modes: any of ic, true, pred, error
  - Choose timestep via --time_index (-1 = all timesteps)
  - 2D field plots with jet colormap
  - 1D slice plots at given pixel rows via --slice_y (e.g. 64 127 200)
  - 1D slice plots at given pixel columns via --slice_x (e.g. 64 127 200)
  - Color scale mode via --scale: global (shared across timesteps) or local (per-timestep)
  - L2 norm of true vs pred shown in slice plot titles
  - Multiple epochs via --epoch (e.g. --epoch 100 200); adds epoch_NNNN/ subdir
  - Scale-independent raw-data dumps: one consolidated fields.h5 per (sample,
    field/Jz) holding all timesteps + the data-loss masks, plus a .txt table per
    slice (B^2-plotter style, with an is_data_point column). These let a PNG be
    re-rendered at any colour scale and tell which pixels/slice points the data
    loss supervised. Written to --data_dir (default: --output_dir) so the
    global/local colour-scale jobs can share ONE copy instead of duplicating it;
    --skip_data_files suppresses them (used by the index_local job).

scale: local or global

Usage:
    python plot_index.py --field b2 e3 --time_index -1 --modes true pred error \\
        --slice_y 64 127 --slice_x 64 127 --epoch 100 200 --plot_jz --scale local \\
        --output_dir ./index_plots --data_dir ./index_data
"""

import argparse
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# German physics-paper style
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "axes.formatter.use_mathtext": True,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "figure.figsize": (5.5, 4.0),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

from physicsnemo.models.fno import FNO
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint
from physicsnemo.sym.hydra import to_absolute_path
from dataloaders import BHACUniformDataset, BHACDataloader
from losses.fourier_derivatives import fourier_derivatives
from losses.finite_diff import fd_derivatives_periodic

dtype = torch.float
torch.set_default_dtype(dtype)

def compute_jz_field(fields, dt, Lx, Ly, diff_type='fourier'):
    """Compute Jz = ∂e3/∂t - ∂b2/∂x + ∂b1/∂y.

    Parameters
    ----------
    fields : Tensor of shape (nt, nx, ny, 7)
        Output fields [u1, u2, b1, b2, p, e3, rho].
    dt : float
        Physical time step.
    Lx, Ly : float
        Physical domain extents.
    diff_type : str
        'fourier' for spectral spatial derivatives, anything else for 2nd-order FD.

    Returns
    -------
    Jz : Tensor of shape (nt, nx, ny)
    """
    b1 = fields[..., 2]   # (nt, nx, ny)
    b2 = fields[..., 3]   # (nt, nx, ny)
    e3 = fields[..., 5]   # (nt, nx, ny)
    nt = e3.shape[0]

    # Temporal derivative of e3: finite differences (central in interior, one-sided at ends)
    e3b = e3.unsqueeze(0)                          # (1, nt, nx, ny)
    e3_t = torch.empty_like(e3b)
    e3_t[:, 0, ...]    = (e3b[:, 1, ...]  - e3b[:, 0, ...])           / dt
    e3_t[:, -1, ...]   = (e3b[:, -1, ...] - e3b[:, -2, ...])          / dt
    e3_t[:, 1:-1, ...] = (e3b[:, 2:, ...]  - e3b[:, :-2, ...]) / (2 * dt)
    e3_t = e3_t[0]                                 # (nt, nx, ny)

    # Spatial derivatives using the configured differentiation method
    b1b = b1.unsqueeze(0)                          # (1, nt, nx, ny)
    b2b = b2.unsqueeze(0)                          # (1, nt, nx, ny)

    if diff_type == 'fourier':
        f_db1 = fourier_derivatives(b1b, [Lx, Ly])
        f_db2 = fourier_derivatives(b2b, [Lx, Ly])
    else:
        f_db1 = fd_derivatives_periodic(b1b, [Lx, Ly])
        f_db2 = fd_derivatives_periodic(b2b, [Lx, Ly])

    b1_y = f_db1[0, nt: 2 * nt, :, :]             # (nt, nx, ny)  ∂b1/∂y
    b2_x = f_db2[0, 0: nt, :, :]                  # (nt, nx, ny)  ∂b2/∂x

    return e3_t - b2_x + b1_y                      # (nt, nx, ny)


FIELD_LABELS = {
    "u1":  r"$u_1$",
    "u2":  r"$u_2$",
    "b1":  r"$B_1$",
    "b2":  r"$B_2$",
    "p":   r"$p$",
    "e3":  r"$E_3$",
    "rho": r"$\rho$",
}


def save_field_2d(data, X, Y, title, save_path, cmap="jet", norm=None):
    """Save a 2D field as a colormesh plot."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    pcm = ax.pcolormesh(X, Y, data, cmap=cmap, shading="auto", norm=norm)
    cb = plt.colorbar(pcm, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def save_field_2d_strided(full_data, X, Y, coarse_factor, title, save_path,
                          cmap="jet", norm=None, bad_color="0.6"):
    """Save a full-resolution field showing ONLY every ``coarse_factor``-th pixel.

    Unlike the downsampled "smooth" low-res view (which collapses the field to an
    (N/cf)^2 image and can read like a blurred picture), this keeps the original
    NxN grid and colours only the pixels at ``[::cf, ::cf]`` — exactly the points
    the coarse data loss supervises — while every other pixel is rendered in a
    flat gray. It makes the *striding* of the coarse data loss explicit.
    """
    masked = np.ma.masked_all(full_data.shape, dtype=full_data.dtype)
    masked[::coarse_factor, ::coarse_factor] = full_data[::coarse_factor, ::coarse_factor]

    cmap_obj = mpl.colormaps[cmap].copy()
    cmap_obj.set_bad(bad_color)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.set_facecolor(bad_color)
    pcm = ax.pcolormesh(X, Y, masked, cmap=cmap_obj, shading="nearest", norm=norm)
    plt.colorbar(pcm, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def save_slice_1d(x_coords, slices_dict, title, save_path, xlabel=r"$x$",
                   l2_norm=None, marker_x=None, marker_vals=None,
                   marker_label="Data supervision"):
    """Plot 1D slices for the requested modes on one axis.

    Parameters
    ----------
    x_coords : 1-D array
    slices_dict : dict  mode_name -> 1-D array
    l2_norm : float or None
        If provided, appended to the title.
    marker_x, marker_vals : 1-D arrays or None
        If both provided, draw small red dots at (marker_x, marker_vals)
        on top of the curves to mark the locations where the data loss is
        enforced.  Dot size is matched to the rcParams line width.
    """
    fig, ax = plt.subplots()
    colors = {"ic": "0.5", "true": "k", "pred": "C0", "error": "C3"}
    linestyles = {"ic": "-", "true": "-", "pred": "--", "error": ":"}
    labels = {"ic": "IC", "true": "BHAC", "pred": "FNO", "error": "Error"}
    for mode, vals in slices_dict.items():
        ax.plot(
            x_coords, vals,
            color=colors.get(mode, "C0"),
            linestyle=linestyles.get(mode, "-"),
            label=labels.get(mode, mode),
        )
    if marker_x is not None and marker_vals is not None:
        ms = mpl.rcParams["lines.linewidth"] * 2.0
        ax.plot(marker_x, marker_vals, linestyle="", marker="o",
                color="red", markersize=ms, markeredgewidth=0,
                zorder=5, label=marker_label)
    ax.set_xlabel(xlabel)
    full_title = title
    if l2_norm is not None:
        full_title += f",  $L_2 = {l2_norm:.4e}$"
    ax.set_ylabel(title)
    ax.set_title(full_title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# --------------------------------------------------------------------------
# Scale-independent raw-data dumps (.h5 fields + .txt slices)
#
# The PNGs differ between the global- and local-scale variants only by their
# colour normalisation; the underlying numbers are identical.  These dumps hold
# the raw arrays so a PNG can be re-rendered at any scale, and they carry the
# data-loss mask so one can tell which pixels/timesteps/slice points the data
# loss actually supervised (conceptually the same marker as in the B^2 plotter).
# They are written ONCE (by the index_global job) into a shared directory rather
# than duplicated per colour-scale variant.
# --------------------------------------------------------------------------

def _slice_data_mask(n, coarse_factor, supervised_in_time):
    """1 where the (coarse) data loss supervises a slice point, else 0.

    A point ``i`` along a slice is supervised iff the timestep is supervised by
    the time stride AND ``i`` lies on the coarse grid (``i % coarse_factor == 0``).
    This mirrors the red marker dots drawn on the slice PNGs.
    """
    m = np.zeros(n, dtype=np.uint8)
    if supervised_in_time:
        m[::coarse_factor] = 1
    return m


def save_slice_txt(coords, slices_dict, save_path, is_data_point=None,
                   coord_label="x", header_lines=None):
    """Write the raw 1-D slice data as a text table (B^2-plotter style).

    Columns: ``coord_label``, one column per mode in ``slices_dict`` (in order),
    and optionally ``is_data_point`` (1 where the data loss supervises this
    point, else 0).  ``header_lines`` are written as leading ``# ...`` comments.
    """
    modes = list(slices_dict.keys())
    with open(save_path, "w") as f:
        for h in (header_lines or []):
            f.write(f"# {h}\n")
        cols = [coord_label] + modes + (["is_data_point"] if is_data_point is not None else [])
        f.write("# columns: " + "  ".join(cols) + "\n")
        for i in range(len(coords)):
            row = [f"{float(coords[i]):.8e}"]
            row += [f"{float(slices_dict[m][i]):.8e}" for m in modes]
            if is_data_point is not None:
                row.append(str(int(is_data_point[i])))
            f.write("\t".join(row) + "\n")


def write_fields_h5(save_path, x, y, t, fields_by_mode, coarse_factor,
                    data_loss_stride, attrs, ic_field=None):
    """Dump the scale-independent raw 2-D data for one (sample, field/quantity).

    Stores every timestep of the provided modes (e.g. true/pred/error) plus the
    optional IC, the grid coordinates, the real time axis, and the data-loss
    masks.  ``data_mask_spatial`` (nx, ny) marks the pixels at ``[::cf, ::cf]``
    that the coarse data loss supervises; ``data_mask_time`` (nt,) marks the
    timesteps at ``[::stride]`` it supervises in time.
    """
    any_field = next(iter(fields_by_mode.values()))
    nt, nx, ny = any_field.shape
    data_mask_spatial = np.zeros((nx, ny), dtype=np.uint8)
    data_mask_spatial[::coarse_factor, ::coarse_factor] = 1
    data_mask_time = np.zeros(nt, dtype=np.uint8)
    data_mask_time[::data_loss_stride] = 1
    with h5py.File(save_path, "w") as f:
        f.create_dataset("x", data=np.asarray(x))
        f.create_dataset("y", data=np.asarray(y))
        f.create_dataset("t", data=np.asarray(t))
        for mode, arr in fields_by_mode.items():
            f.create_dataset(mode, data=np.asarray(arr))
        if ic_field is not None:
            f.create_dataset("ic", data=np.asarray(ic_field))
        f.create_dataset("data_mask_spatial", data=data_mask_spatial)
        f.create_dataset("data_mask_time", data=data_mask_time)
        for k, v in attrs.items():
            f.attrs[k] = v


# ---- CLI parsing (before Hydra) ----
def parse_extra_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--field", nargs="+", default=["b2"],
                        help="Field name(s) to plot (u1,u2,b1,b2,p,e3,rho); accepts multiple")
    parser.add_argument("--time_index", type=int, default=-1,
                        help="Timestep index to plot (-1 = all)")
    parser.add_argument("--modes", nargs="+", default=["true", "pred", "error"],
                        help="What to plot: ic, true, pred, error")
    parser.add_argument("--slice_y", type=int, nargs="+", default=None,
                        help="Pixel row indices for 1D slice plots (e.g. 64 127 200)")
    parser.add_argument("--slice_x", type=int, nargs="+", default=None,
                        help="Pixel column indices for 1D slice plots (e.g. 64 127 200)")
    parser.add_argument("--scale", type=str, default="global",
                        choices=["global", "local"],
                        help="Color scale mode: global (shared across timesteps) "
                             "or local (per-timestep), both referenced on true data")
    parser.add_argument("--epoch", type=int, nargs="*", default=None,
                        help="Checkpoint epoch(s) to load (e.g. --epoch 100 200); "
                             "omit for latest")
    parser.add_argument("--output_dir", type=str, default="./index_plots")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Root for the scale-independent raw data (.h5 fields "
                             "+ .txt slices). Defaults to --output_dir. Lets the "
                             "index_global / index_local jobs share ONE copy of "
                             "the data instead of duplicating it per colour scale.")
    parser.add_argument("--skip_data_files", action="store_true",
                        help="Do not write the .h5/.txt raw-data files (only render "
                             "PNGs). Used by the index_local job so the "
                             "scale-independent data is written once by index_global.")
    parser.add_argument("--no_epoch_subdir", action="store_true",
                        help="Do not add an epoch_NNNN/ subdirectory under the output "
                             "and data dirs. Use when the caller already places the "
                             "output under an epoch-specific directory.")
    parser.add_argument("--plot_jz", action="store_true",
                        help="Also plot Jz = de3/dt - db2/dx + db1/dy "
                             "using the diff_type from the config")
    parser.add_argument("--lowres_style", type=str, default="smooth",
                        choices=["smooth", "gray"],
                        help="How to render the low-res (data_loss_coarse_factor>1) "
                             "field plots: 'smooth' = the downsampled (N/cf)^2 image "
                             "(true/pred/error); 'gray' = keep the full NxN grid but "
                             "show ONLY every cf-th DATA pixel (the points the coarse "
                             "data loss supervises), greying out the rest. 'gray' plots "
                             "the BHAC data only.")
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return known


ARGS = parse_extra_args()


@hydra.main(version_base="1.3", config_path="config", config_name="mhd_bhac.yaml")
def main(cfg: DictConfig) -> None:
    snap = os.path.join(cfg.train_params.ckpt_path, "config.yaml")
    if os.path.isfile(snap):
        print(f"[config] loaded snapshot {snap}")
        cfg = OmegaConf.load(snap)
    else:
        print(f"[config] no snapshot at {snap}; using live config/*.yaml")
    OmegaConf.set_struct(cfg, False)

    DistributedManager.initialize()
    dist = DistributedManager()

    field_names = ARGS.field
    time_index = ARGS.time_index
    modes = ARGS.modes
    slice_ys = ARGS.slice_y or []
    slice_xs = ARGS.slice_x or []
    scale_mode = ARGS.scale
    plot_jz = ARGS.plot_jz
    lowres_style = ARGS.lowres_style
    output_dir = os.path.abspath(ARGS.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # data_root: where the scale-independent .h5/.txt land. Defaults to the PNG
    # output_dir (self-contained standalone runs); the orchestrated jobs point it
    # at a shared index_data/ dir so global+local don't duplicate the raw data.
    data_root = os.path.abspath(ARGS.data_dir) if ARGS.data_dir else output_dir
    write_data = not ARGS.skip_data_files

    # epoch_list: list of ints to load, or [None] meaning "latest"
    # use_epoch_subdir: add epoch_NNNN/ layer when epochs are explicitly specified,
    # unless the caller already nests by epoch (--no_epoch_subdir).
    epoch_list = ARGS.epoch if ARGS.epoch else [None]
    use_epoch_subdir = bool(ARGS.epoch) and not ARGS.no_epoch_subdir

    model_params = cfg.model_params
    dataset_params = cfg.dataset_params
    loss_params = cfg.loss_params
    names = list(dataset_params.fields)

    for fn in field_names:
        if fn not in names:
            print(f"Unknown field '{fn}'. Available: {names}")
            return

    ckpt_path = cfg.train_params.ckpt_path

    # Build validation dataset / dataloader
    dataset_val = BHACUniformDataset(
        dataset_params.data_dir,
        output_names=dataset_params.output_names,
        file_name=dataset_params.file_name,
        num_train=dataset_params.num_train,
        num_test=dataset_params.num_test,
        use_train=False,
    )
    mhd_dataloader_val = BHACDataloader(
        dataset_val,
        sub_x=dataset_params.sub_x,
        sub_t=dataset_params.sub_t,
        ind_x=dataset_params.ind_x,
        ind_t=dataset_params.ind_t,
        ind_t_start=dataset_params.ind_t_start,
    )
    dataloader_val, _ = mhd_dataloader_val.create_dataloader(
        batch_size=1,
        shuffle=False,
        num_workers=cfg.val_loader_params.num_workers,
        pin_memory=cfg.val_loader_params.pin_memory,
        distributed=False,
    )
    t_real = mhd_dataloader_val.t_real

    # Build model (weights loaded per epoch below)
    model = FNO(
        in_channels=model_params.in_dim,
        out_channels=model_params.out_dim,
        decoder_layers=model_params.decoder_layers,
        decoder_layer_size=model_params.fc_dim,
        dimension=model_params.dimension,
        latent_channels=model_params.layers,
        num_fno_layers=model_params.num_fno_layers,
        num_fno_modes=model_params.modes,
        padding=[model_params.pad_z, model_params.pad_y, model_params.pad_x],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input_norm = torch.tensor(model_params.input_norm).to(device)
    output_norm = torch.tensor(model_params.output_norm).to(device)

    # Jz differentiation config
    diff_type = loss_params.get("diff_type", "fourier")
    Lx = float(loss_params.Lx)
    Ly = float(loss_params.Ly)
    coarse_factor = int(loss_params.get("data_loss_coarse_factor", 1))
    data_loss_stride = int(loss_params.get("data_loss_stride", 1))

    print(f"Fields: {field_names}")
    print(f"Epochs: {epoch_list}")
    print(f"Time index: {'all' if time_index == -1 else time_index}")
    print(f"Modes: {modes}")
    print(f"Scale mode: {scale_mode}")
    if plot_jz:
        print(f"Jz: enabled  (diff_type={diff_type}, Lx={Lx}, Ly={Ly})")
    if slice_ys:
        print(f"Slice at pixel y={slice_ys}")
    if slice_xs:
        print(f"Slice at pixel x={slice_xs}")
    print(f"Output: {output_dir}")
    if write_data:
        print(f"Raw data dir: {data_root}")
    else:
        print("Raw data: skipped (--skip_data_files)")

    label_for_mode = {"true": "BHAC", "pred": "FNO", "error": "Error"}

    # ---- Outer loop: epochs ----
    for epoch_spec in epoch_list:
        loaded_epoch = load_checkpoint(
            ckpt_path, model, optimizer=None, scheduler=None,
            epoch=epoch_spec, device=device,
        )
        print(f"\n=== Loaded checkpoint from epoch {loaded_epoch} ===")
        model.eval()

        if use_epoch_subdir:
            epoch_dir = os.path.join(output_dir, f"epoch_{loaded_epoch:04d}")
            data_epoch_dir = os.path.join(data_root, f"epoch_{loaded_epoch:04d}")
        else:
            epoch_dir = output_dir
            data_epoch_dir = data_root
        os.makedirs(epoch_dir, exist_ok=True)

        with torch.no_grad():
            for sample_idx, (inputs, outputs) in enumerate(dataloader_val):
                inputs = inputs.type(dtype).to(device)
                outputs = outputs.type(dtype).to(device)

                pred = (
                    model((inputs / input_norm).permute(0, 4, 1, 2, 3)).permute(
                        0, 2, 3, 4, 1
                    )
                    * output_norm
                )

                pred_s = pred[0].cpu()      # (nt, nx, ny, 7)
                true_s = outputs[0].cpu()   # (nt, nx, ny, 7)
                inp_s = inputs[0].cpu()     # (nt, nx, ny, 11)

                # Eta value (channel 10, constant over domain) — same as plot_B2_onValidation.py
                eta_val = inp_s[0, 0, 0, 10].item()
                eta_str = rf"$\eta={eta_val:.2e}$"

                Nt = pred_s.shape[0]

                # Grid coordinates
                x = inp_s[0, :, 0, 1].numpy()
                y = inp_s[0, 0, :, 2].numpy()
                X, Y = np.meshgrid(x, y, indexing="ij")

                Nfields = true_s.shape[-1]
                ic_data = inp_s[0, ..., 3:3 + Nfields].numpy()  # (nx, ny, 7)

                # Determine which timesteps to plot
                if time_index == -1:
                    t_indices = list(range(Nt))
                else:
                    t_indices = [time_index]

                # ---- Inner loop: fields ----
                for field_name in field_names:
                    field_idx = names.index(field_name)
                    field_label = FIELD_LABELS.get(field_name, field_name)

                    true_field = true_s[..., field_idx].numpy()   # (nt, nx, ny)
                    pred_field = pred_s[..., field_idx].numpy()
                    err_field = - pred_field + true_field
                    ic_field = ic_data[..., field_idx]             # (nx, ny)

                    # Global color normalization from ground truth (all timesteps)
                    global_vmin = float(true_field.min())
                    global_vmax = float(true_field.max())
                    global_val_norm = Normalize(vmin=global_vmin, vmax=global_vmax)
                    global_err_abs = max(abs(float(err_field.min())), abs(float(err_field.max())))
                    global_err_norm = Normalize(vmin=-global_err_abs, vmax=global_err_abs)

                    sample_dir = os.path.join(epoch_dir, f"sample_{sample_idx:03d}", field_name)
                    os.makedirs(sample_dir, exist_ok=True)
                    if slice_ys or slice_xs:
                        slice_dir = os.path.join(sample_dir, "slices")
                        os.makedirs(slice_dir, exist_ok=True)

                    # Shared scale-independent raw data for this (sample, field):
                    # one consolidated fields.h5 (all timesteps + data-loss masks)
                    # plus per-slice .txt tables. Written once (index_global).
                    data_field_dir = os.path.join(
                        data_epoch_dir, f"sample_{sample_idx:03d}", field_name)
                    data_slice_dir = os.path.join(data_field_dir, "slices")
                    if write_data:
                        os.makedirs(data_field_dir, exist_ok=True)
                        write_fields_h5(
                            os.path.join(data_field_dir, "fields.h5"),
                            x, y, np.asarray(t_real)[:true_field.shape[0]],
                            {"true": true_field, "pred": pred_field, "error": err_field},
                            coarse_factor, data_loss_stride,
                            attrs={
                                "eta": eta_val, "epoch": loaded_epoch,
                                "sample": sample_idx, "field": field_name,
                                "coarse_factor": coarse_factor,
                                "data_loss_stride": data_loss_stride,
                                "Lx": Lx, "Ly": Ly,
                            },
                            ic_field=ic_field,
                        )
                        if slice_ys or slice_xs:
                            os.makedirs(data_slice_dir, exist_ok=True)

                    # --- IC plots (only once, independent of timestep) ---
                    if "ic" in modes:
                        save_field_2d(
                            ic_field, X, Y,
                            title=f"{eta_str}, {field_label}, Initial Condition",
                            save_path=os.path.join(sample_dir, "ic.png"),
                            norm=global_val_norm,
                        )
                        ic_hdr = lambda axis, pos: [
                            f"eta = {eta_val:.6e}", f"epoch = {loaded_epoch}",
                            f"sample = {sample_idx}", f"field = {field_name}",
                            f"slice = {axis}={pos}", "t = initial_condition",
                            f"coarse_factor = {coarse_factor}",
                            f"data_loss_stride = {data_loss_stride}",
                        ]
                        for sy in slice_ys:
                            save_slice_1d(
                                x, {"ic": ic_field[:, sy]},
                                title=f"{eta_str}, {field_label}",
                                save_path=os.path.join(slice_dir, f"ic_slice_y{sy}.png"),
                            )
                            if write_data:
                                save_slice_txt(
                                    x, {"ic": ic_field[:, sy]},
                                    os.path.join(data_slice_dir, f"ic_slice_y{sy}.txt"),
                                    is_data_point=_slice_data_mask(len(x), coarse_factor, True),
                                    coord_label="x", header_lines=ic_hdr("y_index", sy))
                        for sx in slice_xs:
                            save_slice_1d(
                                y, {"ic": ic_field[sx, :]},
                                title=f"{eta_str}, {field_label}",
                                save_path=os.path.join(slice_dir, f"ic_slice_x{sx}.png"),
                                xlabel=r"$y$",
                            )
                            if write_data:
                                save_slice_txt(
                                    y, {"ic": ic_field[sx, :]},
                                    os.path.join(data_slice_dir, f"ic_slice_x{sx}.txt"),
                                    is_data_point=_slice_data_mask(len(y), coarse_factor, True),
                                    coord_label="y", header_lines=ic_hdr("x_index", sx))
                        print(f"  sample {sample_idx:03d} [{field_name}]: IC saved")

                    # --- Timestep plots ---
                    for ti in t_indices:
                        t_val = float(t_real[ti])

                        data_for_mode = {
                            "true":  true_field[ti],
                            "pred":  pred_field[ti],
                            "error": err_field[ti],
                        }

                        # Choose color normalization based on scale mode
                        if scale_mode == "global":
                            val_norm = global_val_norm
                            err_norm = global_err_norm
                        else:  # local
                            local_vmin = float(true_field[ti].min())
                            local_vmax = float(true_field[ti].max())
                            val_norm = Normalize(vmin=local_vmin, vmax=local_vmax)
                            local_err_abs = max(abs(float(err_field[ti].min())),
                                                abs(float(err_field[ti].max())))
                            err_norm = Normalize(vmin=-local_err_abs, vmax=local_err_abs)

                        norm_for_mode = {
                            "true":  val_norm,
                            "pred":  val_norm,
                            "error": err_norm,
                        }

                        for mode in modes:
                            if mode == "ic":
                                continue
                            save_field_2d(
                                data_for_mode[mode], X, Y,
                                title=f"{eta_str}, {field_label}, {label_for_mode[mode]}, $t={t_val:.3f}$, epoch {loaded_epoch}",
                                save_path=os.path.join(sample_dir, f"t{ti:03d}_{mode}.png"),
                                norm=norm_for_mode[mode],
                            )

                        # --- Slice plots ---
                        for sy in slice_ys:
                            y_val = y[sy]
                            slices = {}
                            for mode in modes:
                                if mode == "error":
                                    continue
                                if mode == "ic":
                                    slices["ic"] = ic_field[:, sy]
                                else:
                                    slices[mode] = data_for_mode[mode][:, sy]
                            l2 = float(np.linalg.norm(
                                true_field[ti, :, sy] - pred_field[ti, :, sy]))
                            save_slice_1d(
                                x, slices,
                                title=f"{eta_str}, {field_label} at $y={y_val:.3f}$, $t={t_val:.3f}$, epoch {loaded_epoch}",
                                save_path=os.path.join(slice_dir, f"t{ti:03d}_slice_y{sy}.png"),
                                l2_norm=l2,
                            )
                            if write_data:
                                save_slice_txt(
                                    x, slices,
                                    os.path.join(data_slice_dir, f"t{ti:03d}_slice_y{sy}.txt"),
                                    is_data_point=_slice_data_mask(
                                        len(x), coarse_factor, ti % data_loss_stride == 0),
                                    coord_label="x", header_lines=[
                                        f"eta = {eta_val:.6e}", f"epoch = {loaded_epoch}",
                                        f"sample = {sample_idx}", f"field = {field_name}",
                                        f"slice = y_index={sy}, y={y_val:.6e}",
                                        f"t = {t_val:.6e}", f"l2_true_pred = {l2:.6e}",
                                        f"coarse_factor = {coarse_factor}",
                                        f"data_loss_stride = {data_loss_stride}"])

                        for sx in slice_xs:
                            x_val = x[sx]
                            slices = {}
                            for mode in modes:
                                if mode == "error":
                                    continue
                                if mode == "ic":
                                    slices["ic"] = ic_field[sx, :]
                                else:
                                    slices[mode] = data_for_mode[mode][sx, :]
                            l2 = float(np.linalg.norm(
                                true_field[ti, sx, :] - pred_field[ti, sx, :]))
                            save_slice_1d(
                                y, slices,
                                title=f"{eta_str}, {field_label} at $x={x_val:.3f}$, $t={t_val:.3f}$, epoch {loaded_epoch}",
                                save_path=os.path.join(slice_dir, f"t{ti:03d}_slice_x{sx}.png"),
                                xlabel=r"$y$",
                                l2_norm=l2,
                            )
                            if write_data:
                                save_slice_txt(
                                    y, slices,
                                    os.path.join(data_slice_dir, f"t{ti:03d}_slice_x{sx}.txt"),
                                    is_data_point=_slice_data_mask(
                                        len(y), coarse_factor, ti % data_loss_stride == 0),
                                    coord_label="y", header_lines=[
                                        f"eta = {eta_val:.6e}", f"epoch = {loaded_epoch}",
                                        f"sample = {sample_idx}", f"field = {field_name}",
                                        f"slice = x_index={sx}, x={x_val:.6e}",
                                        f"t = {t_val:.6e}", f"l2_true_pred = {l2:.6e}",
                                        f"coarse_factor = {coarse_factor}",
                                        f"data_loss_stride = {data_loss_stride}"])

                        print(f"  sample {sample_idx:03d} [{field_name}]: t={ti} saved")

                    # --- Low-res field plots (when data_loss_coarse_factor > 1) ---
                    if coarse_factor > 1:
                        x_lr = x[::coarse_factor]
                        y_lr = y[::coarse_factor]
                        X_lr, Y_lr = np.meshgrid(x_lr, y_lr, indexing="ij")

                        true_field_lr = true_field[:, ::coarse_factor, ::coarse_factor]
                        pred_field_lr = pred_field[:, ::coarse_factor, ::coarse_factor]
                        err_field_lr  = err_field[:,  ::coarse_factor, ::coarse_factor]

                        global_vmin_lr = float(true_field_lr.min())
                        global_vmax_lr = float(true_field_lr.max())
                        global_val_norm_lr = Normalize(vmin=global_vmin_lr, vmax=global_vmax_lr)
                        global_err_abs_lr = max(abs(float(err_field_lr.min())),
                                                abs(float(err_field_lr.max())))
                        global_err_norm_lr = Normalize(vmin=-global_err_abs_lr, vmax=global_err_abs_lr)

                        sample_dir_lr = os.path.join(epoch_dir, f"sample_{sample_idx:03d}",
                                                     f"{field_name}_lowres")
                        os.makedirs(sample_dir_lr, exist_ok=True)

                        # Slice plots use FULL-resolution fields and overlay
                        # small red dots on the FNO curve at the coarse-grid
                        # positions where the data loss is enforced.
                        # User-supplied slice indices are snapped to the
                        # nearest coarse-grid line so the dots always have a
                        # row/column to land on.
                        Nx_full, Ny_full = true_field.shape[1], true_field.shape[2]

                        def _snap_to_coarse(idx, k, N):
                            j = (idx + k // 2) // k
                            j_max = (N - 1) // k
                            j = max(0, min(j, j_max))
                            return j * k

                        data_slice_dir_lr = os.path.join(
                            data_field_dir + "_lowres", "slices")
                        if (slice_ys or slice_xs):
                            slice_dir_lr = os.path.join(sample_dir_lr, "slices")
                            os.makedirs(slice_dir_lr, exist_ok=True)
                            if write_data:
                                os.makedirs(data_slice_dir_lr, exist_ok=True)

                        # IC standalone slices at snapped coarse-grid positions
                        # (no FNO/BHAC curve here -> no red dots)
                        if "ic" in modes:
                            for sy in slice_ys:
                                sy_eff = _snap_to_coarse(sy, coarse_factor, Ny_full)
                                y_val = y[sy_eff]
                                save_slice_1d(
                                    x, {"ic": ic_field[:, sy_eff]},
                                    title=f"{eta_str}, {field_label} at $y={y_val:.3f}$",
                                    save_path=os.path.join(
                                        slice_dir_lr, f"ic_slice_y{sy_eff}.png"),
                                )
                                if write_data:
                                    save_slice_txt(
                                        x, {"ic": ic_field[:, sy_eff]},
                                        os.path.join(data_slice_dir_lr, f"ic_slice_y{sy_eff}.txt"),
                                        is_data_point=_slice_data_mask(len(x), coarse_factor, True),
                                        coord_label="x", header_lines=[
                                            f"eta = {eta_val:.6e}", f"epoch = {loaded_epoch}",
                                            f"sample = {sample_idx}", f"field = {field_name}",
                                            f"slice = y_index={sy_eff}, y={y_val:.6e}",
                                            "t = initial_condition",
                                            f"coarse_factor = {coarse_factor}",
                                            f"data_loss_stride = {data_loss_stride}"])
                            for sx in slice_xs:
                                sx_eff = _snap_to_coarse(sx, coarse_factor, Nx_full)
                                x_val = x[sx_eff]
                                save_slice_1d(
                                    y, {"ic": ic_field[sx_eff, :]},
                                    title=f"{eta_str}, {field_label} at $x={x_val:.3f}$",
                                    save_path=os.path.join(
                                        slice_dir_lr, f"ic_slice_x{sx_eff}.png"),
                                    xlabel=r"$y$",
                                )
                                if write_data:
                                    save_slice_txt(
                                        y, {"ic": ic_field[sx_eff, :]},
                                        os.path.join(data_slice_dir_lr, f"ic_slice_x{sx_eff}.txt"),
                                        is_data_point=_slice_data_mask(len(y), coarse_factor, True),
                                        coord_label="y", header_lines=[
                                            f"eta = {eta_val:.6e}", f"epoch = {loaded_epoch}",
                                            f"sample = {sample_idx}", f"field = {field_name}",
                                            f"slice = x_index={sx_eff}, x={x_val:.6e}",
                                            "t = initial_condition",
                                            f"coarse_factor = {coarse_factor}",
                                            f"data_loss_stride = {data_loss_stride}"])

                        for ti in t_indices:
                            t_val = float(t_real[ti])

                            data_for_mode_lr = {
                                "true":  true_field_lr[ti],
                                "pred":  pred_field_lr[ti],
                                "error": err_field_lr[ti],
                            }

                            if scale_mode == "global":
                                val_norm_lr = global_val_norm_lr
                                err_norm_lr = global_err_norm_lr
                            else:  # local
                                local_vmin_lr = float(true_field_lr[ti].min())
                                local_vmax_lr = float(true_field_lr[ti].max())
                                val_norm_lr = Normalize(vmin=local_vmin_lr, vmax=local_vmax_lr)
                                local_err_abs_lr = max(abs(float(err_field_lr[ti].min())),
                                                       abs(float(err_field_lr[ti].max())))
                                err_norm_lr = Normalize(vmin=-local_err_abs_lr, vmax=local_err_abs_lr)

                            norm_for_mode_lr = {
                                "true":  val_norm_lr,
                                "pred":  val_norm_lr,
                                "error": err_norm_lr,
                            }

                            if lowres_style == "gray":
                                # Full NxN grid, only every cf-th DATA pixel
                                # coloured (the supervised points), rest gray.
                                # Data (BHAC) only \u2014 predictions are not strided.
                                if scale_mode == "global":
                                    gray_norm = global_val_norm
                                else:
                                    gray_norm = Normalize(
                                        vmin=float(true_field[ti].min()),
                                        vmax=float(true_field[ti].max()))
                                save_field_2d_strided(
                                    true_field[ti], X, Y, coarse_factor,
                                    title=(f"{eta_str}, {field_label}, BHAC, "
                                           f"$t={t_val:.3f}$, epoch {loaded_epoch} "
                                           f"[data loss \u00d7{coarse_factor}, strided]"),
                                    save_path=os.path.join(
                                        sample_dir_lr, f"t{ti:03d}_true_strided.png"),
                                    norm=gray_norm,
                                )
                            else:
                                for mode in modes:
                                    if mode == "ic":
                                        continue
                                    save_field_2d(
                                        data_for_mode_lr[mode], X_lr, Y_lr,
                                        title=(f"{eta_str}, {field_label}, {label_for_mode[mode]}, "
                                               f"$t={t_val:.3f}$, epoch {loaded_epoch} "
                                               f"[low-res \u00d7{coarse_factor}]"),
                                        save_path=os.path.join(sample_dir_lr, f"t{ti:03d}_{mode}.png"),
                                        norm=norm_for_mode_lr[mode],
                                    )

                            # --- Full-resolution slice plots with red dots ---
                            data_for_mode_full = {
                                "true":  true_field[ti],
                                "pred":  pred_field[ti],
                                "error": err_field[ti],
                            }
                            # Red dots only appear on timesteps that are
                            # actually supervised by the data loss.
                            ti_supervised = (ti % data_loss_stride == 0)

                            for sy in slice_ys:
                                sy_eff = _snap_to_coarse(sy, coarse_factor, Ny_full)
                                y_val = y[sy_eff]
                                slices = {}
                                for mode in modes:
                                    if mode == "error":
                                        continue
                                    if mode == "ic":
                                        slices["ic"] = ic_field[:, sy_eff]
                                    else:
                                        slices[mode] = data_for_mode_full[mode][:, sy_eff]
                                if ti_supervised:
                                    marker_x = x[::coarse_factor]
                                    marker_vals = pred_field[ti, ::coarse_factor, sy_eff]
                                else:
                                    marker_x = None
                                    marker_vals = None
                                l2 = float(np.linalg.norm(
                                    true_field[ti, :, sy_eff] - pred_field[ti, :, sy_eff]))
                                save_slice_1d(
                                    x, slices,
                                    title=(f"{eta_str}, {field_label} at $y={y_val:.3f}$, "
                                           f"$t={t_val:.3f}$, epoch {loaded_epoch} "
                                           f"[data loss ×{coarse_factor}]"),
                                    save_path=os.path.join(
                                        slice_dir_lr, f"t{ti:03d}_slice_y{sy_eff}.png"),
                                    l2_norm=l2,
                                    marker_x=marker_x,
                                    marker_vals=marker_vals,
                                )
                                if write_data:
                                    save_slice_txt(
                                        x, slices,
                                        os.path.join(data_slice_dir_lr, f"t{ti:03d}_slice_y{sy_eff}.txt"),
                                        is_data_point=_slice_data_mask(len(x), coarse_factor, ti_supervised),
                                        coord_label="x", header_lines=[
                                            f"eta = {eta_val:.6e}", f"epoch = {loaded_epoch}",
                                            f"sample = {sample_idx}", f"field = {field_name}",
                                            f"slice = y_index={sy_eff}, y={y_val:.6e}",
                                            f"t = {t_val:.6e}", f"l2_true_pred = {l2:.6e}",
                                            f"coarse_factor = {coarse_factor}",
                                            f"data_loss_stride = {data_loss_stride}"])

                            for sx in slice_xs:
                                sx_eff = _snap_to_coarse(sx, coarse_factor, Nx_full)
                                x_val = x[sx_eff]
                                slices = {}
                                for mode in modes:
                                    if mode == "error":
                                        continue
                                    if mode == "ic":
                                        slices["ic"] = ic_field[sx_eff, :]
                                    else:
                                        slices[mode] = data_for_mode_full[mode][sx_eff, :]
                                if ti_supervised:
                                    marker_x = y[::coarse_factor]
                                    marker_vals = pred_field[ti, sx_eff, ::coarse_factor]
                                else:
                                    marker_x = None
                                    marker_vals = None
                                l2 = float(np.linalg.norm(
                                    true_field[ti, sx_eff, :] - pred_field[ti, sx_eff, :]))
                                save_slice_1d(
                                    y, slices,
                                    title=(f"{eta_str}, {field_label} at $x={x_val:.3f}$, "
                                           f"$t={t_val:.3f}$, epoch {loaded_epoch} "
                                           f"[data loss ×{coarse_factor}]"),
                                    save_path=os.path.join(
                                        slice_dir_lr, f"t{ti:03d}_slice_x{sx_eff}.png"),
                                    xlabel=r"$y$",
                                    l2_norm=l2,
                                    marker_x=marker_x,
                                    marker_vals=marker_vals,
                                )
                                if write_data:
                                    save_slice_txt(
                                        y, slices,
                                        os.path.join(data_slice_dir_lr, f"t{ti:03d}_slice_x{sx_eff}.txt"),
                                        is_data_point=_slice_data_mask(len(y), coarse_factor, ti_supervised),
                                        coord_label="y", header_lines=[
                                            f"eta = {eta_val:.6e}", f"epoch = {loaded_epoch}",
                                            f"sample = {sample_idx}", f"field = {field_name}",
                                            f"slice = x_index={sx_eff}, x={x_val:.6e}",
                                            f"t = {t_val:.6e}", f"l2_true_pred = {l2:.6e}",
                                            f"coarse_factor = {coarse_factor}",
                                            f"data_loss_stride = {data_loss_stride}"])

                        print(f"  sample {sample_idx:03d} [{field_name}]: low-res (×{coarse_factor}) saved")

                # --- Jz plots (field-independent) ---
                if plot_jz:
                    dt = float(t_real[1] - t_real[0])
                    true_jz = compute_jz_field(true_s, dt, Lx, Ly, diff_type).numpy()  # (nt, nx, ny)
                    pred_jz = compute_jz_field(pred_s, dt, Lx, Ly, diff_type).numpy()
                    err_jz  = true_jz - pred_jz

                    jz_dir = os.path.join(epoch_dir, f"sample_{sample_idx:03d}", "Jz")
                    os.makedirs(jz_dir, exist_ok=True)
                    if slice_ys or slice_xs:
                        jz_slice_dir = os.path.join(jz_dir, "slices")
                        os.makedirs(jz_slice_dir, exist_ok=True)

                    # Shared scale-independent raw Jz data. Jz is a diagnostic
                    # derived from the (supervised) fields, so the masks flag
                    # where the underlying fields were supervised.
                    data_jz_dir = os.path.join(
                        data_epoch_dir, f"sample_{sample_idx:03d}", "Jz")
                    data_jz_slice_dir = os.path.join(data_jz_dir, "slices")
                    if write_data:
                        os.makedirs(data_jz_dir, exist_ok=True)
                        write_fields_h5(
                            os.path.join(data_jz_dir, "fields.h5"),
                            x, y, np.asarray(t_real)[:true_jz.shape[0]],
                            {"true": true_jz, "pred": pred_jz, "error": err_jz},
                            coarse_factor, data_loss_stride,
                            attrs={
                                "eta": eta_val, "epoch": loaded_epoch,
                                "sample": sample_idx, "field": "Jz",
                                "coarse_factor": coarse_factor,
                                "data_loss_stride": data_loss_stride,
                                "Lx": Lx, "Ly": Ly, "diff_type": diff_type,
                            },
                        )
                        if slice_ys or slice_xs:
                            os.makedirs(data_jz_slice_dir, exist_ok=True)

                    # Global color normalization for Jz
                    global_jz_vmin = float(true_jz.min())
                    global_jz_vmax = float(true_jz.max())
                    global_jz_norm = Normalize(vmin=global_jz_vmin, vmax=global_jz_vmax)
                    global_jz_err_abs = max(abs(float(err_jz.min())), abs(float(err_jz.max())))
                    global_jz_err_norm = Normalize(vmin=-global_jz_err_abs, vmax=global_jz_err_abs)

                    jz_label = r"$J_z$"
                    jz_data_for_mode = {"true": true_jz, "pred": pred_jz, "error": err_jz}

                    for ti in t_indices:
                        t_val = float(t_real[ti])

                        if scale_mode == "global":
                            jz_val_norm = global_jz_norm
                            jz_err_norm = global_jz_err_norm
                        else:  # local
                            local_jz_vmin = float(true_jz[ti].min())
                            local_jz_vmax = float(true_jz[ti].max())
                            jz_val_norm = Normalize(vmin=local_jz_vmin, vmax=local_jz_vmax)
                            local_jz_err_abs = max(abs(float(err_jz[ti].min())),
                                                   abs(float(err_jz[ti].max())))
                            jz_err_norm = Normalize(vmin=-local_jz_err_abs, vmax=local_jz_err_abs)

                        jz_norm_for_mode = {
                            "true":  jz_val_norm,
                            "pred":  jz_val_norm,
                            "error": jz_err_norm,
                        }

                        for mode in modes:
                            if mode == "ic":
                                continue
                            save_field_2d(
                                jz_data_for_mode[mode][ti], X, Y,
                                title=f"{eta_str}, {jz_label}, {label_for_mode[mode]}, $t={t_val:.3f}$, epoch {loaded_epoch}",
                                save_path=os.path.join(jz_dir, f"t{ti:03d}_{mode}.png"),
                                norm=jz_norm_for_mode[mode],
                            )

                        for sy in slice_ys:
                            y_val = y[sy]
                            slices = {}
                            for mode in modes:
                                if mode in ("ic", "error"):
                                    continue
                                slices[mode] = jz_data_for_mode[mode][ti, :, sy]
                            l2 = float(np.linalg.norm(true_jz[ti, :, sy] - pred_jz[ti, :, sy]))
                            save_slice_1d(
                                x, slices,
                                title=f"{eta_str}, {jz_label} at $y={y_val:.3f}$, $t={t_val:.3f}$, epoch {loaded_epoch}",
                                save_path=os.path.join(jz_slice_dir, f"t{ti:03d}_slice_y{sy}.png"),
                                l2_norm=l2,
                            )
                            if write_data:
                                save_slice_txt(
                                    x, slices,
                                    os.path.join(data_jz_slice_dir, f"t{ti:03d}_slice_y{sy}.txt"),
                                    is_data_point=_slice_data_mask(
                                        len(x), coarse_factor, ti % data_loss_stride == 0),
                                    coord_label="x", header_lines=[
                                        f"eta = {eta_val:.6e}", f"epoch = {loaded_epoch}",
                                        f"sample = {sample_idx}", "field = Jz",
                                        f"slice = y_index={sy}, y={y_val:.6e}",
                                        f"t = {t_val:.6e}", f"l2_true_pred = {l2:.6e}",
                                        f"coarse_factor = {coarse_factor}",
                                        f"data_loss_stride = {data_loss_stride}"])

                        for sx in slice_xs:
                            x_val = x[sx]
                            slices = {}
                            for mode in modes:
                                if mode in ("ic", "error"):
                                    continue
                                slices[mode] = jz_data_for_mode[mode][ti, sx, :]
                            l2 = float(np.linalg.norm(true_jz[ti, sx, :] - pred_jz[ti, sx, :]))
                            save_slice_1d(
                                y, slices,
                                title=f"{eta_str}, {jz_label} at $x={x_val:.3f}$, $t={t_val:.3f}$, epoch {loaded_epoch}",
                                save_path=os.path.join(jz_slice_dir, f"t{ti:03d}_slice_x{sx}.png"),
                                xlabel=r"$y$",
                                l2_norm=l2,
                            )
                            if write_data:
                                save_slice_txt(
                                    y, slices,
                                    os.path.join(data_jz_slice_dir, f"t{ti:03d}_slice_x{sx}.txt"),
                                    is_data_point=_slice_data_mask(
                                        len(y), coarse_factor, ti % data_loss_stride == 0),
                                    coord_label="y", header_lines=[
                                        f"eta = {eta_val:.6e}", f"epoch = {loaded_epoch}",
                                        f"sample = {sample_idx}", "field = Jz",
                                        f"slice = x_index={sx}, x={x_val:.6e}",
                                        f"t = {t_val:.6e}", f"l2_true_pred = {l2:.6e}",
                                        f"coarse_factor = {coarse_factor}",
                                        f"data_loss_stride = {data_loss_stride}"])

                    print(f"  sample {sample_idx:03d}: Jz saved")

                print(f"Sample {sample_idx:03d} done.")

    print(f"\nDone. All plots saved under {output_dir}")
    if write_data:
        print(f"Raw data (.h5 fields + .txt slices) saved under {data_root}")


if __name__ == "__main__":
    main()
