#!/usr/bin/env python3
"""
Plot all timesteps from an FNO-style HDF5 file produced by your BHAC pipeline.

Expected datasets:
  - fields: (T, C, Ny, Nx)
  - t: (T,)
  - x: (Nx,)
  - y: (Ny,)
  - varnames: (C,) strings

Also supported:
  - eta: stored as a channel in 'fields' with name 'eta' in 'varnames'.
         (Assumed constant; printed in plot header.)

Outputs:
  outdir/<var>/<var>_t0000.png, ... for all timesteps

Usage:
  python plot_fno_h5.py --input fno_uniform_level1.h5 --var rho --outdir pngs
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Path to .h5 file")
    p.add_argument("--var", "-v", required=True, help="Variable name to plot (must be in varnames)")
    p.add_argument("--outdir", "-o", default="pngs", help="Output directory")
    p.add_argument("--dpi", type=int, default=150, help="PNG DPI")
    p.add_argument("--robust", action="store_true",
                   help="Use 1st/99th percentiles for vmin/vmax (helps with outliers)")
    p.add_argument("--vmin", type=float, default=None, help="Manual vmin (overrides robust)")
    p.add_argument("--vmax", type=float, default=None, help="Manual vmax (overrides robust)")
    p.add_argument("--per-timestep-scale", action="store_true",
                   help="Scale each timestep independently (default: fixed scale across all timesteps)")
    return p.parse_args()


def _decode_varnames(varnames_arr):
    return [vn.decode("utf-8") if isinstance(vn, (bytes, np.bytes_)) else str(vn) for vn in varnames_arr]


def _get_eta_scalar(fields: np.ndarray, varnames: list[str]) -> float | None:
    """
    Returns a scalar eta (mean over all entries) if 'eta' exists in varnames, else None.
    Warns if eta is not (approximately) constant.
    """
    if "eta" not in varnames:
        return None

    ieta = varnames.index("eta")
    eta_field = fields[:, ieta, :, :]  # (T,Ny,Nx)
    eta_val = float(np.nanmean(eta_field))

    # Only warn if it's not approximately constant.
    if not np.allclose(eta_field, eta_val, rtol=1e-6, atol=1e-12, equal_nan=True):
        eta_min = float(np.nanmin(eta_field))
        eta_max = float(np.nanmax(eta_field))
        print(f"Warning: eta is not constant; using mean eta={eta_val:g} (range {eta_min:g}..{eta_max:g}).")

    return eta_val


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_root = Path(args.outdir)

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    with h5py.File(in_path, "r") as f:
        fields = f["fields"][:]  # (T,C,Ny,Nx)
        t = f["t"][:] if "t" in f else np.arange(fields.shape[0], dtype=np.float32)
        x = f["x"][:] if "x" in f else np.arange(fields.shape[3], dtype=np.float32)
        y = f["y"][:] if "y" in f else np.arange(fields.shape[2], dtype=np.float32)
        varnames = f["varnames"][:]

    varnames = _decode_varnames(varnames)

    if fields.ndim != 4:
        raise ValueError(f"Expected fields to have 4 dims (T,C,Ny,Nx), got shape {fields.shape}")

    T, C, Ny, Nx = fields.shape

    if args.var not in varnames:
        raise ValueError(f"Variable '{args.var}' not found. Available: {', '.join(varnames)}")

    eta_val = _get_eta_scalar(fields, varnames)
    eta_part = f"  |  eta={eta_val:g}" if eta_val is not None else ""

    c_idx = varnames.index(args.var)
    data = fields[:, c_idx, :, :]  # (T,Ny,Nx)

    out_dir = out_root / args.var
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine global scaling (unless per-timestep)
    vmin = args.vmin
    vmax = args.vmax
    if not args.per_timestep_scale and (vmin is None or vmax is None):
        flat = data.reshape(-1)
        if args.robust:
            if vmin is None:
                vmin = float(np.nanpercentile(flat, 1))
            if vmax is None:
                vmax = float(np.nanpercentile(flat, 99))
        else:
            if vmin is None:
                vmin = float(np.nanmin(flat))
            if vmax is None:
                vmax = float(np.nanmax(flat))

    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    for ti in range(T):
        frame = data[ti]

        # Per-timestep scaling (optional)
        if args.per_timestep_scale and (args.vmin is None or args.vmax is None):
            if args.robust:
                vvmin = float(np.nanpercentile(frame, 1)) if args.vmin is None else args.vmin
                vvmax = float(np.nanpercentile(frame, 99)) if args.vmax is None else args.vmax
            else:
                vvmin = float(np.nanmin(frame)) if args.vmin is None else args.vmin
                vvmax = float(np.nanmax(frame)) if args.vmax is None else args.vmax
        else:
            vvmin, vvmax = vmin, vmax

        fig, ax = plt.subplots()
        im = ax.imshow(
            frame,
            origin="lower",
            extent=extent,
            aspect="auto",
            vmin=vvmin,
            vmax=vvmax,
        )
        ax.set_title(f"{args.var}{eta_part}  |  t={float(t[ti]):g}  (index {ti})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)

        out_path = out_dir / f"{args.var}_t{ti:04d}.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {T} PNGs to: {out_dir}")
    print(f"Variable '{args.var}' is channel {c_idx} of {C}: {varnames}")
    if eta_val is not None:
        print(f"eta = {eta_val:g}")


if __name__ == "__main__":
    main()
