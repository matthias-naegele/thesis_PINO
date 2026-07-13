#!/usr/bin/env python3
"""
Plot derived field (rho * lfac) for all timesteps from an FNO-style HDF5 file.

Expected datasets:
  - fields: (T, C, Ny, Nx)
  - t: (T,)
  - x: (Nx,)
  - y: (Ny,)
  - varnames: (C,) strings

Also supported:
  - eta: stored as a channel in 'fields' with name 'eta' in 'varnames'.
         (Assumed constant; printed in plot header.)

Usage:
  python plot_rho_lfac.py --input fno_uniform_level1.h5 --outdir pngs --robust
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


def decode_varnames(varnames_arr):
    out = []
    for v in varnames_arr:
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def get_eta_scalar(fields: np.ndarray, varnames: list[str]) -> float | None:
    """
    Returns a scalar eta (mean over all entries) if 'eta' exists in varnames, else None.
    Warns if eta is not (approximately) constant.
    """
    if "eta" not in varnames:
        return None

    ieta = varnames.index("eta")
    eta_field = fields[:, ieta, :, :]  # (T,Ny,Nx)
    eta_val = float(np.nanmean(eta_field))

    if not np.allclose(eta_field, eta_val, rtol=1e-6, atol=1e-12, equal_nan=True):
        eta_min = float(np.nanmin(eta_field))
        eta_max = float(np.nanmax(eta_field))
        print(f"Warning: eta is not constant; using mean eta={eta_val:g} (range {eta_min:g}..{eta_max:g}).")

    return eta_val


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Path to .h5 file")
    p.add_argument("--outdir", "-o", default="pngs", help="Output directory")
    p.add_argument("--dpi", type=int, default=150, help="PNG DPI")
    p.add_argument("--robust", action="store_true",
                   help="Use 1st/99th percentiles for color scaling (global across time)")
    p.add_argument("--vmin", type=float, default=None, help="Manual vmin (overrides robust/global)")
    p.add_argument("--vmax", type=float, default=None, help="Manual vmax (overrides robust/global)")
    p.add_argument("--per-timestep-scale", action="store_true",
                   help="Scale each timestep independently (default: fixed scale across all timesteps)")
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    with h5py.File(in_path, "r") as f:
        fields = f["fields"][:]  # (T,C,Ny,Nx)
        t = f["t"][:] if "t" in f else np.arange(fields.shape[0], dtype=np.float32)
        x = f["x"][:] if "x" in f else np.arange(fields.shape[3], dtype=np.float32)
        y = f["y"][:] if "y" in f else np.arange(fields.shape[2], dtype=np.float32)
        varnames = decode_varnames(f["varnames"][:]) if "varnames" in f else None

    if fields.ndim != 4:
        raise ValueError(f"Expected fields to have 4 dims (T,C,Ny,Nx), got {fields.shape}")
    if varnames is None:
        raise ValueError("Dataset 'varnames' not found; cannot locate channels for rho and lfac.")

    needed = ["rho", "lfac"]
    for name in needed:
        if name not in varnames:
            raise ValueError(f"Missing '{name}' in varnames. Available: {', '.join(varnames)}")

    eta_val = get_eta_scalar(fields, varnames)
    eta_part = f"  |  eta={eta_val:g}" if eta_val is not None else ""

    irho = varnames.index("rho")
    ilfac = varnames.index("lfac")

    rho = fields[:, irho, :, :]   # (T,Ny,Nx)
    lfac = fields[:, ilfac, :, :] # (T,Ny,Nx)
    q = rho * lfac                # (T,Ny,Nx)

    out_dir = Path(args.outdir) / "rho_times_lfac"
    out_dir.mkdir(parents=True, exist_ok=True)

    extent = [float(x.min()), float(x.max()), float(y.min()), float(y.max())]

    # Global scaling unless per-timestep
    vmin = args.vmin
    vmax = args.vmax
    if not args.per_timestep_scale and (vmin is None or vmax is None):
        flat = q.reshape(-1)
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

    for ti in range(q.shape[0]):
        frame = q[ti]

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
        ax.set_title(f"rho*lfac{eta_part}  |  t={float(t[ti]):g}  (index {ti})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)

        out_path = out_dir / f"rho_times_lfac_t{ti:04d}.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote {q.shape[0]} PNGs to: {out_dir}")
    print(f"Used channels: rho -> {irho}, lfac -> {ilfac}")
    if eta_val is not None:
        print(f"eta = {eta_val:g}")


if __name__ == "__main__":
    main()
