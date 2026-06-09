#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Matthias Nägele.
# SPDX-License-Identifier: Apache-2.0
"""
Print per-timestep and per-channel statistics for an FNO-style HDF5 file.

For each channel in [u1, u2, b1, b2, p, e3, rho, eta] it reports:
  - mean pixel value per timestep
  - max pixel value per timestep
  - 90th percentile pixel value per timestep  (border to top 10%)

Also prints a summary table (mean over all timesteps) at the end.

Usage:
  python test_norms.py --input fno_uniform_level1.h5
  python test_norms.py --input fno_uniform_level1.h5 --vars u1 b1 rho
  python test_norms.py --input fno_uniform_level1.h5 --no-per-timestep
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


DEFAULT_VARS = ["u1", "u2", "b1", "b2", "p", "e3", "rho", "eta"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Path to .h5 file")
    p.add_argument(
        "--vars", "-v", nargs="+", default=DEFAULT_VARS,
        help=f"Variables to analyse (default: {' '.join(DEFAULT_VARS)})"
    )
    p.add_argument(
        "--no-per-timestep", action="store_true",
        help="Skip per-timestep table; only print summary"
    )
    p.add_argument(
        "--abs", action="store_true",
        help="Operate on absolute values (useful for signed fields like u1, u2)"
    )
    return p.parse_args()


def decode_varnames(arr):
    return [v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v) for v in arr]


def col(width, s):
    return str(s)[:width].ljust(width)


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    with h5py.File(in_path, "r") as f:
        fields = f["fields"][:]          # (T, C, Ny, Nx)
        t_arr  = f["t"][:]  if "t"  in f else np.arange(fields.shape[0], dtype=np.float32)
        varnames = decode_varnames(f["varnames"][:])

    T, C, Ny, Nx = fields.shape
    print(f"\nFile   : {in_path}")
    print(f"Shape  : T={T}  C={C}  Ny={Ny}  Nx={Nx}")
    print(f"Channels in file: {varnames}\n")

    # resolve requested vars
    available = []
    for v in args.vars:
        if v in varnames:
            available.append(v)
        else:
            print(f"  [skip] '{v}' not found in file")

    if not available:
        print("No requested variables found.")
        return

    # ---------- per-channel stats ----------
    # accumulators for summary
    summary = {v: {"mean": [], "max": [], "p90": []} for v in available}

    for var in available:
        cidx = varnames.index(var)
        data = fields[:, cidx, :, :]   # (T, Ny, Nx)

        if args.abs:
            data = np.abs(data)

        print("=" * 72)
        print(f"  Variable: {var}  (channel {cidx})")
        print("=" * 72)

        if not args.no_per_timestep:
            hdr = (
                col(6,  "t_idx")
                + col(12, "t_phys")
                + col(14, "mean")
                + col(14, "max")
                + col(14, "p90")
            )
            print(hdr)
            print("-" * len(hdr))

        for ti in range(T):
            frame = data[ti].ravel()
            mean_ = float(np.nanmean(frame))
            max_  = float(np.nanmax(frame))
            p90_  = float(np.nanpercentile(frame, 90))

            summary[var]["mean"].append(mean_)
            summary[var]["max"].append(max_)
            summary[var]["p90"].append(p90_)

            if not args.no_per_timestep:
                print(
                    col(6,  ti)
                    + col(12, f"{t_arr[ti]:.4g}")
                    + col(14, f"{mean_:.4e}")
                    + col(14, f"{max_:.4e}")
                    + col(14, f"{p90_:.4e}")
                )

        print()

    # ---------- summary table ----------
    print("=" * 72)
    print("  SUMMARY  (mean over all timesteps)")
    print("=" * 72)
    hdr2 = (
        col(8,  "var")
        + col(16, "mean(mean)")
        + col(16, "mean(max)")
        + col(16, "mean(p90)")
        + col(16, "global_max")
    )
    print(hdr2)
    print("-" * len(hdr2))
    for var in available:
        s = summary[var]
        print(
            col(8,  var)
            + col(16, f"{np.mean(s['mean']):.4e}")
            + col(16, f"{np.mean(s['max']):.4e}")
            + col(16, f"{np.mean(s['p90']):.4e}")
            + col(16, f"{np.max(s['max']):.4e}")
        )
    print()

    if args.abs:
        print("Note: statistics computed on |value| (--abs flag active)")


if __name__ == "__main__":
    main()
