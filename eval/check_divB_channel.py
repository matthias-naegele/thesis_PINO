#!/usr/bin/env python3
"""
Analyse the divB channel from an FNO-style HDF5 file.

Prints at specified frame indices:
  - mean(|divB|)
  - max(|divB|)
  - MSE(divB, 0)  i.e. mean(divB^2)

Usage:
  python check_divB_channel.py --input fno_uniform_level1.h5
  python check_divB_channel.py --input fno_uniform_level1.h5 --frames 0 8 20 40 100
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


TARGET_FRAMES = [0, 8, 20, 40, 100, 200, 300, 400]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="Path to .h5 file")
    p.add_argument("--frames", "-f", type=int, nargs="+", default=TARGET_FRAMES,
                   help="Frame indices to report (0-based)")
    p.add_argument("--chunk", type=int, default=10,
                   help="Number of timesteps to load at once (tune for RAM vs 4000x4000)")
    return p.parse_args()


def decode_varnames(varnames_arr):
    return [v.decode("utf-8") if isinstance(v, (bytes, np.bytes_)) else str(v)
            for v in varnames_arr]


def analyse_divb_at_indices(h5path: Path, c_idx: int, indices: list[int],
                             t_arr: np.ndarray, chunk: int) -> dict[int, dict]:
    """
    Read divB for the required frame indices in chunked passes to keep
    memory manageable at 4000x4000.

    Returns dict: frame_index -> {mean_abs, max_abs, mse}
    """
    results = {}
    needed = set(indices)

    with h5py.File(h5path, "r") as f:
        T = f["fields"].shape[0]

        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            batch_indices = [i for i in range(start, end) if i in needed]
            if not batch_indices:
                continue

            frames = f["fields"][batch_indices, c_idx, :, :]  # (k, Ny, Nx)

            for local_i, global_i in enumerate(batch_indices):
                frame = frames[local_i].astype(np.float64)
                mean_abs = float(np.mean(np.abs(frame)))
                max_abs  = float(np.max(np.abs(frame)))
                mse      = float(np.mean(frame ** 2))
                results[global_i] = {
                    "mean_abs": mean_abs,
                    "max_abs":  max_abs,
                    "mse":      mse,
                    "t":        float(t_arr[global_i]),
                }

    return results


def main():
    args = parse_args()
    h5path = Path(args.input)

    if not h5path.exists():
        raise FileNotFoundError(h5path)

    # ── metadata pass ────────────────────────────────────────────────────────
    with h5py.File(h5path, "r") as f:
        varnames = decode_varnames(f["varnames"][:])
        T, C, Ny, Nx = f["fields"].shape
        t_arr = f["t"][:] if "t" in f else np.arange(T, dtype=np.float64)

    if "divB" not in varnames:
        raise ValueError(f"'divB' not found in file. Available channels: {varnames}")

    c_idx = varnames.index("divB")

    print(f"File       : {h5path}")
    print(f"Grid       : {Nx} x {Ny}  |  T={T} timesteps  |  {C} channels")
    print(f"divB index : channel {c_idx}")
    print(f"t range    : {t_arr[0]:.4g} → {t_arr[-1]:.4g}")
    print()

    # ── validate requested frame indices ─────────────────────────────────────
    requested = []
    for fi in args.frames:
        if fi < 0 or fi >= T:
            print(f"  Warning: frame index {fi} out of range [0, {T-1}] — skipping.")
        else:
            requested.append(fi)

    if not requested:
        raise ValueError("No valid frame indices to analyse.")

    # ── chunked analysis ──────────────────────────────────────────────────────
    print("Computing divB statistics …")
    stats = analyse_divb_at_indices(h5path, c_idx, requested, t_arr, args.chunk)

    # ── report ────────────────────────────────────────────────────────────────
    col = 14
    header = (f"{'frame':>6}  {'t':>{col}}"
              f"  {'mean|divB|':>{col}}  {'max|divB|':>{col}}  {'MSE(divB,0)':>{col}}")
    sep = "-" * len(header)
    print()
    print(header)
    print(sep)

    for fi in requested:
        s = stats[fi]
        print(f"{fi:>6d}  {s['t']:>{col}.6g}"
              f"  {s['mean_abs']:>{col}.6e}  {s['max_abs']:>{col}.6e}  {s['mse']:>{col}.6e}")

    print(sep)
    print()

    # ── aggregate summary ─────────────────────────────────────────────────────
    all_mse      = [stats[i]["mse"]      for i in requested]
    all_max      = [stats[i]["max_abs"]  for i in requested]
    all_mean_abs = [stats[i]["mean_abs"] for i in requested]

    print(f"Over the {len(requested)} sampled frames:")
    print(f"  max  max|divB|  = {max(all_max):.6e}")
    print(f"  mean mean|divB| = {np.mean(all_mean_abs):.6e}")
    print(f"  max  MSE        = {max(all_mse):.6e}")

    # ── scaling check ─────────────────────────────────────────────────────────
    N_eff = min(Nx, Ny)
    print()
    print(f"CT scaling reference  (1/N²  for N={N_eff}): {1.0/N_eff**2:.6e}")
    print(f"max|divB| / (1/N²)  = {max(all_max) * N_eff**2:.4f}   (expect O(1) if healthy)")


if __name__ == "__main__":
    main()
