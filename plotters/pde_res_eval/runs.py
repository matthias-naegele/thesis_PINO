# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry of the four canonical "2x2" runs evaluated by pde_res_eval.

The 2x2 grid is {256p, 512p} x {coarse8, stride8}. For the 512p runs the
_zusatz (enlarged-dataset, 38 train / 11 val sims) variants are used; their
data-only warm-up ends at epoch 60 instead of 100 (the whole schedule is
rescaled by ~0.6, see the run configs).

`warmup_epoch` is the last epoch of the data-only phase (physics weights
still zero) and always lands on a checkpoint (ckpt_freq divides it). The
"final" epoch is by default the latest checkpoint in the run's ckpt dir.

`fig_time_index` is the default frame (0-based, within the run's loaded
window) for the residual-map figures:
  - stride8 runs: the middle of a data-unsupervised range, i.e. halfway
    between two supervised anchor frames (anchors sit at multiples of
    data_loss_stride=8, so mid-gap frames are ==4 mod 8).
  - coarse8 runs: supervision is spatially coarse but present at every
    timestep, so the frame is free; we use the run's established plot frame.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunSpec:
    key: str            # short CLI name
    config_name: str    # config/<config_name>.yaml, also the ckpt dir leaf
    label: str          # human-readable run description
    warmup_epoch: int   # last data-only epoch (end of phase 0)
    fig_time_index: int # default frame for residual-map figures
    fig_time_note: str  # why that frame


RUNS = {
    "coarse8_256p": RunSpec(
        key="coarse8_256p",
        config_name="coarse8_t0_t2_5_256p",
        label="256p coarse8, t=0..2.5",
        warmup_epoch=100,
        fig_time_index=96,
        fig_time_note=(
            "coarse supervision exists at every frame; 96 is the run's "
            "canonical plot frame (wandb_plot_index_t), late in the window "
            "where the flow is structured"
        ),
    ),
    "stride8_256p": RunSpec(
        key="stride8_256p",
        config_name="stride8_t0_t2_5_256p",
        label="256p stride8, t=0..2.5",
        warmup_epoch=100,
        fig_time_index=92,
        fig_time_note=(
            "mid-gap unsupervised frame (anchors at 88 and 96), late in the "
            "window where the flow is structured"
        ),
    ),
    "coarse8_512p": RunSpec(
        key="coarse8_512p",
        config_name="coarse8_t9_25_t10_512p_zusatz",
        label="512p coarse8, t=9.25..10 (zusatz)",
        warmup_epoch=60,
        fig_time_index=15,
        fig_time_note=(
            "coarse supervision exists at every frame; 15 (t=9.625) is the "
            "frame the earlier residual-map investigation used"
        ),
    ),
    "stride8_512p": RunSpec(
        key="stride8_512p",
        config_name="stride8_t9_25_t10_512p_zusatz",
        label="512p stride8, t=9.25..10 (zusatz)",
        warmup_epoch=60,
        fig_time_index=12,
        fig_time_note=(
            "mid-gap unsupervised frame (anchors at 8 and 16, t=9.55) — the "
            "middle of a data-unsupervised range"
        ),
    ),
}


def resolve_run_keys(arg):
    """Turn the --run CLI value ('all' or a run key) into a list of keys."""
    if arg == "all":
        return list(RUNS.keys())
    if arg not in RUNS:
        raise SystemExit(
            f"Unknown run '{arg}'. Choose from: all, {', '.join(RUNS.keys())}"
        )
    return [arg]
