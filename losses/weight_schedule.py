# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Declarative per-epoch loss-weight schedule.

This module is the single source of truth for how the data / PDE / constraint
loss weights ramp over training. It is intentionally free of any heavy
dependency (no torch / physicsnemo), so the ramp arithmetic can be unit-tested
on its own and so the *whole* multi-phase ramp can live in a Hydra config
instead of being scripted as a sequence of separate training runs in a SLURM
launcher.

A schedule is an ordered list of *phases*. Each phase declares how many epochs
it lasts and how each weight ramps during it:

    weight_schedule:
      - epochs: 100                       # phase 0: no ramp (e.g. data-only)
      - epochs: 400                       # phase 1
        set: {pde_weight: 5.0e-6, constraint_weight: 4.0e-4}
        pde_weight_ramp_epoch: 12         # bump pde every 12 epochs ...
        pde_weight_ramp_increment: 1.0e-5 # ... by this much
        constraint_weight_ramp_epoch: 12
        constraint_weight_ramp_increment: 8.0e-4
      - epochs: 200                       # phase 2 ...
        ...

Semantics (these exactly reproduce the old per-phase launchers, where every
phase was its own training run restarted from the previous checkpoint):

* The constructor weights of the loss are the *starting* weights, before
  phase 0.
* Ramp cadence is **phase-relative**: each phase counts epochs from its own
  start, so a phase with ``ramp_epoch: 4`` and length ``L`` applies
  ``L // 4`` increments regardless of where it sits on the global timeline.
* ``set`` is a discontinuous override applied at the moment a phase begins
  (e.g. switching physics on by jumping the weight from 0 to a small value).
  It is the only way a weight can change other than by ramping.
* Weights flow continuously across phase boundaries unless the next phase
  overrides them via ``set``.
* Data weight is floored at ``data_floor`` (never goes negative).
"""

# Numeric fields coerced on parse. Coercing to float sidesteps the YAML gotcha
# where ``1e-5`` (no decimal point) is parsed as a *string* rather than a number.
_INT_KEYS = (
    "epochs",
    "pde_weight_ramp_epoch",
    "constraint_weight_ramp_epoch",
    "data_weight_ramp_epoch",
)
_FLOAT_KEYS = (
    "pde_weight_ramp_increment",
    "constraint_weight_ramp_increment",
    "data_weight_ramp_decrement",
)
# Weight names usable in a phase's ``set`` override, mapped to the short keys
# used in the working-weights dict passed around this module.
_SET_KEYS = {
    "data_weight": "data",
    "pde_weight": "pde",
    "constraint_weight": "constraint",
}


def parse_weight_schedule(weight_schedule):
    """Normalize a config schedule into a list of plain dicts.

    Returns ``None`` when no schedule is configured (the loss then falls back
    to its flat single-phase ramp). Accepts plain lists or OmegaConf
    list/dict configs.
    """
    if weight_schedule is None:
        return None
    # OmegaConf ListConfig / DictConfig -> plain python (only if needed).
    if not isinstance(weight_schedule, (list, tuple)):
        try:
            from omegaconf import OmegaConf

            weight_schedule = OmegaConf.to_container(weight_schedule, resolve=True)
        except Exception:
            weight_schedule = list(weight_schedule)
    if len(weight_schedule) == 0:
        return None

    parsed = []
    for raw in weight_schedule:
        phase = dict(raw)
        if "epochs" not in phase:
            raise ValueError(f"weight_schedule phase is missing 'epochs': {raw}")
        for k in _INT_KEYS:
            if phase.get(k) is not None:
                phase[k] = int(phase[k])
        for k in _FLOAT_KEYS:
            if phase.get(k) is not None:
                phase[k] = float(phase[k])
        if phase.get("set") is not None:
            phase["set"] = {
                k: float(v) for k, v in dict(phase["set"]).items() if k in _SET_KEYS
            }
        parsed.append(phase)
    return parsed


def total_epochs(schedule):
    """Sum of all phase lengths (the epoch at which the schedule ends)."""
    return sum(int(p["epochs"]) for p in schedule)


def apply_phase_set(weights, phase):
    """Apply a phase's ``set`` block (a discontinuous override) to ``weights``.

    ``weights`` is a mutable dict with keys ``data`` / ``pde`` / ``constraint``.
    """
    overrides = phase.get("set")
    if not overrides:
        return
    for cfg_key, short_key in _SET_KEYS.items():
        if cfg_key in overrides:
            weights[short_key] = overrides[cfg_key]


def apply_phase_ramp(weights, phase, r, data_floor):
    """Apply one phase's ramp at phase-relative epoch ``r`` (1-indexed)."""
    re = phase.get("pde_weight_ramp_epoch") or 0
    if re > 0 and r % re == 0:
        weights["pde"] += phase.get("pde_weight_ramp_increment") or 0.0
    re = phase.get("constraint_weight_ramp_epoch") or 0
    if re > 0 and r % re == 0:
        weights["constraint"] += phase.get("constraint_weight_ramp_increment") or 0.0
    re = phase.get("data_weight_ramp_epoch") or 0
    if re > 0 and r % re == 0:
        weights["data"] = max(
            data_floor, weights["data"] - (phase.get("data_weight_ramp_decrement") or 0.0)
        )


def step(schedule, weights, epoch, data_floor):
    """Advance ``weights`` one epoch.

    ``epoch`` is the (1-indexed) epoch that just finished. Applies that epoch's
    ramp at its phase-relative position, and, when the epoch is the last of its
    phase, applies the next phase's ``set`` override so the following epoch
    starts cleanly. Epochs past the end of the schedule freeze the weights.
    """
    cum = 0
    for i, phase in enumerate(schedule):
        length = phase["epochs"]
        if epoch <= cum + length:
            apply_phase_ramp(weights, phase, epoch - cum, data_floor)
            if epoch == cum + length and i + 1 < len(schedule):
                apply_phase_set(weights, schedule[i + 1])
            return
        cum += length


def state_after(schedule, init_weights, epoch, data_floor):
    """Return the weights an uninterrupted run would have *for* ``epoch + 1``.

    Replays the schedule from scratch through ``epoch`` completed epochs. Used
    to restore weight state when resuming from a checkpoint. ``init_weights``
    is the dict of starting weights (before phase 0).
    """
    weights = dict(init_weights)
    if schedule:
        apply_phase_set(weights, schedule[0])
        for e in range(1, int(epoch) + 1):
            step(schedule, weights, e, data_floor)
    return weights
