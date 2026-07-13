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

import torch
from .losses import LpLoss
from .fourier_derivatives import fourier_derivatives
from .mhd_pde import MHD_PDE
from .finite_diff import fd_derivatives_periodic
from . import weight_schedule as weight_schedule_lib

class LossMHD_PhysicsNeMo(object):
    "Calculate loss for MHD equations with magnetic field, using physicsnemo derivatives"

    def __init__(
        self,
        Gamma=1.333333333333333333,
        data_weight=1.0,
        ic_weight=0,
        pde_weight=0,
        constraint_weight=0,
        use_data_loss=False,
        use_ic_loss=False,  # ! unused PhysicsNeMo carry-over; see ic_loss()
        use_pde_loss=False,
        use_constraint_loss=False,
        u1_weight=1.0,
        u2_weight=1.0,
        b1_weight=1.0,
        b2_weight=1.0,
        p_weight=1.0,
        e3_weight=1.0,
        rho_weight=1.0,
        FI1_weight=1.0,
        FI2_weight=1.0,
        MO_weight=1.0,
        ES0_weight=1.0,
        ES1_weight=1.0,
        ES2_weight=1.0,
        C1_weight=1.0,
        div_B_weight=1.0,
        Lx=6.28,
        Ly=6.28,
        tend=10,
        use_weighted_mean=False,
        data_loss_stride=1,
        data_loss_coarse_factor=1,
        pde_weight_ramp_epoch=0,
        pde_weight_ramp_increment=0,
        constraint_weight_ramp_epoch=0,
        constraint_weight_ramp_increment=0,
        data_weight_ramp_epoch=0,
        data_weight_ramp_decrement=0,
        data_weight_floor=0.0,
        weight_schedule=None,
        diff_type='fourier',
        **kwargs,
    ):  # add **kwards so that we ignore unexpected kwargs when passing a config dict
        self.Gamma = Gamma
        self.data_weight = data_weight
        self.ic_weight = ic_weight
        self.pde_weight = pde_weight
        self.constraint_weight = constraint_weight
        self.use_data_loss = use_data_loss
        self.use_ic_loss = use_ic_loss
        self.use_pde_loss = use_pde_loss
        self.use_constraint_loss = use_constraint_loss
        self.u1_weight = u1_weight
        self.u2_weight = u2_weight
        self.b1_weight = b1_weight
        self.b2_weight = b2_weight
        self.p_weight = p_weight
        self.e3_weight = e3_weight
        self.rho_weight = rho_weight
        self.FI1_weight = FI1_weight
        self.FI2_weight = FI2_weight
        self.MO_weight = MO_weight
        self.ES0_weight=ES0_weight
        self.ES1_weight=ES1_weight
        self.ES2_weight=ES2_weight
        self.C1_weight=C1_weight
        self.div_B_weight = div_B_weight
        self.Lx = Lx
        self.Ly = Ly
        self.tend = tend
        self.use_weighted_mean = use_weighted_mean
        self.data_loss_stride = data_loss_stride
        self.data_loss_coarse_factor = data_loss_coarse_factor
        # Define 2D MHD PDEs
        self.mhd_pde_eq = MHD_PDE(self.Gamma)
        self.mhd_pde_node = self.mhd_pde_eq.make_nodes()

        # Weight ramp config (epoch-based)
        self.pde_weight_ramp_epoch = pde_weight_ramp_epoch
        self.pde_weight_ramp_increment = pde_weight_ramp_increment
        self.constraint_weight_ramp_epoch = constraint_weight_ramp_epoch
        self.constraint_weight_ramp_increment = constraint_weight_ramp_increment
        self.data_weight_ramp_epoch = data_weight_ramp_epoch
        self.data_weight_ramp_decrement = data_weight_ramp_decrement
        self.data_weight_floor = data_weight_floor
        self._epoch_count = 0
        self.diff_type = diff_type

        if not self.use_data_loss:
            self.data_weight = 0
        if not self.use_ic_loss:
            self.ic_weight = 0
        if not self.use_pde_loss:
            self.pde_weight = 0
        if not self.use_constraint_loss:
            self.constraint_weight = 0

        # ------------------------------------------------------------------
        # Declarative per-epoch weight schedule (optional).
        #
        # When `weight_schedule` is given, it fully drives the data/PDE/
        # constraint weight ramps and the flat `*_ramp_*` knobs above are
        # ignored. The schedule is an ordered list of phases (see
        # losses/weight_schedule.py) that the loss walks through as training
        # progresses, so a multi-phase ramp can live entirely in the config
        # instead of being scripted as a sequence of separate training runs in
        # the launcher. `restore_weight_state` replays it on checkpoint resume.
        # The constructor weights (data/pde/constraint) are the starting
        # values, before phase 0.
        # ------------------------------------------------------------------
        self._weight_schedule = weight_schedule_lib.parse_weight_schedule(weight_schedule)
        self._init_weights = {
            "data": self.data_weight,
            "pde": self.pde_weight,
            "constraint": self.constraint_weight,
        }
        if self._weight_schedule is not None:
            # Apply phase-0 start overrides (if any) so epoch 1 trains with
            # the schedule's intended weights.
            self._set_weights(
                weight_schedule_lib.state_after(
                    self._weight_schedule, self._init_weights, 0, self.data_weight_floor
                )
            )

    def __call__(self, pred, true, inputs):
        loss, loss_dict = self.compute_losses(pred, true, inputs)
        return loss, loss_dict
    def step_weights(self):
        """Call once per epoch (after the epoch) to ramp the loss weights.

        With a `weight_schedule`, the active phase drives the ramp; otherwise
        the flat `*_ramp_*` knobs are used (legacy single-phase behavior).
        """
        self._epoch_count += 1
        if self._weight_schedule is not None:
            weights = {
                "data": self.data_weight,
                "pde": self.pde_weight,
                "constraint": self.constraint_weight,
            }
            weight_schedule_lib.step(
                self._weight_schedule, weights, self._epoch_count, self.data_weight_floor
            )
            self._set_weights(weights)
            return
        if self.pde_weight_ramp_epoch > 0 and self._epoch_count % self.pde_weight_ramp_epoch == 0:
            self.pde_weight += self.pde_weight_ramp_increment
        if self.constraint_weight_ramp_epoch > 0 and self._epoch_count % self.constraint_weight_ramp_epoch == 0:
            self.constraint_weight += self.constraint_weight_ramp_increment
        if self.data_weight_ramp_epoch > 0 and self._epoch_count % self.data_weight_ramp_epoch == 0:
            self.data_weight = max(self.data_weight_floor, self.data_weight - self.data_weight_ramp_decrement)

    def _set_weights(self, weights):
        """Write back a {data, pde, constraint} working-weights dict."""
        self.data_weight = weights["data"]
        self.pde_weight = weights["pde"]
        self.constraint_weight = weights["constraint"]

    def restore_weight_state(self, epoch):
        """Re-derive the loss weights for a run resumed at `epoch`.

        Only meaningful with a `weight_schedule`: replays the per-epoch ramp
        from scratch (via losses/weight_schedule.py) so a checkpoint-resumed
        run continues with exactly the weights it would have had in an
        uninterrupted run. For the legacy flat ramp this is a no-op (preserving
        prior resume behavior)."""
        if self._weight_schedule is None:
            return
        self._epoch_count = int(epoch)
        self._set_weights(
            weight_schedule_lib.state_after(
                self._weight_schedule, self._init_weights, epoch, self.data_weight_floor
            )
        )

    def compute_losses(self, pred, true, inputs):
        "Compute weighted loss and dictionary"
        pred = pred.reshape(true.shape)
        u1 = pred[..., 0]
        u2 = pred[..., 1]
        b1 = pred[..., 2]
        b2 = pred[..., 3]
        p = pred[..., 4]
        e3 = pred[..., 5]
        rho = pred[..., 6]

        loss_dict = {}

        # Data
        if self.use_data_loss:
            loss_data, loss_u1, loss_u2, loss_b1, loss_b2, loss_p, loss_e3, loss_rho = self.data_loss(
                pred, true, return_all_losses=True
            )
            loss_dict["loss_data"] = loss_data
            loss_dict["loss_u1"] = loss_u1
            loss_dict["loss_u2"] = loss_u2
            loss_dict["loss_b1"] = loss_b1
            loss_dict["loss_b2"] = loss_b2
            loss_dict["loss_p"] = loss_p
            loss_dict["loss_e3"] = loss_e3
            loss_dict["loss_rho"] = loss_rho
            # Full-coverage diagnostic: native-resolution data loss over ALL
            # timesteps (no spatial coarsening, no temporal stride). This is the
            # only metric that compares the *entire* domain of pred to true, so
            # it is directly comparable across coarse/stride configs — unlike
            # loss_data (penalized subset) or loss_data_highres/_skipped, which
            # are relative norms over disjoint subsets and cannot be recombined.
            # Logging-only (@torch.no_grad on _data_loss_full_on).
            loss_dict["loss_data_full"] = self._data_loss_full_on(
                pred, true, slice(None)
            )
            # Track full-res loss on skipped (non-penalized) timesteps
            loss_data_skipped = self.skipped_data_loss(pred, true)
            if loss_data_skipped is not None:
                loss_dict["loss_data_skipped"] = loss_data_skipped
            # Track full-res loss on non-skipped (strided) timesteps when
            # the enforced data loss is spatially coarsened. This mirrors
            # loss_data_skipped but on the complementary time indices, and
            # together with loss_data_skipped covers the full time axis at
            # native spatial resolution.
            if self.data_loss_coarse_factor > 1:
                non_skipped_idx = list(range(0, pred.shape[1], self.data_loss_stride))
                loss_dict["loss_data_highres"] = self._data_loss_full_on(
                    pred, true, non_skipped_idx
                )
        else:
            loss_data = 0
        # IC
        # ! Unused, non-adjusted PhysicsNeMo carry-over -- see ic_loss() below.
        if self.use_ic_loss:
            loss_ic, loss_u_ic, loss_v_ic, loss_Bx_ic, loss_By_ic = self.ic_loss(
                pred, inputs, return_all_losses=True
            )
            loss_dict["loss_ic"] = loss_ic
            loss_dict["loss_u_ic"] = loss_u_ic
            loss_dict["loss_v_ic"] = loss_v_ic
            loss_dict["loss_Bx_ic"] = loss_Bx_ic
            loss_dict["loss_By_ic"] = loss_By_ic
        else:
            loss_ic = 0

        # PDE
        if self.use_pde_loss:
            eta = inputs[..., -1] # take eta from the inputs
            FI1, FI2, MO, ES0, ES1, ES2, C1 = self.mhd_pde(u1, u2, b1, b2, p, e3, rho, eta)
            loss_pde, loss_FI1, loss_FI2, loss_MO, loss_ES0, loss_ES1, loss_ES2, loss_C1 = self.mhd_pde_loss(
                FI1, FI2, MO, ES0, ES1, ES2, C1, return_all_losses=True
            )
            loss_dict["loss_pde"] = loss_pde
            loss_dict["loss_FI1"] = loss_FI1
            loss_dict["loss_FI2"] = loss_FI2
            loss_dict["loss_MO"] = loss_MO
            loss_dict["loss_ES0"] = loss_ES0
            loss_dict["loss_ES1"] = loss_ES1
            loss_dict["loss_ES2"] = loss_ES2
            loss_dict["loss_C1"] = loss_C1
        else:
            loss_pde = 0

        # Constraints
        if self.use_constraint_loss:
            div_B = self.mhd_constraint(b1, b2)
            loss_constraint, loss_div_B = self.mhd_constraint_loss(
                div_B, return_all_losses=True
            )
            loss_dict["loss_constraint"] = loss_constraint
            loss_dict["loss_div_B"] = loss_div_B
        else:
            loss_constraint = 0

        if self.use_weighted_mean:
            weight_sum = (
                self.data_weight
                + self.ic_weight
                + self.pde_weight
                + self.constraint_weight
            )
        else:
            weight_sum = 1.0

        loss = (
            self.data_weight * loss_data
            + self.ic_weight * loss_ic
            + self.pde_weight * loss_pde
            + self.constraint_weight * loss_constraint
        ) / weight_sum
        loss_dict["loss"] = loss
        return loss, loss_dict

    def _compute_field_losses(self, pred, true):
        "Compute per-field LpLoss between pred and true"
        lploss = LpLoss(size_average=True)
        losses = []
        for i in range(7):
            losses.append(lploss(pred[..., i], true[..., i]))
        return losses  # [u1, u2, b1, b2, p, e3, rho]

    def _aggregate_field_losses(self, field_losses):
        "Aggregate per-field losses with weights"
        weights = [self.u1_weight, self.u2_weight, self.b1_weight, self.b2_weight,
                   self.p_weight, self.e3_weight, self.rho_weight]
        if self.use_weighted_mean:
            weight_sum = sum(weights)
        else:
            weight_sum = 1.0
        return sum(w * l for w, l in zip(weights, field_losses)) / weight_sum

    def _coarsen_spatial(self, x):
        """Strided-subsample spatial dims (X, Y) by self.data_loss_coarse_factor.

        Shape convention: x is (B, T, X, Y, C). Strided slicing matches the
        dataloader's own `sub_x` subsampling (see dataloaders/dataloaders.py),
        so a coarse factor of k is pointwise equivalent to training on a
        lower-resolution dataset loaded with sub_x=k.
        """
        k = self.data_loss_coarse_factor
        if k <= 1:
            return x
        return x[:, :, ::k, ::k, :]

    def data_loss(self, pred, true, return_all_losses=False):
        "Compute enforced data loss (strided in time, coarsened in space)"
        # Subsample timesteps for data loss (stride=2 means every second timestep)
        pred_sub = pred[:, ::self.data_loss_stride]
        true_sub = true[:, ::self.data_loss_stride]
        # Spatially coarsen (no-op when data_loss_coarse_factor <= 1)
        pred_sub = self._coarsen_spatial(pred_sub)
        true_sub = self._coarsen_spatial(true_sub)

        field_losses = self._compute_field_losses(pred_sub, true_sub)
        loss_data = self._aggregate_field_losses(field_losses)

        if return_all_losses:
            return (loss_data, *field_losses)
        else:
            return loss_data

    @torch.no_grad()
    def _data_loss_full_on(self, pred, true, time_idx):
        """Full-spatial-resolution data loss on a given set of time indices.

        No spatial coarsening. `time_idx` may be a list of indices or a slice.
        Used for logging only — wrapped in @torch.no_grad().
        """
        pred_sel = pred[:, time_idx]
        true_sel = true[:, time_idx]
        field_losses = self._compute_field_losses(pred_sel, true_sel)
        return self._aggregate_field_losses(field_losses)

    @torch.no_grad()
    def skipped_data_loss(self, pred, true):
        "Full-res data loss on skipped (non-penalized) timesteps for tracking only"
        if self.data_loss_stride <= 1:
            return None
        all_indices = set(range(pred.shape[1]))
        strided_indices = set(range(0, pred.shape[1], self.data_loss_stride))
        skipped_indices = sorted(all_indices - strided_indices)
        if not skipped_indices:
            return None
        return self._data_loss_full_on(pred, true, skipped_indices)

    def ic_loss(self, pred, inputs, return_all_losses=False):
        "Compute initial condition loss"
        # ! Non-adjusted carry-over from the original PhysicsNeMo example: this
        # ! path was never used for BHAC-MHD (every config sets use_ic_loss:
        # ! False) and would raise as-is -- it refers to self.u_weight/v_weight/
        # ! Bx_weight/By_weight (the constructor defines u1_weight...), the input
        # ! slice below assumes 4 channels, and line ~411 has a `x = x =` typo.
        # ! Kept for reference; can probably be made functional with small edits.
        lploss = LpLoss(size_average=True)
        ic_pred = pred[:, 0]
        ic_true = inputs[:, 0, ..., 3:]
        u_ic_pred = ic_pred[..., 0]
        v_ic_pred = ic_pred[..., 1]
        Bx_ic_pred = ic_pred[..., 2]
        By_ic_pred = ic_pred[..., 3]

        u_ic_true = ic_true[..., 0]
        v_ic_true = ic_true[..., 1]
        Bx_ic_true = ic_true[..., 2]
        By_ic_true = ic_true[..., 3]

        loss_u_ic = lploss(u_ic_pred, u_ic_true)
        loss_v_ic = lploss(v_ic_pred, v_ic_true)
        loss_Bx_ic = lploss(Bx_ic_pred, Bx_ic_true)
        loss_By_ic = lploss(By_ic_pred, By_ic_true)

        if self.use_weighted_mean:
            weight_sum = weight_sum = (
                self.u_weight + self.v_weight + self.Bx_weight + self.By_weight
            )
        else:
            weight_sum = 1.0

        loss_ic = (
            self.u_weight * loss_u_ic
            + self.v_weight * loss_v_ic
            + self.Bx_weight * loss_Bx_ic
            + self.By_weight * loss_By_ic
        ) / weight_sum

        if return_all_losses:
            return loss_ic, loss_u_ic, loss_v_ic, loss_Bx_ic, loss_By_ic
        else:
            return loss_ic

    def mhd_pde_loss(self, FI1, FI2, MO, ES0, ES1, ES2, C1, return_all_losses=None):
        "Compute PDE loss"
        loss_FI1 = FI1.pow(2).mean()
        loss_FI2 = FI2.pow(2).mean()
        loss_MO = MO.pow(2).mean()
        loss_ES0 = ES0.pow(2).mean()
        loss_ES1 = ES1.pow(2).mean()
        loss_ES2 = ES2.pow(2).mean()
        loss_C1 = C1.pow(2).mean()

        if self.use_weighted_mean:
            weight_sum = (
                self.FI1_weight
                + self.FI2_weight
                + self.MO_weight
                + self.ES0_weight
                + self.ES1_weight
                + self.ES2_weight
                + self.C1_weight
            )
        else:
            weight_sum = 1.0

        loss_pde = (
            self.FI1_weight * loss_FI1
            + self.FI2_weight * loss_FI2
            + self.MO_weight * loss_MO
            + self.ES0_weight * loss_ES0
            + self.ES1_weight * loss_ES1
            + self.ES2_weight * loss_ES2
            + self.C1_weight * loss_C1
        ) / weight_sum

        if return_all_losses:
            return (
                loss_pde,
                loss_FI1,
                loss_FI2,
                loss_MO,
                loss_ES0,
                loss_ES1,
                loss_ES2,
                loss_C1,
           )
        else:
            return loss_pde

    def mhd_constraint_loss(self, div_B, return_all_losses=False):
        "Compute constraint loss"
        loss_div_B = div_B.pow(2).mean()

        if self.use_weighted_mean:
            weight_sum = self.div_B_weight
        else:
            weight_sum = 1.0

        loss_constraint = (
            self.div_B_weight * loss_div_B
        ) / weight_sum

        if return_all_losses:
            return loss_constraint, loss_div_B
        else:
            return loss_constraint

    def mhd_constraint(self, b1, b2):
        "Compute constraints"
        # pred shape: (batch, nt, nx, ny, nfields)
        nt = b1.size(1)        # number of time frames per sample
        nx = b1.size(2)        # spatial resolution in x
        ny = b1.size(3)        # spatial resolution in y
        if self.diff_type == 'fourier':
            f_db1 = fourier_derivatives(b1, [self.Lx, self.Ly])
            f_db2 = fourier_derivatives(b2, [self.Lx, self.Ly])
        else:
            f_db1 = fd_derivatives_periodic(b1, [self.Lx, self.Ly])
            f_db2 = fd_derivatives_periodic(b2, [self.Lx, self.Ly])


        b1_x = f_db1[:, 0:nt, :nx, :ny]
        b2_y = f_db2[:, nt : 2 * nt, :nx, :ny]

        div_B = self.mhd_pde_node[0].evaluate({"b1__x": b1_x, "b2__y": b2_y})["div_B"]

        return div_B

    def mhd_pde(self, u1, u2, b1, b2, p, e3, rho, eta):
        "Compute PDEs for MHD using magnetic field"
        nt = b1.size(1)
        nx = b1.size(2)
        ny = b1.size(3)
        dt = self.tend / (nt - 1)
        dx = self.Lx / nx
        dy = self.Ly / ny

        if self.diff_type == 'fourier':
        # compute fourier derivatives
            f_du1 = fourier_derivatives(u1, [self.Lx, self.Ly])
            f_du2 = fourier_derivatives(u2, [self.Lx, self.Ly])
            f_db1 = fourier_derivatives(b1, [self.Lx, self.Ly])
            f_db2 = fourier_derivatives(b2, [self.Lx, self.Ly])
            f_dp = fourier_derivatives(p, [self.Lx, self.Ly])
            f_de3 = fourier_derivatives(e3, [self.Lx, self.Ly])
            f_drho = fourier_derivatives(rho, [self.Lx, self.Ly])

        else:
        # compute finite diff derivatives
            f_du1 = fd_derivatives_periodic(u1, [self.Lx, self.Ly])
            f_du2 = fd_derivatives_periodic(u2, [self.Lx, self.Ly])
            f_db1 = fd_derivatives_periodic(b1, [self.Lx, self.Ly])
            f_db2 = fd_derivatives_periodic(b2, [self.Lx, self.Ly])
            f_dp = fd_derivatives_periodic(p, [self.Lx, self.Ly])
            f_de3 = fd_derivatives_periodic(e3, [self.Lx, self.Ly])
            f_drho = fd_derivatives_periodic(rho, [self.Lx, self.Ly])


        u1_x = f_du1[:, 0:nt, :nx, :ny]
        u1_y = f_du1[:, nt : 2 * nt, :nx, :ny]
        u2_x = f_du2[:, 0:nt, :nx, :ny]
        u2_y = f_du2[:, nt : 2 * nt, :nx, :ny]
        b1_x = f_db1[:, 0:nt, :nx, :ny]
        b1_y = f_db1[:, nt : 2 * nt, :nx, :ny]
        b2_x = f_db2[:, 0:nt, :nx, :ny]
        b2_y = f_db2[:, nt : 2 * nt, :nx, :ny]
        p_x = f_dp[:, 0:nt, :nx, :ny]
        p_y = f_dp[:, nt : 2 * nt, :nx, :ny]
        e3_x = f_de3[:, 0:nt, :nx, :ny]
        e3_y = f_de3[:, nt : 2 * nt, :nx, :ny]
        rho_x = f_drho[:, 0:nt, :nx, :ny]
        rho_y = f_drho[:, nt : 2 * nt, :nx, :ny]

        # time derivatives: central difference
        u1_t = self.Du_t(u1, dt)
        u2_t = self.Du_t(u2, dt)
        b1_t = self.Du_t(b1, dt)
        b2_t = self.Du_t(b2, dt)
        p_t = self.Du_t(p, dt)
        e3_t = self.Du_t(e3, dt)
        rho_t= self.Du_t(rho, dt)

        # Plug inputs into dictionary
        all_inputs = {
            "u1": u1,
            "u2": u2,
            "u1__x": u1_x,
            "u1__y": u1_y,
            "u2__x": u2_x,
            "u2__y": u2_y,
            "u1__t": u1_t,
            "u2__t": u2_t,
            "b1": b1,
            "b2": b2,
            "b1__x": b1_x,
            "b1__y": b1_y,
            "b2__x": b2_x,
            "b2__y": b2_y,
            "b1__t": b1_t,
            "b2__t": b2_t,
            "p": p,
            "p__x": p_x,
            "p__y": p_y,
            "p__t": p_t,
            "e3": e3,
            "e3__x": e3_x,
            "e3__y": e3_y,
            "e3__t": e3_t,
            "rho": rho,
            "rho__x": rho_x,
            "rho__y": rho_y,
            "rho__t": rho_t,
        }
        # Evaluate equations

        # Faraday Induction I & II
        FI1 = self.mhd_pde_node[1].evaluate({"b1__t": b1_t, "e3__y": e3_y})[
            "FI1"
        ]
        # "e3__y": e3_y[:, 1:-1]
        FI2 = self.mhd_pde_node[2].evaluate({"b2__t": b2_t, "e3__x": e3_x})[
            "FI2"
        ]
        # "e3__x": e3_x[:, 1:-1]

        # Gauss, Maxwell, Ohm
        MO = self.mhd_pde_node[3].evaluate({
            "u1": u1,
            "u2": u2,
            "b1": b1,
            "b2": b2,
            "e3": e3,
            "eta": eta,
            "b1__y": b1_y,
            "b2__x": b2_x,
            "e3__t":e3_t
        })["MO"]
        #for central differencing: "e3__x": e3_x[:, 1:-1]

        # EoS
        ES0 = self.mhd_pde_node[4].evaluate(all_inputs)["ES0"]
        ES1 = self.mhd_pde_node[5].evaluate(all_inputs)["ES1"]
        ES2 = self.mhd_pde_node[6].evaluate(all_inputs)["ES2"]

        # Continuity Eq
        # del_t (rho gamma) + div (rho u) = 0
        C1 = self.mhd_pde_node[7].evaluate({
            "u1": u1,
            "u2": u2,
            "rho": rho,
            "u1__x": u1_x,
            "u2__y": u2_y,
            "u1__t": u1_t,
            "u2__t": u2_t,
            "rho__x": rho_x,
            "rho__y": rho_y,
            "rho__t": rho_t,
        })["C1"]

        return FI1, FI2, MO, ES0, ES1, ES2, C1

    '''
    # central differencing
    # leaves out first and last timestep
    def Du_t(self, u, dt):
        "Compute time derivative"
        u_t = (u[:, 2:] - u[:, :-2]) / (2 * dt)
        return u_t
    '''
    # includes first and last timestep
    def Du_t(self, u, dt):
        "Time derivative with boundary support (shape preserved)"
        u_t = torch.empty_like(u)

        # one-sided at boundaries
        u_t[:, 0, ...]  = (u[:, 1, ...] - u[:, 0, ...]) / dt
        u_t[:, -1, ...] = (u[:, -1, ...] - u[:, -2, ...]) / dt

        # centered in interior
        u_t[:, 1:-1, ...] = (u[:, 2:, ...] - u[:, :-2, ...]) / (2 * dt)

        return u_t
