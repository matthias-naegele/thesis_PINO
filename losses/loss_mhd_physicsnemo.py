# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modifications copyright (c) 2026 Matthias Nägele.
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

class LossMHD_PhysicsNeMo(object):
    "Calculate loss for MHD equations with magnetic field, using physicsnemo derivatives"

    def __init__(
        self,
        Gamma=4.0 / 3.0,
        data_weight=1.0,
        pde_weight=0,
        constraint_weight=0,
        use_data_loss=False,
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
        physics_loss_stride=1,
        pde_weight_ramp_epoch=0,
        pde_weight_ramp_increment=0,
        constraint_weight_ramp_epoch=0,
        constraint_weight_ramp_increment=0,
        data_weight_ramp_epoch=0,
        data_weight_ramp_decrement=0,
        data_weight_floor=0.0,
        diff_type='fourier',
        **kwargs,
    ):  # **kwargs lets us pass a full config dict and ignore extra keys
        self.Gamma = Gamma
        self.data_weight = data_weight
        self.pde_weight = pde_weight
        self.constraint_weight = constraint_weight
        self.use_data_loss = use_data_loss
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
        self.physics_loss_stride = physics_loss_stride
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
        if not self.use_pde_loss:
            self.pde_weight = 0
        if not self.use_constraint_loss:
            self.constraint_weight = 0

    def __call__(self, pred, true, inputs, return_loss_dict=False):
        if not return_loss_dict:
            loss = self.compute_loss(pred, true, inputs)
            return loss
        else:
            loss, loss_dict = self.compute_losses(pred, true, inputs)
            return loss, loss_dict
    def step_weights(self):
        """Call once per epoch to ramp PDE/constraint up and data down."""
        self._epoch_count += 1
        if self.pde_weight_ramp_epoch > 0 and self._epoch_count % self.pde_weight_ramp_epoch == 0:
            self.pde_weight += self.pde_weight_ramp_increment
        if self.constraint_weight_ramp_epoch > 0 and self._epoch_count % self.constraint_weight_ramp_epoch == 0:
            self.constraint_weight += self.constraint_weight_ramp_increment
        if self.data_weight_ramp_epoch > 0 and self._epoch_count % self.data_weight_ramp_epoch == 0:
            self.data_weight = max(self.data_weight_floor, self.data_weight - self.data_weight_ramp_decrement)

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

        # PDE
        if self.use_pde_loss:
            eta = inputs[..., -1] # take eta from the inputs
            FI1, FI2, MO, ES0, ES1, ES2, C1 = self.mhd_pde(u1, u2, b1, b2, p, e3, rho, eta)
            # Apply physics stride: compute loss only on strided timesteps
            s = self.physics_loss_stride
            if s > 1:
                FI1_s, FI2_s, MO_s = FI1[:, ::s], FI2[:, ::s], MO[:, ::s]
                ES0_s, ES1_s, ES2_s, C1_s = ES0[:, ::s], ES1[:, ::s], ES2[:, ::s], C1[:, ::s]
            else:
                FI1_s, FI2_s, MO_s = FI1, FI2, MO
                ES0_s, ES1_s, ES2_s, C1_s = ES0, ES1, ES2, C1
            loss_pde, loss_FI1, loss_FI2, loss_MO, loss_ES0, loss_ES1, loss_ES2, loss_C1 = self.mhd_pde_loss(
                FI1_s, FI2_s, MO_s, ES0_s, ES1_s, ES2_s, C1_s, return_all_losses=True
            )
            loss_dict["loss_pde"] = loss_pde
            loss_dict["loss_FI1"] = loss_FI1
            loss_dict["loss_FI2"] = loss_FI2
            loss_dict["loss_MO"] = loss_MO
            loss_dict["loss_ES0"] = loss_ES0
            loss_dict["loss_ES1"] = loss_ES1
            loss_dict["loss_ES2"] = loss_ES2
            loss_dict["loss_C1"] = loss_C1
            # Track loss on skipped (non-penalized) physics timesteps
            if s > 1:
                loss_pde_skipped = self._skipped_physics_loss(
                    FI1, FI2, MO, ES0, ES1, ES2, C1
                )
                if loss_pde_skipped is not None:
                    loss_dict["loss_pde_skipped"] = loss_pde_skipped
        else:
            loss_pde = 0

        # Constraints
        if self.use_constraint_loss:
            div_B = self.mhd_constraint(b1, b2)
            # Apply same physics stride
            s = self.physics_loss_stride
            if s > 1:
                div_B_s = div_B[:, ::s]
            else:
                div_B_s = div_B
            loss_constraint, loss_div_B = self.mhd_constraint_loss(
                div_B_s, return_all_losses=True
            )
            loss_dict["loss_constraint"] = loss_constraint
            loss_dict["loss_div_B"] = loss_div_B
            # Track loss on skipped constraint timesteps
            if s > 1:
                loss_constraint_skipped = self._skipped_constraint_loss(div_B)
                if loss_constraint_skipped is not None:
                    loss_dict["loss_constraint_skipped"] = loss_constraint_skipped
        else:
            loss_constraint = 0

        if self.use_weighted_mean:
            weight_sum = (
                self.data_weight
                + self.pde_weight
                + self.constraint_weight
            )
        else:
            weight_sum = 1.0

        loss = (
            self.data_weight * loss_data
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

    @torch.no_grad()
    def _skipped_physics_loss(self, FI1, FI2, MO, ES0, ES1, ES2, C1):
        "Compute PDE loss on skipped (non-penalized) timesteps for tracking only"
        s = self.physics_loss_stride
        if s <= 1:
            return None
        nt = FI1.shape[1]
        all_idx = set(range(nt))
        strided_idx = set(range(0, nt, s))
        skipped = sorted(all_idx - strided_idx)
        if not skipped:
            return None
        return self.mhd_pde_loss(
            FI1[:, skipped], FI2[:, skipped], MO[:, skipped],
            ES0[:, skipped], ES1[:, skipped], ES2[:, skipped], C1[:, skipped],
        )

    @torch.no_grad()
    def _skipped_constraint_loss(self, div_B):
        "Compute constraint loss on skipped (non-penalized) timesteps for tracking only"
        s = self.physics_loss_stride
        if s <= 1:
            return None
        nt = div_B.shape[1]
        all_idx = set(range(nt))
        strided_idx = set(range(0, nt, s))
        skipped = sorted(all_idx - strided_idx)
        if not skipped:
            return None
        return self.mhd_constraint_loss(div_B[:, skipped])

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
        nt = b1.size(1)
        nx = b1.size(2)
        ny = b1.size(3)
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
        rho_t = self.Du_t(rho, dt)

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
        FI1 = self.mhd_pde_node[1].evaluate({"b1__t": b1_t, "e3__y": e3_y})["FI1"]
        FI2 = self.mhd_pde_node[2].evaluate({"b2__t": b2_t, "e3__x": e3_x})["FI2"]

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
            "e3__t": e3_t,
        })["MO"]

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

    def Du_t(self, u, dt):
        "Time derivative with boundary support (shape preserved)"
        u_t = torch.empty_like(u)

        # one-sided at boundaries
        u_t[:, 0, ...]  = (u[:, 1, ...] - u[:, 0, ...]) / dt
        u_t[:, -1, ...] = (u[:, -1, ...] - u[:, -2, ...]) / dt

        # centered in interior
        u_t[:, 1:-1, ...] = (u[:, 2:, ...] - u[:, :-2, ...]) / (2 * dt)

        return u_t
