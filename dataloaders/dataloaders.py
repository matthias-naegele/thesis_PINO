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
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import glob
import h5py


class BHACDataloader(Dataset):
    """
    Builds (inputs, outputs) for BHAC MHD-like training:
      inputs: [t,x,y, (u1,u2,b1,b2,p,e3,rho) at t0, eta]  -> 11 channels
      outputs: (u1,u2,b1,b2,p,e3,rho) over time           -> 7 channels
    """

    def __init__(
        self,
        dataset,
        sub_x=1,
        sub_t=1,
        ind_x=None,
        ind_t=None,
        ind_t_start=0,
    ):
        self.dataset = dataset
        self.sub_x = sub_x
        self.sub_t = sub_t
        self.ind_x = ind_x
        self.ind_t = ind_t
        self.ind_t_start = ind_t_start

        t, x, y = dataset.get_coords(0)
        self.x = x[:ind_x:sub_x]
        self.y = y[:ind_x:sub_x]
        # Use the selected start frame as the new t=0 for the model
        t_sel = t[ind_t_start:ind_t:sub_t]
        self.t = t_sel - t_sel[0]
        self.t_real = t_sel  # un-shifted, physical time values

        self.nx = len(self.x)
        self.ny = len(self.y)
        self.nt = len(self.t)

        self.x_slice = slice(0, self.ind_x, self.sub_x)
        self.y_slice = slice(0, self.ind_x, self.sub_x)
        self.t_slice = slice(self.ind_t_start, self.ind_t, self.sub_t)

        # Cache varname->index map (assumes consistent varnames across files)
        self.var_idx = dataset.get_var_index_map(0)

        self.out_names = ["u1", "u2", "b1", "b2", "p", "e3", "rho"]
        self.eta_name = "eta"

        # Resolve indices once
        self.out_idx = [self.var_idx[n] for n in self.out_names]
        self.eta_idx = self.var_idx[self.eta_name]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file = self.dataset[index]

        with h5py.File(file, "r") as f:
            F = f["fields"]  # BHAC export layout is (T, C, Ny, Nx)

            # outputs: stack 7 channels across time
            outs = []
            for k in self.out_idx:
                # Read as (nt, ny, nx) then transpose -> (nt, nx, ny)
                arr = F[self.t_slice, k, self.y_slice, self.x_slice]
                outs.append(torch.from_numpy(arr).permute(0, 2, 1))
            outputs = torch.stack(outs, dim=-1)  # (nt,nx,ny,7)

            # initial condition (t=0)
            # initial condition = first kept frame (ind_t_start-th frame)
            data0 = outputs[0].reshape(1, self.nx, self.ny, 7).repeat(self.nt, 1, 1, 1)

            # eta: take t=0 and broadcast over time
            eta0 = F[self.ind_t_start, self.eta_idx, self.y_slice, self.x_slice]
            eta0 = torch.from_numpy(eta0).permute(1, 0).reshape(1, self.nx, self.ny, 1)
            eta_rep = eta0.repeat(self.nt, 1, 1, 1)

        # coords
        grid_t = torch.from_numpy(self.t).reshape(self.nt, 1, 1, 1).repeat(1, self.nx, self.ny, 1)
        grid_x = torch.from_numpy(self.x).reshape(1, self.nx, 1, 1).repeat(self.nt, 1, self.ny, 1)
        grid_y = torch.from_numpy(self.y).reshape(1, 1, self.ny, 1).repeat(self.nt, self.nx, 1, 1)

        inputs = torch.cat([grid_t, grid_x, grid_y, data0, eta_rep], dim=-1)  # (nt,nx,ny,11)
        return inputs, outputs

    def create_dataloader(self, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, distributed=False):
        if distributed:
            sampler = torch.utils.data.DistributedSampler(self)
            dl = DataLoader(self, batch_size=batch_size, shuffle=False, sampler=sampler,
                            num_workers=num_workers, pin_memory=pin_memory)
        else:
            sampler = None
            dl = DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=pin_memory)
        return dl, sampler
