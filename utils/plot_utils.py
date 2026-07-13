# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import matplotlib.pyplot as plt
import plotly.express as px


def plot_predictions_mhd(
    pred,
    true,
    inputs,
    index_t=-1,
    names=[],
    save_path=None,
    save_suffix=None,
    font_size=None,
    sci_limits=None,
    shading="auto",
    cmap="jet",
):
    "Plots images of predictions and absolute error"
    if font_size is not None:
        plt.rcParams.update({"font.size": font_size})

    if sci_limits is not None:
        plt.rcParams.update({"axes.formatter.limits": sci_limits})
    # Plot
    fig = plt.figure(figsize=(24, 5 * len(names)))

    # Make plots for each field
    for index, name in enumerate(names):
        Nt, Nx, Ny, Nfields = pred.shape
        u_pred = pred[index_t, ..., index]
        u_true = true[index_t, ..., index]
        u_err = u_pred - u_true

        Nfields = true.shape[-1]  # excludes eta by construction (eta is not in true)
        initial_data = inputs[0, ..., 3:3 + Nfields]

        u0 = initial_data[..., index]

        x = inputs[0, :, 0, 1]
        y = inputs[0, 0, :, 2]
        X, Y = torch.meshgrid(x, y, indexing="ij")
        t = inputs[index_t, 0, 0, 0]

        plt.subplot(len(names), 4, index * 4 + 1)
        plt.pcolormesh(X, Y, u0, cmap=cmap, shading=shading)
        plt.colorbar()
        plt.title(f"Intial Condition ${name}_0(x,y)$")
        plt.tight_layout()
        plt.axis("square")
        plt.axis("off")

        plt.subplot(len(names), 4, index * 4 + 2)
        plt.pcolormesh(X, Y, u_true, cmap=cmap, shading=shading)
        plt.colorbar()
        plt.title(f"Exact ${name}(x,y,t={t:.2f})$")
        plt.tight_layout()
        plt.axis("square")
        plt.axis("off")

        plt.subplot(len(names), 4, index * 4 + 3)
        plt.pcolormesh(X, Y, u_pred, cmap=cmap, shading=shading)
        plt.colorbar()
        plt.title(f"Predict ${name}(x,y,t={t:.2f})$")
        plt.axis("square")
        plt.tight_layout()
        plt.axis("off")

        plt.subplot(len(names), 4, index * 4 + 4)
        plt.pcolormesh(X, Y, u_pred - u_true, cmap=cmap, shading=shading)
        plt.colorbar()
        plt.title(f"Absolute Error ${name}(x,y,t={t:.2f})$")
        plt.tight_layout()
        plt.axis("square")
        plt.axis("off")

    if save_path is not None:
        if save_suffix is not None:
            figure_path = f"{save_path}_{save_suffix}.png"
        else:
            figure_path = f"{save_path}.png"
        plt.savefig(figure_path, bbox_inches="tight")
    # plt.show()
    # return fig
    plt.close()


def plot_predictions_mhd_plotly(
    pred,
    true,
    inputs,
    index=0,
    index_t=-1,
    name="u",
    save_path=None,
    font_size=None,
    shading="auto",
    cmap="jet",
):
    "Plots images of predictions and absolute error to be saved to wandb"
    Nt, Nx, Ny, Nfields = pred.shape
    u_pred = pred[index_t, ..., index]
    u_true = true[index_t, ..., index]

    Nfields = true.shape[-1]
    ic = inputs[0, ..., 3:3 + Nfields]

    u_ic = ic[..., index]
    u_err = u_pred - u_true

    x = inputs[0, :, 0, 1]
    y = inputs[0, 0, :, 2]
    X, Y = torch.meshgrid(x, y, indexing="ij")
    t = inputs[index_t, 0, 0, 0]

    zmin = u_true.min().item()
    zmax = u_true.max().item()
    labels = {"color": name}

    # Initial Conditions
    title_ic = f"{name}0"
    fig_ic = px.imshow(
        u_ic,
        binary_string=False,
        color_continuous_scale=cmap,
        labels=labels,
        title=title_ic,
    )
    fig_ic.update_xaxes(showticklabels=False)
    fig_ic.update_yaxes(showticklabels=False)

    # Predictions
    title_pred = f"Predict {name}: t={t:.2f}"
    fig_pred = px.imshow(
        u_pred,
        binary_string=False,
        color_continuous_scale=cmap,
        labels=labels,
        title=title_pred,
    )
    fig_pred.update_xaxes(showticklabels=False)
    fig_pred.update_yaxes(showticklabels=False)

    # Ground Truth
    title_true = f"Exact {name}: t={t:.2f}"
    fig_true = px.imshow(
        u_true,
        binary_string=False,
        color_continuous_scale=cmap,
        labels=labels,
        title=title_true,
    )
    fig_true.update_xaxes(showticklabels=False)
    fig_true.update_yaxes(showticklabels=False)

    # Ground Truth
    title_err = f"Error {name}: t={t:.2f}"
    fig_err = px.imshow(
        u_err,
        binary_string=False,
        color_continuous_scale=cmap,
        labels=labels,
        title=title_err,
    )
    fig_err.update_xaxes(showticklabels=False)
    fig_err.update_yaxes(showticklabels=False)

    return fig_ic, fig_pred, fig_true, fig_err
