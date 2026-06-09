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

import hydra
from omegaconf import DictConfig
import torch
import plotly
import os

from torch.nn.parallel import DistributedDataParallel
from omegaconf import OmegaConf

from physicsnemo.models.fno import FNO
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import (
    PythonLogger,
    LaunchLogger,
)
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.sym.hydra import to_absolute_path
from dataloaders import BHACUniformDataset, BHACDataloader

from losses import LossMHD_PhysicsNeMo
from torch.optim import AdamW
from utils.plot_utils import plot_predictions_mhd, plot_predictions_mhd_plotly
import wandb

dtype = torch.float
torch.set_default_dtype(dtype)


@hydra.main(version_base="1.3", config_path="config", config_name="mhd_bhac.yaml")
def main(cfg: DictConfig) -> None:
    """Train a Fourier Neural Operator on 2D MHD trajectories produced by BHAC.

    The model is trained as a PINO: a supervised data loss on the FNO output is
    combined with PDE-residual losses (ideal MHD: continuity, momentum, induction,
    energy) and a soft divergence-free constraint on B. Configuration is fully
    Hydra-driven via ``config/mhd_bhac.yaml``.
    """

    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    log = PythonLogger(name="mhd_pino")
    log.file_logging()

    # if multiple ranks, start only one wandb log
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    wandb_dir = cfg.wandb_params.wandb_dir
    wandb_project = cfg.wandb_params.wandb_project
    wandb_group = cfg.wandb_params.wandb_group
    plot_index_t = cfg.wandb_params.wandb_plot_index_t

    initialize_wandb(
        project=wandb_project,
        entity=cfg.wandb_params.wandb_entity,
        mode=cfg.wandb_params.wandb_mode if rank == 0 else "disabled",
        group=wandb_group,
        config=dict(cfg),
        results_dir=wandb_dir,
    )

    LaunchLogger.initialize(use_wandb=cfg.use_wandb)  # PhysicsNeMo launch logger

    # Load config file parameters
    model_params = cfg.model_params
    dataset_params = cfg.dataset_params
    train_loader_params = cfg.train_loader_params
    val_loader_params = cfg.val_loader_params
    test_loader_params = cfg.test_loader_params
    loss_params = cfg.loss_params
    optimizer_params = cfg.optimizer_params
    train_params = cfg.train_params
    wandb_params = cfg.wandb_params

    load_ckpt = cfg.load_ckpt
    output_dir = cfg.output_dir
    use_wandb = cfg.use_wandb

    output_dir = to_absolute_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data_dir = dataset_params.data_dir
    ckpt_path = train_params.ckpt_path

    # Construct dataloaders
    dataset_train = BHACUniformDataset(
        dataset_params.data_dir,
        output_names=dataset_params.output_names,
        file_name=dataset_params.file_name,
        num_train=dataset_params.num_train,
        num_test=dataset_params.num_test,
        use_train=True,
    )
    dataset_val = BHACUniformDataset(
        dataset_params.data_dir,
        output_names=dataset_params.output_names,
        file_name=dataset_params.file_name,
        num_train=dataset_params.num_train,
        num_test=dataset_params.num_test,
        use_train=False,
    )


    mhd_dataloader_train = BHACDataloader(
        dataset_train,
        sub_x=dataset_params.sub_x,
        sub_t=dataset_params.sub_t,
        ind_x=dataset_params.ind_x,
        ind_t=dataset_params.ind_t,
        ind_t_start=dataset_params.ind_t_start,
    )
    mhd_dataloader_val = BHACDataloader(
        dataset_val,
        sub_x=dataset_params.sub_x,
        sub_t=dataset_params.sub_t,
        ind_x=dataset_params.ind_x,
        ind_t=dataset_params.ind_t,
        ind_t_start=dataset_params.ind_t_start,
    )

    dataloader_train, sampler_train = mhd_dataloader_train.create_dataloader(
        batch_size=train_loader_params.batch_size,
        shuffle=train_loader_params.shuffle,
        num_workers=train_loader_params.num_workers,
        pin_memory=train_loader_params.pin_memory,
        distributed=dist.distributed,
    )
    dataloader_val, sampler_val = mhd_dataloader_val.create_dataloader(
        batch_size=val_loader_params.batch_size,
        shuffle=val_loader_params.shuffle,
        num_workers=val_loader_params.num_workers,
        pin_memory=val_loader_params.pin_memory,
        distributed=dist.distributed,
    )

    # define FNO model
    # Convert `modes` to a plain Python int/list. If it is a ListConfig
    # (because `modes` was given as a list in the Hydra config, e.g.
    # `model_params.modes=[8,12,12]`), physicsnemo.Module would otherwise
    # fail to JSON-serialize it in save_checkpoint.
    modes_cfg = model_params.modes
    if isinstance(modes_cfg, int):
        num_fno_modes = modes_cfg
    else:
        num_fno_modes = list(OmegaConf.to_container(modes_cfg, resolve=True))

    model = FNO(
        in_channels=model_params.in_dim,
        out_channels=model_params.out_dim,
        decoder_layers=model_params.decoder_layers,
        decoder_layer_size=model_params.fc_dim,
        dimension=model_params.dimension,
        latent_channels=model_params.layers,
        num_fno_layers=model_params.num_fno_layers,
        num_fno_modes=num_fno_modes,
        padding=[model_params.pad_z, model_params.pad_y, model_params.pad_x],
    ).to(dist.device)

    # Set up DistributedDataParallel if using more than a single process.
    # The `distributed` property of DistributedManager can be used to
    # check this.
    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],  # Set the device_id to be
                # the local rank of this process on
                # this node
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # Construct optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        betas=optimizer_params.betas,
        lr=optimizer_params.lr,
        weight_decay=optimizer_params.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=optimizer_params.milestones, gamma=optimizer_params.gamma
    )

    # Derive tend from the dataloader's already-computed real time axis
    loss_params.tend = float(mhd_dataloader_train.t_real[-1] - mhd_dataloader_train.t_real[0])
    print(f"[auto] tend = {loss_params.tend:.6f}")

    # Construct Loss class
    mhd_loss = LossMHD_PhysicsNeMo(**loss_params)

    # Load model from checkpoint (if exists)
    loaded_epoch = 0
    if load_ckpt:
        loaded_epoch = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, device=dist.device
        )

    # Training Loop
    epochs = train_params.epochs
    ckpt_freq = train_params.ckpt_freq
    names = dataset_params.fields
    input_norm = torch.tensor(model_params.input_norm).to(dist.device)
    output_norm = torch.tensor(model_params.output_norm).to(dist.device)

    for epoch in range(max(1, loaded_epoch + 1), epochs + 1):
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader_train),
            epoch_alert_freq=1,
        ) as log:
            if dist.distributed:
                sampler_train.set_epoch(epoch)

            # Train Loop
            model.train()

            for i, (inputs, outputs) in enumerate(dataloader_train):
                inputs = inputs.type(torch.FloatTensor).to(dist.device)
                outputs = outputs.type(torch.FloatTensor).to(dist.device)
                optimizer.zero_grad()
                pred = (
                    model((inputs / input_norm).permute(0, 4, 1, 2, 3)).permute(
                        0, 2, 3, 4, 1
                    )
                    * output_norm
                )
                loss, loss_dict = mhd_loss(pred, outputs, inputs, return_loss_dict=True)
                loss.backward()
                optimizer.step()

                loss_dict = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
                log.log_minibatch(loss_dict)

            mhd_loss.step_weights()
            log.log_epoch({
                "Learning Rate": optimizer.param_groups[0]["lr"],
                "pde_weight": mhd_loss.pde_weight,
                "constraint_weight": mhd_loss.constraint_weight,
                "data_weight": mhd_loss.data_weight,
            })
            scheduler.step()

        with LaunchLogger("valid", epoch=epoch) as log:
            # Val loop
            model.eval()
            val_loss_dict = {}
            plot_count = 0
            plot_dict = {name: {} for name in names}
            with torch.no_grad():
                for i, (inputs, outputs) in enumerate(dataloader_val):
                    inputs = inputs.type(dtype).to(dist.device)
                    outputs = outputs.type(dtype).to(dist.device)

                    # Compute Predictions
                    pred = (
                        model((inputs / input_norm).permute(0, 4, 1, 2, 3)).permute(
                            0, 2, 3, 4, 1
                        )
                        * output_norm
                    )
                    # Compute Loss
                    loss, loss_dict = mhd_loss(
                        pred, outputs, inputs, return_loss_dict=True
                    )

                    loss_dict = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
                    log.log_minibatch(loss_dict)

                    # Get prediction plots to log for wandb
                    # Do for number of batches specified in the config file
                    if (i < wandb_params.wandb_num_plots) and (
                        epoch % wandb_params.wandb_plot_freq == 0
                    ):
                        # Add all predictions in batch
                        for j, _ in enumerate(pred):
                            # Make plots for each field
                            for index, name in enumerate(names):
                                # Generate figure
                                if use_wandb:
                                    figs = plot_predictions_mhd_plotly(
                                        pred[j].cpu(),
                                        outputs[j].cpu(),
                                        inputs[j].cpu(),
                                        index=index,
                                        name=name,
                                        index_t=plot_index_t,
                                    )
                                    # Add figure to plot dict
                                    plot_dict[name].update({
                                        f"{name}/{plot_type}-{plot_count}": wandb.Html(plotly.io.to_html(fig))
                                        for plot_type, fig in zip(wandb_params.wandb_plot_types, figs)
                                    })

                            plot_count += 1

                    # Get prediction plots and save images locally
                    if (i < 2) and (epoch % wandb_params.wandb_plot_freq == 0):
                        # Add all predictions in batch
                        for j, _ in enumerate(pred):
                            # Generate figure
                            plot_predictions_mhd(
                                pred[j].cpu(),
                                outputs[j].cpu(),
                                inputs[j].cpu(),
                                names=names,
                                index_t=plot_index_t,
                                save_path=os.path.join(
                                    output_dir,
                                    "MHD_" + str(dist.rank),
                                ),
                                save_suffix=i,
                            )

            if use_wandb and epoch % wandb_params.wandb_plot_freq == 0:
                wandb.log({"plots": plot_dict})

            if epoch % ckpt_freq == 0 and dist.rank == 0:
                save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch=epoch)


if __name__ == "__main__":
    main()
