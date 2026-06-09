# SPDX-FileCopyrightText: Copyright (c) 2026 Matthias Nägele.
# SPDX-License-Identifier: Apache-2.0
"""
Plot domain-averaged B^2 vs time for validation data.

Loads all checkpoints from the checkpoint directory (defined in the Hydra
config at train_params.ckpt_path), evaluates the FNO model on the validation
set, and produces one plot per validation sample per checkpoint showing
<B^2>(t) for both the true solution and the model prediction.

Usage:
    python plot_B2_onValidation.py --output_dir ./b2_plots
"""
 
import argparse
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import re
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
 
# German physics style: serif font, inward ticks on all sides
mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "axes.formatter.use_mathtext": True,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "figure.figsize": (5.5, 4.0),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
 
from physicsnemo.models.fno import FNO
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint
from physicsnemo.sym.hydra import to_absolute_path
from dataloaders import BHACUniformDataset, BHACDataloader
 
dtype = torch.float
torch.set_default_dtype(dtype)
 
 
def find_checkpoint_epochs(ckpt_path):
    """Find all available checkpoint epochs in the directory.
 
    Checkpoints are saved as  {ModelName}.0.{epoch}.mdlus  (model)
    and  checkpoint.0.{epoch}.pt  (training state).
    We glob the model files and extract epoch numbers.
    """
    pattern = os.path.join(ckpt_path, "*.0.*.mdlus")
    files = glob.glob(pattern)
    epochs = []
    for f in files:
        basename = os.path.basename(f)
        m = re.match(r".+\.0\.(\d+)\.mdlus$", basename)
        if m:
            epochs.append(int(m.group(1)))
    epochs.sort()
    return epochs
 
 
def compute_domain_avg_b_squared(fields):
    """Compute <B^2>(t) = spatial average of (b1^2 + b2^2) at each timestep.
 
    Parameters
    ----------
    fields : Tensor of shape (nt, nx, ny, 7)
        Output fields ordered as [u1, u2, b1, b2, p, e3, rho].
 
    Returns
    -------
    b2_avg : ndarray of shape (nt,)
    """
    b1 = fields[..., 2]  # (nt, nx, ny)
    b2 = fields[..., 3]  # (nt, nx, ny)
    b_squared = b1 ** 2 + b2 ** 2  # (nt, nx, ny)
    b2_avg = b_squared.mean(dim=(1, 2))  # (nt,)
    return b2_avg.numpy()
 
 
# Parse --output_dir from argv before Hydra consumes the args
def parse_output_dir():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output_dir", type=str, default="./b2_plots",
                        help="Directory to save plots (default: ./b2_plots)")
    known, remaining = parser.parse_known_args()
    # Put remaining args back so Hydra can parse them
    sys.argv = [sys.argv[0]] + remaining
    return known.output_dir
 
 
OUTPUT_DIR = parse_output_dir()
 
 
@hydra.main(version_base="1.3", config_path="config", config_name="mhd_bhac.yaml")
def main(cfg: DictConfig) -> None:
    snap = os.path.join(cfg.train_params.ckpt_path, "config.yaml")
    if os.path.isfile(snap):
        print(f"[config] loaded snapshot {snap}")
        cfg = OmegaConf.load(snap)
    else:
        print(f"[config] no snapshot at {snap}; using live config/*.yaml")
    OmegaConf.set_struct(cfg, False)

    DistributedManager.initialize()
    dist = DistributedManager()
 
    # Checkpoint path from the yaml config
    ckpt_path = cfg.train_params.ckpt_path
 
    # Output dir from CLI --output_dir
    output_dir = os.path.abspath(OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
 
    model_params = cfg.model_params
    dataset_params = cfg.dataset_params
 
    # Build validation dataset / dataloader
    dataset_val = BHACUniformDataset(
        dataset_params.data_dir,
        output_names=dataset_params.output_names,
        file_name=dataset_params.file_name,
        num_train=dataset_params.num_train,
        num_test=dataset_params.num_test,
        use_train=False,
    )
    mhd_dataloader_val = BHACDataloader(
        dataset_val,
        sub_x=dataset_params.sub_x,
        sub_t=dataset_params.sub_t,
        ind_x=dataset_params.ind_x,
        ind_t=dataset_params.ind_t,
        ind_t_start=dataset_params.ind_t_start,
    )
    dataloader_val, _ = mhd_dataloader_val.create_dataloader(
        batch_size=1,
        shuffle=False,
        num_workers=cfg.val_loader_params.num_workers,
        pin_memory=cfg.val_loader_params.pin_memory,
        distributed=False,
    )
 
    # Print which files are used for validation
    print(f"\nValidation files ({len(dataset_val)} total):")
    for i, f in enumerate(dataset_val.files):
        print(f"  [{i}] {f}")
    print()
 
    # Build model
    model = FNO(
        in_channels=model_params.in_dim,
        out_channels=model_params.out_dim,
        decoder_layers=model_params.decoder_layers,
        decoder_layer_size=model_params.fc_dim,
        dimension=model_params.dimension,
        latent_channels=model_params.layers,
        num_fno_layers=model_params.num_fno_layers,
        num_fno_modes=model_params.modes,
        padding=[model_params.pad_z, model_params.pad_y, model_params.pad_x],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
 
    input_norm = torch.tensor(model_params.input_norm).to(device)
    output_norm = torch.tensor(model_params.output_norm).to(device)
 
    # Data loss stride: the model is supervised on every N-th timestep
    data_loss_stride = cfg.loss_params.get("data_loss_stride", 1)

    # Real (un-shifted) time axis — already computed by the dataloader, no extra I/O
    t_real = mhd_dataloader_val.t_real
 
    # Discover checkpoints
    epochs = find_checkpoint_epochs(ckpt_path)
    if not epochs:
        print(f"No checkpoints found in {ckpt_path}")
        return
    print(f"Found {len(epochs)} checkpoints: {epochs}")
    print(f"Checkpoint path: {ckpt_path}")
    print(f"Output directory: {output_dir}")
 
    # Process each checkpoint
    for ckpt_epoch in epochs:
        print(f"\n=== Checkpoint epoch {ckpt_epoch} ===")
        loaded_epoch = load_checkpoint(
            ckpt_path, model, optimizer=None, scheduler=None,
            epoch=ckpt_epoch, device=device,
        )
        print(f"  Loaded epoch {loaded_epoch}")
        model.eval()
 
        ckpt_dir = os.path.join(output_dir, f"epoch_{ckpt_epoch:04d}")
        os.makedirs(ckpt_dir, exist_ok=True)
 
        with torch.no_grad():
            for sample_idx, (inputs, outputs) in enumerate(dataloader_val):
                inputs = inputs.type(dtype).to(device)
                outputs = outputs.type(dtype).to(device)
 
                # Forward pass
                pred = (
                    model((inputs / input_norm).permute(0, 4, 1, 2, 3)).permute(
                        0, 2, 3, 4, 1
                    )
                    * output_norm
                )
 
                # Extract single sample (batch_size=1)
                pred_s = pred[0].cpu()      # (nt, nx, ny, 7)
                true_s = outputs[0].cpu()   # (nt, nx, ny, 7)
                inp_s = inputs[0].cpu()     # (nt, nx, ny, 11)
 
                # Time axis: use the real (un-shifted) time values
                t = t_real
 
                # Eta value (channel 10 of inputs, constant over domain)
                eta_val = inp_s[0, 0, 0, 10].item()
 
                # Domain-averaged B^2
                b2_true = compute_domain_avg_b_squared(true_s)
                b2_pred = compute_domain_avg_b_squared(pred_s)
 
                # Indices of supervised (strided) timesteps
                stride_idx = np.arange(0, len(t), data_loss_stride)
 
                # Plot
                fig, ax = plt.subplots()
                ax.plot(t, b2_true, label="BHAC", color="k")
                ax.plot(t, b2_pred, label="FNO", color="C0", linestyle="--")
                # Mark supervised timesteps only when the data loss actually
                # strides time (stride==1 means every timestep is supervised,
                # so the dots would just trace the FNO curve).
                if data_loss_stride > 1:
                    ax.scatter(t[stride_idx], b2_pred[stride_idx],
                               color="red", s=18, zorder=5,
                               label=rf"Data supervision")
                ax.set_xlabel(r"$t$")
                ax.set_ylabel(r"$\langle B^2 \rangle$")
                ax.set_title(
                    rf"$\eta = {eta_val:.2e}$, Epoch {ckpt_epoch}, "
                    f"Sample {sample_idx}, Stride {data_loss_stride}"
                )

                ax.set_yscale("log")
                ax.set_xlim(left=t_real[0])
                ax.set_ylim(1, 11)
                # fix to 0, 0 at the bottom
                ax.set_xlim(left=0)
                ax.set_ylim(bottom=1)

                ax.legend(loc="upper right")
                plt.tight_layout()
 
                save_path = os.path.join(
                    ckpt_dir, f"b2_avg_sample_{sample_idx:03d}.png"
                )
                plt.savefig(save_path)
                plt.close(fig)
 
                # ---- Save data to .txt file ----
                txt_path = os.path.join(
                    ckpt_dir, f"b2_avg_sample_{sample_idx:03d}.txt"
                )
                is_strided = np.zeros(len(t), dtype=int)
                is_strided[stride_idx] = 1
                with open(txt_path, "w") as f:
                    f.write(f"# eta = {eta_val:.6e}\n")
                    f.write(f"# epoch = {ckpt_epoch}\n")
                    f.write(f"# sample = {sample_idx}\n")
                    f.write(f"# data_loss_stride = {data_loss_stride}\n")
                    f.write(f"# columns: t  b2_true  b2_pred  is_stride_point\n")
                    for i in range(len(t)):
                        f.write(f"{t[i]:.8e}\t{b2_true[i]:.8e}\t{b2_pred[i]:.8e}\t{is_strided[i]}\n")
 
                print(f"  Saved {save_path}")
                print(f"  Saved {txt_path}")
 
    print(f"\nDone. All plots saved under {output_dir}")
 
 
if __name__ == "__main__":
    main()
