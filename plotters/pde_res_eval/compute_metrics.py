# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PDE/divB residual losses: validation DATA vs FNO prediction, per run.

For each of the 2x2 runs this computes the individual mean-square residual
losses (FI1, FI2, MO, ES0, ES1, ES2, C1, divB) over the FULL validation set
for three "models":

  data          the BHAC ground truth itself, passed through the identical
                residual operators used in training (the reference scale),
  fno_warmup    the FNO at the end of the data-only warm-up (physics
                weights still zero; epoch 100, or 60 for the zusatz runs),
  fno_final     the FNO after the physics ramp (latest checkpoint by
                default).

All numbers are in exactly the metric logged to wandb during training
(valid/loss_FI1 ... loss_C1, loss_div_B): same loss code, same snapshot
config, mean squared residual, averaged over the val set. The truth column
is therefore directly comparable to the logged FNO curves.

NOTE the aggregate `pde` row is the training objective's weighted sum
(FI 10 / MO 1000 / ES 1e-4 / C1 1) — a training artifact dominated by MO,
included only for cross-checking against the wandb `loss_pde` curve. Use
the individual rows for any physics statement.

Run on the HPC (venv active), e.g.:
  python plotters/pde_res_eval/compute_metrics.py --run all
  python plotters/pde_res_eval/compute_metrics.py --run coarse8_512p --final-epoch 1680

Outputs (per run) under figs/pde_res_eval_out/<run_key>/:
  metrics_<config>_val.csv   machine-readable table
  metrics_<config>_val.md    same table in markdown, with ratio columns
plus a combined figs/pde_res_eval_out/metrics_summary.md over all runs evaluated
in the invocation.
"""

import argparse
import os

import torch

from common import (  # noqa: E402 (sys.path set up in common)
    QUANTITIES,
    REPO_ROOT,
    build_model,
    build_val_dataloader,
    load_model_epoch,
    load_run_config,
    mean_square,
    predict,
    residual_fields,
)
from losses import LossMHD_PhysicsNeMo
from runs import RUNS, resolve_run_keys

ROWS = QUANTITIES + ["pde"]  # 'pde' = weighted aggregate, see module docstring


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default="all",
                        help=f"all or one of: {', '.join(RUNS)}")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup-epoch", type=int, default=None,
                        help="Override the run's end-of-warm-up epoch")
    parser.add_argument("--final-epoch", type=int, default=None,
                        help="Checkpoint epoch for the +physics column "
                             "(default: latest checkpoint)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Evaluate only the first N val samples (default: all)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Dataloader workers")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "figs" / "pde_res_eval_out"))
    return parser.parse_args()


@torch.no_grad()
def weighted_pde_aggregate(loss_fn, ms):
    """Recombine individual mean-square losses into the training loss_pde."""
    return (
        loss_fn.FI1_weight * ms["FI1"]
        + loss_fn.FI2_weight * ms["FI2"]
        + loss_fn.MO_weight * ms["MO"]
        + loss_fn.ES0_weight * ms["ES0"]
        + loss_fn.ES1_weight * ms["ES1"]
        + loss_fn.ES2_weight * ms["ES2"]
        + loss_fn.C1_weight * ms["C1"]
    )


@torch.no_grad()
def accumulate_losses(dataloader, loss_fn, device, num_samples,
                      model=None, input_norm=None, output_norm=None):
    """Mean of the individual mean-square residual losses over the val set.

    With model=None the residuals are evaluated on the ground truth itself;
    otherwise on the model's prediction. eta always comes from the inputs.
    """
    running = {q: 0.0 for q in QUANTITIES}
    n = 0
    for batch_idx, (inputs, outputs) in enumerate(dataloader):
        if num_samples is not None and batch_idx >= num_samples:
            break
        inputs = inputs.to(device, dtype=torch.float32)
        outputs = outputs.to(device, dtype=torch.float32)
        if model is None:
            fields = outputs
        else:
            fields = predict(model, inputs, input_norm, output_norm)
        res = residual_fields(loss_fn, fields, inputs[..., -1])
        for q in QUANTITIES:
            running[q] += mean_square(res[q])
        n += 1
    ms = {q: running[q] / n for q in QUANTITIES}
    ms["pde"] = weighted_pde_aggregate(loss_fn, ms)
    return ms, n


def write_tables(out_dir, spec, cfg, source, columns, n_samples, device):
    """Write the per-run CSV + markdown tables; return the markdown text."""
    config_name = spec.config_name
    dp = cfg.dataset_params
    header_lines = [
        f"run = {spec.key} ({spec.label})",
        f"config = {config_name}  [{source}]",
        f"split = val, n_samples = {n_samples}, device = {device}",
        f"window: ind_t_start = {dp.ind_t_start}, ind_t = {dp.ind_t}; "
        f"tend = {cfg.loss_params.tend}, diff_type = {cfg.loss_params.diff_type}",
        f"data loss: stride = {cfg.loss_params.data_loss_stride}, "
        f"coarse_factor = {cfg.loss_params.data_loss_coarse_factor}",
        "metric = mean squared residual, identical to the wandb "
        "valid/loss_* curves (same loss code + snapshot config)",
        "'pde' row = training-weighted aggregate (FI 10 / MO 1000 / ES 1e-4 / "
        "C1 1) - cross-check against wandb loss_pde only, not a physics metric",
    ]
    col_names = list(columns.keys())  # e.g. ["data", "fno_ep0060", "fno_ep1680"]

    csv_path = os.path.join(out_dir, f"metrics_{config_name}_val.csv")
    with open(csv_path, "w") as f:
        for h in header_lines:
            f.write(f"# {h}\n")
        f.write("quantity," + ",".join(col_names) + "\n")
        for q in ROWS:
            f.write(q + "," + ",".join(f"{columns[c][q]:.6e}" for c in col_names) + "\n")

    fno_cols = [c for c in col_names if c != "data"]
    md_path = os.path.join(out_dir, f"metrics_{config_name}_val.md")
    lines = [f"### {spec.label} — `{config_name}`", ""]
    lines += [f"> {h}" for h in header_lines[2:]]
    lines.append("")
    lines.append("| quantity | " + " | ".join(col_names)
                 + " | " + " | ".join(f"{c}/data" for c in fno_cols) + " |")
    lines.append("|---" * (1 + len(col_names) + len(fno_cols)) + "|")
    for q in ROWS:
        vals = [f"{columns[c][q]:.3e}" for c in col_names]
        ratios = [
            f"{columns[c][q] / columns['data'][q]:.3f}"
            if columns["data"][q] != 0 else "n/a"
            for c in fno_cols
        ]
        lines.append(f"| {q} | " + " | ".join(vals) + " | " + " | ".join(ratios) + " |")
    lines.append("")
    md_text = "\n".join(lines)
    with open(md_path, "w") as f:
        f.write(md_text + "\n")
    print(f"[out] {csv_path}\n[out] {md_path}")
    return md_text


def main():
    args = parse_args()
    keys = resolve_run_keys(args.run)
    summary_parts = []

    for key in keys:
        spec = RUNS[key]
        print(f"\n===== {spec.key}: {spec.config_name} =====")
        cfg, ckpt_path, source = load_run_config(spec.config_name)
        print(f"[config] {source}")
        loss_fn = LossMHD_PhysicsNeMo(**cfg.loss_params)
        dataloader, _ = build_val_dataloader(cfg, num_workers=args.num_workers)

        # Reference: residuals of the validation data itself.
        print("[data] evaluating residuals of the ground truth ...")
        columns = {}
        columns["data"], n_samples = accumulate_losses(
            dataloader, loss_fn, args.device, args.num_samples
        )
        print(f"[data] done ({n_samples} val samples)")

        # FNO at end of warm-up (data-only) and after the physics ramp.
        model, input_norm, output_norm = build_model(cfg, args.device)
        warmup_req = args.warmup_epoch if args.warmup_epoch is not None else spec.warmup_epoch
        for tag, epoch_req in [("data-only", warmup_req), ("+physics", args.final_epoch)]:
            loaded = load_model_epoch(model, ckpt_path, epoch_req, args.device)
            if epoch_req is not None and loaded != epoch_req:
                print(f"[WARN] requested epoch {epoch_req} but checkpoint "
                      f"loader returned epoch {loaded} — column is labeled "
                      f"with the LOADED epoch")
            col = f"fno_ep{loaded:04d}"
            print(f"[fno] evaluating {tag} prediction at epoch {loaded} ...")
            columns[col], _ = accumulate_losses(
                dataloader, loss_fn, args.device, args.num_samples,
                model=model, input_norm=input_norm, output_norm=output_norm,
            )

        out_dir = os.path.join(args.out_dir, spec.key)
        os.makedirs(out_dir, exist_ok=True)
        summary_parts.append(
            write_tables(out_dir, spec, cfg, source, columns, n_samples, args.device)
        )

    if len(keys) > 1:
        summary_path = os.path.join(args.out_dir, "metrics_summary.md")
        with open(summary_path, "w") as f:
            f.write("# PDE/divB residual losses: validation data vs FNO "
                    "prediction (all runs)\n\n")
            f.write("\n\n".join(summary_parts) + "\n")
        print(f"\n[out] combined summary: {summary_path}")


if __name__ == "__main__":
    main()
