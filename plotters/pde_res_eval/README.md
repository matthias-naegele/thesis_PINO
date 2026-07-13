# pde_res_eval — PDE/divB residuals: validation data vs FNO prediction

**Goal:** quantify that the physics-trained FNO's prediction scores **lower
PDE-residual and ∇·B losses than the validation data itself**, in exactly the
metric logged during training, and show with residual maps that the surviving
residual (of both data and prediction) concentrates on the shock fronts.

Caveat to carry into any figure caption: *lower residual ≠ "more physical than
BHAC"*. The data is the physical reference; this measures self-consistency
under the discretized residual operator (spectral derivatives, the training
time stencil), not correctness.

Everything reuses the training code paths (`losses/LossMHD_PhysicsNeMo`, the
BHAC val dataloader, the FNO construction of `plot_index.py`) and the run's
**snapshot config** from its checkpoint dir, so the numbers are directly
comparable to the wandb `valid/loss_FI1 … loss_C1, loss_div_B` curves: same
spectral derivatives, same time stencil, same `dt = tend/(nt−1)`, mean squared
residual, averaged over the full validation set.

## The 2×2 runs

Defined once in `runs.py` (edit there if the set changes):

| key | config | warm-up ends | default fig frame |
|---|---|---|---|
| `coarse8_256p` | `coarse8_t0_t2_5_256p` | ep 100 | 96 (canonical plot frame) |
| `stride8_256p` | `stride8_t0_t2_5_256p` | ep 100 | 92 (mid-gap, unsupervised) |
| `coarse8_512p` | `coarse8_t9_25_t10_512p_zusatz` | ep 60 | 15 (t=9.625) |
| `stride8_512p` | `stride8_t9_25_t10_512p_zusatz` | ep 60 | 12 (mid-gap, unsupervised) |

For the 512p rows the `_zusatz` (enlarged-dataset) variants are used. The
"+physics" epoch defaults to the **latest checkpoint** in the run's ckpt dir
(override with `--final-epoch`). For stride8 runs the default figure frame is
the **middle of a data-unsupervised range** (halfway between two supervised
anchor frames). Figures use validation **sample 001**.

## How to run (on the HPC, from the repo root)

```bash
sbatch plotters/pde_res_eval/run_all.sh                    # everything, all four runs
RUNS=coarse8_512p sbatch --export=ALL plotters/pde_res_eval/run_all.sh   # one run
```

or the two scripts individually (venv active; GPU used if visible):

```bash
python plotters/pde_res_eval/compute_metrics.py    --run all
python plotters/pde_res_eval/plot_residual_maps.py --run all
python plotters/pde_res_eval/plot_residual_maps.py --run stride8_512p \
    --quantities FI1 divB --time-index 20 --vmax-rms 2   # variants
```

Requirements: the run's checkpoint dir must contain `config.yaml` (the
launcher's snapshot) and checkpoints at the warm-up epoch (100 / 60) and the
final epoch; the BHAC h5 data must be readable at the snapshot's `data_dir`.

## Outputs (`figs/pde_res_eval_out/`, git-ignored)

```
figs/pde_res_eval_out/
  metrics_summary.md                    # all runs' tables in one file
  <run_key>/
    metrics_<config>_val.csv            # quantity, data, fno_ep<warm>, fno_ep<final>
    metrics_<config>_val.md             # same + ratio columns (fno/data)
    figs/
      resmap_<quantity>_<config>_sample001_t<frame>_ep<warm>_vs_ep<final>.png
      resmap_....h5                     # raw signed-residual arrays for re-rendering
```

- **Table rows** are the *individual* mean-square residuals (FI1, FI2, MO,
  ES0–2, C1) plus divB. The `pde` row is the training-weighted aggregate
  (FI 10 / MO 1000 / ES 1e-4 / C1 1) — it is dominated by 1000×MO and is only
  there to cross-check against the wandb `loss_pde` curve; never quote it as a
  physics number.
- The `.h5` next to each PNG stores the raw signed-residual arrays plus the
  grid and metadata, so a figure can be re-rendered at any color scale/colormap
  without touching checkpoints or data (any machine) — write a small matplotlib
  reader if you need a different style than `plot_residual_maps.py`'s default.
- **Maps** are three panels of the **signed** residual on one shared symmetric
  linear scale (diverging `RdBu_r`, ±3×RMS of the data panel):
  BHAC data | FNO end-of-warm-up (data-only) |
  FNO final (+physics), each annotated with its frame RMS. There is
  deliberately no "error" panel — a difference of two residual maps has no
  interpretation.

## Recommendation for the thesis (keep it minimal)

The suite produces 4 tables and 12 maps; for the thesis use **one table and
2–3 maps**:

1. **Table:** the 512p pair (`coarse8_512p` + `stride8_512p`) — it is the
   compute- and architecture-matched pair. Quote FI1, FI2, MO, C1 and divB
   with the ratio column; 256p tables go to the appendix if at all.
2. **Fig A (PDE residual):** `resmap_FI1_...` of `coarse8_512p` at frame 15 —
   both data and prediction light up on the same shock fronts, prediction
   ~10–20× lower (ties into limitation (iv) / weak-form outlook).
3. **Fig B (divB):** `resmap_divB_...` of `coarse8_512p` — the cleaner, more
   strongly wordable half of the claim.
4. *(optional)* `resmap_FI1_...` of `stride8_512p` at the mid-gap frame 12 —
   shows the physics losses doing the work exactly where no data supervision
   exists.

**Framing caveat (word it carefully):** lower residual than the
data does *not* mean "more physical than BHAC" — the data's strong-form
residual is dominated by spectral (Gibbs) ringing at shocks. The legitimate
claims: the physics objective is genuinely achieved (below the data's own
level), the data residual gives the loss curves an absolute reference scale,
and the constraint removes broadband ∇·B violations the data never had.
Quote residual metrics only alongside the data-error metrics.
