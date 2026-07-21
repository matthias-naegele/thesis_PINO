# plotters/

Standalone figure scripts, runnable from anywhere (paths are anchored to the
repo root); outputs go to the git-ignored `figs/`. The collage plotters read
the `plot_index.py` field dumps straight from each run's checkpoint dir on the
HPC (`plotters/collage_h5.py`); `plot_noPDE_wandb_comparison.py` pulls curves
from the wandb API — see each script's docstring.

## plot_noPDE_wandb_comparison.py

**Intent: quantify what the physics losses bought.** For each canonical run
vs. its `*_noPDE` control (same setup, physics never switched on), it fetches
the train/valid loss curves from the wandb API and writes two figures per
pair into `figs/noPDE_comparison/`, named
`<parent>__<metric>__PDE_vs_noPDE.png`: the family's per-timestep full-res
metric (`loss_data_highres` coarse8 / `loss_data_skipped` stride8) and the
cross-config-comparable `loss_data_full`. Blue = PDE run, red = noPDE
control; dashed = train, solid = valid. (coarse8 runs have
`data_loss_stride: 1`, so their `loss_data_highres` covers every timestep and
is the same quantity as `loss_data_full` — the two coarse8 figures coincide.)

Needs internet + `wandb login` (e.g. HPC login node):

```bash
python plotters/plot_noPDE_wandb_comparison.py
```

The x-axis runs to the end of each PDE run (the noPDE control is shorter and
just stops where it ends).

It also writes into `figs/zusatz_comparison/` two figures per family (`coarse8`
/ `stride8`) — the family's per-timestep full-res metric and `loss_data_full` —
overlaying each `*_zusatz` run (38-sim dataset) on its non-zusatz partner
(23-sim). Their x-axis is "samples seen" (`epoch x num_train`, i.e. batches,
since `batch_size=1`) so the two are comparable despite the zusatz run seeing
more data per epoch — on that axis the gray dashed physics-loss activation
lines of the two runs nearly coincide.

Note: it compares *data* losses only — physics metrics (PDE residual, div-B)
are never logged by the noPDE runs and would need checkpoint-based evaluation
(`pde_res_eval/`, `plot_B2_onValidation.py`).


## plot_coarse8_* and plot_stride8_* (collages)

Qualitative collage figures (BHAC truth | FNO prediction | error) comparing an
early data-only checkpoint against the final physics-ramped model — i.e. what
the PDE + div-B losses bought, shown per field and timestep:

- `plot_coarse8_t9_25_t10_512p.py`, `plot_stride8_t9_25_t10_512p.py`
- `plot_coarse8_t0_t2_5_256p.py`, `plot_stride8_t0_t2_5_256p.py`

Each plotter's `RUN` constant is the config name. They read the consolidated
`fields.h5` dumps `plot_index.py` wrote during the post-training plot jobs,
straight from where they live under the run's checkpoint dir —
`<ckpt_path>/plots/epoch_<E>/index_data/sample_001/<field>/fields.h5`
(`<ckpt_path>` is resolved from the run config + `paths.yaml`; see
`plotters/collage_h5.py`). So they **run on the HPC** (where those dumps live)
and need no local pre-copied data — previously the dumps were hand-copied into
`figs/data/` and read from there, which is gone. Only the epochs `plot_index.py`
actually dumped (the launchers' `submit_index_plots` epochs) are available.

Outputs go to `figs/collages/<name>_figs/<field>/`. Pure h5 read + matplotlib
(no GPU), so run them on the `small_cpu` partition (or a login node):

```bash
sbatch plotters/run_collages.sh                    # all four
python plotters/plot_coarse8_t9_25_t10_512p.py     # just one
```

## pde_res_eval/ (subdirectory)

PDE/div-B residual evaluation of the trained checkpoints: validation data vs
FNO prediction, in exactly the metric logged during training. Unlike the
scripts above, these reload the run configs and checkpoints (via the training
code paths), so they run on the HPC. `run_all.sh` drives the whole 2×2-run
suite; outputs (metrics tables + signed-residual maps) go to the git-ignored
`figs/pde_res_eval_out/`. See `pde_res_eval/README.md` for details.

