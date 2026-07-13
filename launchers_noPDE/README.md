# noPDE control runs

**Experiment: what if the physics losses had never been switched on?**

One control per canonical run: identical setup (same config, seed, data
treatment, optimizer/LR schedule), but the PDE-residual and div-B losses stay
off for the entire run — pure data loss, i.e. the parent's data-only warm-up
continued to the end instead of entering the physics ramp. Comparing a control
to its parent at the same epoch isolates what the physics losses bought (or
cost).

Run length: 600 epochs (360 for the `*_zusatz` pair — same 0.6 rescale as
their parents, keeping the controls samples-matched).

Each launcher is driven by a `config/<parent>_noPDE.yaml` overlay that
inherits the parent config via Hydra `defaults` and overrides only the
control-defining knobs. Same wandb project as the parent; group and ckpt dir
carry the `_noPDE` suffix. No plot jobs are submitted.

Submit from the repo root:

```bash
sbatch launchers_noPDE/<name>_noPDE.sh
```
