"""Locate the ``plot_index.py`` field dumps a collage plotter reads.

The collage plotters (``plot_coarse8_*`` / ``plot_stride8_*``) draw the *same* raw
truth / prediction / error arrays that ``plot_index.py`` writes during the
post-training plot jobs: one consolidated ``fields.h5`` per (epoch, validation
sample, field), holding every timestep plus the data-loss masks and ``eta``.
Those dumps live under each run's checkpoint dir, in the shared ``index_data``
directory the global/local colour-scale jobs both point at::

    <ckpt_path>/plots/epoch_<E>/index_data/sample_<SSS>/<field>/fields.h5

(see ``plot_at_epoch.sh`` -> ``plot_index.py --data_dir``; the epoch is
unpadded, exactly as the launcher passes it).

Historically these h5 were hand-copied onto the laptop under ``figs/data/`` and
the plotters read them there, which chained every thesis figure to a manual scp
step and to whichever local copy happened to be around -- bad for
reproducibility. Instead the plotters now read the dumps straight from their
home under the checkpoint dir, so they run on the HPC (where the checkpoints and
their plot dumps live) with no local pre-copied data.

``<ckpt_path>`` is pulled from the run config (``config/<name>.yaml`` merged
with ``config/paths.yaml``) -- the same single source of truth the launchers
use, so the dumps follow whatever ``paths.yaml`` roots point at.
"""

import os
from pathlib import Path

from omegaconf import OmegaConf

# This module lives in plotters/, one level below the repo root; the run configs
# it reads live in config/ at the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def run_ckpt_path(config_name):
    """Resolved ``train_params.ckpt_path`` for a run config.

    ``config/paths.yaml`` (the machine's ``data_root`` / ``output_root``) is
    composed in first so the ``${output_root}/...`` interpolation in the run
    config resolves -- mirroring how the SLURM launchers derive ``CKPT_PATH``.
    """
    cfg = OmegaConf.load(_REPO_ROOT / "config" / f"{config_name}.yaml")
    paths_yaml = _REPO_ROOT / "config" / "paths.yaml"
    if paths_yaml.is_file():
        cfg = OmegaConf.merge(OmegaConf.load(paths_yaml), cfg)
    return str(cfg.train_params.ckpt_path)


def fields_h5_path(config_name, epoch, field, sample=1):
    """Absolute path to the ``fields.h5`` for one (epoch, sample, field).

    Mirrors the layout ``plot_at_epoch.sh`` + ``plot_index.py`` write. ``field``
    is a model output (``u1``..``rho``) or the derived ``Jz``; ``sample`` is the
    validation-sample index (1 = second sim, the one the figures use).
    """
    return os.path.join(
        run_ckpt_path(config_name),
        "plots", f"epoch_{epoch}", "index_data",
        f"sample_{sample:03d}", field, "fields.h5",
    )
