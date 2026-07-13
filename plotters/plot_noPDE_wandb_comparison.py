"""
Compare each canonical (PDE-ramped) run against its *_noPDE control on the
wandb loss curves: "what did switching the physics losses on buy (or cost)?"

For each of the 6 run pairs, two figures are produced, each showing train and
valid curves of BOTH variants (4 curves; BLUE = PDE parent, RED = noPDE
control; solid = valid, dashed = train):

  fig 1 — the family's "data the loss did NOT see" metric:
            coarse8 runs -> loss_data_highres  (full-res on penalized steps)
            stride8 runs -> loss_data_skipped  (full-res on withheld steps)
  fig 2 — loss_data_full (full resolution, ALL timesteps; the one data metric
          that is directly comparable across coarse/stride configs)

Data comes from the wandb API (the runs trained with wandb_mode=online), so
this script needs internet + a wandb login ("wandb login" once, or WANDB_API_KEY
set). Fire it off anywhere that has that — e.g. the HPC login node:

    python plotters/plot_noPDE_wandb_comparison.py

Entity / project / group are read from the run configs in config/, so this
stays consistent with the launchers by construction. Interrupted+resumed
trainings appear as several wandb runs in one group; the curves are stitched
by epoch (later run wins on overlapping epochs).

The x-axis runs to the end of the PDE parent's history (its full run); the
noPDE control is shorter and its curves simply stop where they end.

On top of the per-pair figures, the *_zusatz runs are overlaid on their
non-zusatz partners (the enlarged 38-sim dataset vs. the 23-sim original), two
figures per family (coarse8, stride8): the family's withheld-data metric
(loss_data_highres coarse8 / loss_data_skipped stride8) and loss_data_full.
Here the x-axis is "samples seen" (epoch x num_train, i.e. batches, since
batch_size=1) rather than epoch, so the 38-sim and 23-sim runs are directly
comparable despite the 38-sim run seeing more data per epoch; the gray dashed
line still marks where the physics losses switch on (one per run — by design
they land at nearly the same samples-seen).

Output (self-describing filenames, git-ignored):
    figs/noPDE_comparison/<parent>__<metric>__PDE_vs_noPDE.png
    figs/zusatz_comparison/<family>__<metric>__zusatz_vs_base__samples_seen.png
"""

import argparse
import os
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from hydra import compose, initialize_config_dir
from matplotlib.ticker import FuncFormatter, LogLocator

# This script lives in plotters/, one level below the repo root. Anchor the
# config/ and figs/ paths to the repo root so it runs the same from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT_DIR = os.path.join(_REPO_ROOT, "figs", "noPDE_comparison")
_ZUSATZ_OUT_DIR = os.path.join(_REPO_ROOT, "figs", "zusatz_comparison")
_RAW_DIR = os.path.join(_OUT_DIR, "raw_data")
_ZUSATZ_RAW_DIR = os.path.join(_ZUSATZ_OUT_DIR, "raw_data")

# X-axis padding (in epochs) left/right of the data range, so curves don't
# touch the plot border.
_EPOCH_PAD = 50

# Legend text for the PDE-vs-noPDE comparison figures, keyed by (variant, ns).
LEGEND_LABEL = {
    ("PDE", "train"): "Loss with PDE; train",
    ("PDE", "valid"): "Loss with PDE; valid",
    ("noPDE", "train"): "Loss without PDE; train",
    ("noPDE", "valid"): "Loss without PDE; valid",
}

sys.path.insert(0, _REPO_ROOT)
from losses import weight_schedule as weight_schedule_lib

# The 6 canonical runs and the family metric for fig 1 (fig 2 is always
# loss_data_full). The noPDE partner is <parent>_noPDE by convention.
PAIRS = [
    ("coarse8_t0_t2_5_256p", "loss_data_highres"),
    ("stride8_t0_t2_5_256p", "loss_data_skipped"),
    ("stride8_t9_25_t10_512p", "loss_data_skipped"),
    ("stride8_t9_25_t10_512p_zusatz", "loss_data_skipped"),
    ("coarse8_t9_25_t10_512p", "loss_data_highres"),
    ("coarse8_t9_25_t10_512p_zusatz", "loss_data_highres"),
]

# The zusatz overlays: each family's non-zusatz base (23-sim) vs its enlarged
# *_zusatz counterpart (38-sim), plotted against "samples seen" so the two are
# comparable. (base, zusatz, family_metric, family_label).
ZUSATZ_PAIRS = [
    (
        "coarse8_t9_25_t10_512p",
        "coarse8_t9_25_t10_512p_zusatz",
        "loss_data_highres",
        "coarse8",
    ),
    (
        "stride8_t9_25_t10_512p",
        "stride8_t9_25_t10_512p_zusatz",
        "loss_data_skipped",
        "stride8",
    ),
]

Y_LABELS = {
    "loss_data_full": "Loss on full res. data",
    "loss_data_skipped": "Loss at locations without Data-Supervision",
    "loss_data_highres": "Loss at locations without Data-Supervision",
}

NAMESPACES = ("train", "valid")
# BLUE = first variant, RED = second (PDE parent vs noPDE control; or, for the
# zusatz overlays, base vs zusatz).
VARIANT_COLOR = {"PDE": "tab:blue", "noPDE": "tab:red"}



def line_style(color, namespace):
    """Per-namespace line style: train dashed (faded), valid solid."""
    if namespace == "train":
        return dict(color=color, linestyle="--", alpha=0.7)
    return dict(color=color, linestyle="-")


def wandb_target(config_name):
    """(entity, project, group) for one run, from its config in config/.

    Composed via Hydra (not a plain YAML load) because the *_noPDE configs are
    overlays that inherit entity/project from their parent via `defaults`.
    """
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(_REPO_ROOT, "config")
    ):
        wp = compose(config_name=config_name).wandb_params
    return str(wp.wandb_entity), str(wp.wandb_project), str(wp.wandb_group)


def pde_activation_epoch(config_name):
    """First epoch of the PDE parent's `weight_schedule` where `pde_weight`
    becomes nonzero (the end of the data-only warm-up). Replays the same
    declarative schedule the training loop uses (`losses/weight_schedule.py`),
    so this stays correct if a config's warm-up length ever changes. Returns
    ``None`` if the config has no `weight_schedule`.
    """
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(_REPO_ROOT, "config")
    ):
        lp = compose(config_name=config_name).loss_params
    schedule = weight_schedule_lib.parse_weight_schedule(lp.get("weight_schedule"))
    if not schedule:
        return None
    weights = {
        "data": float(lp.get("data_weight", 1.0)),
        "pde": float(lp.get("pde_weight", 0.0)),
        "constraint": float(lp.get("constraint_weight", 0.0)),
    }
    weight_schedule_lib.apply_phase_set(weights, schedule[0])
    if weights["pde"] > 0:
        return 0
    data_floor = float(lp.get("data_weight_floor", 0.0))
    for epoch in range(1, weight_schedule_lib.total_epochs(schedule) + 1):
        weight_schedule_lib.step(schedule, weights, epoch, data_floor)
        if weights["pde"] > 0:
            return epoch
    return None


def num_train(config_name):
    """Number of training sims for a run (its `dataset_params.num_train`).

    With batch_size=1 and one sim per dataset item (see dataloaders/), this is
    exactly the number of batches — "samples seen" — per epoch, so it converts
    the epoch axis into the samples-seen axis that makes the 23-sim and 38-sim
    (zusatz) runs comparable.
    """
    with initialize_config_dir(
        version_base=None, config_dir=os.path.join(_REPO_ROOT, "config")
    ):
        return int(compose(config_name=config_name).dataset_params.num_train)


def fetch_group_history(api, entity, project, group):
    """All logged rows of every run in the group, as one list of dicts.

    Runs are ordered oldest-first so that, when stitching by epoch later,
    values from a resumed (newer) run overwrite the tail of the interrupted
    one.
    """
    runs = sorted(
        api.runs(f"{entity}/{project}", filters={"group": group}),
        key=lambda r: r.created_at,
    )
    if not runs:
        raise SystemExit(
            f"ERROR: no wandb runs found for group '{group}' in "
            f"{entity}/{project} (did the run start / is the group name right?)"
        )
    rows = []
    for run in runs:
        # samples is an upper bound; runs are <= a few thousand epochs, so
        # this returns the full history without wandb's down-sampling.
        rows.extend(run.history(samples=100000, pandas=False))
    return rows


def series(rows, namespace, metric):
    """Stitched (epochs, values) for e.g. namespace='train', metric='loss_data_full'.

    LaunchLogger writes metrics namespaced per phase; match any key of the
    form '<namespace>/.../<metric>' (exact tail match, so loss_data does not
    also match loss_data_full). The epoch is taken from the same row.
    """

    def match(key):
        parts = key.split("/")
        return len(parts) >= 2 and parts[0] == namespace and parts[-1] == metric

    by_epoch = {}
    for row in rows:
        metric_keys = [k for k in row if match(k) and row[k] is not None]
        if not metric_keys:
            continue
        epoch = next(
            (row[k] for k in (f"{namespace}/epoch", "epoch") if row.get(k) is not None),
            None,
        )
        if epoch is None:
            continue
        by_epoch[float(epoch)] = float(row[metric_keys[0]])
    epochs = sorted(by_epoch)
    return epochs, [by_epoch[e] for e in epochs]


def _log_tick_label(y, _pos):
    """Tick label for the log y-axis: 10^i and 5x10^i, nothing else."""
    import math

    if y <= 0:
        return ""
    exponent = math.floor(math.log10(y))
    mantissa = y / 10**exponent
    if abs(mantissa - 1.0) < 1e-6:
        return f"$10^{{{exponent}}}$"
    if abs(mantissa - 5.0) < 1e-6:
        return f"$5{{\\times}}10^{{{exponent}}}$"
    return ""


def style_log_yaxis(ax, metric):
    """Shared log y-axis dressing: labeled ticks at 10^i AND 5x10^i (an explicit
    formatter — matplotlib's log formatters drop the non-decade labels once the
    axis spans a few decades), plus the y-label. Ticks point inward on all four
    sides and there is no grid underlay."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(_log_tick_label))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10, subs=(2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0))
    )
    ax.set_ylabel(Y_LABELS.get(metric, metric))
    ax.tick_params(
        axis="both", which="both", direction="in",
        top=True, bottom=True, left=True, right=True,
    )


def write_raw_data(path, xlabel, curves):
    """Dump the plotted curves to a plain-text file next to the figure, so the
    figure is reproducible from the raw numbers alone. ``curves`` is a list of
    (label, xs, ys) triples; each is written as its own labeled block since the
    curves don't share a common x-grid."""
    lines = []
    for label, xs, ys in curves:
        lines.append(f"# {label}")
        lines.append(f"# {xlabel}\tvalue")
        lines.extend(f"{x}\t{y}" for x, y in zip(xs, ys))
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def plot_pair(parent, metric, histories, activation_epoch):
    """One figure: train+valid of PDE parent (blue) vs noPDE control (red).

    The x-axis runs to the end of the PDE parent's history (its full run) —
    the noPDE control is shorter and simply stops where its curve ends.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    pde_max_epoch = 0
    raw_curves = []
    for variant in ("PDE", "noPDE"):
        for ns in NAMESPACES:
            epochs, values = series(histories[variant], ns, metric)
            if not epochs:
                print(f"    WARNING: no '{ns}/{metric}' data for {variant}", flush=True)
                continue
            if variant == "PDE":
                pde_max_epoch = max(pde_max_epoch, max(epochs))
            label = LEGEND_LABEL[(variant, ns)]
            ax.plot(
                epochs, values, label=label,
                **line_style(VARIANT_COLOR[variant], ns),
            )
            raw_curves.append((label, epochs, values))

    if activation_epoch is not None:
        ax.axvline(
            activation_epoch,
            color="gray",
            linestyle="--",
            label="Activation of\nPhysics-Loss",
        )

    if pde_max_epoch:
        ax.set_xlim(-_EPOCH_PAD, pde_max_epoch + _EPOCH_PAD)
    style_log_yaxis(ax, metric)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.set_title(r"relative $L_2$ loss on BHAC data")
    fig.tight_layout()
    out = os.path.join(_OUT_DIR, f"{parent}__{metric}__PDE_vs_noPDE.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, _REPO_ROOT)}", flush=True)
    raw_out = os.path.join(_RAW_DIR, f"{parent}__{metric}__PDE_vs_noPDE.txt")
    write_raw_data(raw_out, "epoch", raw_curves)
    print(f"  wrote {os.path.relpath(raw_out, _REPO_ROOT)}", flush=True)


def plot_zusatz(base, zusatz, metric, family, histories, samples_per_epoch, activation_epochs):
    """One figure: train+valid of the non-zusatz base (blue) vs its *_zusatz
    counterpart (red), on a "samples seen" (epoch x num_train) x-axis so the
    38-sim and 23-sim runs are comparable."""
    fig, ax = plt.subplots(figsize=(8, 5))
    variants = [
        (base, "base (23 sims)", "tab:blue"),
        (zusatz, "zusatz (38 sims)", "tab:red"),
    ]
    raw_curves = []
    for config_name, label, color in variants:
        spe = samples_per_epoch[config_name]
        for ns in NAMESPACES:
            epochs, values = series(histories[config_name], ns, metric)
            if not epochs:
                print(f"    WARNING: no '{ns}/{metric}' data for {label}", flush=True)
                continue
            samples_seen = [e * spe for e in epochs]
            ax.plot(
                samples_seen, values,
                label=f"{label} {ns}", **line_style(color, ns),
            )
            raw_curves.append((f"{label} {ns}", samples_seen, values))

    # Physics-loss activation marker for each run (one axvline per run; by
    # design they land at nearly the same samples-seen, so they overlap).
    activation_label = "Activation of\nPhysics-Loss"
    for config_name, _, _ in variants:
        activation_epoch = activation_epochs[config_name]
        if activation_epoch is None:
            continue
        ax.axvline(
            activation_epoch * samples_per_epoch[config_name],
            color="gray",
            linestyle="--",
            label=activation_label,
        )
        activation_label = None  # label the marker only once in the legend

    style_log_yaxis(ax, metric)
    ax.set_xlabel("samples seen")
    ax.legend()
    ax.set_title(
        f"{family}: zusatz (38 sims) vs base (23 sims)\n"
        f"{metric} (train dashed / valid solid)"
    )
    fig.tight_layout()
    out = os.path.join(
        _ZUSATZ_OUT_DIR, f"{family}__{metric}__zusatz_vs_base__samples_seen.png"
    )
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, _REPO_ROOT)}", flush=True)
    raw_out = os.path.join(
        _ZUSATZ_RAW_DIR, f"{family}__{metric}__zusatz_vs_base__samples_seen.txt"
    )
    write_raw_data(raw_out, "samples_seen", raw_curves)
    print(f"  wrote {os.path.relpath(raw_out, _REPO_ROOT)}", flush=True)


def main():
    argparse.ArgumentParser(description=__doc__.split("\n\n")[0]).parse_args()

    import wandb

    api = wandb.Api()
    os.makedirs(_RAW_DIR, exist_ok=True)
    os.makedirs(_ZUSATZ_RAW_DIR, exist_ok=True)

    # Cache history by (entity, project, group) so the base runs shared between
    # the noPDE-comparison and zusatz-overlay figures are fetched only once.
    history_cache = {}

    def get_history(config_name):
        entity, project, group = wandb_target(config_name)
        key = (entity, project, group)
        if key not in history_cache:
            print(f"  fetching {config_name}: {entity}/{project} group={group}", flush=True)
            history_cache[key] = fetch_group_history(api, entity, project, group)
        return history_cache[key]

    for parent, family_metric in PAIRS:
        print(f"== {parent} ==", flush=True)
        activation_epoch = pde_activation_epoch(parent)
        histories = {
            "PDE": get_history(parent),
            "noPDE": get_history(f"{parent}_noPDE"),
        }
        for metric in (family_metric, "loss_data_full"):
            plot_pair(parent, metric, histories, activation_epoch)

    for base, zusatz, family_metric, family in ZUSATZ_PAIRS:
        print(f"== zusatz overlay: {family} ==", flush=True)
        histories = {base: get_history(base), zusatz: get_history(zusatz)}
        samples_per_epoch = {base: num_train(base), zusatz: num_train(zusatz)}
        activation_epochs = {
            base: pde_activation_epoch(base),
            zusatz: pde_activation_epoch(zusatz),
        }
        for metric in (family_metric, "loss_data_full"):
            plot_zusatz(
                base, zusatz, metric, family,
                histories, samples_per_epoch, activation_epochs,
            )


if __name__ == "__main__":
    sys.exit(main())
