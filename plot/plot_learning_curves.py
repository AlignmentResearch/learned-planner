# %%
import collections
import os
import pathlib
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import wandb
from matplotlib import pyplot as plt


# %%
def extract_per_step(df, suffix="successes"):
    is_success = re.compile(f"{difficulty}/([0-9]{{2}})_episode_{suffix}")
    planning_cols = [(c, int(m.group(1))) for c in df.columns if (m := is_success.match(c)) is not None]

    keys, numbers = map(list, zip(*planning_cols))
    per_step = df[keys]
    per_step.columns = numbers
    return per_step


# %%

wandb.init(mode="disabled")

plots_dir = pathlib.Path("plots/")
plots_dir.mkdir(exist_ok=True)

style = {
    "font.family": "serif",
    "font.serif": "Times New Roman",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "legend.fontsize": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (3.25, 2),
    "figure.constrained_layout.use": True,
    "axes.grid": True,
}
matplotlib.rcParams.update(style)


baseline_steps = best_steps = 100000  # absurdly high number so using it before definition errors

# %%

base_dir = Path("learning_curves")

drc33_csv = pd.read_csv(base_dir / "drc_33/jl6bq8ih.test.csv", index_col="step")
resnet_csv = pd.read_csv(base_dir / "resnet/8ul1b23e.test.csv", index_col="step")

# %%

difficulty = "valid_medium"
MARKER = None


fig, axes = plt.subplots(2, 1, figsize=(3.25, 2.5), sharex=True, height_ratios=[3, 1])

per_step = extract_per_step(drc33_csv)
for i in range(len(per_step.T)):
    this_step_proportion = i / len(per_step.T)
    per_step.iloc[:, i].plot(
        color=plt.get_cmap("viridis")(this_step_proportion), ax=axes[0], legend=False, linewidth=1, marker=MARKER
    )

resnet_csv[f"{difficulty}/00_episode_successes"].plot(color="C1", ax=axes[0], label="resnet", marker=MARKER)

(per_step.max(axis=1) - per_step.min(axis=1)).plot(ax=axes[1], marker=MARKER)


axes[1].set_xlabel("Environment steps, training")
axes[0].set_ylabel("Val-medium solved")
axes[1].set_ylabel("Plan. Effect")
axes[0].legend(ncols=3, prop={"size": 8})
axes[0].set_xlim((0.0, int(per_step.index.max())))  # type: ignore

plt.savefig(plots_dir / "valid_curve.pdf", format="pdf")


# %% Training curves compared to Deepmind

ARCHES = ["drc_33", "resnet", "drc_11"]
PLOT_DEEPMIND_ERRORS = False

fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5), sharex=True)

for difficulty, diff_label, ax in [
    ("test_unfiltered", "Test-unfiltered solved", axes[0]),
    ("valid_medium", "Val-medium solved", axes[1]),
]:
    for arch_i, arch in enumerate(ARCHES):
        arch_values = []
        arch_csv = None
        for fname in os.listdir(base_dir / arch):
            if fname.endswith("test.csv"):
                arch_csv = extract_per_step(pd.read_csv(base_dir / arch / fname, index_col="step"))
                arch_values.append(arch_csv.loc[:, 0].to_numpy())

        max = np.max(arch_values, 0)
        min = np.min(arch_values, 0)
        median = np.median(arch_values, 0)

        assert arch_csv is not None
        xs = arch_csv.index

        ax.plot(xs, median, label=arch, color=f"C{arch_i}")
        ax.fill_between(xs, min, max, color=f"C{arch_i}", alpha=0.2)
        ax.set_ylabel(diff_label)
        ax.set_xlabel("Environment steps, training")


axes[0].legend()
axes[0].set_ylim((0.8, 1.01))


dm_csv = pd.read_csv("extract_gdm/dm-learning-curves-resized.csv", index_col=0).sort_index()

arch_colors = {
    "drc_33": "#377eb8",
    "drc_11": "#984ea3",
    "resnet": "#a65628",
}

ax = axes[0]

for arch_i, arch in enumerate(ARCHES):
    column_name = arch_colors[arch]
    dm_csv[column_name].dropna().plot(ax=ax, color=f"C{arch_i}", label="", ls="--")

    if PLOT_DEEPMIND_ERRORS:
        minmax = dm_csv[[f"{column_name}_min", f"{column_name}_max"]].dropna()
        minmax.columns = ["min", "max"]
        ax.fill_between(minmax.index, minmax["min"], minmax["max"], color=f"C{arch_i}", alpha=0.2)


ax.plot([], [], color="gray", label="deepmind", ls="--")
ax.legend()

plt.savefig(plots_dir / "test_and_valid_learning_curves.pdf")

# %% Planning effect during training

ARCHES = ["drc_33", "drc_11"]

fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.5), sharex=True)

for difficulty, diff_label, ax in [
    ("test_unfiltered", "Test-unfiltered planning at 12", axes[0]),
    ("valid_medium", "Val-medium planning at 12", axes[1]),
]:
    for arch_i, arch in enumerate(ARCHES):
        arch_values = []
        arch_csv = None
        for fname in os.listdir(base_dir / arch):
            if fname.endswith("test.csv"):
                arch_csv = extract_per_step(pd.read_csv(base_dir / arch / fname, index_col="step"))
                # diff begins
                planning_effect = arch_csv[12] - arch_csv[0]
                arch_values.append(planning_effect)
                # diff ends

        max = np.max(arch_values, 0)
        min = np.min(arch_values, 0)
        median = np.median(arch_values, 0)

        assert arch_csv is not None
        xs = arch_csv.index

        ax.plot(xs, median, label=arch, color=f"C{arch_i}")
        ax.fill_between(xs, min, max, color=f"C{arch_i}", alpha=0.2, label="")
        ax.set_ylabel(diff_label)
        ax.set_xlabel("Environment steps, training")

ax.legend()
plt.savefig(plots_dir / "planning_at_12.pdf")


# %% Planning effect per steps for various points in training

indices = [20029440, 100147200, 200294400, 2002944000]


fig, axes_grid = plt.subplots(1, len(indices), figsize=(6.75, 2.0), sharex=True, sharey=True)

for difficulty, diff_label, axes in [
    ("valid_medium", "Val-medium planning effect", list(axes_grid.flat)),
]:
    arch_values = collections.defaultdict(list)
    for arch_i, arch in enumerate(ARCHES):
        for fname in os.listdir(base_dir / arch):
            if fname.endswith("test.csv"):
                arch_csv = extract_per_step(pd.read_csv(base_dir / arch / fname, index_col="step"))
                # diff begins
                for idx in indices:
                    planning_effect = arch_csv.loc[idx, :] - arch_csv.loc[idx, 0]
                    arch_values[idx].append(planning_effect)
                # diff ends

        for idx, ax in zip(indices, axes):
            max = np.max(arch_values[idx], 0)
            min = np.min(arch_values[idx], 0)
            median = np.median(arch_values[idx], 0)

            planning_steps = [0, 2, 4, 8, 12, 16, 24, 32]
            xs = planning_steps

            ax.plot(xs, median, label=arch, color=f"C{arch_i}")
            ax.fill_between(xs, min, max, color=f"C{arch_i}", alpha=0.2, label="")
            ax.set_title(f"Planning at {idx//int(1e6)}M steps")
            ax.set_xlabel("Planning time")
            ax.set_xticks(xs)
            ax.set_xticklabels(planning_steps)

plt.savefig(plots_dir / "planning_over_training")
