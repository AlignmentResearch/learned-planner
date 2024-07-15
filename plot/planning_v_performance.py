import re
from pathlib import Path

import matplotlib
import wandb
from matplotlib import pyplot as plt

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
}
matplotlib.rcParams.update(style)


api = wandb.Api()
group = "069-evaluate-scatter-plot"
subsets = ["test_unfiltered", "planning_medium", "hard"]

runs = api.runs(
    "farai/lp-cleanba",
    filters={
        "group": group,
        "state": "finished",
        "summary_metrics.test_unfiltered/00_episode_successes": {"$gt": 0},
    },
)

print("Total runs:", len(runs))
baseline_performances = {k: [] for k in subsets}
planning_effects = {k: [] for k in subsets}
net_paths = []
for run in runs:
    net_paths.append(Path(run.config["load_other_run"]))
    try:
        for subset in subsets:
            baseline_performance = run.summary_metrics[f"{subset}/00_episode_successes"]
            baseline_performances[subset].append(baseline_performance)
            all_step_successes = [v for k, v in run.summary_metrics.items() if re.match(subset + r"/\d+_episode_successes", k)]
            planning_effect = max(all_step_successes) - baseline_performance
            planning_effects[subset].append(planning_effect)
    except KeyError as e:
        print(f"KeyError: {e}")
        print(net_paths[-1])
        continue

print(net_paths)

baseline_performances = {k: [100 * v for v in vs] for k, vs in baseline_performances.items()}
planning_effects = {k: [100 * v for v in vs] for k, vs in planning_effects.items()}

for subset in subsets:
    # scatter plot of planning effect (y) vs baseline performance (x)
    plt.scatter(baseline_performances[subset], planning_effects[subset], label=subset)

# add a figsize of (6.5, 4)
plt.gcf().set_size_inches(6.5, 4)
plt.xlabel("Baseline performance (%)")
plt.ylabel("Planning effect (%)")
plt.legend()
plt.savefig("planning_v_performance_combined.png")
plt.show()

# scatter plots in subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, subset in enumerate(subsets):
    axs[i].scatter(baseline_performances[subset], planning_effects[subset])
    axs[i].set_title(subset)
# label for the whole figure
fig.supxlabel("Baseline performance (%)")
fig.supylabel("Planning effect (%)")
fig.tight_layout()
plt.savefig("planning_v_performance_subplots.png")
plt.show()
