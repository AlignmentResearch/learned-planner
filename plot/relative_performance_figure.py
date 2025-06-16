# %%


import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from learned_planner import LP_DIR

plots_dir = LP_DIR / "plot" / "plots"
plots_dir.mkdir(exist_ok=True)

style = {
    "font.family": "serif",
    "font.serif": "Times New Roman",
    "mathtext.fontset": "cm",
    "font.size": 8,
    "legend.fontsize": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": (3.25, 1.8),
    "figure.constrained_layout.use": True,
}
matplotlib.rcParams.update(style)


def obtain_successes(network_name: str, dataset_name: str) -> pd.Series:
    df = pd.read_csv(LP_DIR / "plot" / "data" / f"{network_name}_{dataset_name}_success_across_thinking_steps.csv", index_col="Step")
    run_name = str(df.columns[0]).split(" - ")[0]
    out = df[f"{run_name} - {dataset_name}/00_episode_successes"]
    out.name = f"{network_name} - {dataset_name}"
    return out  # type: ignore


_, axes = plt.subplots(1, 2, sharex=True, sharey=True)

pretty = {"test_unfiltered": "test unfiltered", "hard": "hard"}

for i, pair in enumerate([("test_unfiltered", "valid_medium"), ("hard", "valid_medium")]):
    for net in ["drc33", "resnet", "drc11"]:
        x = obtain_successes(net, pair[0])
        y = obtain_successes(net, pair[1])
        axes[i].plot(x.values, y.values, label=net, linestyle="", marker="+")
        axes[i].set_xlabel("Success (" + pretty[pair[0]] + ")")
        axes[i].set_xticks([0, 0.5, 1.0])

axes[0].set_ylabel("Success (val. medium)")
axes[0].legend()
plt.savefig(
    plots_dir / "relative_performance.pdf",
    bbox_inches="tight",
)
