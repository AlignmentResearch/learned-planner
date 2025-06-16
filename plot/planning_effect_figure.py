# %%
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import learned_planner.interp.plot  # noqa

dataset_name = "valid_medium"
steps_to_think_for_pe = [0, 2, 4, 8, 12, 16]
network_name = "drc_33"
plots_dir = pathlib.Path("./")

success_rates = {}
csv_file = pathlib.Path(__file__).parent / "data" / f"{network_name}.csv"
df = pd.read_csv(csv_file, index_col="step")
df = df * 100
select_columns = [dataset_name in col and col.endswith("_episode_successes_mean") for col in df.columns]
df_mean = df.loc[:, select_columns]
new_cols = [int(re.search(r"(\d+)_episode_successes_mean$", col).group(1)) for col in df_mean.columns]  # type: ignore
df_mean.columns = new_cols

success_rates[network_name] = df_mean.iloc[-1].copy()

df_mean = df_mean[steps_to_think_for_pe]
csv_file = pathlib.Path(__file__).parent / "data" / "resnet.csv"
df_resnet = pd.read_csv(csv_file, index_col="step")

fig, axes = plt.subplots(2, 1, figsize=(3.25, 2.5), sharex=True, height_ratios=[3, 1])
x = df_mean.index
for i in range(len(df_mean.T)):
    this_step_proportion = i / len(df_mean.T)
    df_mean.iloc[:, i].plot(color=plt.get_cmap("viridis")(this_step_proportion), ax=axes[0], legend=False)
    steps_to_think = steps_to_think_for_pe[i]
    axes[0].fill_between(
        x,
        df.loc[:, f"{dataset_name}/{steps_to_think:02d}_episode_successes_min"],
        df.loc[:, f"{dataset_name}/{steps_to_think:02d}_episode_successes_max"],
        alpha=0.2,
        color=plt.get_cmap("viridis")(this_step_proportion),
    )

(df_resnet[f"{dataset_name}/00_episode_successes_mean"] * 100).plot(color="C1", ax=axes[0], label="resnet")
axes[0].fill_between(
    df_resnet.index,
    df_resnet[f"{dataset_name}/00_episode_successes_min"] * 100,
    df_resnet[f"{dataset_name}/00_episode_successes_max"] * 100,
    alpha=0.2,
    color="C1",
)

# Planning Effect
for ds_name in ["valid_medium", "hard"]:
    planning_effect = []
    seed, num_seeds = 0, 0
    while True:
        try:
            success_rate_for_seed = df[[f"{ds_name}/{stt:02d}_episode_successes_{seed}" for stt in steps_to_think_for_pe]]
            planning_effect.append(success_rate_for_seed.max(axis=1) - success_rate_for_seed.min(axis=1))
            seed += 1
            num_seeds += 1
        except KeyError:
            break
    print(f"num_seeds: {num_seeds}")
    planning_effect = np.array(planning_effect)
    color = "C0" if ds_name == "valid_medium" else "C3"
    axes[1].plot(x, planning_effect.mean(axis=0), label=ds_name, color=color)
    axes[1].fill_between(x, planning_effect.min(axis=0), planning_effect.max(axis=0), alpha=0.2, color=color)

axes[0].grid(True)
axes[1].grid(True)

axes[1].set_xlabel("Environment steps, training")

axes[0].set_ylabel(r"% solved")
axes[1].set_ylabel("Planning Effect")
axes[0].legend(ncols=3, prop={"size": 8})
axes[0].set_xlim((998400.0, 2002944000.0))
# set xticks
axes[0].set_xticks([7e7, 5e8, 1e9, 1.5e9, 2e9])
# set xticklabels
axes[0].set_xticklabels(["70M", "500M", "1B", "1.5B", "2B"])
axes[1].legend(ncols=2, prop={"size": 8})
plt.savefig(plots_dir / f"fig1_{dataset_name}_with_valid_hard_plan_effect.pdf", format="pdf")
plt.show()
plt.close()

# %%
