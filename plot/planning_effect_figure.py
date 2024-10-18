# %%
import pathlib
import re

import matplotlib.pyplot as plt
import pandas as pd

import learned_planner.interp.plot  # noqa

steps_to_think_for_pe = [0, 2, 4, 8, 12, 16]
network_name = "drc33"
plots_dir = pathlib.Path("./")

success_rates = {}
dataset_name = "valid_medium"
csv_file = pathlib.Path(__file__).parent / "data" / f"{network_name}_{dataset_name}_success_across_thinking_steps.csv"
df = pd.read_csv(csv_file, index_col="Step")
select_columns = [col.endswith("_episode_successes") for col in df.columns]
df = df.loc[:, select_columns]
run_name = df.columns[0].split(" - ")[0]
new_cols = [int(re.search(r"(\d+)_episode_successes$", col).group(1)) for col in df.columns]  # type: ignore
df.columns = new_cols

success_rates[network_name] = df.iloc[-1].copy()

df = df[steps_to_think_for_pe]
csv_file = pathlib.Path(__file__).parent / "data" / f"resnet_{dataset_name}_success_across_thinking_steps.csv"
df_resnet = pd.read_csv(csv_file, index_col="Step")

per_step = df * 100
# per_step = per_step - per_step.loc[0]

fig, axes = plt.subplots(2, 1, figsize=(3.25, 2.5), sharex=True, height_ratios=[3, 1])

for i in range(len(per_step.T)):
    this_step_proportion = i / len(per_step.T)
    per_step.iloc[:, i].plot(color=plt.get_cmap("viridis")(this_step_proportion), ax=axes[0], legend=False)

resnet_run_name = str(df_resnet.columns[0]).split(" - ")[0]
(df_resnet[f"{resnet_run_name} - {dataset_name}/00_episode_successes"] * 100).plot(color="C1", ax=axes[0], label="resnet")
(per_step.max(axis=1) - per_step[0]).plot(ax=axes[1], label="valid_medium", color="C0")

dataset_name = "hard"
csv_file = pathlib.Path(__file__).parent / "data" / f"{network_name}_{dataset_name}_success_across_thinking_steps.csv"
df = pd.read_csv(csv_file, index_col="Step")
select_columns = [col.endswith("_episode_successes") for col in df.columns]
df = df.loc[:, select_columns]
run_name = df.columns[0].split(" - ")[0]
new_cols = [int(re.search(r"(\d+)_episode_successes$", col).group(1)) for col in df.columns]  # type: ignore
df.columns = new_cols

success_rates[network_name] = df.iloc[-1].copy()

df = df[steps_to_think_for_pe]
csv_file = pathlib.Path(__file__).parent / "data" / f"resnet_{dataset_name}_success_across_thinking_steps.csv"
df_resnet = pd.read_csv(csv_file, index_col="Step")

per_step = df * 100
(per_step.max(axis=1) - per_step[0]).plot(ax=axes[1], label="hard", color="C3")

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
plt.savefig(plots_dir / "fig1_with_valid_hard_plan_effect.pdf", format="pdf")
plt.show()
plt.close()
