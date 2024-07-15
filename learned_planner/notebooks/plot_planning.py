import re

import matplotlib.pyplot as plt
import pandas as pd

# Take a CSV exported from weights and biases tables, like
# https://wandb.ai/farai/learned-planners/groups/120-evaluate-119-5d/table
# runs = pd.read_csv("planning-runs-119-30h.csv")
runs = pd.read_csv("~/Downloads/wandb_export_2024-04-02T12_42_47.186-07_00.csv")

runs = runs[runs["cmd.env.difficulty"] == "medium"]

success_rate_columns = re.compile("^train/([0-9][0-9])_steps/eval/success_rate$")
run_label = re.compile("^.*-([a-z0-9]{8})/files/.*$")

succs = runs[list(filter(success_rate_columns.match, runs.columns))]
names = runs["cmd.load_path"].map(lambda s: run_label.match(s).groups(1)[0])

succs = succs.T
succs.columns = names
succs.index = succs.index.map(lambda s: int(success_rate_columns.match(s).groups(1)[0]))

plt.style.use("ggplot")
succs.plot()
plt.show()
