import argparse
import pathlib

import torch as th

from learned_planner.interp.train_probes import ActivationsDataset, DatasetStore, set_seed  # noqa: F401

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True)

set_seed(42)

args = parser.parse_args()
gamma_values = [0.99, 1.0]

for gamma_value in gamma_values:
    for labels_type in ["true_value", "pred_value", "reward", "next_target_pos", "success"]:
        if gamma_value in gamma_values[1:] and labels_type != "true_value":
            continue
        for num_points in [5]:
            acts_ds = ActivationsDataset(
                pathlib.Path(args.dataset_path) / "8_think_step",
                labels_type=labels_type,
                num_data_points=num_points,
                fetch_all_boxing_data_points=True,
                gamma_value=gamma_value,
            )
            if labels_type != "true_value":
                name = f"/8ts_{labels_type}_{num_points}.pt"
            else:
                name = f"/8ts_{labels_type}_{num_points}_gamma_{gamma_value}.pt"
            th.save(acts_ds, args.dataset_path + name)
            print("Saved", name)
