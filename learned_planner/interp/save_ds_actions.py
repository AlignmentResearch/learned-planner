import argparse
import pathlib
from time import time

import torch as th

from learned_planner.interp.train_probes import (  # noqa: F401  # pyright: ignore
    ActivationsDataset,
    DatasetStore,
    set_seed,
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--balance", action="store_true")
parser.add_argument("--skip_walls", action="store_true")
parser.add_argument("--ts", action="store_true")
parser.add_argument("--nozip", action="store_true")
parser.add_argument("--multioutput", action="store_true")
parser.add_argument("--internal_act", action="store_true")


args = parser.parse_args()
gamma_values = [0.99, 1.0]
ts = "8" if args.ts else "0"
for gamma_value in gamma_values:
    # for labels_type in ["true_value", "pred_value", "reward", "next_target_pos", "success"]:
    # for labels_type in ["pred_value"]:
    # for labels_type in ["actions_for_probe_0"]:
    for action in range(4):
        labels_type = f"actions_for_probe_{action}_False"
        # for labels_type in ["agents_future_direction_map", "boxes_future_direction_map", "next_box", "next_target"]:
        # for labels_type in ["agents_future_direction_map", "boxes_future_direction_map"]:
        for skip in [0]:
            # for labels_type in ["agents_future_position_map"]:
            if gamma_value in gamma_values[1:] and labels_type != "true_value":
                continue
            if args.multioutput:
                assert "direction" in labels_type
            keys = [".*hook_h$", ".*hook_c$"]
            if args.internal_act:
                keys = [".*hook_i$", ".*hook_j$", ".*hook_f$", ".*hook_o$"]
            for num_points in [5, 20000]:
                if labels_type != "true_value":
                    name = f"/{ts}ts_{labels_type}_{num_points}{'_balance' if args.balance else ''}_skip{skip}.pt"
                else:
                    name = f"/{ts}ts_{labels_type}_{num_points}_gamma_{gamma_value}_skip{skip}.pt"
                if args.skip_walls:
                    name = name.replace(".pt", "_skipwalls.pt")
                if args.multioutput:
                    name = name.replace(".pt", "_multioutput.pt")
                if args.internal_act:
                    name = name.replace(".pt", "_keys-ijfo.pt")
                set_seed(42)
                train_ds = ActivationsDataset(
                    pathlib.Path(args.dataset_path) / (ts + "_think_step"),
                    labels_type=labels_type,
                    keys=keys,
                    num_data_points=num_points,
                    fetch_all_boxing_data_points=True,
                    gamma_value=gamma_value,
                    balance_classes=args.balance,
                    skip_first_n=skip,
                    skip_walls=args.skip_walls,
                    multioutput=args.multioutput,
                    train=True,
                    seed=42,
                )
                train_levels = set(train_ds.level_files)
                t0 = time()
                th.save(train_ds, args.dataset_path + name, _use_new_zipfile_serialization=not args.nozip)
                del train_ds
                print(f"Saved {name} in", time() - t0)
