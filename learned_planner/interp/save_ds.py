import argparse
import pathlib
from time import time

import torch as th

from learned_planner.interp.train_probes import ActivationsDataset, DatasetStore, set_seed  # noqa: F401  # pyright: ignore

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--labels_type", type=str, required=True, choices=[
    "true_value", "pred_value", "reward", "agent_in_a_cycle", "agents_future_position_map", "agents_future_direction_map", 
    "boxes_future_direction_map", "next_box", "next_target",
])
parser.add_argument("--num_datapoints", type=int, default=5000)
parser.add_argument("--skip_first_n", type=int, default=5)
parser.add_argument("--balance", action="store_true")
parser.add_argument("--skip_walls", action="store_true")
parser.add_argument("--ts", action="store_true")
parser.add_argument("--nozip", action="store_true")
parser.add_argument("--multioutput", action="store_true")
parser.add_argument("--internal_act", action="store_true")

args = parser.parse_args()

gamma_values = [0.99, 1.0]
labels_type = args.labels_type
skip = args.skip_first_n
num_points = args.num_datapoints
ts = "6" if args.ts else "0"

for gamma_value in gamma_values:
    if gamma_value in gamma_values[1:] and labels_type != "true_value":
        continue
    if args.multioutput:
        assert "direction" in labels_type
    keys = [".*hook_h$", ".*hook_c$"]
    if args.internal_act:
        keys = [".*hook_i$", ".*hook_j$", ".*hook_f$", ".*hook_o$"]
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
    train_name = name.replace(".pt", "_train.pt")
    th.save(train_ds, args.dataset_path + train_name, _use_new_zipfile_serialization=not args.nozip)
    del train_ds
    print(f"Saved {train_name} in", time() - t0)
    set_seed(42)
    test_ds = ActivationsDataset(
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
        train=False,
        seed=42,
    )
    assert train_levels.intersection(test_ds.level_files) == set()
    t0 = time()
    test_name = name.replace(".pt", "_test.pt")
    th.save(test_ds, args.dataset_path + test_name, _use_new_zipfile_serialization=not args.nozip)
    print(f"Saved {test_name} in", time() - t0)
    del test_ds
