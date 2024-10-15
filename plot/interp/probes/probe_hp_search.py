import json
from pathlib import Path

import wandb
from matplotlib import pyplot as plt

from learned_planner.interp.train_probes import TrainOn

api = wandb.Api()

groups = [
    # "01-probe-agent-in-a-cycle",
    # "02-probe-future-positions",
    # "03-probe-boxes-future-position",
    # "04-probe-agents-future-direction",
    # "05-probe-boxes-future-direction",
    # "06-probe-next-box",
    # "07-probe-next-target",
    # "08-probe-alternative-boxes-directions",
    # "09-probe-action",
    # "10-probe-action-hookc",
    "11-probe-value",
    # "12-probe-true-value",
]
dataset_name_filter = "multioutput"  # set None to disable filtering
dataset_name_filter = None

plot_acc = True  # set True to plot accuracy (used for 11-probe-value)

split_options = [
    "layer",
    "weight_decay",
    "solver",
    "sklearn_l1_ratio",
    "sklearn_class_weight",
    "dataset",
    "hooks",
    "mean_pool_grid",
]
for group in groups:
    runs = api.runs(
        "farai/learned-planners",
        filters={
            "created_at": {"$gt": "2024-09-17"},
            "group": group,
        },
    )
    print("Total runs:", len(runs))
    if len(runs) == 0:
        raise ValueError("No runs found")
    run_stats = []
    f1_is_loss = False
    for run in runs:
        if run.state != "finished":
            continue
        config_dict = json.loads(run.json_config)
        dataset = config_dict["cmd"]["value"]["dataset_path"].split("/")[-1].split(".")[0]
        if dataset_name_filter and dataset_name_filter not in dataset:
            continue
        summary = {k.replace("x-all_y-all_", ""): v for k, v in run.summary.items()}
        train_on_args = config_dict["cmd"]["value"]["train_on"]
        train_on_args.pop("square_x", None), train_on_args.pop("square_y", None)
        train_on = TrainOn(**train_on_args)
        probe_subkey = str(train_on)
        # for backward compatibility
        probe_subkey_without_mpg = probe_subkey.split("_mpg-")[0]
        probe_subkey_wo_ds = probe_subkey_without_mpg.split("_ds-")[0]
        keys_to_try = [probe_subkey, probe_subkey_without_mpg, probe_subkey_wo_ds]
        layer = train_on.layer
        hooks = tuple(train_on.hooks)
        weight_decay = config_dict["cmd"]["value"]["weight_decay"]
        solver = config_dict["cmd"]["value"]["sklearn_solver"]

        f1, l0, acc = None, None, None
        while f1 is None and len(keys_to_try) > 0:
            curr_key = keys_to_try.pop(0)
            f1 = summary.get(f"test/{curr_key}/f1", None)
            l0 = summary.get(f"{curr_key}/nonzero_weights", None)
        if f1 is None:
            try:
                f1 = summary[f"test/{probe_subkey}/loss"]
                acc = summary.get(f"test/{probe_subkey}/accuracy", None)
                f1_is_loss = True
                l0 = summary.get(f"{probe_subkey}/nonzero_weights", None)
            except KeyError:
                continue
        f1 = float(f1)
        if f1 != f1:  # nan
            continue
        if plot_acc and acc is None:
            continue
        else:
            acc = float(acc)

        run_stats.append(
            {
                "id": run.id,
                "name": run.name,
                "layer": layer,
                "hooks": hooks,
                "weight_decay": weight_decay,
                "solver": solver,
                "f1": f1,
                "nonzero_weights": l0,
                "dataset": dataset,
                "acc": acc,
            }
        )
        for split_option in split_options:
            if split_option in config_dict["cmd"]["value"]:
                run_stats[-1][split_option] = config_dict["cmd"]["value"][split_option]
            elif split_option in config_dict["cmd"]["value"]["train_on"]:
                run_stats[-1][split_option] = config_dict["cmd"]["value"]["train_on"][split_option]

    run_stats = sorted(run_stats, key=lambda x: x["f1"], reverse=not f1_is_loss)
    for run_stat in run_stats:
        print(
            run_stat["dataset"],
            run_stat["id"],
            run_stat["name"],
            run_stat["f1"],
            int(run_stat["nonzero_weights"]),
            run_stat["layer"],
            sep=",",
            end="",
        )
        if plot_acc:
            print(",", run_stat["acc"], sep="", end="")
        print()

    print("Finished runs:", len(run_stats))

    if len(run_stats) == 0:
        continue

    print("Min nonzero weights:", min([x["nonzero_weights"] for x in run_stats]))
    # run_stats = [x for x in run_stats if x["layer"] != -1]
    # run_stats = [x for x in run_stats if x["nonzero_weights"] < 1000]
    # split_bys = ["layer", "weight_decay"] + (["sklearn_l1_ratio"] if "elastic" in group else ["solver"])
    split_bys = []
    for split_option in split_options:
        if split_option not in run_stats[0]:
            continue
        if all([x.get(split_option, None) == run_stats[0][split_option] for x in run_stats]):
            continue
        split_bys.append(split_option)

    fig, axs = plt.subplots(1, len(split_bys), figsize=(5 * len(split_bys), 5))
    axs = [axs] if len(split_bys) == 1 else axs
    for i, split_by in enumerate(split_bys):
        ax = axs[i]
        try:
            unique_vals = sorted(set([x.get(split_by, None) for x in run_stats]))
        except TypeError:
            unique_vals = set([x.get(split_by, None) for x in run_stats])
        for split_val in unique_vals:
            split_runs = [x for x in run_stats if x.get(split_by, None) == split_val]
            ax.scatter(
                [x["nonzero_weights"] for x in split_runs],
                [x["acc" if plot_acc else "f1"] for x in split_runs],
                label=str(split_val),
            )
        ax.set_xlabel("Nonzero weights")
        ax.set_ylabel("test/" + ("acc" if plot_acc else "loss" if f1_is_loss else "f1"))
        ax.legend(loc="lower right")
        ax.set_title(split_by)

    fig.suptitle(group)
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / f"{group}.png")
