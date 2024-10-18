import argparse
import glob
import multiprocessing
import pathlib
import pickle
import subprocess
import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from sae_lens import SAE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve
from torch.nn import functional as F

from learned_planner.interp.collect_dataset import DatasetStore
from learned_planner.interp.train_probes import TrainOn
from learned_planner.interp.utils import encode_with_sae, predict

hook_template = "features_extractor.cell_list.{layer}.{hook}"


def one_hot(actions: th.Tensor):
    assert len(actions.shape) == 1
    return F.one_hot(actions, num_classes=4)


def plt_confusion_matrix(preds, labels, name="confusion_matrix.png"):
    n_classes = 4 if combined else 2
    cm = confusion_matrix(labels, preds, labels=range(n_classes), normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(n_classes))
    disp.plot()
    plt.savefig(name)


def process_file(file, combined=False):
    with warnings.catch_warnings(action="ignore"):
        ds = DatasetStore.load(file)
    actions = ds.get_actions(only_env_steps=True)
    if not combined:
        actions = one_hot(actions)
    cache = {k: ds.get_cache(key=k, only_env_steps=True) for k in keys}
    if args.feature_from == "sae":
        acts = encode_with_sae(probe_or_sae, cache, is_concatenated_cache=True)
        action_preds = acts[..., action_features].detach().cpu().numpy()
        assert len(action_preds.shape) == 4, f"Got {action_preds.shape}"
        action_preds = action_preds.sum(axis=(1, 2))
        action_preds = negative_multiplier * action_preds
    elif args.feature_from == "probe":
        action_preds = [predict(cache, probe, probe_info, 0, is_concatenated_cache=True) for probe in probe_or_sae]
        action_preds = np.stack(action_preds, axis=-1).sum(axis=(-2, -3))
        if action_preds.shape[1] == 1:
            actions = actions[:, [probe_info.action_idx]]
        else:
            actions = actions[:, : action_preds.shape[1]]

    elif args.feature_from == "channel":
        acts = cache[hook_template.format(layer=args.layer, hook=hook_names[0])]
        action_preds = acts[:, action_features].sum(axis=(-1, -2))
        action_preds = negative_multiplier * action_preds
    else:
        raise ValueError("Invalid feature_from")

    assert action_preds.shape == actions.shape
    return action_preds, actions


description = """This script can evaluate action features from SAE, probes, or channels. Example commands:
Layer 2 Probes
python -W ignore evaluate_sae_features.py --feature_from probe --wandb_id 4astgbed,6tndl9lb,dhejplrt,vaou9qcc --layer 2 --dataset_name actions_for_probe_0
All Layer Probes
python -W ignore evaluate_sae_features.py --feature_from probe --wandb_id vly1xddx,tngkza1u,l914ti2k,9gttt42q --layer -1 --dataset_name actions_for_probe_0
SAE
python -W ignore evaluate_sae_features.py --feature_from sae --wandb_id ho6ob1tk
Channel
python -W ignore evaluate_sae_features.py --feature_from channel --layer 2 --hooks hook_h --action_features 29,8,27,3 --negative_features 2
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset_path", type=str, default="/training/activations_dataset/valid_medium/0_think_step/")
    parser.add_argument("--feature_from", type=str, default="sae", choices=["sae", "probe", "channel"])
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--wandb_id", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="boxes_future_direction_map")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--hooks", type=str, default="hook_h,hook_c", help="Activation hook for probe or channel")
    parser.add_argument("--action_features", type=str, default="304,187,244,385")
    parser.add_argument(
        "--negative_features", type=str, default="", help="Action indices for the features whose activations should be negated"
    )
    # parser.add_argument("--action_features", type=str, default="165,187,244,385")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    wandb_id = args.wandb_id
    path = args.path

    if args.layer < 0:
        keys = [hook_template.format(layer=layer, hook=hook) for layer in range(3) for hook in args.hooks.split(",")]
    else:
        keys = [hook_template.format(layer=args.layer, hook=hook) for hook in args.hooks.split(",")]

    if path != "" and wandb_id != "":
        raise ValueError("Cannot specify both path and wandb_id")

    if args.feature_from != "channel":
        if wandb_id != "":
            if args.feature_from == "sae" or args.feature_from == "probe":
                commands = [
                    f"/training/find{args.feature_from}.sh {single_wandb_id}" for single_wandb_id in wandb_id.split(",")
                ]
            else:
                raise ValueError("Invalid feature_from")
            path = [subprocess.run(command, shell=True, capture_output=True, text=True).stdout.strip() for command in commands]
            path = [pathlib.Path(p) for p in path if p]
        else:
            path = [pathlib.Path(p) for p in path.split(",")]
        assert len(path) >= 1 and all([p.exists() for p in path]), f"Invalid path='{args.path}' and wandb_id='{args.wandb_id}'"

    if args.feature_from == "sae":
        probe_or_sae = SAE.load_from_pretrained(path[0])
        keys = [probe_or_sae.cfg.hook_name]
    elif args.feature_from == "probe":
        probe_or_sae = []
        for p in path:
            with open(p, "rb") as f:
                probe_or_sae.append(pickle.load(f))

    action_features = [int(f) for f in args.action_features.split(",")]
    assert len(action_features) == 4, "Please provide 4 action features"
    negative_features = [int(f) for f in args.negative_features.split(",") if f]
    negative_multiplier = np.array([-1 if i in negative_features else 1 for i in range(4)]).reshape(1, -1)

    dataset_path = pathlib.Path(args.dataset_path)
    hook_names = args.hooks.split(",")
    probe_info = TrainOn(layer=args.layer, dataset_name=args.dataset_name, hooks=hook_names)

    files = glob.glob(str(dataset_path / "*.pkl"))

    combined = False
    map_fn = partial(process_file, combined=combined)

    if args.num_workers <= 1:
        results = [map_fn(f) for f in files]
    else:
        with multiprocessing.Pool(args.num_workers) as pool:
            results = list(pool.imap(map_fn, files))

    all_preds = np.concatenate([r[0] for r in results if len(r[0]) > 0])
    gts = np.concatenate([r[1] for r in results if len(r[0]) > 0])

    ft_name = ["UP", "DOWN", "LEFT", "RIGHT"]
    print("Precision,Recall,F1,Threshold")
    for i in range(gts.shape[1]):
        act_idx = probe_info.action_idx if gts.shape[1] == 1 else i
        p, r, t = precision_recall_curve(gts[:, i], all_preds[:, i])
        f1 = 2 * p * r / (p + r)
        best_idx = np.argmax(f1)
        print(f"{p[best_idx]},{r[best_idx]},{f1[best_idx]},{t[best_idx]}")
        if args.plot:
            plt.plot(r, p)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Feature {ft_name[act_idx]}")
            plt.savefig(f"precision_recall_curve_{args.feature_from}_{act_idx}.png")
            plt.close()
            # plt_confusion_matrix(all_preds[:, i], gts[:, i], name=f"confusion_matrix_{i}.png")
            # metrics.update(get_metrics(all_preds[:, i], gts[:, i], classification=True, key_prefix=f"action_{i}"))
