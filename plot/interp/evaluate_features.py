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


def process_file(file, combined=False, feature_type="action", channel_layer: int = 2, offset_for_target_fts=(0, 0)):
    with warnings.catch_warnings(action="ignore"):
        ds = DatasetStore.load(file)
    action_type = feature_type == "action"
    if action_type:
        labels = ds.get_actions(only_env_steps=True)
    else:
        only_unsolved = "unsolved" in feature_type
        only_solved = "solved" in feature_type and not only_unsolved
        if "target" in feature_type:
            labels_t = (
                ds.get_target_positions_from_obs(ds.obs, return_map=True, only_unsolved=only_unsolved, only_solved=only_solved)
                .unsqueeze(-1)
                .numpy()
            )
            labels = labels_t
        if "box" in feature_type:
            labels_b = (
                ds.get_box_positions(return_map=True, only_unsolved=only_unsolved, only_solved=only_solved)
                .unsqueeze(-1)
                .numpy()
            )
            labels = labels_b
        if "target" in feature_type and "box" in feature_type:
            labels = labels_t | labels_b
    if action_type and not combined:
        labels = one_hot(labels)
    cache = {k: ds.get_cache(key=k, only_env_steps=True) for k in keys}
    if args.feature_from == "sae":
        acts = encode_with_sae(probe_or_sae, cache, is_concatenated_cache=True)
        ft_preds = acts[..., feature_indices].detach().cpu().numpy()
        ft_preds = negative_multiplier * ft_preds
    elif args.feature_from == "probe":
        ft_preds = [predict(cache, probe, probe_info, 0, is_concatenated_cache=True) for probe in probe_or_sae]
        ft_preds = np.stack(ft_preds, axis=-1)
        if action_type and ft_preds.shape[1] == 1:
            labels = labels[..., [probe_info.action_idx]]
        else:
            labels = labels[..., : ft_preds.shape[1]]
    elif args.feature_from == "channel":
        acts = cache[hook_template.format(layer=channel_layer, hook=hook_names[0])]
        ft_preds = np.transpose(acts[:, feature_indices], (0, 2, 3, 1))
        ft_preds = negative_multiplier * ft_preds
    else:
        raise ValueError("Invalid feature_from")

    if action_type:
        assert len(ft_preds.shape) == 4 and ft_preds.shape[1] == ft_preds.shape[2] == 10, f"Got {ft_preds.shape}"
        ft_preds = ft_preds.sum(axis=(1, 2))
    else:
        # rotate ft_preds by offset_for_target_fts
        ft_preds = np.roll(ft_preds, offset_for_target_fts, axis=(1, 2))
        labels = np.repeat(labels, ft_preds.shape[-1], axis=-1)

    assert ft_preds.shape == labels.shape, f"Got {ft_preds.shape} and {labels.shape}"
    return ft_preds, labels


def call_process_file(num_workers, files, map_fn):
    if num_workers <= 1:
        results = [map_fn(f) for f in files]
    else:
        with multiprocessing.Pool(num_workers) as pool:
            results = list(pool.imap(map_fn, files))
    return results


def get_best_feature(gts, all_preds, plot=False):
    p, r, t = precision_recall_curve(gts, all_preds, drop_intermediate=False)
    p, r = 100 * p, 100 * r
    f1 = 2 * p * r / (p + r + 1e-8)
    best_idx = np.argmax(f1)

    if plot:
        plt.plot(r, p)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Feature {action_ft_name[act_idx]}")
        plt.savefig(f"precision_recall_curve_{args.feature_from}_{act_idx}.png")
        plt.close()
    return p[best_idx], r[best_idx], f1[best_idx], t[best_idx]


def get_best_features(num_workers, gts, all_preds, plot=False):
    num_fts = gts.shape[1]
    num_workers = min(num_workers, num_fts)
    map_fn = partial(get_best_feature, plot=plot)
    if num_workers <= 1:
        results = [map_fn(gts[:, ft_idx], all_preds[:, ft_idx]) for ft_idx in range(num_fts)]
    else:
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.starmap(map_fn, [(gts[:, ft_idx], all_preds[:, ft_idx]) for ft_idx in range(num_fts)])
    return results


description = """This script can evaluate action/target features from SAE, probes, or channels. Example commands:
Layer 2 Probes
python -W ignore evaluate_features.py --feature_from probe --wandb_id 4astgbed,6tndl9lb,dhejplrt,vaou9qcc --layer 2 --dataset_name actions_for_probe_0
All Layer Probes
python -W ignore evaluate_features.py --feature_from probe --wandb_id vly1xddx,tngkza1u,l914ti2k,9gttt42q --layer -1 --dataset_name actions_for_probe_0
SAE
python -W ignore evaluate_features.py --feature_from sae --wandb_id ho6ob1tk
Channel
python -W ignore evaluate_features.py --feature_from channel --layer 2 --hooks hook_h --feature_indices 29,8,27,3 --negative_features 2
python -W ignore evaluate_features.py --feature_from channel --layer -1 --hooks hook_h --feature_indices -1 --feature_type unsolved_target
python -W ignore evaluate_features.py --feature_from channel --layer -1 --hooks hook_h --feature_indices -1 --negative_features '*' --feature_type unsolved_target

"""

if __name__ == "__main__":
    solved_or_unsolved = ["solved_", "unsolved_", ""]
    ft_types = ["target", "box", "target_box", "box_target"]  # target_box and box_target are the same
    feature_choices = [f"{s}{ft}" for s in solved_or_unsolved for ft in ft_types]
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset_path", type=str, default="/training/activations_dataset/valid_medium/0_think_step/")
    parser.add_argument("--num_files", type=int, default=-1)
    parser.add_argument(
        "--feature_type",
        type=str,
        default="action",
        choices=["action"] + feature_choices,
    )
    parser.add_argument("--feature_from", type=str, default="sae", choices=["sae", "probe", "channel"])
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--wandb_id", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="boxes_future_direction_map")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--hooks", type=str, default="hook_h,hook_c", help="Activation hook for probe or channel")
    parser.add_argument(
        "--feature_indices",
        type=str,
        default="304,187,244,385",
        help="Indices for the features to evaluate. -1 to search for best feature.",
    )
    parser.add_argument(
        "--negative_features",
        type=str,
        default="",
        help="Indices for the features whose activations should be negated. Provide * to negate all features.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        nargs=2,
        default=[0, 0],
        help="Feature offset in the grid (for target-based features). (-1, -1) to search for all offsets.",
    )
    # parser.add_argument("--feature_indices", type=str, default="165,187,244,385")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--plot", action="store_true")

    args = parser.parse_args()

    wandb_id = args.wandb_id
    path = args.path

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

    if args.layer < 0:
        keys = [hook_template.format(layer=layer, hook=hook) for layer in range(3) for hook in args.hooks.split(",")]
    else:
        keys = [hook_template.format(layer=args.layer, hook=hook) for hook in args.hooks.split(",")]
    if args.feature_from == "sae":
        probe_or_sae = SAE.load_from_pretrained(path[0])
        keys = [probe_or_sae.cfg.hook_name]
    elif args.feature_from == "probe":
        probe_or_sae = []
        for p in path:
            with open(p, "rb") as f:
                probe_or_sae.append(pickle.load(f))

    if args.feature_indices == "-1":
        feature_indices = slice(None)
    else:
        feature_indices = [int(f) for f in args.feature_indices.split(",")]
    if args.feature_type == "action":
        assert len(feature_indices) == 4, "Please provide 4 action features"

    if args.negative_features == "":
        negative_multiplier = 1
    elif args.negative_features == "*":
        negative_multiplier = -1
    else:
        negative_features = [int(f) for f in args.negative_features.split(",") if f]
        negative_multiplier = np.array([-1 if i in negative_features else 1 for i in range(len(feature_indices))]).reshape(
            1, -1
        )
    dataset_path = pathlib.Path(args.dataset_path)
    hook_names = args.hooks.split(",")
    probe_info = TrainOn(layer=args.layer, dataset_name=args.dataset_name, hooks=hook_names)

    files = glob.glob(str(dataset_path / "*.pkl"))[: (args.num_files if args.num_files > 0 else None)]

    combined = False
    map_fn = partial(
        process_file,
        combined=combined,
        feature_type=args.feature_type,
        channel_layer=args.layer,
        offset_for_target_fts=args.offset,
    )
    all_preds, gts = [], []
    all_layers, first_layer = range(3) if args.layer < 0 else [args.layer], 0 if args.layer < 0 else args.layer
    all_offsets = [(i, j) for i in range(-1, 2) for j in range(-1, 2)] if args.offset == [-1, -1] else [args.offset]
    num_features = 0
    for offset in all_offsets:
        for layer in all_layers:
            keys = [hook_template.format(layer=layer, hook=hook) for hook in hook_names]
            map_fn.keywords["channel_layer"] = layer
            map_fn.keywords["offset_for_target_fts"] = offset
            results = call_process_file(args.num_workers, files, map_fn)
            num_features = results[0][0].shape[-1]
            all_preds.append(np.concatenate([r[0] for r in results if len(r[0]) > 0]).reshape(-1, num_features))
            gts.append(np.concatenate([r[1] for r in results if len(r[0]) > 0]).reshape(-1, num_features))
    all_preds = np.concatenate(all_preds, axis=-1)
    gts = np.concatenate(gts, axis=-1)

    action_ft_name = ["UP", "DOWN", "LEFT", "RIGHT"]
    print("Precision,Recall,F1,Threshold,Feature,Offset")
    all_p, all_r, all_f1, all_t, all_ft_names = [], [], [], [], []

    plot = args.plot and args.feature_type == "action"

    all_metrics = get_best_features(args.num_workers, gts, all_preds, plot=plot)

    for offset in all_offsets:
        for layer in all_layers:
            ft_prefix = f"L{layer}" + ("F" if args.feature_from == "sae" else "C")

            for i in range(num_features):
                ft_idx = i + (layer - first_layer) * num_features
                act_idx = probe_info.action_idx if args.feature_from == "probe" and num_features == 1 else ft_idx
                p, r, f1, t = all_metrics[ft_idx]
                all_ft_names.append(f"{ft_prefix}{i}")

                if feature_indices != slice(None):
                    print(f"{p}, {r}, {f1}, {t}, {ft_prefix}{feature_indices[i]}, {offset}")

    if feature_indices == slice(None):
        best_feature = np.argmax([f1 for _, _, f1, _ in all_metrics])
        p, r, f1, t = all_metrics[best_feature]
        offset = all_offsets[best_feature // (num_features * len(all_layers))]
        print(f"{p:.1f}, {r:.1f}, {f1:.1f}, {t:.2f}, {all_ft_names[best_feature]}, {offset}")
