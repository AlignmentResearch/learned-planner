import argparse
import pathlib
import pickle

import einops
import numpy as np
import torch as th
from matplotlib import pyplot as plt
from scipy.stats import bootstrap
from sklearn.multioutput import MultiOutputClassifier

import learned_planner.interp.plot  # noqa
from learned_planner import MODEL_PATH_IN_REPO, ON_CLUSTER
from learned_planner.interp.collect_dataset import DatasetStore
from learned_planner.interp.train_probes import TrainOn
from learned_planner.interp.utils import get_boxoban_cfg, load_jax_model_to_torch, load_probe, play_level
from learned_planner.policies import download_policy_from_huggingface

MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--difficulty", type=str, default="medium")
parser.add_argument("--split", type=str, default="valid")
parser.add_argument("--thinking_steps", type=int, default=6)
parser.add_argument("--num_levels", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=100)
parser.add_argument(
    "--probe_path",
    type=str,
    default="probes/best/boxes_future_direction_map_l-all.pkl",
    help="Path of the probe on disk or on learned-planner huggingface repo",
)
parser.add_argument("--probe_wandb_id", type=str, default="")
parser.add_argument("--dataset_name", type=str, default="boxes_future_direction_map")
parser.add_argument("--output_base_path", type=str, default="iclr_logs/plan_recall/", help="Path to save plots and cache.")

args = parser.parse_args()
difficulty = args.difficulty
split = args.split
if split.lower() == "none" or split.lower() == "null" or not split:
    split = None
thinking_steps = args.thinking_steps
num_levels = args.num_levels
num_envs = args.num_envs

max_steps = 80

boxo_cfg = get_boxoban_cfg(num_envs=num_envs, episode_steps=thinking_steps + max_steps)
boxo_env = boxo_cfg.make()
cfg_th, policy_th = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)

probe, grid_wise = load_probe(args.probe_path, args.probe_wandb_id)
probe_info = TrainOn(grid_wise=grid_wise, dataset_name=args.dataset_name)
probes, probe_infos = [probe], [probe_info]
multioutput = isinstance(probe, MultiOutputClassifier)
if multioutput:
    raise NotImplementedError


def plan_predictions(policy_th=policy_th, probes=probes, probe_infos=probe_infos):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    policy_th = policy_th.to(device)

    all_preds = np.zeros((num_levels, thinking_steps * 3, 10, 10), dtype=int)
    all_labels = np.zeros((num_levels, 10, 10), dtype=int)

    for i in range(int(np.ceil(num_levels / num_envs))):
        out = play_level(
            boxo_env,
            policy_th=policy_th,
            probes=probes,
            probe_train_ons=probe_infos,
            internal_steps=True,
            thinking_steps=thinking_steps,
            max_steps=thinking_steps + max_steps,
        )
        curr_levels = min(num_levels - i * num_envs, num_envs)
        plan = einops.rearrange(out.probe_outs[0], "t i b h w -> b (t i) h w")[:curr_levels, : thinking_steps * 3]
        batch_slice = slice(i * num_envs, (i + 1) * num_envs)
        dss = [
            DatasetStore(None, out.obs[thinking_steps : thinking_steps + out.lengths[i], i].cpu(), solved=out.solved[i].item())
            for i in range(curr_levels)
        ]
        labels = np.array([getattr(ds, probe_info.dataset_name)(multioutput=multioutput)[0].numpy() for ds in dss])
        assert len(labels.shape) == 3, f"Got {labels.shape}"

        all_preds[batch_slice] = plan
        all_labels[batch_slice] = labels

    all_preds = np.transpose(all_preds, (1, 0, 2, 3))
    return all_preds, all_labels

if ON_CLUSTER:
    args.output_base_path = pathlib.Path("/training/") / args.output_base_path
args.output_base_path = pathlib.Path(args.output_base_path)
save_dir = args.output_base_path / f"{args.dataset_name}/{difficulty}_{split}/"
(save_dir / "plots").mkdir(parents=True, exist_ok=True)
if ON_CLUSTER and (save_dir / f"all_preds_{num_levels}.npy").exists():
    print("Loading from cache")
    all_preds = np.load(save_dir / f"all_preds_{num_levels}.npy")
    all_labels = np.load(save_dir / f"all_labels_{num_levels}.npy")
else:
    all_preds, all_labels = plan_predictions()
    np.save(save_dir / f"all_preds_{num_levels}.npy", all_preds)
    np.save(save_dir / f"all_labels_{num_levels}.npy", all_labels)

all_labels = all_labels[None, ...].repeat(thinking_steps * 3, axis=0)
assert all_preds.shape == all_labels.shape, f"Got {all_preds.shape} and {all_labels.shape}"

rng = np.random.default_rng(seed=42)

all_preds = all_preds.reshape(thinking_steps * 3, -1)
all_labels = all_labels.reshape(thinking_steps * 3, -1)


def get_confidence_interval(data, statistic=np.mean, paired=False):
    return bootstrap(
        # (data,),
        data,
        statistic=statistic,
        random_state=rng,
        n_resamples=1000,
        method="basic",
        paired=paired,
    ).confidence_interval


def get_recall(labels, preds):
    return 100 * (labels[labels != -1] == preds[labels != -1]).mean()


def get_precision(labels, preds):
    return 100 * (labels[preds != -1] == preds[preds != -1]).mean()


def get_f1(labels, preds):
    prec = get_precision(labels, preds)
    rec = get_recall(labels, preds)
    return 2 * prec * rec / (prec + rec)


mean_recall = [get_recall(labels, preds) for labels, preds in zip(all_labels, all_preds)]
mean_recall_ci = [
    get_confidence_interval((labels, preds), statistic=get_recall, paired=True) for labels, preds in zip(all_labels, all_preds)
]

mean_precision = [get_precision(labels, preds) for labels, preds in zip(all_labels, all_preds)]
mean_precision_ci = [
    get_confidence_interval((labels, preds), statistic=get_precision, paired=True)
    for labels, preds in zip(all_labels, all_preds)
]

mean_f1 = [get_f1(labels, preds) for labels, preds in zip(all_labels, all_preds)]
mean_f1_ci = [
    get_confidence_interval((labels, preds), statistic=get_f1, paired=True) for labels, preds in zip(all_labels, all_preds)
]

with open(save_dir / "plots" / "plan_recall.pkl", "wb") as f:
    pickle.dump((mean_recall, mean_recall_ci, mean_precision, mean_precision_ci, mean_f1, mean_f1_ci), f)

# exclude_internal_steps = False
fig, ax1 = plt.subplots(1, 1, figsize=(2.0, 1.6))

linewidth, alpha = 0.8, 0.3
ax1.plot(mean_recall, label="Recall", linewidth=linewidth)
ax1.fill_between(range(len(mean_recall)), [ci.low for ci in mean_recall_ci], [ci.high for ci in mean_recall_ci], alpha=alpha)

ax1.plot(mean_precision, label="Precision", linewidth=linewidth)
ax1.fill_between(
    range(len(mean_precision)), [ci.low for ci in mean_precision_ci], [ci.high for ci in mean_precision_ci], alpha=alpha
)

ax1.plot(mean_f1, label="F1", linewidth=linewidth)
ax1.fill_between(range(len(mean_f1)), [ci.low for ci in mean_f1_ci], [ci.high for ci in mean_f1_ci], alpha=alpha)

ax1.set_xticks(np.arange(2, thinking_steps * 3, 3), minor=False)
ax1.set_xticks(np.arange(0, thinking_steps * 3, 1), minor=True)
ax1.set_xticklabels(np.arange(1, thinking_steps + 1), minor=False)
ax1.set_xlabel("Thinking Steps")
ax1.grid(True)
plt.legend()
plt.savefig(save_dir / "plots" / "plan_recall.pdf")
