import argparse
import os
import pathlib
import pickle

import einops
import numpy as np
import torch as th
from cleanba.environments import BoxobanConfig, EnvpoolBoxobanConfig
from gym_sokoban.envs.sokoban_env import CHANGE_COORDINATES
from matplotlib import pyplot as plt
from scipy.stats import bootstrap
from sklearn.multioutput import MultiOutputClassifier

import learned_planner.interp.plot  # noqa
from learned_planner.interp.collect_dataset import DatasetStore
from learned_planner.interp.train_probes import TrainOn
from learned_planner.interp.utils import load_jax_model_to_torch, load_probe, play_level
from learned_planner.policies import download_policy_from_huggingface

on_cluster = os.path.exists("/training")
LP_DIR = pathlib.Path(__file__).parent.parent.parent

MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000/"  # DRC(3, 3) 2B checkpoint
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)
if on_cluster:
    BOXOBAN_CACHE = pathlib.Path("/training/.sokoban_cache/")
else:
    BOXOBAN_CACHE = pathlib.Path(__file__).parent.parent.parent / "training/.sokoban_cache/"

parser = argparse.ArgumentParser()
parser.add_argument("--difficulty", type=str, default="medium")
parser.add_argument("--split", type=str, default="valid")
parser.add_argument("--thinking_steps", type=int, default=6)
parser.add_argument("--num_levels", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--probe_path", type=str, default="")
parser.add_argument("--probe_wandb_id", type=str, default="vb6474rg")
parser.add_argument("--dataset_name", type=str, default="boxes_future_direction_map")

args = parser.parse_args()
difficulty = args.difficulty
split = args.split
if split.lower() == "none" or split.lower() == "null" or not split:
    split = None
thinking_steps = args.thinking_steps
num_levels = args.num_levels
num_envs = args.num_envs

extra_kwargs = dict()
if on_cluster:
    cfg_cls = EnvpoolBoxobanConfig
    extra_kwargs = dict(load_sequentially=True)
else:
    cfg_cls = BoxobanConfig
    extra_kwargs = dict(asynchronous=False, tinyworld_obs=True)

boxo_cfg = cfg_cls(
    cache_path=BOXOBAN_CACHE,
    num_envs=num_envs,
    max_episode_steps=thinking_steps,
    min_episode_steps=thinking_steps,
    difficulty=difficulty,
    split=split,
    **extra_kwargs,
)
boxo_env = boxo_cfg.make()
cfg_th, policy_th = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)

probe, grid_wise = load_probe(args.probe_path, args.probe_wandb_id)
probe_info = TrainOn(grid_wise=grid_wise, dataset_name=args.dataset_name)
probes, probe_infos = [probe], [probe_info]
multioutput = isinstance(probe, MultiOutputClassifier)
if multioutput:
    raise NotImplementedError


def non_empty_squares_in_plan(plan):
    assert plan.ndim == (5 if multioutput else 4), f"Got {plan.shape}"
    if multioutput:
        raise NotImplementedError

    non_empty_squares = (plan >= 0).sum(axis=(-1, -2))  # -1 is empty square
    return non_empty_squares


def continuous_chains_in_plan(plan, boxes, targets=None):
    """Total Continuous chain length starting from boxes"""
    assert plan.ndim == (5 if multioutput else 4), f"Got {plan.shape}"
    if multioutput:
        raise NotImplementedError

    total_chain_length = np.zeros(plan.shape[:2])
    total_intersects, total_ends_on = np.zeros(plan.shape[:2]), np.zeros(plan.shape[:2])
    for batch in range(plan.shape[0]):
        batch_targets = targets[batch] if targets is not None else None
        for seq in range(plan.shape[1]):
            for box in boxes[batch]:
                chain_lenth, intersects, ends_on = chain_length_from_box(plan[batch, seq], box, batch_targets)
                total_chain_length[batch, seq] += chain_lenth
                total_intersects[batch, seq] += intersects
                total_ends_on[batch, seq] += ends_on
    total_intersects /= boxes.shape[1]
    total_ends_on /= boxes.shape[1]
    return total_chain_length, total_intersects, total_ends_on


def chain_length_from_box(plan, box, targets=None):
    """Continuous chain length starting from box"""
    assert plan.ndim == 2, f"Got {plan.shape}"
    current_direction = plan[*box]
    chain_length = 0
    covered = set([10 * box[0] + box[1]])
    intersects_target, ends_on_target = False, False
    while current_direction != -1:
        chain_length += 1
        new_box = box + CHANGE_COORDINATES[current_direction]
        current_direction = plan[*new_box]
        if 10 * new_box[0] + new_box[1] in covered:
            break
        covered.add(10 * new_box[0] + new_box[1])
        if targets is not None:
            if (new_box == targets).all(axis=1).any():
                if current_direction == -1:
                    ends_on_target = True
                else:
                    intersects_target = True

    return chain_length, intersects_target, ends_on_target


def plan_quality(policy_th=policy_th, probes=probes, probe_infos=probe_infos):
    non_empty_squares = np.zeros((num_levels, thinking_steps * 3))
    continuous_chains = np.zeros((num_levels, thinking_steps * 3))
    interects_target = np.zeros((num_levels, thinking_steps * 3))
    ends_on_target = np.zeros((num_levels, thinking_steps * 3))

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    policy_th = policy_th.to(device)

    for i in range(int(np.ceil(num_levels / num_envs))):
        out = play_level(
            boxo_env,
            policy_th=policy_th,
            probes=probes,
            probe_train_ons=probe_infos,
            internal_steps=True,
            thinking_steps=thinking_steps,
            max_steps=thinking_steps,
        )
        curr_levels = min(num_levels - i * num_envs, num_envs)
        plan = einops.rearrange(out.probe_outs[0], "t i b h w -> b (t i) h w")[:curr_levels]
        boxes = np.stack([DatasetStore.get_box_position_per_step(out.obs[0, i].cpu()) for i in range(curr_levels)])
        targets = np.stack([DatasetStore.get_target_positions_from_obs(out.obs[0, i].cpu()) for i in range(curr_levels)])
        batch_slice = slice(i * num_envs, (i + 1) * num_envs)
        non_empty_squares[batch_slice] = non_empty_squares_in_plan(plan)
        chain_lengths, intersects, ends_on = continuous_chains_in_plan(plan, boxes, targets)
        continuous_chains[batch_slice] = chain_lengths
        interects_target[batch_slice] = intersects
        ends_on_target[batch_slice] = ends_on

    return non_empty_squares, continuous_chains, interects_target, ends_on_target


save_dir = pathlib.Path("/training/iclr_logs/") if on_cluster else LP_DIR / "plot/interp/"
save_dir = save_dir / f"plan_quality/{args.dataset_name}/{difficulty}_{split}/"
(save_dir / "plots").mkdir(parents=True, exist_ok=True)
if on_cluster and (save_dir / f"non_empty_squares_{num_levels}.npy").exists():
    print("Loading from cache")
    non_empty_squares = np.load(save_dir / f"non_empty_squares_{num_levels}.npy")
    continuous_chains = np.load(save_dir / f"continuous_chains_{num_levels}.npy")
    interects_target = np.load(save_dir / f"interects_target_{num_levels}.npy")
    ends_on_target = np.load(save_dir / f"ends_on_target_{num_levels}.npy")
else:
    non_empty_squares, continuous_chains, interects_target, ends_on_target = plan_quality()
    np.save(save_dir / f"non_empty_squares_{num_levels}.npy", non_empty_squares)
    np.save(save_dir / f"continuous_chains_{num_levels}.npy", continuous_chains)
    np.save(save_dir / f"interects_target_{num_levels}.npy", interects_target)
    np.save(save_dir / f"ends_on_target_{num_levels}.npy", ends_on_target)

rng = np.random.default_rng(seed=42)


def get_confidence_interval(data):
    return bootstrap(
        (data,), statistic=np.mean, random_state=rng, n_resamples=1000, vectorized=True, method="basic"
    ).confidence_interval


non_empty_squares_ci = get_confidence_interval(non_empty_squares)
continuous_chains_ci = get_confidence_interval(continuous_chains)

non_empty_squares = non_empty_squares.mean(axis=0)
continuous_chains = continuous_chains.mean(axis=0)

exclude_internal_steps = False
if exclude_internal_steps:
    fig, [ax1, ax2] = plt.subplots(1, 2)
else:
    fig, ax1 = plt.subplots(1, 1, figsize=(2.0, 1.6))


with open(save_dir / "plots" / "plan_quality.pkl", "wb") as f:
    pickle.dump(
        {
            "non_empty_squares": non_empty_squares,
            "continuous_chains": continuous_chains,
            "non_empty_squares_ci": non_empty_squares_ci,
            "continuous_chains_ci": continuous_chains_ci,
        },
        f,
    )

linewidth, alpha = 0.8, 0.3
ax1.plot(non_empty_squares, label="Positive squares", linewidth=linewidth)
ax1.fill_between(range(len(non_empty_squares)), non_empty_squares_ci.low, non_empty_squares_ci.high, alpha=alpha)
ax1.plot(continuous_chains, label="Cont. chain length", linewidth=linewidth)
ax1.fill_between(range(len(continuous_chains)), continuous_chains_ci.low, continuous_chains_ci.high, alpha=alpha)

ax1.set_xticks(np.arange(2, thinking_steps * 3, 3), minor=False)
ax1.set_xticks(np.arange(0, thinking_steps * 3, 1), minor=True)
ax1.set_xticklabels(np.arange(1, thinking_steps + 1), minor=False)
ax1.set_xlabel("Thinking Steps")
ax1.grid(True)
if exclude_internal_steps:
    ax1.set_title("Including internal steps")


if exclude_internal_steps:
    non_empty_squares = non_empty_squares[2::3]
    continuous_chains = continuous_chains[2::3]
    ax2.plot(non_empty_squares, label="Non-empty squares", linewidth=linewidth)
    ax2.fill_between(
        range(len(non_empty_squares)), non_empty_squares_ci.low[2::3], non_empty_squares_ci.high[2::3], alpha=alpha
    )
    ax2.plot(continuous_chains, label="Cont. chain length", linewidth=linewidth)
    ax2.fill_between(
        range(len(continuous_chains)), continuous_chains_ci.low[2::3], continuous_chains_ci.high[2::3], alpha=alpha
    )
    ax2.set_title("Excluding internal steps")
plt.legend(handlelength=1)
plt.savefig(save_dir / "plots" / "plan_quality.pdf")

interects_target_ci = get_confidence_interval(interects_target)
ends_on_target_ci = get_confidence_interval(ends_on_target)

interects_target = interects_target.mean(axis=0)
ends_on_target = ends_on_target.mean(axis=0)

fig, ax1 = plt.subplots(1, 1)
ax1.plot(interects_target, label="Intersects with target", linewidth=linewidth)
ax1.fill_between(range(len(interects_target)), interects_target_ci.low, interects_target_ci.high, alpha=alpha)
ax1.plot(ends_on_target, label="Ends on target", linewidth=linewidth)
ax1.fill_between(range(len(ends_on_target)), ends_on_target_ci.low, ends_on_target_ci.high, alpha=alpha)

ax1.set_xticks(np.arange(2, thinking_steps * 3, 3), minor=False)
ax1.set_xticks(np.arange(0, thinking_steps * 3, 1), minor=True)
ax1.set_xticklabels(np.arange(1, thinking_steps + 1), minor=False)
ax1.set_xlabel("Thinking Steps")
ax1.grid(True)
plt.legend()
plt.savefig(save_dir / "plots" / "chains_on_target.pdf")
