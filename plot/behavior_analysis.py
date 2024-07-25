# %%
import argparse
import contextlib
import dataclasses
import json
import os
import pathlib
import pickle
import re
import tarfile
from functools import partial
from typing import Any, Dict

import databind
import farconf
import flax
import huggingface_hub
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import wandb
from cairosvg import svg2png
from cleanba.cleanba_impala import make_optimizer, unreplicate
from cleanba.config import Args
from cleanba.convlstm import ConvLSTMConfig
from cleanba.environments import BoxobanConfig, EnvConfig, EnvpoolBoxobanConfig
from cleanba.network import Policy
from flax.training.train_state import TrainState
from matplotlib import pyplot as plt
from render_svg import episode_obs_to_svgs
from tqdm import tqdm

wandb.init(mode="disabled")

style = {
    "font.family": "serif",
    "font.serif": "Times New Roman",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "legend.fontsize": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (3.25, 2),
    "figure.constrained_layout.use": True,
}
matplotlib.rcParams.update(style)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--local_or_hgf_repo_path",
    type=str,
    default="drc33/bkynosqi/cp_2002944000",
    help="Path to a local directory containing model files or"
    " a relative path to a model on the learned-planner huggingface hub:"
    " https://huggingface.co/AlignmentResearch/learned-planner/",
)
parser.add_argument("--output_base_path", type=str, default="/training/cleanba/logs/", help="Path to save plots and cache.")
parser.add_argument(
    "--boxoban_cache_path",
    type=str,
    default="/training/.sokoban_cache/",
    help="Path containing the boxoban levels.",
)
parser.add_argument("--split", type=str, default="valid", choices=["valid", "test", ""])
parser.add_argument("--difficulty", type=str, default="medium", choices=["unfiltered", "medium", "hard"])
parser.add_argument("--use_envpool", action="store_true")
parser.add_argument("--save_video", action="store_true")

args = parser.parse_args()
local_or_hgf_repo_path = args.local_or_hgf_repo_path
output_base_path = args.output_base_path
boxoban_cache_path = args.boxoban_cache_path
split = args.split if args.split else None
difficulty = args.difficulty
envpool = args.use_envpool
save_video = args.save_video

output_base_path = pathlib.Path(output_base_path)

if not pathlib.Path(local_or_hgf_repo_path).exists():
    try:
        output_base_path = output_base_path / "_".join(local_or_hgf_repo_path.split("/"))
        local_or_hgf_repo_path = huggingface_hub.snapshot_download(
            "AlignmentResearch/learned-planner",
            allow_patterns=[local_or_hgf_repo_path + "/*"],
        )
    except huggingface_hub.errors.HFValidationError:
        raise ValueError(f"Model {local_or_hgf_repo_path} not found in local cache or on the hub")

local_or_hgf_repo_path = pathlib.Path(local_or_hgf_repo_path)

plots_dir = output_base_path / "plots"
plots_dir.mkdir(exist_ok=True, parents=True)

baseline_steps = best_steps = 100000  # absurdly high number so using it before definition errors


# %%
def eval_with_noop(
    envs,
    max_steps,
    policy,
    get_action_fn,
    params,
    key,
    level_file_idx,
    level_idx,
    noop_start,
    noop_length,
    thinking_steps=0,
    action_at_cycle_end=None,
):
    key, carry_key, obs_reset_key = jax.random.split(key, 3)
    metrics = {}

    episode_starts_no = jnp.zeros(envs.num_envs, dtype=jnp.bool_)
    temperature = 0.0

    reset_key, sub_reset_key = jax.random.split(obs_reset_key)
    reset_seed = int(jax.random.randint(sub_reset_key, (), minval=0, maxval=2**31 - 2))
    obs, level_infos = envs.reset(seed=reset_seed, options={"level_file_idx": level_file_idx, "level_idx": level_idx})
    assert envs.num_envs == 1, "multiple envs not supported with reset using level index."
    assert (
        level_infos["level_file_idx"] == level_file_idx
    ), f"Expected level_file_idx {level_file_idx}, got {level_infos['level_file_idx']}"
    assert level_infos["level_idx"] == level_idx, f"Expected level_idx {level_idx}, got {level_infos['level_idx']}"

    # reset the carry here so we can use `episode_starts_no` later
    carry = policy.apply(params, carry_key, obs.shape, method=policy.initialize_carry)

    eps_done = np.zeros(envs.num_envs, dtype=np.bool_)
    episode_success = np.zeros(envs.num_envs, dtype=np.bool_)
    episode_returns = np.zeros(envs.num_envs, dtype=np.float64)
    episode_lengths = np.zeros(envs.num_envs, dtype=np.int64)
    episode_obs = np.zeros((max_steps + 1, *obs.shape), dtype=np.int64)
    episode_acts = np.zeros((max_steps, envs.num_envs), dtype=np.int64)
    episode_rewards = np.zeros((max_steps, envs.num_envs), dtype=np.float64)

    episode_obs[0] = obs
    i = 0
    thinking_steps = thinking_steps if thinking_steps > 0 else noop_length

    while not eps_done:
        if i >= max_steps:
            break

        if i == noop_start:
            # Update the carry with the same observation many times
            for _ in range(thinking_steps):
                carry, _, _, key = get_action_fn(params, carry, obs, episode_starts_no, key, temperature=temperature)

        if i == noop_start and action_at_cycle_end is not None:
            action = [action_at_cycle_end]
        else:
            carry, action, _, key = get_action_fn(params, carry, obs, episode_starts_no, key, temperature=temperature)

        cpu_action = np.asarray(action)
        obs, rewards, terminated, truncated, infos = envs.step(cpu_action)
        episode_returns[~eps_done] += rewards[~eps_done]
        episode_lengths[~eps_done] += 1
        episode_success[~eps_done] |= terminated[~eps_done]  # If episode terminates it's a success

        episode_obs[i + 1, ~eps_done] = obs[~eps_done]
        episode_acts[i, ~eps_done] = cpu_action[~eps_done]
        episode_rewards[i, ~eps_done] = rewards[~eps_done]

        # Set as done the episodes which are done
        eps_done |= truncated | terminated
        i += 1

    metrics["episode_returns"] = episode_returns
    metrics["episode_lengths"] = episode_lengths
    metrics["episode_success"] = episode_success
    metrics["episode_obs"] = episode_obs.squeeze(1)[: episode_lengths.item()] if episode_obs.shape[1] == 1 else episode_obs
    metrics["episode_acts"] = episode_acts
    metrics["episode_rewards"] = episode_rewards
    metrics["level_infos"] = level_infos
    return metrics


@dataclasses.dataclass
class EvalConfig:
    env: EnvConfig
    n_episode_multiple: int = 1
    steps_to_think: list[int] = dataclasses.field(default_factory=lambda: [0])
    temperature: float = 0.0

    safeguard_max_episode_steps: int = 30000

    def run(self, policy: Policy, get_action_fn, params, *, key: jnp.ndarray) -> dict[str, float]:
        assert isinstance(self.env, EnvpoolBoxobanConfig)
        key, carry_key = jax.random.split(key, 2)
        max_steps = min(self.safeguard_max_episode_steps, self.env.max_episode_steps)
        episode_starts_no = jnp.zeros(self.env.num_envs, dtype=jnp.bool_)

        metrics = {}
        try:
            for steps_to_think in tqdm(self.steps_to_think):
                all_episode_returns = []
                all_episode_lengths = []
                all_episode_successes = []
                all_obs = []
                all_acts = []
                all_rewards = []
                all_level_infos = []
                # envs = dataclasses.replace(self.env, seed=env_seed).make()
                for minibatch_idx in range(self.n_episode_multiple):
                    # Re-create the environments, so we start at the beginning of the batch
                    with contextlib.closing(self.env.make()) as envs:
                        obs, level_infos = envs.reset()
                        # Reset more than once so we get to the Nth batch of levels
                        for _ in range(minibatch_idx):
                            obs, level_infos = envs.reset()

                        # reset the carry here so we can use `episode_starts_no` later
                        carry = policy.apply(params, carry_key, obs.shape, method=policy.initialize_carry)

                        # Update the carry with the initial observation many times
                        for think_step in range(steps_to_think):
                            carry, _, _, key = get_action_fn(
                                params, carry, obs, episode_starts_no, key, temperature=self.temperature
                            )

                        eps_done = np.zeros(envs.num_envs, dtype=np.bool_)
                        episode_success = np.zeros(envs.num_envs, dtype=np.bool_)
                        episode_returns = np.zeros(envs.num_envs, dtype=np.float64)
                        episode_lengths = np.zeros(envs.num_envs, dtype=np.int64)
                        episode_obs = np.zeros((max_steps + 1, *obs.shape), dtype=np.int64)
                        episode_acts = np.zeros((max_steps, envs.num_envs), dtype=np.int64)
                        episode_rewards = np.zeros((max_steps, envs.num_envs), dtype=np.float64)

                        episode_obs[0] = obs
                        i = 0
                        while not np.all(eps_done):
                            if i >= self.safeguard_max_episode_steps:
                                break
                            carry, action, _, key = get_action_fn(
                                params, carry, obs, episode_starts_no, key, temperature=self.temperature
                            )

                            cpu_action = np.asarray(action)
                            obs, rewards, terminated, truncated, infos = envs.step(cpu_action)
                            episode_returns[~eps_done] += rewards[~eps_done]
                            episode_lengths[~eps_done] += 1
                            episode_success[~eps_done] |= terminated[~eps_done]  # If episode terminates it's a success

                            episode_obs[i + 1, ~eps_done] = obs[~eps_done]
                            episode_acts[i, ~eps_done] = cpu_action[~eps_done]
                            episode_rewards[i, ~eps_done] = rewards[~eps_done]

                            # Set as done the episodes which are done
                            eps_done |= truncated | terminated
                            i += 1

                        all_episode_returns.append(episode_returns)
                        all_episode_lengths.append(episode_lengths)
                        all_episode_successes.append(episode_success)

                        all_obs += [episode_obs[: episode_lengths[i], i] for i in range(envs.num_envs)]
                        all_acts += [episode_acts[: episode_lengths[i], i] for i in range(envs.num_envs)]
                        all_rewards += [episode_rewards[: episode_lengths[i], i] for i in range(envs.num_envs)]

                        all_level_infos.append(level_infos)

                all_episode_returns = np.concatenate(all_episode_returns)
                all_episode_lengths = np.concatenate(all_episode_lengths)
                all_episode_successes = np.concatenate(all_episode_successes)
                if isinstance(self.env, BoxobanConfig):
                    all_level_infos = {
                        k: np.concatenate([d[k] for d in all_level_infos])
                        for k in all_level_infos[0].keys()
                        if not k.startswith("_")
                    }
                else:
                    all_level_infos = {
                        k: np.concatenate([d[k] for d in all_level_infos]) for k in all_level_infos[0].keys() if "level" in k
                    }
                    total = set(zip(all_level_infos["level_file_idx"], all_level_infos["level_idx"]))
                    print(f"Total levels: {len(total)}")

                metrics.update(
                    {
                        f"{steps_to_think:02d}_episode_returns": float(np.mean(all_episode_returns)),
                        f"{steps_to_think:02d}_episode_lengths": float(np.mean(all_episode_lengths)),
                        f"{steps_to_think:02d}_episode_successes": float(np.mean(all_episode_successes)),
                        f"{steps_to_think:02d}_num_episodes": len(all_episode_returns),
                        f"{steps_to_think:02d}_all_episode_info": dict(
                            episode_returns=all_episode_returns,
                            episode_lengths=all_episode_lengths,
                            episode_successes=all_episode_successes,
                            episode_obs=all_obs,
                            episode_acts=all_acts,
                            episode_rewards=all_rewards,
                            level_infos=all_level_infos,
                        ),
                    }
                )
                print(f"Success rate for {steps_to_think} steps: {np.mean(all_episode_successes)}")
        finally:
            envs.close()
        return metrics


all_episode_info: Dict[int, Dict[str, Any]]


def save_level_video(level_idx, base_dir="./", force=False):
    base_dir = pathlib.Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    file_path = base_dir / f"{level_idx}.mp4"
    if file_path.exists() and not force:
        return
    obs_baseline = np.moveaxis(all_episode_info[baseline_steps_idx]["episode_obs"][level_idx], 1, 3)
    obs_best = np.moveaxis(all_episode_info[best_steps_idx]["episode_obs"][level_idx], 1, 3)
    num_obs_baseline = len(obs_baseline)
    num_obs_best = len(obs_best)
    max_obs = max(num_obs_baseline, num_obs_best)
    fig, axs = plt.subplots(1, 2)
    ax1, ax2 = axs
    ax1.set_title(f"{steps_to_think[baseline_steps_idx]} think steps")
    ax2.set_title(f"{steps_to_think[best_steps_idx]} think steps")
    im1 = ax1.imshow(obs_baseline[0])
    im2 = ax2.imshow(obs_best[0])
    title = fig.suptitle(f"Level {level_idx}: Step 0")

    def update_frame(j):
        baseline_img = obs_baseline[min(len(obs_baseline) - 1, j)]
        # ax1.imshow(baseline_img)
        im1.set(data=baseline_img)
        best_img = obs_best[min(len(obs_best) - 1, j)]
        # ax2.imshow(best_img)
        im2.set(data=best_img)
        title.set_text(f"Level {level_idx}: Step {j}")
        return (im1, im2, title)

    anim = animation.FuncAnimation(
        fig,
        update_frame,  # type: ignore
        frames=max_obs,
        interval=1,
        repeat=False,
    )
    plt.tight_layout()
    anim.save(file_path, fps=3)
    print(f"Level {level_idx} saved")


def load_train_state(dir: pathlib.Path, env):
    with open(dir / "cfg.json", "r") as f:
        args_dict = json.load(f)
    try:
        update_step = args_dict["update_step"]
    except KeyError:
        update_step = 1
    try:
        loaded_cfg = args_dict["cfg"]
    except KeyError:
        loaded_cfg = args_dict

    try:
        args = farconf.from_dict(loaded_cfg, Args)
    except databind.core.converter.ConversionError as e:
        if (m := re.fullmatch(r"^encountered extra keys: \{(.*)\}$", e.message)) is not None:
            keys_to_remove = {k.strip().strip("'") for k in m.group(1).split(",")}
            print("Removing keys ", keys_to_remove)
            for k in keys_to_remove:
                del loaded_cfg[k]
            args = farconf.from_dict(loaded_cfg, Args)
        else:
            raise

    _, _, params = args.net.init_params(env, jax.random.PRNGKey(1234))

    local_batch_size = int(args.local_num_envs * args.num_steps * args.num_actor_threads * len(args.actor_device_ids))

    target_state = TrainState.create(
        apply_fn=None,
        params=params,
        tx=make_optimizer(args, params, total_updates=args.total_timesteps // local_batch_size),
    )

    with open(dir / "model", "rb") as f:
        train_state = flax.serialization.from_bytes(target_state, f.read())
    assert isinstance(train_state, TrainState)
    train_state = unreplicate(train_state)
    if isinstance(args.net, ConvLSTMConfig):
        for i in range(args.net.n_recurrent):
            train_state.params["params"]["network_params"][f"cell_list_{i}"]["fence"]["kernel"] = np.sum(
                train_state.params["params"]["network_params"][f"cell_list_{i}"]["fence"]["kernel"],
                axis=2,
                keepdims=True,
            )
    return args, train_state, update_step


def load_policy(path, prng_key=jax.random.PRNGKey(0)):
    args, train_state, _ = load_train_state(path, env_cfg.env.make())
    policy, _, _ = args.net.init_params(env_cfg.env.make(), prng_key)
    get_action_fn = jax.jit(partial(policy.apply, method=policy.get_action), static_argnames="temperature")
    params = train_state.params
    return policy, get_action_fn, params


def get_svg(serial_idx, timestep=0):
    svg_dir = pathlib.Path("svg/")
    svg_dir.mkdir(exist_ok=True)
    obs = all_episode_info[0]["episode_obs"][serial_idx][timestep]
    obs = np.moveaxis(obs, 0, 2)
    assert obs.shape == (10, 10, 3)
    # save as svg
    fig, ax = plt.subplots()
    ax.imshow(obs)
    ax.axis("off")
    file_idx = all_episode_info[0]["level_infos"]["level_file_idx"][serial_idx]
    lev_idx = all_episode_info[0]["level_infos"]["level_idx"][serial_idx]
    plt.savefig(svg_dir / f"{serial_idx}_time-{timestep}_file-{file_idx}_level-{lev_idx}.svg", format="svg")
    plt.show()


# %%
dataset_name = difficulty if split is None else f"{split}_{difficulty}"
steps_to_think_all = [0, 1, 2, 4, 6, 8, 10, 12, 16]
episode_steps = 120


def get_cfg():
    common_args = dict(
        cache_path=pathlib.Path(boxoban_cache_path),
        min_episode_steps=episode_steps,
        max_episode_steps=episode_steps,
        split=split,
        difficulty=difficulty,
        seed=0,
    )
    if not envpool:
        common_args["tinyworld_obs"] = True
        common_args["asynchronous"] = False
        if dataset_name == "test_unfiltered":
            return EvalConfig(
                BoxobanConfig(
                    num_envs=100,
                    **common_args,
                ),
                n_episode_multiple=10,  # only 1000 levels in unfil test set
                steps_to_think=steps_to_think_all,
            )
        elif dataset_name == "valid_medium":
            return EvalConfig(
                BoxobanConfig(
                    num_envs=500,
                    **common_args,
                ),
                n_episode_multiple=10,
                steps_to_think=steps_to_think_all,
            )
        elif dataset_name == "hard":
            return EvalConfig(
                BoxobanConfig(
                    num_envs=119,
                    **common_args,
                ),
                n_episode_multiple=28,
                steps_to_think=steps_to_think_all,
            )
        else:
            return EvalConfig(
                BoxobanConfig(
                    num_envs=500,
                    **common_args,
                ),
                n_episode_multiple=1,
                steps_to_think=steps_to_think_all,
            )
    else:
        common_args["load_sequentially"] = True
        if dataset_name == "test_unfiltered":
            return EvalConfig(
                EnvpoolBoxobanConfig(
                    num_envs=100,
                    n_levels_to_load=1000,
                    **common_args,
                ),
                n_episode_multiple=10,  # only 1000 levels in unfil test set
                steps_to_think=steps_to_think_all,
            )
        elif dataset_name == "valid_medium":
            return EvalConfig(
                EnvpoolBoxobanConfig(
                    num_envs=500,
                    n_levels_to_load=5000,
                    **common_args,
                ),
                n_episode_multiple=10,
                steps_to_think=steps_to_think_all,
            )
        elif dataset_name == "hard":
            return EvalConfig(
                EnvpoolBoxobanConfig(
                    num_envs=119,
                    n_levels_to_load=3332,
                    **common_args,
                ),
                n_episode_multiple=28,
                steps_to_think=steps_to_think_all,
            )
        else:
            return EvalConfig(
                EnvpoolBoxobanConfig(
                    num_envs=500,
                    **common_args,
                ),
                n_episode_multiple=1,
                steps_to_think=steps_to_think_all,
            )


env_cfg = get_cfg()

# %% [markdown]
# ### Planning Effect during Training

# %%

steps_to_think_for_pe = [0, 2, 4, 8, 12, 16]
network_names = ["drc33", "drc11"]

success_rates = {}

for network_name in network_names:
    df = pd.read_csv(f"data/{network_name}_{dataset_name}_success_across_thinking_steps.csv", index_col="Step")
    select_columns = [col.endswith("_episode_successes") for col in df.columns]
    df = df.loc[:, select_columns]
    run_name = df.columns[0].split(" - ")[0]
    new_cols = [int(re.search(r"(\d+)_episode_successes$", col).group(1)) for col in df.columns]
    df.columns = new_cols

    success_rates[network_name] = df.iloc[-1].copy()

    df = df[steps_to_think_for_pe]

    df_resnet = pd.read_csv(f"data/resnet_{dataset_name}_success_across_thinking_steps.csv", index_col="Step")

    per_step = df
    # per_step = per_step - per_step.loc[0]

    fig, axes = plt.subplots(2, 1, figsize=(3.25, 2.5), sharex=True, height_ratios=[3, 1])

    for i in range(len(per_step.T)):
        this_step_proportion = i / len(per_step.T)
        per_step.iloc[:, i].plot(color=plt.get_cmap("viridis")(this_step_proportion), ax=axes[0], legend=False)

    resnet_run_name = df_resnet.columns[0].split(" - ")[0]
    df_resnet[f"{resnet_run_name} - {dataset_name}/00_episode_successes"].plot(color="C1", ax=axes[0], label="resnet")

    (per_step.max(axis=1) - per_step[0]).plot(ax=axes[1])

    axes[0].grid(True)
    axes[1].grid(True)

    axes[1].set_xlabel("Environment steps, training")
    y_label = dict(test_unfiltered="Test-unfiltered", valid_medium="Val-medium", hard="Hard")[dataset_name]
    axes[0].set_ylabel(y_label + " solved")
    axes[1].set_ylabel("Plan. Effect")
    axes[0].legend(ncols=3, prop={"size": 8})
    axes[0].set_xlim((998400.0, 2002944000.0))

    plt.savefig(plots_dir / f"{network_name}_{dataset_name}_curve.pdf", format="pdf")
    plt.show()
    plt.close()

    success_rates["resnet"] = df_resnet.iloc[-1].iloc[0]
    print("success rate for resnet", dataset_name, ":", success_rates["resnet"])
    print("success rate for", network_name, dataset_name, ":", success_rates[network_name][0])


# %%
for network_name in ["drc33", "drc11"]:
    episode_successes = [success_rates[network_name][step] for step in steps_to_think_all]

    fig, ax1 = plt.subplots(figsize=(3.25, 2))
    ax1.grid(True)
    ax1.set_xlabel("Number of extra thinking steps at episode start")
    ax1.set_ylabel("Success Rate")
    x_steps_to_think = np.array(steps_to_think_all)
    ax1.plot(
        x_steps_to_think[: len(episode_successes)],
        episode_successes,
        color="C0",
        label="DRC(1, 1)" if network_name == "drc11" else "DRC(3, 3)",
    )
    ax1.tick_params(axis="y")

    x_min = 0.04
    x_max = 1 - x_min

    ax1.axhline(success_rates["resnet"], color="C1", linestyle="dotted", label="ResNet")
    ax1.set_xlim(x_steps_to_think[0], x_steps_to_think[-1])
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x)))
    ax1.xaxis.set_major_locator(ticker.FixedLocator(x_steps_to_think))
    ax1.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    # ax1.legend(bbox_to_anchor=(1.1, 1.4, -0.1, -0.1), ncol=2)

    plt.tight_layout()
    plt.savefig(plots_dir / f"success_vs_steps_to_think_{network_name}_{dataset_name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    print("Planning effect for", network_name, dataset_name, ":", max(episode_successes) - episode_successes[0])

# %%[markdown]
### Medium level success rate when DRC and ResNet are same on test set

# %%
df_drc_val = pd.read_csv("data/drc33_valid_medium_success_across_thinking_steps.csv", index_col="Step")
df_resnet_val = pd.read_csv("data/resnet_valid_medium_success_across_thinking_steps.csv", index_col="Step")
df_drc_test = pd.read_csv("data/drc33_test_unfiltered_success_across_thinking_steps.csv", index_col="Step")
df_resnet_test = pd.read_csv("data/resnet_test_unfiltered_success_across_thinking_steps.csv", index_col="Step")
resnet_best_on_test = df_resnet_test.iloc[-1].iloc[0]

steps_when_drc_resnet_same = df_drc_test[df_drc_test.iloc[:, 0] >= resnet_best_on_test].iloc[0].name
print("Training Steps where DRC33 is same as best ResNet on test:", steps_when_drc_resnet_same)
print(
    f"DRC33 @ {steps_when_drc_resnet_same//(10**6)}M val medium success rate:",
    df_drc_val.loc[steps_when_drc_resnet_same].iloc[0],
)
print("ResNet val medium success rate (best):", df_resnet_val.iloc[-1].iloc[0])
print("best thinking step val medium success rate:", max(df_drc_val.loc[steps_when_drc_resnet_same]))

# %%[markdown]
### Load Cache (or run the evaluation and get the cache)

# %%

policy_key, env_key = jax.random.split(jax.random.PRNGKey(0), 2)
if not (output_base_path / f"{dataset_name}_log_dict.pkl").exists():
    output_base_path.mkdir(parents=True, exist_ok=True)
    policy, get_action_fn, params = load_policy(local_or_hgf_repo_path, policy_key)
    log_dict = env_cfg.run(policy, get_action_fn, params, key=env_key)
    all_episode_info = [log_dict.pop(f"{step_to_think:02d}_all_episode_info") for step_to_think in steps_to_think_all]

    with open(output_base_path / f"{dataset_name}_log_dict.pkl", "wb") as f:
        pickle.dump(log_dict, f)
    with open(output_base_path / f"{dataset_name}_all_episode_info.pkl", "wb") as f:
        pickle.dump(all_episode_info, f)

else:
    print("Loading from cache")
    with open(output_base_path / f"{dataset_name}_log_dict.pkl", "rb") as f:
        log_dict = pickle.load(f)
    with open(output_base_path / f"{dataset_name}_all_episode_info.pkl", "rb") as f:
        all_episode_info = pickle.load(f)


# %%
baseline_steps = 0
best_steps = success_rates["drc33"].idxmax()
steps_to_think = [0, 2, 4, 6, 8, 12]
baseline_steps_idx = steps_to_think.index(baseline_steps)
best_steps_idx = steps_to_think.index(best_steps)
print("Best steps to think:", best_steps)

# %%
num_levels = len(all_episode_info[0]["episode_successes"])
improved_level_list = []
impaired_level_list = []
solved_better_returns = []
solved_worse_returns = []
unsolved_better_same_returns = []
unsolved_worse_returns = []
same_return_and_solve = []
for i in range(len(all_episode_info[0]["episode_successes"])):
    solved_after_thinking = (
        all_episode_info[baseline_steps_idx]["episode_successes"][i] < all_episode_info[best_steps_idx]["episode_successes"][i]
    )
    messed_up_after_thinking = (
        all_episode_info[baseline_steps_idx]["episode_successes"][i] > all_episode_info[best_steps_idx]["episode_successes"][i]
    )

    solved_always = (
        all_episode_info[baseline_steps_idx]["episode_successes"][i]
        and all_episode_info[best_steps_idx]["episode_successes"][i]
    )
    unsolved_always = not (
        all_episode_info[baseline_steps_idx]["episode_successes"][i]
        or all_episode_info[best_steps_idx]["episode_successes"][i]
    )
    better_return = (
        all_episode_info[best_steps_idx]["episode_returns"][i] > all_episode_info[baseline_steps_idx]["episode_returns"][i]
    )
    worse_return = (
        all_episode_info[best_steps_idx]["episode_returns"][i] < all_episode_info[baseline_steps_idx]["episode_returns"][i]
    )
    same_return = (
        all_episode_info[best_steps_idx]["episode_returns"][i] == all_episode_info[baseline_steps_idx]["episode_returns"][i]
    )

    if solved_after_thinking:
        improved_level_list.append(i)
    elif messed_up_after_thinking:
        impaired_level_list.append(i)
    elif solved_always and better_return:
        solved_better_returns.append(i)
    elif solved_always and worse_return:
        solved_worse_returns.append(i)
    elif solved_always and same_return:
        same_return_and_solve.append(i)
    elif unsolved_always and (better_return or same_return):
        unsolved_better_same_returns.append(i)
    elif unsolved_always and worse_return:
        unsolved_worse_returns.append(i)
    else:
        raise ValueError("This should not happen")


# print all fractions
improved_pc = len(improved_level_list) / num_levels * 100
impaired_pc = len(impaired_level_list) / num_levels * 100
solved_better_pc = len(solved_better_returns) / num_levels * 100
solved_worse_pc = len(solved_worse_returns) / num_levels * 100
unsolved_better_same_pc = len(unsolved_better_same_returns) / num_levels * 100
unsolved_worse_pc = len(unsolved_worse_returns) / num_levels * 100
same_return_and_solve_pc = len(same_return_and_solve) / num_levels * 100

print(f"Solved, previously unsolved:\t{improved_pc:.2f}%")
print(f"Unsolved, previously solved:\t{impaired_pc:.2f}%")
print(f"Solved, with better returns:\t{solved_better_pc:.2f}%")
print(f"Solved, with worse returns:\t{solved_worse_pc:.2f}%")
print(f"Solved, with the same returns:\t\t{same_return_and_solve_pc:.2f}%")
print(f"Remaining unsolved, with same or better returns:\t{unsolved_better_same_pc:.2f}%")
print(f"Remaining unsolved, with worse returns:\t{unsolved_worse_pc:.2f}%")

total = (
    improved_pc
    + impaired_pc
    + solved_better_pc
    + solved_worse_pc
    + unsolved_better_same_pc
    + unsolved_worse_pc
    + same_return_and_solve_pc
)
solved_total = improved_pc + solved_better_pc + solved_worse_pc + same_return_and_solve_pc
print(f"Total:\t\t\t\t{total:.2f}%")
print(f"Solved Total:\t\t\t{solved_total:.2f}%")
print(f"Total higher return:\t\t\t{solved_better_pc + improved_pc:.2f}%")
print("better return given solved:", (solved_better_pc + improved_pc) / solved_total * 100)

# latex table for the above
print(
    f"""
\\begin{{tabular}}{{lr}}
\\toprule
\\textsc{{Level categorization}} & \\textsc{{Percentage}} \\\\
\\midrule
Solved, previously unsolved & {improved_pc:.2f} \\\\
Unsolved, previously solved & {impaired_pc:.2f} \\\\
\\midrule
Solved, with better returns & {solved_better_pc:.2f} \\\\
Solved, with the same returns & {same_return_and_solve_pc:.2f} \\\\
Solved, with worse returns & {solved_worse_pc:.2f} \\\\
\\midrule
Unsolved, with same or better returns & {unsolved_better_same_pc:.2f} \\\\
Unsolved, with worse returns & {unsolved_worse_pc:.2f} \\\\
\\bottomrule
\\end{{tabular}}
"""
)


# %%
reward_for_placing_box = 0.9
reward_for_placing_last_box = -0.1 + 1.0 + 10.0

fig, axs = plt.subplots(1, 2, figsize=(3.3, 2), sharey=True, sharex=True)
for ax, condition_on_improved_levels in zip(axs, [False, True]):
    ax.grid(True)
    time_across_think_steps = []
    for j in range(len(steps_to_think)):
        all_rewards = all_episode_info[j]["episode_rewards"]
        if condition_on_improved_levels:
            time_for_placing_boxes = [
                np.where(np.isclose(all_rewards[level_idx], reward_for_placing_box))[0] for level_idx in improved_level_list
            ]
        else:
            time_for_placing_boxes = [
                np.where(np.isclose(reward_array, reward_for_placing_box))[0] for reward_array in all_rewards
            ]
        avg_time_box_placed = [
            np.mean([t[box_idx] for t in time_for_placing_boxes if len(t) > box_idx]) for box_idx in range(3)
        ]
        time_for_placing_last_box = [
            np.where(np.isclose(reward_array, reward_for_placing_last_box))[0] for reward_array in all_rewards
        ]
        time_for_placing_last_box = [e for e in time_for_placing_last_box if len(e) > 0]
        avg_time_box_placed.append(np.mean(time_for_placing_last_box))
        time_across_think_steps.append(avg_time_box_placed)

    ax.plot(steps_to_think, time_across_think_steps)
    if condition_on_improved_levels:
        ax.set_yticks([10, 20, 30, 40, 50])
        # ax.set_xticks([0, 4, 16, 32])
        ax.set_xticks(steps_to_think)
        ax.set_xlabel("(b) On solved levels")
    else:
        ax.set_xlabel("(a) On all levels")
        ax.set_ylabel("Avg timesteps to place the box")
fig.text(0.55, -0.05, "Steps to think", ha="center")

fig.legend(["B1", "B2", "B3", "B4"], bbox_to_anchor=(1.02, 1.2), ncol=4)

plt.savefig(plots_dir / f"{dataset_name}_time_to_box_combined.pdf", format="pdf", bbox_inches="tight")
plt.show()
plt.close()

# %%
all_level_infos = list(
    zip(all_episode_info[0]["level_infos"]["level_file_idx"], all_episode_info[0]["level_infos"]["level_idx"])
)

# %% [markdown]
# ### Cycles

# %%
cycle_starts_within, min_cycle_length, max_cycle_len = 40, 2, 15

level_idx_to_serial_idx = {
    (lfi, li): i
    for i, (lfi, li) in enumerate(
        zip(all_episode_info[0]["level_infos"]["level_file_idx"], all_episode_info[0]["level_infos"]["level_idx"])
    )
}


def get_last_box_time_step(rewards):
    if rewards[-1] == reward_for_placing_last_box:
        return len(rewards) - 1
    try:
        return np.where(np.isclose(rewards, reward_for_placing_box))[0][-1]
    except IndexError:
        return None


def get_cycles(
    all_obs,
    last_box_time_step,
    cycle_starts_within=cycle_starts_within,
    min_cycle_length=min_cycle_length,
    max_cycle_len=max_cycle_len,
):
    assert all_obs.shape[1:] == (3, 10, 10)
    assert last_box_time_step is not None
    all_obs = all_obs[:last_box_time_step]
    all_obs = all_obs.reshape(all_obs.shape[0], 1, *all_obs.shape[1:])
    obs_repeat = np.all(all_obs == all_obs.transpose(1, 0, 2, 3, 4), axis=(2, 3, 4))
    np.fill_diagonal(obs_repeat, False)
    obs_repeat = [np.where(obs_repeat[j])[0] for j in range(min(cycle_starts_within, len(obs_repeat)))]
    # obs_repeat = [
    #     (j, arr[-1] - j)
    #     for j, arr in enumerate(obs_repeat)
    #     if arr.size > 0 and min_cycle_length <= arr[-1] - j
    # ]
    dedup_obs_repeat = []
    i = 0
    # this way of deduplicating will break some 8 shaped cycles into two circles (at different starts)
    while i < len(obs_repeat):
        if obs_repeat[i].size > 0 and min_cycle_length <= obs_repeat[i][-1] - i:
            dedup_obs_repeat.append((i, obs_repeat[i][-1] - i))  # max length cycle starting at i
            i += dedup_obs_repeat[-1][1]
        i += 1

    return dedup_obs_repeat


max_cycles = np.zeros(len(all_episode_info[0]["episode_obs"]), dtype=int)
all_cycles = []
for i in range(len(all_episode_info[0]["episode_obs"])):
    all_obs = all_episode_info[0]["episode_obs"][i]
    last_box_time_step = get_last_box_time_step(all_episode_info[0]["episode_rewards"][i])
    if last_box_time_step is None:
        continue
    obs_repeat = get_cycles(all_obs, last_box_time_step)
    max_cycles[i] = max(cyc_len for _, cyc_len in obs_repeat) if len(obs_repeat) > 0 else 0

    level_file_idx, level_idx = (
        all_episode_info[0]["level_infos"]["level_file_idx"][i],
        all_episode_info[0]["level_infos"]["level_idx"][i],
    )
    if len(obs_repeat) > 0:
        all_cycles.append((level_file_idx, level_idx, obs_repeat))

all_cycles = [(lfi, li, cyc) for lfi, li, cycs in all_cycles for cyc in cycs]
print("Total cycles:", len(all_cycles))

num_cycles_across_stt = []
for stt_i in tqdm(range(len(steps_to_think))):
    num_cycles_stt = 0
    for i in range(len(all_episode_info[stt_i]["episode_obs"])):
        all_obs = all_episode_info[stt_i]["episode_obs"][i]
        last_box_time_step = get_last_box_time_step(all_episode_info[stt_i]["episode_rewards"][i])
        if last_box_time_step is None:
            continue
        obs_repeat = get_cycles(all_obs, last_box_time_step, cycle_starts_within=5)
        num_cycles_stt += len(obs_repeat)
    num_cycles_across_stt.append(num_cycles_stt)

cycle_starts = np.array([cyc[0] for _, _, cyc in all_cycles])
mean_cycle_start = np.mean(cycle_starts)
median_cycle_start = np.median(cycle_starts)

fig, axes = plt.subplots(1, 2, figsize=(3.7, 1.8))

axes[0].hist(cycle_starts, bins=np.max(cycle_starts), density=True)
axes[0].axvline(median_cycle_start, color="green", linestyle="dotted", label=f"Median: {int(median_cycle_start)}")
axes[0].axvline(mean_cycle_start, color="red", linestyle="dotted", label=f"Mean: {mean_cycle_start:.1f}")
axes[0].set_xlabel("Cycle start timestep")
axes[0].set_ylabel("Density")
axes[0].legend()

axes[1].plot(steps_to_think, num_cycles_across_stt)
axes[1].grid(True)
axes[1].set_xticks(steps_to_think)
axes[1].set_yticks(range(0, 9000, 2000))
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1000:.0f}k" if x > 0 else "0"))
axes[1].set_xlabel("Steps to think")
axes[1].set_ylabel("Number of cycles")
plt.savefig(plots_dir / f"{dataset_name}_num_cycles_v_steps_to_think.pdf", format="pdf")
plt.show()
plt.close()
print("Best reduction in cycles:", (num_cycles_across_stt[0] - min(num_cycles_across_stt)) / num_cycles_across_stt[0] * 100)

cycle_lengths = np.array([cyc[1] for _, _, cyc in all_cycles])
mean_cycle_len = np.mean(cycle_lengths)
median_cycle_len = np.median(cycle_lengths)
plt.hist(cycle_lengths, density=True, log=True)
plt.axvline(median_cycle_len, color="green", linestyle="dotted", label=f"Median: {int(median_cycle_len)}")
plt.axvline(mean_cycle_len, color="red", linestyle="dotted", label=f"Mean: {mean_cycle_len:.1f}")
# plt.xscale("log")
plt.xlabel("Cycle length")
plt.ylabel("Density")
plt.legend()
# plt.title("Cycle length distribution")
plt.savefig(plots_dir / f"{dataset_name}_cycle_len_dist.pdf", format="pdf")
plt.show()
plt.close()


# %%
def inside_range(number, range_start, range_end):
    return range_start <= number < range_end


found_in_next_n_steps = np.zeros(len(all_cycles), dtype=np.bool_)
found_at_the_same_step = np.zeros(len(all_cycles), dtype=np.bool_)
performance = 0
baseline_performance = 0
curr_max_steps = 120
cycle_search_cache_file = output_base_path / f"{dataset_name}_cycle_search.pkl"
cache_exists = cycle_search_cache_file.exists()
all_metrics = {}
no_boxing_fails = 0
if cache_exists:
    with open(cycle_search_cache_file, "rb") as f:
        all_metrics = pickle.load(f)
    # level_subset = [(metrics["level_infos"]["level_file_idx"].item(), metrics["level_infos"]["level_idx"].item()) for metrics in all_metrics]
    # all_cycles = [(lfi, li, cyc) for lfi, li, cyc in all_cycles if (lfi, li) in level_subset]
else:
    key = jax.random.PRNGKey(0)
    env_key, eval_key, key = jax.random.split(key, 3)
    env_seed = int(jax.random.randint(env_key, (), minval=0, maxval=2**31 - 2))
    policy, get_action_fn, params = load_policy(local_or_hgf_repo_path, policy_key)
    envs = dataclasses.replace(env_cfg.env, seed=env_seed, num_envs=1).make()

for cyc_idx, (lfi, li, cycle) in tqdm(enumerate(all_cycles), total=len(all_cycles)):
    # curr_max_steps = min(120, cycle[0]+cycle[1])
    if not cache_exists:
        metrics = eval_with_noop(envs, curr_max_steps, policy, get_action_fn, params, eval_key, lfi, li, *cycle)
        all_metrics[(lfi, li, cycle)] = metrics
    else:
        metrics = all_metrics[(lfi, li, cycle)]
    last_box_time_step = get_last_box_time_step(metrics["episode_rewards"].squeeze(1))
    if last_box_time_step is None:
        no_boxing_fails += 1
        continue
    new_cycles = get_cycles(metrics["episode_obs"], last_box_time_step)
    found_in_next_n_steps[cyc_idx] = any(inside_range(cyc[0], cycle[0], cycle[0] + cycle[1]) for cyc in new_cycles)
    found_at_the_same_step[cyc_idx] = cycle[0] in [cyc[0] for cyc in new_cycles]

    baseline_performance += all_episode_info[0]["episode_successes"][level_idx_to_serial_idx[(lfi, li)]]
    performance += metrics["episode_success"][0]

performance /= len(all_cycles)
baseline_performance /= len(all_cycles)

if not cache_exists:
    with open(cycle_search_cache_file, "wb") as f:
        pickle.dump(all_metrics, f)

print(f"Total cycles & {len(all_cycles)} \\\\")
print("\\midrule")
print(f"Cycles at the end of thinking steps & {found_at_the_same_step.sum()/len(all_cycles) * 100:.2f}\\% \\\\")
print(f"Cycles in the next N timesteps & {found_in_next_n_steps.sum()/len(all_cycles) * 100:.2f}\\% \\\\")
print()
print(f"Performance: {performance * 100:.2f}%")
print(f"Baseline Performance: {baseline_performance * 100:.2f}%")
print(f"No boxing fails: {no_boxing_fails}")


# %%
check_actions = 30
same_obs_after_cycle = np.zeros(check_actions, dtype=int)
same_obs_acts_after_cycle = np.zeros(check_actions, dtype=int)
same_obs_after_cycle_solved = np.zeros(check_actions, dtype=int)
num_cycles_used = np.zeros(check_actions, dtype=int)

same_obs_acts_after_cycle_solved = np.zeros(check_actions, dtype=int)
num_cycles_used_solved = np.zeros(check_actions, dtype=int)


for cyc_idx, (lfi, li, cycle) in tqdm(enumerate(all_cycles), total=len(all_cycles)):
    metrics = all_metrics[(lfi, li, cycle)]
    # if found_in_next_n_steps[cyc_idx] or cycle[1] < 3:
    #     continue
    i = level_idx_to_serial_idx[(lfi, li)]
    obs_after_cycle = all_episode_info[0]["episode_obs"][i][cycle[0] + cycle[1] :]
    obs_after_cycle = np.array(obs_after_cycle[:check_actions])
    actions_after_cycle = all_episode_info[0]["episode_acts"][i][cycle[0] + cycle[1] :]
    actions_after_cycle = np.array(actions_after_cycle[:check_actions])

    obs_after_thinking = metrics["episode_obs"][cycle[0] :][:check_actions]
    actions_after_thinking = [act[0] for act in metrics["episode_acts"][cycle[0] :]]
    actions_after_thinking = np.array(actions_after_thinking[:check_actions])
    min_len = min(len(obs_after_cycle), len(obs_after_thinking))
    curr_same_obs_after_cycle = np.array([np.all(obs_after_cycle[i] == obs_after_thinking[i]) for i in range(min_len)]).astype(
        int
    )
    curr_same_acts_after_cycle = (actions_after_cycle[:min_len] == actions_after_thinking[:min_len]).astype(int)

    try:
        first_idx_where_obs_not_same = np.where(curr_same_obs_after_cycle == 0)[0][0]
    except IndexError:
        first_idx_where_obs_not_same = min_len
    same_obs_after_cycle[:first_idx_where_obs_not_same] += 1
    curr_same_obs_acts_after_cycle = curr_same_obs_after_cycle * curr_same_acts_after_cycle
    try:
        first_idx_where_obs_acts_not_same = np.where(curr_same_obs_after_cycle == 0)[0][0]
    except IndexError:
        first_idx_where_obs_acts_not_same = min_len
    same_obs_acts_after_cycle[:first_idx_where_obs_acts_not_same] += 1
    num_cycles_used[:min_len] += 1

    if all_episode_info[0]["episode_successes"][i]:
        same_obs_after_cycle_solved[:first_idx_where_obs_not_same] += 1
        same_obs_acts_after_cycle_solved[:first_idx_where_obs_acts_not_same] += 1
        num_cycles_used_solved[:min_len] += 1

same_obs_after_cycle = same_obs_after_cycle / num_cycles_used
same_obs_acts_after_cycle = same_obs_acts_after_cycle / num_cycles_used

same_obs_after_cycle_solved = same_obs_after_cycle_solved / num_cycles_used_solved
same_obs_acts_after_cycle_solved = same_obs_acts_after_cycle_solved / num_cycles_used_solved

print("% of same obs after cycle after", check_actions, "steps:", same_obs_after_cycle[-1])
print("% of same obs on solved levels after", check_actions, "steps:", same_obs_after_cycle_solved[-1])

x = np.arange(check_actions - 1) + 1
plt.plot(x, same_obs_after_cycle[1:], label="Same state")
# plt.plot(x, same_obs_acts_after_cycle[1:], label="Same state & action")
plt.grid(True)
# plt.xticks([1, 5, 10, 15, 20, 25, 30])
plt.xlabel("Steps after cycle")
# plt.title("on all cycles")
plt.legend()
plt.savefig(plots_dir / f"{dataset_name}_same_obs_after_cycle.pdf", format="pdf")
plt.show()
plt.close()


plt.plot(x, same_obs_after_cycle_solved[1:], label="Same state")
# plt.plot(x, same_obs_acts_after_cycle_solved[1:], label="Same state & action")
plt.grid(True)
# plt.xticks([1, 5, 10, 15, 20, 25, 30])
plt.xlabel("Steps after cycle")
# plt.ylabel("Same states & acts")
# plt.title("on cycles from solved levels")
plt.legend()
plt.savefig(plots_dir / f"{dataset_name}_same_obs_after_cycle_solved_levels.pdf", format="pdf")
plt.show()
plt.close()


# %%
df = pd.DataFrame()
for i in range(len(steps_to_think)):
    df[steps_to_think[i]] = all_episode_info[i]["episode_successes"]

first_solved_at_step = [0] * len(all_episode_info[0]["episode_successes"])
for j, row in df.iterrows():
    this_row = 44
    for i in range(len(row)):
        if bool(row[steps_to_think[i]]):
            this_row = int(steps_to_think[i])
            break
    first_solved_at_step[j] = this_row

# %%
natural_thinking_steps = np.zeros(len(all_episode_info[0]["episode_obs"]), dtype=int)
box_on_target = np.array([254, 95, 56])[:, None, None]

reward_for_placing_box = 0.9


def get_box_on_target_pos(obs):
    check_box_on_target = np.all(obs == box_on_target, axis=0)
    x_pos, y_pos = np.where(check_box_on_target)
    assert len(x_pos) == 1
    return (x_pos[0], y_pos[0])


for i in range(len(all_episode_info[0]["episode_obs"])):
    time_to_box1_wo_think_steps = np.where(all_episode_info[baseline_steps]["episode_rewards"][i] == reward_for_placing_box)[0]

    time_to_box1_w_think_steps = np.where(all_episode_info[best_steps_idx]["episode_rewards"][i] == reward_for_placing_box)[0]
    if len(time_to_box1_wo_think_steps) == 0 or len(time_to_box1_w_think_steps) == 0:
        continue
    time_to_box1_wo_think_steps = time_to_box1_wo_think_steps[0] + 1  # +1 to index correct obs
    time_to_box1_w_think_steps = time_to_box1_w_think_steps[0] + 1

    box_pos_wo_think_steps = get_box_on_target_pos(
        all_episode_info[baseline_steps]["episode_obs"][i][time_to_box1_wo_think_steps]
    )
    box_pos_w_think_steps = get_box_on_target_pos(
        all_episode_info[best_steps_idx]["episode_obs"][i][time_to_box1_w_think_steps]
    )
    if box_pos_w_think_steps == box_pos_wo_think_steps:
        natural_thinking_steps[i] = max(0, time_to_box1_wo_think_steps - time_to_box1_w_think_steps)
        # assert natural_thinking_steps[i] >= 0, f"{natural_thinking_steps[i]} for level {i}"

plt.scatter(natural_thinking_steps, first_solved_at_step)
plt.close()

df = pd.DataFrame()
for i in range(len(steps_to_think)):
    df[steps_to_think[i]] = all_episode_info[i]["episode_returns"]

thinking_much = df.apply(np.argmax, axis=1)

plt.scatter(first_solved_at_step, max_cycles)
plt.close()

df = pd.DataFrame()
df["successes"] = all_episode_info[0]["episode_successes"]
df["successes_think"] = all_episode_info[4]["episode_successes"]

df["lengths"] = all_episode_info[0]["episode_lengths"]
df["lengths_think"] = all_episode_info[4]["episode_lengths"]
df["thinking_much"] = thinking_much  # Peak return
df["natural_steps"] = natural_thinking_steps
df["cycles"] = max_cycles
df

df["diff"] = df.lengths - df.lengths_think

fig, axes = plt.subplots(1, 2, figsize=(3.7, 1.8))


a = df[df["successes"] & df["successes_think"]]

bs = [
    (diff, np.array(a[a["thinking_much"] == diff]["cycles"]))
    for diff in range(a["thinking_much"].min(), a["thinking_much"].max() + 1)
]
positions, data = zip(*bs)
ax = axes[1]
_ = ax.boxplot(data, positions=positions)

a.plot(kind="scatter", x="diff", y="cycles", marker=".", alpha=0.2, ax=axes[0])
a.groupby("thinking_much").mean("cycles").plot(kind="line", y="cycles", marker="+", alpha=1, ax=axes[1], label="mean")

axes[1].set_xticklabels(steps_to_think)
axes[1].set_xlabel("steps at peak return")
axes[0].set_xlabel("len(not thinking) - len(thinking)")
axes[0].set_ylabel("longest state cycle")

plt.savefig(plots_dir / f"{dataset_name}_thinking_substitution.pdf", format="pdf")
plt.show()
plt.close()

# %% [markdown]
# ### Correlation with A* difficulty (# of search steps)

all_search_steps = []
all_optimal_lengths = []
all_optimal_actions = []

# medium_valid_plans = pd.read_csv("data/medium_valid_astar.csv.gz", dtype=str)
# medium_valid_plans = pd.read_csv("data/medium_valid_astar.tar.gz", dtype=str, compression="gzip", names=['File', 'Level', 'Actions', "Steps", "SearchSteps"])

with tarfile.open("data/astar.tar.gz", "r:gz") as tar:
    tar.extractall(path="./")

try:
    split = dataset_name.split("_")[0]
except IndexError:
    split = None
difficulty = dataset_name.split("_")[1]
astar_base_path = pathlib.Path(f"./{difficulty}/{split}/logs/" if split is not None else f"./{difficulty}/logs/")

for i, (file_idx, lev_idx) in enumerate(all_level_infos):
    filename = astar_base_path / f"log_{file_idx:03d}_{lev_idx}.csv"

    # row = medium_valid_plans[
    #     (medium_valid_plans["File"] == f"{file_idx:03d}") & (medium_valid_plans["Level"] == f"{lev_idx:03d}")
    # ].iloc[0]
    row = pd.read_csv(filename, names=["File", "Level", "Actions", "Steps", "SearchSteps"], dtype=str).iloc[0]
    try:
        steps, search_steps = int(row["Steps"].strip()), int(row["SearchSteps"].strip())
        all_optimal_lengths.append(steps)
        all_search_steps.append(search_steps)
    except ValueError:
        all_optimal_lengths.append(len(row["Actions"]))
        all_search_steps.append(int(row["SearchSteps"].strip()))
        all_optimal_actions.append("")
        continue
    except AttributeError:
        all_optimal_lengths.append(-1)
        all_search_steps.append(-1)
    all_optimal_actions.append(row["Actions"].strip())


levels_partition_by_think_steps = [[] for _ in range(len(steps_to_think) + 1)]
print("Number of episodes:", len(all_episode_info[0]["episode_successes"]))
for i in range(len(all_episode_info[0]["episode_successes"])):
    # If the level was not solved, don't put it in the partition
    if all_search_steps[i] == -1:
        continue

    found = False
    for j in range(len(steps_to_think)):
        if all_episode_info[j]["episode_successes"][i]:
            levels_partition_by_think_steps[j].append(i)
            found = True
            break
    if not found:
        levels_partition_by_think_steps[-1].append(i)


x = [step for i, step in enumerate(steps_to_think) if len(levels_partition_by_think_steps[i]) > 10] + ["unsolved"]
avg_search_steps = [
    np.mean([all_search_steps[level_idx] for level_idx in partition])
    for partition in levels_partition_by_think_steps
    if len(partition) > 10
]
fig, ax = plt.subplots()
ax.grid(True)
ax.plot(x, avg_search_steps)
# ylabels 1000s to 1k
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x/1000:.0f}k" if x > 0 else "0"))
plt.xlabel("Solved at steps to think")
plt.ylabel("Avg search steps A*")
plt.savefig(plots_dir / f"{dataset_name}_search_steps_v_steps_to_think.pdf", format="pdf")
plt.show()
plt.close()

avg_opt_len = [
    np.mean([all_optimal_lengths[level_idx] for level_idx in partition])
    for partition in levels_partition_by_think_steps
    if len(partition) > 10
]
fig, ax = plt.subplots()
ax.grid(True)
ax.plot(x, avg_opt_len)
plt.xlabel("Solved at steps to think")
plt.ylabel("Avg Optimal Length")
plt.savefig(plots_dir / f"{dataset_name}_optimal_length_v_steps_to_think.pdf", format="pdf")
plt.show()
plt.close()

# %% [markdown]
# ### Videos

# %%
if save_video:
    # saved = 0
    # for level_idx in improved_level_list:
    #     save_level_video(level_idx, base_dir="thinking_solves_unsolved/")
    #     saved += 1
    #     if saved >= 10:
    #         break

    levels_to_save = [18, 31215, 53, 153, 231]
    for level_idx in tqdm(levels_to_save):
        baseline_obs = np.transpose(all_episode_info[baseline_steps_idx]["episode_obs"][level_idx], (0, 2, 3, 1))
        best_obs = np.transpose(all_episode_info[best_steps_idx]["episode_obs"][level_idx], (0, 2, 3, 1))
        baseline_svgs = episode_obs_to_svgs(baseline_obs, max_len=120)
        best_svgs = episode_obs_to_svgs(best_obs, max_len=120)
        save_dir = plots_dir / "baseline_svgs" / f"level_{level_idx}"
        save_png_dir = plots_dir / "baseline_pngs" / f"level_{level_idx}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_png_dir.mkdir(parents=True, exist_ok=True)

        for i, baseline_svg in enumerate(baseline_svgs):
            with open(save_dir / f"{i:03d}.svg", "w") as f:
                f.write(baseline_svg)
            svg2png(bytestring=baseline_svg, write_to=str(save_png_dir / f"{i:03d}.png"))

        save_dir = plots_dir / "best_svgs" / f"level_{level_idx}"
        save_png_dir = plots_dir / "best_pngs" / f"level_{level_idx}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_png_dir.mkdir(parents=True, exist_ok=True)

        for i, best_svg in enumerate(best_svgs):
            with open(save_dir / f"{i:03d}.svg", "w") as f:
                f.write(best_svg)
            svg2png(bytestring=best_svg, write_to=str(save_png_dir / f"{i:03d}.png"))

    baseline_videos = plots_dir / "videos" / "baseline"
    best_videos = plots_dir / "videos" / "best"
    os.system(f"rm -rf {baseline_videos}")
    os.system(f"rm -rf {best_videos}")
    baseline_videos.mkdir(parents=True, exist_ok=True)
    best_videos.mkdir(parents=True, exist_ok=True)
    for level_idx in levels_to_save:
        save_baseline_png_dir = plots_dir / "baseline_pngs" / f"level_{level_idx}"
        save_best_png_dir = plots_dir / "best_pngs" / f"level_{level_idx}"
        os.system(
            f"ffmpeg -framerate 3 -i '{save_baseline_png_dir / '%03d.png'}' {baseline_videos / f'level_{level_idx}.mp4'}"
        )
        os.system(f"ffmpeg -framerate 3 -i '{save_best_png_dir / '%03d.png'}' {best_videos / f'level_{level_idx}.mp4'}")
