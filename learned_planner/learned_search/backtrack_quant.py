"""Quantitative evaluation of causal intervention on negative activations of backtracking."""

# %%
import dataclasses
import re
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import plotly.io as pio
import torch as th
from cleanba.environments import BoxobanConfig
from gym_sokoban.envs.sokoban_env import CHANGE_COORDINATES
from scipy.stats import bootstrap
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from learned_planner import BOXOBAN_CACHE
from learned_planner.interp.channel_group import get_channel_dict, get_group_channels, split_by_layer
from learned_planner.interp.collect_dataset import DatasetStore
from learned_planner.interp.offset_fns import apply_inv_offset_lc, offset_yx
from learned_planner.interp.plot import plotly_feature_vis
from learned_planner.interp.render_svg import BOX, TARGET
from learned_planner.interp.utils import load_jax_model_to_torch, play_level
from learned_planner.policies import download_policy_from_huggingface

try:
    pio.kaleido.scope.mathjax = None  # Disable MathJax to remove the loading message
except AttributeError:
    pass
pio.renderers.default = "notebook"
th.set_printoptions(sci_mode=False, precision=2)

# %%

MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000"  # DRC(3, 3) 2B checkpoint
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

boxo_cfg = BoxobanConfig(
    cache_path=BOXOBAN_CACHE,
    num_envs=1,
    max_episode_steps=200,
    min_episode_steps=200,
    asynchronous=False,
    tinyworld_obs=True,
    split=None,
    difficulty="hard",
    seed=42,
)
model_cfg, model = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)

orig_state_dict = deepcopy(model.state_dict())

envs = boxo_cfg.make()

if th.cuda.is_available():
    model = model.to("cuda")


def restore_model():
    model.load_state_dict(orig_state_dict)


# %% Print model's hook points
def recursive_children(model):
    for c in model.children():
        yield c
        yield from recursive_children(c)


[c.name for c in recursive_children(model) if isinstance(c, HookPoint)]


def standardize_channel(channel_value, channel_info: tuple[int, int] | dict):
    """Standardize the channel value based on its sign and index."""
    assert len(channel_value.shape) >= 2, f"Invalid channel value shape: {channel_value.shape}"
    if isinstance(channel_info, tuple):
        l, c = channel_info
        channel_dict = get_channel_dict(l, c)
    else:
        channel_dict = channel_info
    channel_value = apply_inv_offset_lc(channel_value, channel_dict["layer"], channel_dict["idx"], last_dim_grid=True)
    sign = channel_dict["sign"]
    if isinstance(sign, str):
        assert sign in ["+", "-"], f"Invalid sign: {sign}"
        sign = 1 if sign == "+" else -1
    elif not isinstance(sign, int):
        raise ValueError(f"Invalid sign type: {type(sign)}")
    return channel_value * sign


# %%

batch_size = 128
num_envs = 512
fwd_hooks = []
boxo_cfg = dataclasses.replace(boxo_cfg, num_envs=batch_size, seed=42)
envs = boxo_cfg.make()

debug = False

all_obs = []
all_seq_lens = []
all_cache = []
level_infos = []
ds_list = []
for batch_idx in tqdm(range(0, num_envs // batch_size)):
    envs = boxo_cfg.make()
    for _ in range(batch_idx):
        envs.reset()
    play_out = play_level(envs, model, fwd_hooks=fwd_hooks, re_hook_filter="hook_h")

    select_idx = th.ones(batch_size, dtype=bool)
    current_seq_lens = []
    for env_idx in range(batch_size):
        try:
            env_steps = min(th.where(play_out.rewards[:, env_idx] > 0)[0][-1].item(), play_out.lengths[env_idx].item())
            all_seq_lens.append(env_steps)
            pops = []
            for k, v in play_out.cache.items():
                if len(v.shape) != 5:
                    pops.append(k)
                    continue
                v[3 * env_steps :, env_idx] = 0
            [play_out.cache.pop(k) for k in pops]
            if debug:
                play_out.obs[env_steps:, env_idx] = 0
            ds_list.append(DatasetStore.load_from_play_output(play_out, env_idx))
        except IndexError:
            select_idx[env_idx] = False
    current_seq_lens = th.tensor(current_seq_lens)

    if debug:
        all_cache.append({k: v[:, select_idx] for k, v in play_out.cache.items() if len(v.shape) == 5})
        all_obs.append(play_out.obs[:, select_idx])
    else:
        all_cache.append({k: v[:, select_idx] for k, v in play_out.cache.items() if "hook_h" in k})
    infos = list(zip(play_out.info["level_file_idx"], play_out.info["level_idx"]))
    level_infos += [inf for env_idx, inf in enumerate(infos) if select_idx[env_idx]]

del play_out
all_cache = {k: th.cat([th.as_tensor(cache[k]) for cache in all_cache], dim=1) for k in all_cache[0].keys()}

# all_cache = {
#     k: th.nn.utils.rnn.pad_sequence([th.as_tensor(cache[k]) for cache in all_cache]).flatten(1, 2) for k in all_cache[0].keys()
# }
if debug:
    all_obs = th.nn.utils.rnn.pad_sequence(all_obs).flatten(1, 2)

# %%

NP_CHANGE_COORDINATES = {k: np.array(v) for k, v in CHANGE_COORDINATES.items()}

box_group = get_group_channels("box", return_dict=True)
threshold = 0.5
adjacent_threshold = 0.1
sqs_to_probe = 1
final_dataset = []
for dir_idx, channel_dicts in enumerate(box_group):
    for c_dict in tqdm(channel_dicts):
        l, c, long_term = c_dict["layer"], c_dict["idx"], c_dict["long-term"]
        acts = standardize_channel(all_cache[f"features_extractor.cell_list.{l}.hook_h"][:, :, c], c_dict)
        acts_below_thresh = acts[..., 1:-1, 1:-1] < -threshold
        if acts_below_thresh.any().item():
            indices = th.where(acts_below_thresh)
            for it in range(len(indices[0])):
                step_i, batch_i, y_i, x_i = indices[0][it], indices[1][it], indices[2][it], indices[3][it]
                if step_i >= all_seq_lens[batch_i.item()] * 3 * 0.99:
                    # print(step_i, all_seq_lens[batch_i.item()] * 3)
                    continue
                y_i, x_i = y_i + 1, x_i + 1  # +1 since we do 1:-1 above on acts
                square = np.array([y_i, x_i])
                ds = ds_list[batch_i.item()]
                # if not ds.is_wall(y_i, x_i):
                #     continue
                # if ds.is_target(y_i, x_i):
                if ds.obs[0, :, y_i, x_i].eq(th.tensor(TARGET)).all().item():
                    continue
                forward_sqs = np.array([square + s * NP_CHANGE_COORDINATES[dir_idx] for s in range(1, sqs_to_probe + 1)])
                backward_sqs = np.array([square - s * NP_CHANGE_COORDINATES[dir_idx] for s in range(1, sqs_to_probe + 1)])
                lfi, li = level_infos[batch_i.item()]
                try:
                    forward_acts = acts[step_i, batch_i, forward_sqs[:, 0], forward_sqs[:, 1]]
                    forward_acts_next_step = acts[step_i + 1, batch_i, forward_sqs[:, 0], forward_sqs[:, 1]]

                    prev_step_act = forward_acts.sum().item()
                    next_step_act = forward_acts_next_step.sum().item()
                    if (
                        prev_step_act > (sqs_to_probe) * adjacent_threshold
                        and next_step_act < prev_step_act / 2
                        # and not ds.is_wall(forward_sqs[0, 0], forward_sqs[0, 1])
                        and not ds.obs[(step_i // 3), :, forward_sqs[0, 0], forward_sqs[0, 1]].eq(th.tensor(BOX)).all().item()
                        and not ds.obs[(step_i // 3) - 1, :, forward_sqs[0, 0], forward_sqs[0, 1]]
                        .eq(th.tensor(BOX))
                        .all()
                        .item()
                    ):
                        final_dataset.append(
                            (
                                dir_idx,
                                l,
                                c,
                                long_term,
                                step_i.item(),
                                batch_i.item(),
                                lfi,
                                li,
                                y_i.item(),
                                x_i.item(),
                                "forward",
                                prev_step_act,
                                next_step_act,
                            )
                        )
                        continue
                except IndexError:
                    pass
                try:
                    backward_acts = acts[step_i, batch_i, backward_sqs[:, 0], backward_sqs[:, 1]]
                    backward_acts_next_step = acts[step_i + 1, batch_i, backward_sqs[:, 0], backward_sqs[:, 1]]

                    prev_step_act = backward_acts.sum().item()
                    next_step_act = backward_acts_next_step.sum().item()
                    if (
                        prev_step_act > (sqs_to_probe) * adjacent_threshold
                        and next_step_act < prev_step_act / 2
                        # and not ds.is_wall(backward_sqs[0, 0], backward_sqs[0, 1])
                        and not ds.obs[(step_i // 3), :, backward_sqs[0, 0], backward_sqs[0, 1]]
                        .eq(th.tensor(BOX))
                        .all()
                        .item()
                        and not ds.obs[(step_i // 3) - 1, :, backward_sqs[0, 0], backward_sqs[0, 1]]
                        .eq(th.tensor(BOX))
                        .all()
                        .item()
                    ):
                        final_dataset.append(
                            (
                                dir_idx,
                                l,
                                c,
                                long_term,
                                step_i.item(),
                                batch_i.item(),
                                lfi,
                                li,
                                y_i.item(),
                                x_i.item(),
                                "backward",
                                prev_step_act,
                                next_step_act,
                            )
                        )
                        continue
                except IndexError:
                    pass

df = pd.DataFrame(
    final_dataset,
    columns=[
        "direction",
        "layer",
        "channel",
        "long-term",
        "step",
        "batch_idx",
        "lfi",
        "li",
        "y",
        "x",
        "chain_direction",
        "og_prev_act",
        "og_next_act",
    ],
)

print("Total samples:", len(df))

# %%
envs = dataclasses.replace(boxo_cfg, num_envs=1, seed=42).make()
box_channels = get_group_channels("box_agent", return_dict=True, exclude_nfa_mpa=True)
layer_wise_direction = [split_by_layer([dir_channels]) for dir_channels in box_channels]
successes = 0
all_prev_acts = -np.ones(len(df))
all_next_acts = -np.ones(len(df))


def abs_hook(inp, hook, direction, int_y_start, int_y_end, int_x_start, int_x_end):
    layer = int(hook.name.split(".")[2])
    channels = layer_wise_direction[direction][layer]
    for c_dict in channels:
        idx = c_dict["idx"]
        offset = offset_yx(0, 0, [idx], layer)
        offset_y, offset_x = offset[0][0], offset[1][0]
        inp[:, idx, offset_y + int_y_start : offset_y + int_y_end, offset_x + int_x_start : offset_x + int_x_end] = (
            inp[:, idx, offset_y + int_y_start : offset_y + int_y_end, offset_x + int_x_start : offset_x + int_x_end].abs()
            * c_dict["sign"]
        )
        # inp[:, idx, offset_y + int_y_start : offset_y + int_y_end, offset_x + int_x_start : offset_x + int_x_end] = c_dict[
        #     "sign"
        # ]
    return inp


total = 0

patched_obs = []
patched_cache = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    lfi, li = row["lfi"], row["li"]
    reset_opts = {"level_file_idx": lfi, "level_idx": li}

    l, c = row["layer"], row["channel"]
    y, x = row["y"], row["x"]
    direction, step = row["direction"], row["step"]
    env_step, tick = step // 3, step % 3

    int_y_start, int_y_end, int_x_start, int_x_end = y, y + 1, x, x + 1
    if direction in [0, 1]:
        int_x_start -= 1
        int_x_end += 1
    else:
        int_y_start -= 1
        int_y_end += 1

    fwd_hooks = [
        (
            f"features_extractor.cell_list.{layer}.{hook_type}.{pos}.{int_step}",
            partial(
                abs_hook,
                direction=direction,
                int_y_start=int_y_start,
                int_y_end=int_y_end,
                int_x_start=int_x_start,
                int_x_end=int_x_end,
            ),
        )
        for layer in [0, 1, 2]
        for pos in [0]
        for int_step in [0, 1, 2]
        for hook_type in ["hook_h"]
    ]

    patched_play = play_level(
        envs,
        model,
        reset_opts,
        fwd_hooks=fwd_hooks,
        hook_steps=list(range(env_step - 1, env_step + 2)),
        max_steps=env_step + 2,
        re_hook_filter="hook_h",
    )
    patched_obs.append(patched_play.obs)
    patched_cache.append(patched_play.cache)
    acts = standardize_channel(patched_play.cache[f"features_extractor.cell_list.{l}.hook_h"][:, 0, c], (l, c))

    dir_sign = 1 if row["chain_direction"] == "forward" else -1
    square = np.array([y, x])
    probe_sqs = np.array([square + dir_sign * s * NP_CHANGE_COORDINATES[direction] for s in range(1, sqs_to_probe + 1)])

    all_prev_acts[i] = acts[step, probe_sqs[:, 0], probe_sqs[:, 1]].sum()
    all_next_acts[i] = acts[step + 1, probe_sqs[:, 0], probe_sqs[:, 1]].sum()

    if all_next_acts[i] > row["og_prev_act"] / 2:
        successes += 1
    total += 1

df["patch_prev_act"] = all_prev_acts
df["patch_next_act"] = all_next_acts

df.to_csv("/training/iclr_logs/backtrack_quant.csv")

print(successes / total * 100, successes, total)


# %%


def calculate_ci(subset):
    success_condition = subset["patch_next_act"] > (subset["og_prev_act"] / 2)
    success_rate = success_condition.mean()

    data_for_bootstrap = (success_condition.to_numpy(),)

    res = bootstrap(data_for_bootstrap, np.mean, confidence_level=0.95, method="basic", n_resamples=1000, random_state=42)

    lower_ci = res.confidence_interval.low
    upper_ci = res.confidence_interval.high
    delta = max(success_rate - lower_ci, upper_ci - success_rate)
    return success_rate, delta


success_rate, delta = calculate_ci(df[(df["long-term"])])
print(f"95% CI for long-term: {success_rate:.1%} \pm {delta:.1%}")

success_rate, delta = calculate_ci(df[(~df["long-term"])])
print(f"95% CI for short-term: {success_rate:.1%} \pm {delta:.1%}")

# subset[subset["patch_next_act"] < subset["og_prev_act"] / 2].sort_values("og_prev_act").head(20)


# %%

# for direction in range(4):
#     subset = df[df["direction"] == direction]
#     success = (subset["patch_next_act"] > subset["og_prev_act"] / 2).sum()
#     print(f"Rate {direction}:", success / len(subset) * 100, len(subset))

# df[df["patch_next_act"] < df["og_prev_act"] / 2].head(20)
# %%
if debug:
    obs_idx = 28

    layer_values = {}
    hook_type = "h"
    for k, v in all_cache.items():
        if m := re.match(f"^.*([0-9]+)\\.hook_([{hook_type}])$", k):
            layer_values[int(m.group(1))] = v[: 3 * all_seq_lens[obs_idx]]

    # desired_groups = ["B up", "B down", "B left", "B right"]
    # desired_groups = get_group_channels("No label", return_dict=True)
    # desired_groups = get_group_channels("T", return_dict=True)
    # desired_groups = get_group_channels("Other", return_dict=True)
    # desired_groups = get_group_channels("nfa", return_dict=True)
    desired_groups = get_group_channels("box", return_dict=True)
    # desired_groups = [desired_groups[1], desired_groups[3]]

    channels = []
    labels = []

    for group in desired_groups:
        for layer in group:
            acts = layer_values[layer["layer"]][:, obs_idx, layer["idx"], :, :]
            if hook_type == "h":
                acts = standardize_channel(acts, layer)
            channels.append(acts)
            labels.append(f"L{layer['layer']}{hook_type.upper()}{layer['idx']}")

    channels = np.stack(channels, 1)

    toy_obs_repeated = all_obs[: all_seq_lens[obs_idx], obs_idx].repeat_interleave(3, 0).numpy()

    fig = plotly_feature_vis(channels, toy_obs_repeated, feature_labels=labels)
    fig.update_layout(height=800)
    fig.show()

# %%
if debug:
    idx = 0
    layer_values = {}
    hook_type = "h"
    for k, v in patched_cache[idx].items():
        if m := re.match(f"^.*([0-9]+)\\.hook_([{hook_type}])$", k):
            layer_values[int(m.group(1))] = v[:, 0]

    # desired_groups = ["B up", "B down", "B left", "B right"]
    # desired_groups = get_group_channels("No label", return_dict=True)
    # desired_groups = get_group_channels("T", return_dict=True)
    # desired_groups = get_group_channels("Other", return_dict=True)
    # desired_groups = get_group_channels("nfa", return_dict=True)
    desired_groups = get_group_channels("box", return_dict=True)
    # desired_groups = [desired_groups[1], desired_groups[3]]

    channels = []
    labels = []

    for group in desired_groups:
        for layer in group:
            acts = layer_values[layer["layer"]][:, layer["idx"], :, :]
            if hook_type == "h":
                acts = standardize_channel(acts, layer)
            channels.append(acts)
            labels.append(f"L{layer['layer']}{hook_type.upper()}{layer['idx']}")

    channels = np.stack(channels, 1)

    toy_obs_repeated = patched_obs[idx][:, 0].repeat_interleave(3, 0).numpy()

    fig = plotly_feature_vis(channels, toy_obs_repeated, feature_labels=labels)
    fig.update_layout(height=800)
    fig.show()

# %%
