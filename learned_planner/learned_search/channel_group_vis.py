"""Visualize a single channel activations across all groups."""

# %%
import dataclasses
import re
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch as th
from cleanba.environments import BoxobanConfig

from learned_planner import LP_DIR
from learned_planner.interp.channel_group import get_channel_dict, get_group_channels
from learned_planner.interp.offset_fns import apply_inv_offset_lc
from learned_planner.interp.plot import apply_style, plotly_feature_vis
from learned_planner.interp.utils import load_jax_model_to_torch, play_level
from learned_planner.notebooks.emacs_plotly_render import set_plotly_renderer
from learned_planner.policies import download_policy_from_huggingface

set_plotly_renderer("emacs")
pio.renderers.default = "notebook"
th.set_printoptions(sci_mode=False, precision=2)


# %%
MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000"  # DRC(1, 1) 2B checkpoint
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)


boxo_cfg = BoxobanConfig(
    cache_path=LP_DIR / "alternative-levels/levels/",
    num_envs=1,
    max_episode_steps=120,
    min_episode_steps=120,
    asynchronous=False,
    tinyworld_obs=True,
    split="train",
    difficulty="unfiltered",
    dim_room=(10, 10),
)

model_cfg, model = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)
reps = model_cfg.features_extractor.repeats_per_step

orig_state_dict = deepcopy(model.state_dict())
if MODEL_PATH_IN_REPO == "drc11/eue6pax7/cp_2002944000":
    print("Reducing conv_hh weights")
    comp = model.features_extractor.cell_list[0]
    comp.conv_ih.weight.data[:, 32:64] += comp.conv_hh.weight.data
    comp.conv_hh.weight.data.zero_()
    reduced_state_dict = deepcopy(model.state_dict())


def restore_model(orig_state_dict=orig_state_dict):
    model.load_state_dict(orig_state_dict)


def map_key(key, channel=None, keep_ticks=False, sum_pos=True):
    channel = "" if channel is None else channel
    if "hook_pre_model" in key:
        return f"ENC{channel}"
    layer = int(key.split(".")[2])
    layer_type = key.split(".")[3][5:].upper()
    ret = f"L{layer}{layer_type}{channel}"
    if keep_ticks:
        ret += f"T{key.split('.')[-1]}"
    if not sum_pos:
        pos_idx = int(key.split(".")[-2]) * 3 + int(key.split(".")[-1])
        ret += f"P{pos_idx}"
    return ret


def plot_group(toy_cache, toy_obs_repeated, group_name="box", hook_type="h"):
    layer_values = {}
    if isinstance(toy_cache, dict):
        for k, v in toy_cache.items():
            if m := re.match(f"^.*([0-9]+)\\.hook_([{hook_type}])$", k):
                layer_values[int(m.group(1))] = v
    elif isinstance(toy_cache, list):
        for i, (h, c) in enumerate(toy_cache):
            layer_values[i] = h
    else:
        raise ValueError(f"Incorrect type: {type(toy_cache)}")

    desired_groups = get_group_channels(group_name, return_dict=True)

    channels = []
    labels = []

    for group in desired_groups:
        for layer in group:
            channels.append(layer_values[layer["layer"]][:, layer["idx"], :, :])
            labels.append(f"L{layer['layer']}{hook_type.upper()}{layer['idx']}")

    channels = np.stack(channels, 1)
    fig = plotly_feature_vis(channels, toy_obs_repeated, feature_labels=labels, common_channel_norm=True)
    fig.update_layout(height=800)
    return fig


# %%
envs = dataclasses.replace(boxo_cfg, num_envs=1).make()
thinking_steps = 0
max_steps = 120

size = 10
walls = [(i, 0) for i in range(size)]
walls += [(i, size - 1) for i in range(size)]
walls += [(0, i) for i in range(1, size - 1)]
walls += [(size - 1, i) for i in range(1, size - 1)]

walls += [(y, x) for y in range(4, 7) for x in range(3, 7)]
walls += [(y, 7) for y in range(1, 4)]

boxes = [(3, 5)]
targets = [(1, 8)]
player = (1, 2)
toy_reset_opt = dict(walls=walls, boxes=boxes, targets=targets, player=player)
obs = envs.reset(options=toy_reset_opt)[0][0]
plt.imshow(np.transpose(obs, (1, 2, 0)))
plt.show()
# %% PLAY TOY LEVEL
play_toy = True
two_levels = False
level_reset_opt = {"level_file_idx": 25, "level_idx": 0}
thinking_steps = 0

max_steps = 120
size = 10


def get_hook_info(short_key, set_ticks=True):
    layer = int(short_key[1])
    hook_type = short_key[2].lower()
    if "T" in short_key:
        split_key = short_key.split("T")
        tick_pos = [int(short_key.split("T")[-1])] if set_ticks else None
        prune_channel = split_key[0][3:]
    else:
        tick_pos = None
        prune_channel = short_key[3:]
    return layer, hook_type, int(prune_channel), tick_pos


fwd_hooks = []
reset_opts = toy_reset_opt if play_toy else level_reset_opt
toy_out = play_level(
    envs,
    model,
    reset_opts=reset_opts,
    # reset_opts=level_reset_opt,
    thinking_steps=thinking_steps,
    fwd_hooks=fwd_hooks,
    max_steps=max_steps,
    hook_steps=list(range(thinking_steps, max_steps)) if thinking_steps > 0 else -1,
    # probes=[probe],
    # probe_train_ons=[probe_info],
    probe_logits=True,
    internal_steps=True,
)

toy_cache = toy_out.cache
toy_cache = {k: v.squeeze(1) for k, v in toy_cache.items() if len(v.shape) == 5}
toy_obs = toy_out.obs.squeeze(1)
toy_obs_repeated = toy_obs.repeat_interleave(reps, 0).numpy()
print("Total len:", toy_obs.shape[0], toy_cache["features_extractor.cell_list.0.hook_h"].shape[0])


# %%


def standardize_channel(channel_value, channel_info: tuple[int, int] | dict):
    """Standardize the channel value based on its sign and index."""
    assert len(channel_value.shape) >= 2, f"Invalid channel value shape: {channel_value.shape}"
    if isinstance(channel_info, tuple):
        l, c = channel_info
        channel_dict = get_channel_dict(l, c)
    else:
        channel_dict = channel_info
    channel_value = apply_inv_offset_lc(channel_value, channel_dict["layer"], channel_dict["idx"], last_dim_grid=True)
    sign = channel_dict.get("sign", 1)
    if isinstance(sign, str):
        assert sign in ["+", "-"], f"Invalid sign: {sign}"
        sign = 1 if sign == "+" else -1
    elif not isinstance(sign, int):
        raise ValueError(f"Invalid sign type: {type(sign)}")
    return channel_value * sign


apply_style((5.4, 2.0), px_use_default=False)
labels = [
    "Box left",
    "Box down",
    "Box right",
    "Box up",
    "Agent left",
    "Agent down",
    "Agent right",
    "Agent up",
    "GNA",
    "PNA",
    "Target",
    "Combined plan",
    "No-label",
]
channel_indices = [
    (1, 27),
    (1, 17),
    (0, 17),
    (0, 24),
    # agent
    (2, 31),
    (1, 18),
    (2, 5),
    (1, 29),
    # gna/pna
    (2, 26),
    (2, 3),
    # target/misc/no-label
    (0, 6),
    (2, 14),
    (0, 3),
]
ticks = [9] * 4 + [16] * 4 + [1, 2] + [0] + [20] + [9]
channels = []
fig_labels = []

for label, (l, c), tick in zip(labels, channel_indices, ticks):
    acts = toy_cache[f"features_extractor.cell_list.{l}.hook_h"][tick, c, :, :][None]
    acts = standardize_channel(acts, (l, c))
    channels.append(acts)
    fig_labels.append(label + f"<br>(L{l}H{c})")
channels = np.stack(channels, 1)
fig = plotly_feature_vis(
    channels,
    toy_obs[0][None],
    feature_labels=fig_labels,
    common_channel_norm=True,
)

fig.add_trace(
    go.Scatter(
        x=[None],
        y=[None],  # Invisible points
        mode="markers",
        marker=dict(
            colorscale="Viridis",
            cmin=-1,
            cmax=1,
            colorbar=dict(
                title="Normalized Activation",
                titleside="right",
                thickness=8,
                x=1.00,  # Position to the right of plot area
                xanchor="left",
                y=0.5,
                yanchor="middle",
                lenmode="fraction",  # Length relative to plot area
                len=0.75,
            ),
        ),
        showlegend=False,
    )
)
fig.update_layout(margin=dict(l=0, r=0, t=0, b=15, pad=0))
fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, ticks="")
fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, ticks="")

shift = -100
fig.for_each_annotation(lambda a: a.update(yshift=shift + 8 if "Observation" in a.text else shift))
fig.write_image("../../new_plots/group_visualization.svg")
fig.show()
# fig.update_layout(height=800)

# %%
