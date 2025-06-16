"""circuit discovery related experiments for drc11 and drc33.

Activation patching code for tutorial level at the end of the notebook.
"""

# %%
import dataclasses
import re
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
import torch as th
from cleanba.environments import BoxobanConfig
from tqdm import tqdm

from learned_planner import BOXOBAN_CACHE
from learned_planner.interp.act_patch_utils import (
    activation_patching_sq_wise,
    corrupt_obs,
    get_cache_and_probs,
    get_obs,
    mse_loss_fn,
    skip_cache_key,
)
from learned_planner.interp.channel_group import get_group_channels, get_group_connections, layer_groups
from learned_planner.interp.plot import plotly_feature_vis
from learned_planner.interp.render_svg import WALL
from learned_planner.interp.utils import join_cache_across_steps, load_jax_model_to_torch, play_level
from learned_planner.interp.weight_utils import get_conv_weights, visualize_top_conv_inputs
from learned_planner.notebooks.emacs_plotly_render import set_plotly_renderer
from learned_planner.policies import download_policy_from_huggingface

set_plotly_renderer("emacs")
pio.renderers.default = "notebook"
th.set_printoptions(sci_mode=False, precision=2)


# %%
# MODEL_PATH_IN_REPO = "drc11/eue6pax7/cp_2002944000"  # DRC(1, 1) 2B checkpoint
MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000"  # DRC(1, 1) 2B checkpoint
# MODEL_PATH_IN_REPO = "resnet/syb50iz7/cp_2002944000"  # ResNet model 2B checkpoint
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

boxo_cfg = BoxobanConfig(
    cache_path=BOXOBAN_CACHE,
    num_envs=2,
    max_episode_steps=120,
    min_episode_steps=120,
    asynchronous=False,
    tinyworld_obs=True,
    split=None,
    difficulty="hard",
)
# boxo_cfg = BoxobanConfig(
#     cache_path=LP_DIR / "alternative-levels/levels/",
#     num_envs=1,
#     max_episode_steps=120,
#     min_episode_steps=120,
#     asynchronous=False,
#     tinyworld_obs=True,
#     split="train",
#     difficulty="unfiltered",
#     dim_room=(10, 10),
# )

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


def map_to_normal_key(short_key):
    if short_key.startswith("ENC"):
        return "features_extractor.hook_pre_model"
    return f"features_extractor.cell_list.{int(short_key[1])}.hook_{short_key[2].lower()}"


# probe, _ = load_probe("probes/best/boxes_future_direction_map_l-all.pkl")
# probe_info = TrainOn(dataset_name="boxes_future_direction_map")
# %%
envs = dataclasses.replace(boxo_cfg, num_envs=1).make()
thinking_steps = 0
max_steps = 50


def run_policy_reset(num_steps, envs, policy):
    new_obs, _ = envs.reset()
    obs = [new_obs]
    carry = policy.recurrent_initial_state(envs.num_envs)
    all_false = th.zeros(envs.num_envs, dtype=th.bool)
    for _ in range(num_steps - 1):
        action, _value, something, carry = policy(th.as_tensor(new_obs), carry, all_false)
        new_obs, _, term, trunc, _ = envs.step(action.detach().cpu().numpy())
        assert not (np.any(term) or np.any(trunc))
        obs.append(new_obs)
    return th.as_tensor(np.stack(obs))


seq_len = 10

# %%
restore_model(orig_state_dict)
# restore_model(reduced_state_dict)
# model.features_extractor.cell_list[0].conv_ih.weight.data[:, 32:96].zero_()
# non_zero_channel = [12, 10, 26, 18]
# model.features_extractor.cell_list[0].conv_ih.weight.data[non_zero_channel] = reduced_state_dict[
#     "features_extractor.cell_list.0.conv_ih.weight"
# ][non_zero_channel]

# %% PLAY TOY LEVEL
play_toy = True
two_levels = True
level_reset_opt = {"level_file_idx": 8, "level_idx": 2}
thinking_steps = 0

max_steps = 50
size = 10


def zig_zag():
    walls = [(i, 0) for i in range(size)]
    walls += [(i, size - 1) for i in range(size)]
    walls += [(0, i) for i in range(1, size - 1)]
    walls += [(size - 1, i) for i in range(1, size - 1)]

    walls += [(y, y + 2) for y in range(1, size - 2)]
    walls += [(x + 3, x) for x in range(1, size - 4)]

    player = (1, 1)
    target = [(8, 7)]
    boxes = [(2, 2)]
    obs = envs.reset(options=dict(walls=walls, boxes=boxes, targets=target, player=player))[0]
    obs = obs[0].transpose(1, 2, 0)
    plt.imshow(obs)
    plt.show()
    return walls, boxes, target, player


walls = [(i, 0) for i in range(size)]
walls += [(i, size - 1) for i in range(size)]
walls += [(0, i) for i in range(1, size - 1)]
walls += [(size - 1, i) for i in range(1, size - 1)]

walls += [(y, x) for y in range(4, 6) for x in range(3, 6)]

boxes = [(3, 2)]
targets = [(8, 8)]
player = (1, 1)

# boxes = [(4, 4)]
# targets = [(8, 8)]
# player = (8, 7)


# walls += [(y, 5) for y in range(2, 7)]
# boxes = [(3, 2)]
# targets = [(3, 6)]
# player = (1, 7)

# walls, boxes, targets, player = zig_zag()


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

# mean_done = 0


toy_reset_opt = dict(walls=walls, boxes=boxes, targets=targets, player=player)
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
print("Total len:", len(toy_obs), toy_cache["features_extractor.cell_list.0.hook_h"].shape[0] // 3)

if two_levels:
    play_toy = True
    # boxes = [(2, 2)]
    # targets = [(8, 8)]
    # player = (1, 1)

    boxes = [(7, 7)]
    targets = [(2, 1)]
    player = (8, 8)
    mean_done = 0
    level_reset_opt["level_idx"] += 1
    toy_reset_opt = dict(walls=walls, boxes=boxes, targets=targets, player=player)
    reset_opts = toy_reset_opt if play_toy else level_reset_opt
    toy_out2 = play_level(
        envs,
        model,
        reset_opts=reset_opts,
        # reset_opts={**level_reset_opt, "level_idx": level_reset_opt["level_idx"] + 1},
        thinking_steps=thinking_steps,
        fwd_hooks=fwd_hooks,
        max_steps=max_steps,
        hook_steps=list(range(thinking_steps, max_steps)) if thinking_steps > 0 else -1,
        # probes=[probe],
        # probe_train_ons=[probe_info],
        probe_logits=True,
        internal_steps=True,
    )
    toy_cache2 = toy_out2.cache
    toy_cache2 = {k: v.squeeze(1) for k, v in toy_cache2.items() if len(v.shape) == 5}
    toy_obs2 = toy_out2.obs.squeeze(1)

    toy_cache = {k: np.concatenate([v, toy_cache2[k]], axis=0) for k, v in toy_cache.items()}
    toy_obs = th.cat([toy_obs, toy_obs2], dim=0)
    toy_obs_repeated = toy_obs.repeat_interleave(reps, 0).numpy()

    play_toy = False

    boxes = [(7, 7)]
    targets = [(2, 1)]
    player = (8, 8)
    mean_done = 0
    level_reset_opt["level_idx"] += 1
    toy_reset_opt = dict(walls=walls, boxes=boxes, targets=targets, player=player)
    reset_opts = toy_reset_opt if play_toy else level_reset_opt
    toy_out2 = play_level(
        envs,
        model,
        reset_opts=reset_opts,
        # reset_opts={**level_reset_opt, "level_idx": level_reset_opt["level_idx"] + 1},
        thinking_steps=thinking_steps,
        fwd_hooks=fwd_hooks,
        max_steps=max_steps,
        hook_steps=list(range(thinking_steps, max_steps)) if thinking_steps > 0 else -1,
        # probes=[probe],
        # probe_train_ons=[probe_info],
        probe_logits=True,
        internal_steps=True,
    )
    toy_cache2 = toy_out2.cache
    toy_cache2 = {k: v.squeeze(1) for k, v in toy_cache2.items() if len(v.shape) == 5}
    toy_obs2 = toy_out2.obs.squeeze(1)

    toy_cache = {k: np.concatenate([v, toy_cache2[k]], axis=0) for k, v in toy_cache.items()}
    toy_obs = th.cat([toy_obs, toy_obs2], dim=0)
    toy_obs_repeated = toy_obs.repeat_interleave(reps, 0).numpy()

print("Total len:", toy_obs.shape[0], toy_cache["features_extractor.cell_list.0.hook_h"].shape[0])

# %%
restore_model()

# %%

# layer_values = {}

# for k, v in toy_cache.items():
#     if m := re.match("^.*([0-9]+)\\.hook_([h])$", k):
#         layer_values[int(m.group(1))] = v

# desired_groups_box = ["B up", "B down", "B left", "B right"]
# desired_groups_agent = ["A up", "A down", "A left", "A right"]

# desired_groups = desired_groups_box + desired_groups_agent


# channels = []
# labels = []
# group_channels = []
# for group in desired_groups:
#     group_channels.append([])
#     for layer in layer_groups[group]:
#         group_channels[-1].append((layer["layer"], layer["idx"]))
#         print(group, layer)
#         channels.append(layer_values[layer["layer"]][:, layer["idx"], :, :])
#         labels.append(f"{group} L{layer['layer']}H{layer['idx']}")

# group_channels = [g + group_channels[i + 4] for i, g in enumerate(group_channels) if i < 4]

group_channels = get_group_channels("box_agent")
group_connections = get_group_connections(group_channels)

restore_model()
# ablate connections from B down to B right
for inplc, outlc in group_connections[1][3]:
    inpl, inc = inplc
    outl, outc = outlc
    outc_ijfo = [idx * 32 + outc for idx in range(4)]
    if inpl == outl:
        model.features_extractor.cell_list[outl].conv_hh.weight.data[outc_ijfo, inc] = 0.0
    else:
        model.features_extractor.cell_list[outl].conv_ih.weight.data[outc_ijfo, inc + 32] = 0.0

# ablate connections from B right to B down
for inplc, outlc in group_connections[3][1]:
    inpl, inc = inplc
    outl, outc = outlc
    outc_ijfo = [idx * 32 + outc for idx in range(4)]
    if inpl == outl:
        model.features_extractor.cell_list[outl].conv_hh.weight.data[outc_ijfo, inc] = 0.0
    else:
        model.features_extractor.cell_list[outl].conv_ih.weight.data[outc_ijfo, inc + 32] = 0.0


# %%
play_toy = False
thinking_steps = 6
show_obs = False
size = 10
walls = [(i, 0) for i in range(size)]
walls += [(i, size - 1) for i in range(size)]
walls += [(0, i) for i in range(1, size - 1)]
walls += [(size - 1, i) for i in range(1, size - 1)]


# boxes = [(4, 6), (4, 3)]
# targets = [(4, 8), (1, 6)]

boxes = [(4, 4)]
targets = [(4, 8)]
player = (8, 1)

fwd_hooks = []

# mean_done = 0

toy_reset_opt = dict(walls=walls, boxes=boxes, targets=targets, player=player)
level_reset_opt = {"level_file_idx": 3, "level_idx": 3}
if show_obs:
    obs = envs.reset(options=toy_reset_opt)[0]
    plt.imshow(obs[0].transpose(1, 2, 0))
    plt.xticks(range(10))
else:

    def intervention(cache, hook):
        cache[:, 26, 1:8, 4] = -2.0
        # cache[:, 21, 0, 4:8] = -2.0
        return cache

    fwd_hooks = [("features_extractor.cell_list.0.hook_h.0.0", intervention)]

    reset_opts = toy_reset_opt if play_toy else level_reset_opt
    toy_out = play_level(
        envs,
        model,
        reset_opts=reset_opts,
        # reset_opts=level_reset_opt,
        thinking_steps=thinking_steps,
        max_steps=max_steps,
        # probes=[probe],
        # probe_train_ons=[probe_info],
        probe_logits=True,
        internal_steps=True,
        # fwd_hooks=fwd_hooks,
        fwd_hooks=None,
        # hook_steps=list(range(thinking_steps, 3)),
        hook_steps=[0],
    )

    toy_cache = join_cache_across_steps([toy_out.cache])
    toy_cache = {
        k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, *v.shape[-3:])
        for k, v in toy_cache.items()
        if len(v.shape) == 6
    }
    toy_obs = toy_out.obs.squeeze(1)
    print("Total len:", len(toy_obs))

    layer, batch_no = 0, 0
    batch_size = 7 + 3 * 8
    hcijfo = "h"
    show_ticks = True
    tick = reps - 1
    toy_all_channels = toy_cache[f"features_extractor.cell_list.{layer}.hook_{hcijfo}"][
        :, batch_no * batch_size : (batch_no + 1) * batch_size
    ]
    if not show_ticks:
        toy_all_channels = toy_all_channels[tick::reps]
    fig = plotly_feature_vis(
        toy_all_channels,
        toy_obs_repeated if show_ticks else toy_obs,
        feature_labels=[f"L{layer}{hcijfo.upper()}{batch_no * batch_size + i}" for i in range(batch_size)],
        show=True,
    )

# %%

pio.templates.default = "plotly"

layer_values = {}
hook_type = "i"
for k, v in toy_cache.items():
    if m := re.match(f"^.*([0-9]+)\\.hook_([{hook_type}])$", k):
        layer_values[int(m.group(1))] = v

# desired_groups = ["B up", "B down", "B left", "B right"]
# desired_groups = get_group_channels("No label", return_dict=True)
# desired_groups = get_group_channels("T", return_dict=True)
# desired_groups = get_group_channels("Other", return_dict=True)
desired_groups = get_group_channels("nfa", return_dict=True)
# desired_groups = get_group_channels("box", return_dict=True)
# desired_groups = [desired_groups[1], desired_groups[3]]

channels = []
labels = []

for group in desired_groups:
    for layer in group:
        channels.append(layer_values[layer["layer"]][:, layer["idx"], :, :])
        labels.append(f"L{layer['layer']}{hook_type.upper()}{layer['idx']}")

channels = np.stack(channels, 1)

fig = plotly_feature_vis(channels, toy_obs_repeated, feature_labels=labels)
fig.update_layout(height=800)
fig.show()


# %%

layer_values = {}

for k, v in toy_cache.items():
    if m := re.match("^.*([0-9]+)\\.hook_([h])$", k):
        layer_values[int(m.group(1))] = v

desired_groups = ["B up", "B down", "B left", "B right"]

channels = []
labels = []

for group in desired_groups:
    for layer in layer_groups[group]:
        channels.append(layer_values[layer["layer"]][:, layer["idx"], :, :])
        labels.append(f"{group} L{layer['layer']}H{layer['idx']}")

channels = np.stack(channels, 1)

fig = plotly_feature_vis(channels, toy_obs, feature_labels=labels)
fig.update_layout(height=800)
fig.show()

# %% Visualize a bunch of features

for k, v in toy_cache.items():
    if m := re.match("^.*hook_([h])$", k):
        fig = plotly_feature_vis(v, toy_obs, k, m.group(1).upper())
        fig.update_layout(height=800)
        fig.show()

# %% Highly activating forget channels
max_forget = np.stack(
    [toy_cache[f"features_extractor.cell_list.{layer}.hook_f"].max(axis=(2, 3)).mean(axis=0) for layer in range(3)]
)
max_forget_sort = np.argsort(max_forget, axis=1)[:, ::-1]
max_forget_sort_val = np.sort(max_forget, axis=1)[:, ::-1]
print(max_forget_sort[:, :16])
print(max_forget_sort_val[:, :16])


# %%
plot_layer, plot_channel = 2, 26
# tick = reps - 1
tick = 0
show_ticks = True
keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "C", "I", "J", "F", "O"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, plot_channel] for key in keys], axis=1)
# if not show_ticks:
#     toy_all_channels_for_lc = toy_all_channels_for_lc[tick::3]

# repeat obs 3 along first dimension
fig = plotly_feature_vis(
    toy_all_channels_for_lc,
    toy_obs_repeated,
    feature_labels=[k.rsplit(".")[-1] for k in keys],
    common_channel_norm=True,
)
fig.show()
# %%
# %%  fence feature
layer, batch_no = 0, 0
batch_size = 7 + 3 * 8
hcijfo = "h"
show_ticks = True
tick = reps - 1
toy_all_channels = toy_cache[f"features_extractor.cell_list.{layer}.hook_fence_conv"][
    :, batch_no * batch_size : (batch_no + 1) * batch_size
]
if not show_ticks:
    toy_all_channels = toy_all_channels[tick::reps]

fig = plotly_feature_vis(
    toy_all_channels,
    toy_obs,
    feature_labels=[f"L{layer}Fence{batch_no * batch_size + i}" for i in range(batch_size)],
    height=800,
    common_channel_norm=True,
)
fig.show()

# %%  batch: use this to visualize all channels of a layer
layer, batch_no = 0, 0
batch_size = 7 + 3 * 8
hcijfo = "h"
show_ticks = True
tick = reps - 1
toy_all_channels = toy_cache[f"features_extractor.cell_list.{layer}.hook_{hcijfo}"][
    :, batch_no * batch_size : (batch_no + 1) * batch_size
]
if not show_ticks:
    toy_all_channels = toy_all_channels[tick::reps]

toy_obs_to_plot = np.concatenate([toy_obs[:3], toy_obs[66:69]], axis=0)
toy_all_channels_to_plot = np.concatenate([toy_all_channels[:3], toy_all_channels[66:69]], axis=0)

fig = plotly_feature_vis(
    toy_all_channels,
    toy_obs,
    feature_labels=[f"L{layer}{hcijfo.upper()}{batch_no * batch_size + i}" for i in range(batch_size)],
    height=800,
    common_channel_norm=True,
)
fig.show()
# %% conv ih hh for ijo
plot_layer, plot_channel = 2, 4

keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["conv_ih", "conv_hh"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, 32 * ijo + plot_channel] for key in keys for ijo in [0, 1, 3]], axis=1)
# repeat obs 3 along first dimension
fig = plotly_feature_vis(
    toy_all_channels_for_lc,
    toy_obs,
    feature_labels=[k.rsplit(".")[-1][5:] + "_" + "ijfo"[ijo] for k in keys for ijo in [0, 1, 3]],
)
fig.show()
# %% h/c/i/j/f/o
plot_layer, plot_channel = 0, 31

keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "C", "I", "J", "F", "O"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, plot_channel] for key in keys], axis=1)
# repeat obs 3 along first dimension
fig = plotly_feature_vis(toy_all_channels_for_lc, toy_obs_repeated, feature_labels=[k.rsplit(".")[-1] for k in keys])
fig.show()

# %%
# enc_channels = [14, 2, 15, 17, 1]
# enc_channels = list(range(15, 30))
enc_channels = list(range(15))

toy_all_channels_for_lc = np.stack(
    [toy_cache["features_extractor.hook_pre_model"][:, enc_channel] for enc_channel in enc_channels], axis=1
)
# repeat obs 3 along first dimension
fig = plotly_feature_vis(toy_all_channels_for_lc, toy_obs, feature_labels=list(map(str, enc_channels)), show=True)

# %% visualize_top_conv_inputs
plot_layer, plot_channel, ih, ijfo, inp_types = 2, 4, True, "f", "lh"


def ijfo_idx(ijfo):
    return ["i", "j", "f", "o"].index(ijfo)


toy_all_channels_for_lc, top_channels, values = visualize_top_conv_inputs(
    plot_layer,
    plot_channel,
    out_type=ijfo,
    model=model,
    cache=toy_cache,
    ih=ih,
    num_channels=6 + 1 * 8,
    inp_types=inp_types,
    top_channel_sum=True,
)
plot_channel = 32 * ijfo_idx(ijfo) + plot_channel
toy_all_channels_for_lc = toy_all_channels_for_lc.numpy()
fig = plotly_feature_vis(
    toy_all_channels_for_lc,
    toy_obs,
    feature_labels=[f"{c}: {v:.2f}" for c, v in zip(top_channels, values)],  # + ["ih" if ih else "hh"],
    common_channel_norm=True,
    height=800,
)
fig.show()
# %%

# %%
get_conv_weights(1, 18, 2, model, inp_types="lh", ih=True)
# %%
pio.renderers.default = "notebook"
batch_no = 0
full_keys = []
toy_all_channels = np.stack([toy_cache["features_extractor.hook_pre_model"][:, i] for i in range(31)], axis=1)
fig = plotly_feature_vis(toy_all_channels, toy_obs, feature_labels=[f"{i}" for i in range(31)], show=True)

# %% VIS ALL FUTURE CHANNELS
down_channels = []
# future_channels = down_channels
action_channels = []
pre_mp_action_channels = []
all_action_channels = action_channels + pre_mp_action_channels
future_channels = []
# %%
# to_plot = all_action_channels
to_plot = down_channels
# to_plot = future_channels
toy_all_channels = np.stack(
    [toy_cache[f"features_extractor.cell_list.{layer}.hook_h"][:, channel] for layer, channel in to_plot],
    axis=1,
)
fig = plotly_feature_vis(
    toy_all_channels,
    toy_obs,
    feature_labels=[f"L{layer}H{channel}" for layer, channel in to_plot],
    show=True,
)

# %% Run model on given obs of toy level (useful after editing the model (directly or through hooks))
zero_carry = model.recurrent_initial_state(1)
eps_start = th.zeros((len(toy_out.obs), 1), dtype=th.bool)
(actions_solo, _, _, _), cache_solo = model.run_with_cache(toy_out.obs, zero_carry, eps_start)

zero_carry = model.recurrent_initial_state(1)
eps_start = th.zeros((len(toy_out.obs), 1), dtype=th.bool)
(actions_solo2, _, _, _), cache_solo2 = model.run_with_cache(toy_out2.obs, zero_carry, eps_start)

cache_solo = join_cache_across_steps([cache_solo])
cache_solo2 = join_cache_across_steps([cache_solo2])

cache_solo = {k: v.squeeze(1) for k, v in cache_solo.items() if len(v.shape) == 5}
cache_solo2 = {k: v.squeeze(1) for k, v in cache_solo2.items() if len(v.shape) == 5}
toy_cache = {k: np.concatenate([v, cache_solo2[k]], axis=0) for k, v in cache_solo.items()}

# %% Activation patching

grid_batch_size = 10
skip_walls = True
channel_batch_size = 1
cross_squares, cross_channels = True, True

num_layers = len(model.features_extractor.cell_list)

clean_input = get_obs(8, 2, envs)[None]
corrupted_input = corrupt_obs(clean_input, 5, 3, "floor")[None]

clean_cache, clean_log_probs = get_cache_and_probs(clean_input, model)
corrupted_cache, corrupted_log_probs = get_cache_and_probs(corrupted_input, model)
print("Clean:", clean_log_probs.argmax().item())
print("Corr:", corrupted_log_probs.argmax().item())
assert clean_log_probs.argmax() != corrupted_log_probs.argmax(), f"Same predicted action: {clean_log_probs.argmax()}"

baseline_loss = mse_loss_fn(corrupted_log_probs, clean_log_probs, corrupted_log_probs)

losses = {}
keys = list(clean_cache.keys())
# keys = [f"features_extractor.cell_list.{layer}.hook_h" for layer in range(num_layers)]
# keys = ["features_extractor.hook_pre_model"]
# keys = ["features_extractor.cell_list.2.hook_layer_input"]
keys = [f"features_extractor.cell_list.{layer}.hook_h" for layer in range(1, 3)]
# %%
for key in tqdm(keys):
    if skip_cache_key(key, skip_pre_model=False):
        continue
    pre_model = "pre_model" in key
    reps_key = 1 if pre_model else reps
    losses_for_key = th.ones((reps_key, 32, 10, 10)) * baseline_loss
    patched_preds = th.zeros((reps_key, 32, 10, 10))
    for sq_y in range(0, 10, grid_batch_size):
        y_slice = range(sq_y, sq_y + grid_batch_size)
        for sq_x in range(0, 10, grid_batch_size):
            x_slice = range(sq_x, sq_x + grid_batch_size)
            if skip_walls and np.all(clean_input[0, :, y_slice, x_slice] == WALL[:, None, None]):
                continue

            for int_step in range(reps_key):
                for channel in range(0, 32, channel_batch_size):
                    channel_slice = slice(channel, channel + channel_batch_size)
                    loss, patched_log_probs, _ = activation_patching_sq_wise(
                        [(key, channel_slice, y_slice, x_slice, int_step)],
                        corrupted_input,
                        corrupted_log_probs,
                        clean_log_probs,
                        clean_cache,
                    )
                    print(loss)
                    losses_for_key[
                        int_step, channel_slice, slice(y_slice.start, y_slice.stop), slice(x_slice.start, x_slice.stop)
                    ] = loss
                    patched_preds[int_step, channel_slice, y_slice, x_slice] = patched_log_probs.argmax().item()
    losses[map_key(key)] = (losses_for_key, patched_preds)


# %%
key = "L1H"
# key = "ENC"
fig = plotly_feature_vis(
    losses[key][0].numpy(),
    clean_input.repeat(reps, 0),
    feature_labels=[f"{key}{i}" for i in range(32)],
    common_channel_norm=True,
    zmin=0,
)
fig.show()

# %%


def sort_loss(loss):
    # loss shape: (1, 32, 10, 10). flatten all dim and return the modded indices along each dim
    assert loss.shape == (1, 32, 10, 10), f"Invalid loss shape: {loss.shape}"
    flatten_argsort = loss.flatten().argsort()
    return th.stack(th.unravel_index(flatten_argsort, loss.shape), dim=1)


key = "L0H"
min_loss_sort = sort_loss(losses[key])[:1]
print(min_loss_sort)
loss, patched_log_probs = activation_patching_sq_wise(
    [(map_to_normal_key(key), channel, sq_y, sq_x) for _, channel, sq_y, sq_x in min_loss_sort],
    corrupted_input,
    corrupted_log_probs,
    clean_log_probs,
    clean_cache,
)
print(loss)
print("Patched:", patched_log_probs.argmax().item())
# %%
# %%

box_wm_l1h = np.array([0, 4, 6, 10, 25])
# hook_prev_layer_hidden, hook_layer_input, hook_input_h
cross_squares, cross_channels = False, False
l1_act_channels = np.array([18, 17])
y_slice = [3, 4]
x_slice = [4, 4]
# y_slice, x_slice, cross_squares = range(10), range(10), True

loss, patched_log_probs, patch_cache = activation_patching_sq_wise(
    [
        (
            f"features_extractor.cell_list.{layer}.{hook_type}",
            # np.concatenate([l1_act_channels + off for off in [0,32,96]]),
            # l1_act_channels,
            [17, 18] + [19],
            # [4,23],
            # y_slice,
            # x_slice,
            # range(32),
            # box_wm_l1h,
            range(10),
            range(10),
            int_step,
        )
        for layer in [2]
        for int_step in [1]
        # for hook_type in ["hook_conv_ih", "hook_conv_hh"]
        for hook_type in [
            "hook_prev_layer_hidden",
        ]
    ],
    # + [("features_extractor.cell_list.0.hook_h", [5, 6, 9, 10, 16, 17, 26, 27], range(10), range(10), 0)],
    corrupted_input,
    corrupted_log_probs,
    clean_log_probs,
    clean_cache,
    model=model,
    # cross_squares=cross_squares,
    # cross_channels=cross_channels,
    cross_squares=True,
    cross_channels=True,
)
print(loss)
# print(patched_log_probs)
# print("Patched:", patched_log_probs.argmax().item())

# %%
cache_diff = {k: patch_cache[k] - corrupted_cache[k] for k in patch_cache}
cache_diff = join_cache_across_steps([cache_diff])
for layer in range(3):
    fig = plotly_feature_vis(
        cache_diff[f"features_extractor.cell_list.{layer}.hook_h"].squeeze(1),
        corrupted_input.squeeze(0).repeat(reps, 0),
        feature_labels=[f"{k}" for k in range(32)],
        common_channel_norm=True,
        zmin=-0.5,
    )
    fig.show()
