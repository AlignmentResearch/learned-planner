"""Majority of circuit discovery related experiments were performed in this notebook.

Changes from acdc.py: The `loss_fn` function is changed to calculate the MSE between the activations of a particular channel
of the clean and corrupted inputs. It is used to attribute the output of a channel to all the other channels.

The rest of the script plays the network on 2 levels (can be toy levels) and visualizes the activations of the network
at different layers and channels.
"""
# %%
import dataclasses
import re
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict

import numpy as np
import plotly.express as px
import plotly.io as pio
import torch as th
from cleanba.environments import BoxobanConfig
from transformer_lens.hook_points import HookPoint

from learned_planner import BOXOBAN_CACHE
from learned_planner.interp.collect_dataset import join_cache_across_steps
from learned_planner.interp.plot import plotly_feature_vis
from learned_planner.interp.train_probes import TrainOn
from learned_planner.interp.utils import load_jax_model_to_torch, load_probe, play_level
from learned_planner.interp.weight_utils import get_conv_weights, visualize_top_conv_inputs
from learned_planner.notebooks.emacs_plotly_render import set_plotly_renderer
from learned_planner.policies import download_policy_from_huggingface

set_plotly_renderer("emacs")
pio.renderers.default = "notebook"
th.set_printoptions(sci_mode=False, precision=2)


# %%
MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000"  # DRC(3, 3) 2B checkpoint
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

boxes_direction_probe_file = Path(
    "/training/TrainProbeConfig/05-probe-boxes-future-direction/wandb/run-20240813_184417-vb6474rg/local-files/probe_l-all_x-all_y-all_c-all.pkl"
)

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
model_cfg, model = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)

orig_state_dict = deepcopy(model.state_dict())


def restore_model():
    model.load_state_dict(orig_state_dict)


def map_key(key, channel):
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
    print(short_key)
    return f"features_extractor.cell_list.{int(short_key[1])}.hook_{short_key[2].lower()}"


probe, _ = load_probe("probes/best/boxes_future_direction_map_l-all.pkl")
probe_info = TrainOn(dataset_name="boxes_future_direction_map")
# %%


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

envs = dataclasses.replace(boxo_cfg, num_envs=512).make()


# %%
clean_obs = run_policy_reset(seq_len, envs, model)
corrupted_obs = run_policy_reset(seq_len, envs, model)

zero_carry = model.recurrent_initial_state(envs.num_envs)
eps_start = th.zeros((seq_len, envs.num_envs), dtype=th.bool)
eps_start[0, :] = True

(clean_actions, clean_values, clean_log_probs, _), clean_cache = model.run_with_cache(clean_obs, zero_carry, eps_start)
# Create corrupted activations which are other random levels. The hope is that, over a large data set, they will change
# the output enough times and in different enough ways that we can correctly attribute to things in the latest layer
_, corrupted_cache = model.run_with_cache(corrupted_obs, zero_carry, eps_start)

# %%
# key_pattern = [rf".*hook_{v}\.\d\.\d$" for v in ["h", "c", "i", "j", "f", "o"]]
key_pattern = [rf".*hook_{v}\.\d\.\d$" for v in ["h"]]
key_pattern += [r".*hook_pre_model$"]


def save_cache_hook(inputs: th.Tensor, hook: HookPoint):
    global cache_on_attr_run
    cache_on_attr_run[str(hook.name)] = inputs


def prune_hook(inputs: th.Tensor, hook: HookPoint, prune_channel=None):
    if prune_channel is None:
        return th.zeros_like(inputs)
    inputs[:, prune_channel] = 0
    return inputs


def multiplier_hook(inputs: th.Tensor, hook: HookPoint, prune_channel=None, multiplier=1.0):
    if prune_channel is None:
        return multiplier * inputs
    inputs[:, prune_channel] = multiplier * inputs[:, prune_channel]
    return inputs


mean_done = 0


def prune_hook_mean(inputs: th.Tensor, hook: HookPoint, prune_channel=None):
    global mean_done
    mean_done += 1
    if mean_done % 3 != 2:
        return inputs
    mean_acts = clean_cache[str(hook.name)].mean(0)
    if prune_channel is None:
        return mean_acts[None, ...]
    inputs[:, prune_channel] = mean_acts[prune_channel]
    return inputs


# %%

cache_on_attr_run = {}


def interpolate_rnn_inputs(alpha, am1_tuple, a_tuple):
    am1, _, _ = am1_tuple
    a, _, _ = a_tuple

    return ((1 - alpha) * am1 + alpha * a, *am1_tuple[1:])


def add_attributions(
    attributions: Dict[str, th.Tensor],
    loss_fn,
    model,
    clean_inputs: Any,
    corrupted_inputs: Any,
    clean_cache: Dict[str, th.Tensor],
    corrupted_minus_clean_cache: Dict[str, th.Tensor],
    *,
    ablate_at_every_hook: bool = False,
    n_gradients_to_integrate: int = 5,
    disable_recurrent_state: bool = False,
) -> Dict[str, th.Tensor]:
    """Attributes the output to every parameter in `clean_cache`. The attribution strength is added go `attributions`
    and returned. The `attributions` parameter is useful for computing the attribution in several minibatches.

    """
    assert n_gradients_to_integrate >= 1

    def set_corrupted_hook(inputs: th.Tensor, hook: HookPoint):
        nonlocal alpha
        if ablate_at_every_hook:
            desired = clean_cache[str(hook.name)] + alpha * corrupted_minus_clean_cache[str(hook.name)]
            return inputs + (desired - inputs).detach()
        return None

    def save_gradient_hook(grad: th.Tensor, hook: HookPoint):
        nonlocal attributions
        batch_dim = 1 if "pre_model" in str(hook.name) else 0
        attr = (grad.detach() * corrupted_minus_clean_cache[str(hook.name)]).sum(batch_dim).detach() / n_gradients_to_integrate
        try:
            attributions[str(hook.name)].add_(attr)
        except KeyError:
            attributions[str(hook.name)] = attr

    keys = clean_cache.keys()
    keys = [fk for fk in keys if any(re.match(k, fk) for k in key_pattern)]

    fwd_hooks = [(k, set_corrupted_hook) for k in keys]
    bwd_hooks = [(k, save_gradient_hook) for k in keys]
    if disable_recurrent_state:
        fwd_hooks += [
            (
                f"features_extractor.cell_list.{layer}.{hook_type}.{pos}.{int_pos}",
                partial(prune_hook, prune_channel=None),
            )
            for pos in range(1)
            for int_pos in range(1)
            for layer in range(3)
            for hook_type in ["hook_input_h", "hook_input_c"]
        ]
        fwd_hooks += [
            (
                "features_extractor.cell_list.0.hook_prev_layer_hidden.0.0",
                partial(prune_hook, prune_channel=None),
            )
        ]

    with model.input_dependent_hooks_context(*clean_inputs, fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
        with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
            for k in range(1, n_gradients_to_integrate):
                alpha = k / max(1, n_gradients_to_integrate - 1)

                # Check that computation is right
                assert 0.0 <= alpha <= 1.0
                if k == 0:
                    assert alpha == 0.0
                if n_gradients_to_integrate > 1 and alpha == n_gradients_to_integrate - 1:
                    assert alpha == 1.0

                loss_fn(interpolate_rnn_inputs(alpha, clean_inputs, corrupted_inputs)).backward()
                model.zero_grad()
    return attributions


comp_hook_name = "features_extractor.cell_list.0.hook_conv_hh"
comp_channel = 14 + 32 * 3
comp_ticks = range(1, 3)
positions = range(10)


def loss_fn(inputs):
    global cache_on_attr_run
    obs, init_carry, eps_start = inputs
    # dist, _carry = model.get_distribution(obs, init_carry, eps_start)
    # logits = dist.distribution.log_prob(clean_actions)
    cache_on_attr_run = {}
    fwd_hooks = [(f"{comp_hook_name}.{pos}.{int_pos}", save_cache_hook) for pos in positions for int_pos in comp_ticks]
    with model.input_dependent_hooks_context(*inputs, fwd_hooks=fwd_hooks, bwd_hooks=[]):
        with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=[]):
            model(obs, init_carry, eps_start)
    clean_acts = th.stack(
        [clean_cache[f"{comp_hook_name}.{pos}.{int_pos}"][:, comp_channel] for pos in positions for int_pos in comp_ticks],
        dim=0,
    )
    corr_acts = th.stack(
        [
            cache_on_attr_run[f"{comp_hook_name}.{pos}.{int_pos}"][:, comp_channel]
            for pos in positions
            for int_pos in comp_ticks
        ],
        dim=0,
    )
    bw_tensor = ((clean_acts - corr_acts) ** 2).sum(dim=(-1, -2)).mean()
    print(bw_tensor)
    # bw_tensor = logits.sum()
    return bw_tensor


# %%

attributions = add_attributions(
    {},
    loss_fn,
    model,
    clean_inputs=(clean_obs, zero_carry, eps_start),
    corrupted_inputs=(corrupted_obs, zero_carry, eps_start),
    clean_cache={k: v.detach() for k, v in clean_cache.items()},
    corrupted_minus_clean_cache={k: (v - corrupted_cache[k]).detach() for k, v in clean_cache.items()},
    ablate_at_every_hook=False,
    n_gradients_to_integrate=2,
    disable_recurrent_state=True,
)

# %%
mean_attributions = {}
mean_attributions_channels = {}
do_abs = True
keep_ticks = False
sum_pos = True

for k, v in attributions.items():
    if keep_ticks:
        new_key = k.rsplit(".", 2)
        new_key[-2] = "_"
        new_key = ".".join(new_key)
        if new_key in mean_attributions:
            mean_attributions[new_key] += v.abs() if do_abs else v
        else:
            mean_attributions[new_key] = v.abs() if do_abs else v
    else:
        new_key = k if (not sum_pos) or "hook_pre_model" in k else k.rsplit(".", 1)[0]
        if new_key in mean_attributions:
            mean_attributions[new_key] += v.abs() if do_abs else v
        else:
            mean_attributions[new_key] = v.abs() if do_abs else v
for k, v in mean_attributions.items():
    if "hook_pre_model" in k:
        v = v.sum(0)
    for c in range(v.shape[0]):
        mean_attributions_channels[map_key(k, c)] = v[c].mean().item()

mean_attributions_channels = sorted(mean_attributions_channels.items(), key=lambda x: x[1], reverse=True)
# %%
mean_attributions_channels[:25]

# %% SINGLE ENV

# envs = dataclasses.replace(boxo_cfg, num_envs=1, difficulty="unfiltered").make()
envs = dataclasses.replace(boxo_cfg, num_envs=1).make()

# %%
# fwd_hooks = [
#     (f"features_extractor.cell_list.0.hook_h.0.{int_pos}", partial(prune_hook, prune_channel=[17])) for int_pos in range(3)
# ]
for channel in range(1):
    diffs = 0
    for level_idx in range(1, 2):
        fwd_hooks = []
        out = play_level(
            envs,
            model,
            reset_opts=dict(level_file_idx=0, level_idx=level_idx),
            thinking_steps=0,
            fwd_hooks=fwd_hooks,
        )

        cache = join_cache_across_steps([out.cache])
        cache = {
            k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, 32, 10, 10)
            for k, v in cache.items()
            if len(v.shape) == 6
        }
        toy_cache = cache
        toy_obs = out.obs.squeeze(1)
        toy_obs = toy_obs.repeat_interleave(3, 0).numpy()
        len_before = len(toy_obs)

        # fwd_hooks = [
        #     (f"features_extractor.cell_list.2.hook_h.0.{int_pos}", partial(prune_hook, prune_channel=[channel]))
        #     for int_pos in range(3)
        # ]
    #     fwd_hooks = [
    #         (
    #             f"features_extractor.cell_list.0.hook_h.0.{int_pos}",
    #             partial(multiplier_hook, prune_channel=[channel], multiplier=5.0),
    #         )
    #         for int_pos in range(3)
    #     ]
    #     # fwd_hooks = []
    #     out = play_level(
    #         envs,
    #         model,
    #         reset_opts=dict(level_file_idx=0, level_idx=level_idx),
    #         thinking_steps=0,
    #         fwd_hooks=fwd_hooks,
    #     )

    #     cache = join_cache_across_steps([out.cache])
    #     cache = {
    #         k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, 32, 10, 10)
    #         for k, v in cache.items()
    #         if len(v.shape) == 6
    #     }
    #     toy_cache = cache
    #     toy_obs = out.obs.squeeze(1)
    #     toy_obs = toy_obs.repeat_interleave(3, 0).numpy()

    #     if len(toy_obs) > len_before and len(toy_obs) != 3 * 120:
    #         diffs += len(toy_obs) - len_before
    # print(channel, diffs / (20 * 3))

# %%
restore_model()
# model.features_extractor.cell_list[1].conv_ih.weight.data[32 * 3 + 25][32 * 0 : 32 * 1] = 0
# model.features_extractor.cell_list[1].conv_ih.weight.data[32 * 3 + 25][32 * 2 : 32 * 3] = 0
# model.features_extractor.cell_list[1].conv_ih.weight.data[32*3 + 25][32*1 + 6] = orig_state_dict["features_extractor.cell_list.1.conv_ih.weight"][32*3 + 25][32*1 + 6]

# %% PLAY TOY LEVEL

envs = dataclasses.replace(boxo_cfg, num_envs=1).make()
thinking_steps = 0
max_steps = 50
size = 10
walls = [(i, 0) for i in range(size)]
walls += [(i, size - 1) for i in range(size)]
walls += [(0, i) for i in range(1, size - 1)]
walls += [(size - 1, i) for i in range(1, size - 1)]

walls += [(y, x) for y in range(4, 6) for x in range(3, 6)]

boxes = [(3, 2)]
targets = [(8, 8)]
player = (1, 1)


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


prune_keys = [
    # ("L1H21T1", 0.02339771017432213),
    # ("L1H24T1", 0.022313304245471954),
    # ("L1H25T0", 0.019601810723543167),
    # ("L2I4T0", 0.018496893346309662),
    # ("L1H25T1", 0.0171161238104105),
    # ("L2H4T0", 0.01660654880106449),
    # ("L1J25T0", 0.016340060159564018),
    # ("L2H14T0", 0.016049379482865334),
    # ("L1I24T1", 0.015933256596326828),
    # ("L1O25T0", 0.015400230884552002),
    # ("L1O21T1", 0.014821980148553848),
    # ("L2O5T0", 0.014550766907632351),
    # ("L1O10T0", 0.014471199363470078),
    # ("L1H10T0", 0.014086911454796791),
    # ("L1I25T0", 0.014048771932721138),
    # ("L0O5T1", 0.013477562926709652),
    # ("L1H6T1", 0.013180520385503769),
    # ("L0H5T1", 0.013009640388190746),
    # ("L1J25T1", 0.012674916535615921),
    # ("L1H5T1", 0.012244414538145065),
    # ("L0H6T0", 0.01222571637481451),
    # ("L1O25T1", 0.011620094999670982),
    # ("L1H25T2", 0.011552662588655949),
    # ("L1O10T1", 0.011412248946726322),
    # ("L1I25T1", 0.011186463758349419),
    # ("L1O10T2", 0.010042699985206127),
    # ("L1O9T1", 0.00985176581889391),
    # ("L1H18T1", 0.009820137172937393),
    # ("L1O6T1", 0.009801537729799747),
    # ("L1H10T1", 0.00979539193212986),
]
prune_keys = [k for k, _ in prune_keys]

fwd_hooks = []
# fwd_hooks = [
#     (
#         f"features_extractor.cell_list.{layer}.hook_{hook_type}.{pos}.{int_pos}",
#         partial(prune_hook, prune_channel=prune_channel),
#     )
#     for pos in range(1)
#     # for hook_name, prune_channel in [("hook_h", 25), ("hook_h", 21), ("hook_h", 10)]
#     for layer, hook_type, prune_channel, tick_pos in [get_hook_info(short_key, set_ticks=False) for short_key in prune_keys]
#     for int_pos in (tick_pos if tick_pos is not None else range(0, 3))
# ]

# fwd_hooks = [
#     (
#         f"features_extractor.cell_list.{layer}.{hook_type}.{pos}.{int_pos}",
#         partial(prune_hook, prune_channel=None),
#     )
#     for pos in range(1)
#     for int_pos in range(1)
#     for layer in range(3)
#     for hook_type in ["hook_input_h", "hook_input_c"]
# ]
# fwd_hooks += [
#     (
#         "features_extractor.cell_list.0.hook_prev_layer_hidden.0.0",
#         partial(prune_hook, prune_channel=None),
#     )
# ]
# fwd_hooks += [
#     (f"features_extractor.cell_list.0.hook_h.0.{int_pos}", partial(prune_hook, prune_channel=[26])) for int_pos in range(3)
# ]

# mean_done = 0
level_reset_opt = {"level_file_idx": 0, "level_idx": 2}
toy_out = play_level(
    envs,
    model,
    # reset_opts=dict(walls=walls, boxes=boxes, targets=targets, player=player),
    reset_opts=level_reset_opt,
    thinking_steps=thinking_steps,
    fwd_hooks=fwd_hooks,
    max_steps=max_steps,
    hook_steps=list(range(thinking_steps, max_steps)) if thinking_steps > 0 else -1,
    probes=[probe],
    probe_train_ons=[probe_info],
    probe_logits=True,
    internal_steps=True,
)

toy_cache = join_cache_across_steps([toy_out.cache])
toy_cache = {
    k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, *v.shape[-3:])
    for k, v in toy_cache.items()
    if len(v.shape) == 6
}
toy_obs = toy_out.obs.squeeze(1)


boxes = [(7, 7)]
targets = [(1, 1)]
player = (8, 8)
mean_done = 0
toy_out2 = play_level(
    envs,
    model,
    # reset_opts=dict(walls=walls, boxes=boxes, targets=targets, player=player),
    reset_opts={**level_reset_opt, "level_idx": level_reset_opt["level_idx"] + 1},
    thinking_steps=thinking_steps,
    fwd_hooks=fwd_hooks,
    max_steps=max_steps,
    hook_steps=list(range(thinking_steps, max_steps)) if thinking_steps > 0 else -1,
    probes=[probe],
    probe_train_ons=[probe_info],
    probe_logits=True,
    internal_steps=True,
)
toy_cache2 = join_cache_across_steps([toy_out2.cache])
toy_cache2 = {
    k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, *v.shape[-3:])
    for k, v in toy_cache2.items()
    if len(v.shape) == 6
}
toy_obs2 = toy_out2.obs.squeeze(1)

toy_cache = {k: np.concatenate([v, toy_cache2[k]], axis=0) for k, v in toy_cache.items()}
toy_obs_non_rep = th.cat([toy_obs, toy_obs2], dim=0)
toy_obs = toy_obs_non_rep.repeat_interleave(3, 0).numpy()
print("Total len:", len(toy_obs))

# %%
batch_no = 0
short_keys = [k for k, v in mean_attributions_channels[batch_no * 15 : (batch_no + 1) * 15] if not k.startswith("ENC")]
toy_all_channels = np.stack(
    [toy_cache[map_to_normal_key(short_key)][:, int(short_key.split("P")[0][3:])] for short_key in short_keys], axis=1
)
fig = plotly_feature_vis(toy_all_channels, toy_obs, feature_labels=short_keys, show=True)

# %%
plot_layer, plot_channel = 0, 9
tick = 2
show_ticks = True
keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "C", "I", "J", "F", "O"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, plot_channel] for key in keys], axis=1)
if not show_ticks:
    toy_all_channels_for_lc = toy_all_channels_for_lc[tick::3]
# repeat obs 3 along first dimension
fig = plotly_feature_vis(
    toy_all_channels_for_lc,
    toy_obs if show_ticks else toy_obs_non_rep,
    feature_labels=[k.rsplit(".")[-1] for k in keys],
    show=True,
)
# %%  batch
layer, batch_no = 0, 20
tick = 2
toy_all_channels = toy_cache[f"features_extractor.cell_list.{layer}.hook_h"][:, batch_no * 15 : (batch_no + 1) * 15]
if not show_ticks:
    toy_all_channels = toy_all_channels[tick::3]
fig = plotly_feature_vis(
    toy_all_channels,
    toy_obs if show_ticks else toy_obs_non_rep,
    feature_labels=[f"L{layer}H{batch_no * 15 + i}" for i in range(15)],
    show=True,
)


# %% conv ih hh for ijo
plot_layer, plot_channel = 1, 18

keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["conv_ih", "conv_hh"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, 32 * ijo + plot_channel] for key in keys for ijo in [0, 1, 3]], axis=1)
# repeat obs 3 along first dimension
fig = plotly_feature_vis(
    toy_all_channels_for_lc,
    toy_obs,
    feature_labels=[k.rsplit(".")[-1][5:] + "_" + "ijfo"[ijo] for k in keys for ijo in [0, 1, 3]],
    show=True,
)

# %% h/c/i/j/f/o
plot_layer, plot_channel = 2, 19

keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "C", "I", "J", "F", "O"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, plot_channel] for key in keys], axis=1)
# repeat obs 3 along first dimension
fig = plotly_feature_vis(toy_all_channels_for_lc, toy_obs, feature_labels=[k.rsplit(".")[-1] for k in keys], show=True)

# %%
# enc_channels = [14, 2, 15, 17, 1]
enc_channels = list(range(15, 30))

toy_all_channels_for_lc = np.stack(
    [toy_cache["features_extractor.hook_pre_model"][:, enc_channel] for enc_channel in enc_channels], axis=1
)
# repeat obs 3 along first dimension
fig = plotly_feature_vis(toy_all_channels_for_lc, toy_obs_non_rep, feature_labels=list(map(str, enc_channels)), show=True)

# %% visualize_top_conv_inputs
plot_layer, plot_channel, ih, ijfo, inp_types = 1, 18, True, "o", "lh"


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
    show=True,
    common_channel_norm=True,
)

# %%
get_conv_weights(1, 18, 2, model, inp_types="lh", ih=True)
# %%
pio.renderers.default = "notebook"
batch_no = 0
full_keys = []
toy_all_channels = np.stack([toy_cache["features_extractor.hook_pre_model"][:, i] for i in range(31)], axis=1)
fig = plotly_feature_vis(toy_all_channels, toy_obs_non_rep, feature_labels=[f"{i}" for i in range(31)], show=True)

# %% VIS ALL FUTURE CHANNELS
down_channels = [(0, 2), (0, 14), (0, 28), (1, 17), (1, 18), (2, 3), (2, 4)]
# future_channels = down_channels
action_channels = [(2, 3), (2, 8), (2, 27), (2, 29)]
pre_mp_action_channels = [(2, 28), (2, 4), (2, 23), (2, 26)]
all_action_channels = action_channels + pre_mp_action_channels
future_channels = [
    (0, 2),
    (0, 14),
    (0, 11),
    (0, 17),
    (0, 26),
    (1, 4),
    (1, 7),
    (1, 9),
    (1, 10),
    (1, 15),
    (1, 17),
    (1, 18),
    (1, 21),
    (1, 24),
    (1, 25),
    (2, 1),
    (2, 4),
    (2, 5),
    (2, 9),
    (2, 17),
]
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
# %%
all_activations = []
for layer, channel in future_channels:
    # Stack all positions and internal positions for this channel
    acts = th.stack(
        [
            clean_cache[f"features_extractor.cell_list.{layer}.hook_h.{pos}.{int_pos}"][:, channel]
            for pos in range(10)
            for int_pos in range(3)
        ],
        dim=0,
    )
    all_activations.append(acts.flatten().numpy())

# Convert to numpy array and compute correlation matrix in one step
all_activations = np.array(all_activations)
correlation_matrix = np.abs(np.corrcoef(all_activations))
np.fill_diagonal(correlation_matrix, 0)

# %%
correlation_matrix[np.triu_indices(correlation_matrix.shape[0], k=1)] = 0
fig = px.imshow(
    np.abs(correlation_matrix),
    x=[f"L{layer}H{channel}" for layer, channel in future_channels],
    y=[f"L{layer}H{channel}" for layer, channel in future_channels],
)
fig.show()

# %% Correlation between I/J/F/O
all_ijfo = [[], [], [], []]
for layer in range(3):
    for idx, hook_type in enumerate(["i", "j", "f", "o"]):
        # Stack all positions and internal positions for this channel
        acts = th.cat(
            [
                clean_cache[f"features_extractor.cell_list.{layer}.hook_{hook_type}.{pos}.{int_pos}"]
                for pos in range(10)
                for int_pos in range(3)
            ],
            dim=0,
        )
        all_ijfo[idx].append(np.transpose(acts.numpy(), (1, 0, 2, 3)))

# Convert to numpy array and compute correlation matrix in one step
all_ijfo = np.array(all_ijfo)
# correlation between I/J/F/O for each layer for each channel
all_ijfo = all_ijfo.reshape(4 * 3 * 32, -1)
correlation_matrix_ijfo = np.abs(np.corrcoef(all_ijfo))
np.fill_diagonal(correlation_matrix_ijfo, 0)
correlation_matrix_ijfo = correlation_matrix_ijfo.reshape(4, 3, 32, 4, 3, 32)
# average correlation between I/J/F/O for each layer for each channel
correlation_matrix_ijfo = correlation_matrix_ijfo.sum(axis=(0, 3))
correlation_matrix_ijfo = correlation_matrix_ijfo / 12

# %% which channels break the model
breaking_channels = []
# for idx, (l, c) in enumerate([(2, c) for c in range(32)]):
for idx, (ly, c) in enumerate(future_channels):
    walls = [(i, 0) for i in range(size)]
    walls += [(i, size - 1) for i in range(size)]
    walls += [(0, i) for i in range(1, size - 1)]
    walls += [(size - 1, i) for i in range(1, size - 1)]

    walls += [(y, x) for y in range(4, 6) for x in range(3, 6)]

    boxes = [(3, 2)]
    targets = [(8, 8)]
    player = (1, 1)
    fwd_hooks = [
        (
            f"features_extractor.cell_list.{layer}.{hook_type}.{pos}.{int_pos}",
            partial(prune_hook, prune_channel=None),
        )
        for pos in range(1)
        for int_pos in range(1)
        for layer in range(3)
        for hook_type in ["hook_input_h", "hook_input_c"]
    ]
    fwd_hooks += [
        (
            "features_extractor.cell_list.0.hook_prev_layer_hidden.0.0",
            partial(prune_hook, prune_channel=None),
        )
    ]
    fwd_hooks += [
        (f"features_extractor.cell_list.{ly}.hook_h.0.{int_pos}", partial(prune_hook, prune_channel=[c]))
        for int_pos in range(3)
    ]

    # mean_done = 0
    toy_out = play_level(
        envs,
        model,
        reset_opts=dict(walls=walls, boxes=boxes, targets=targets, player=player),
        thinking_steps=0,
        fwd_hooks=fwd_hooks,
        max_steps=30,
    )

    toy_cache = join_cache_across_steps([toy_out.cache])
    toy_cache = {
        k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, *v.shape[-3:])
        for k, v in toy_cache.items()
        if len(v.shape) == 6
    }
    toy_obs = toy_out.obs.squeeze(1)

    boxes = [(7, 7)]
    targets = [(1, 1)]
    player = (8, 8)
    mean_done = 0
    toy_out2 = play_level(
        envs,
        model,
        reset_opts=dict(walls=walls, boxes=boxes, targets=targets, player=player),
        thinking_steps=0,
        fwd_hooks=fwd_hooks,
        max_steps=30,
    )
    toy_cache2 = join_cache_across_steps([toy_out2.cache])
    toy_cache2 = {
        k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, *v.shape[-3:])
        for k, v in toy_cache2.items()
        if len(v.shape) == 6
    }
    toy_obs2 = toy_out2.obs.squeeze(1)

    toy_cache = {k: np.concatenate([v, toy_cache2[k]], axis=0) for k, v in toy_cache.items()}
    toy_obs_non_rep = th.cat([toy_obs, toy_obs2], dim=0)
    toy_obs = toy_obs_non_rep.repeat_interleave(3, 0).numpy()
    print("Total len:", len(toy_obs), f"| L{ly}H{c}")
    if len(toy_obs) >= 180:
        breaking_channels.append(idx)

# %%
breaking_channels_corr = correlation_matrix[np.ix_(breaking_channels, breaking_channels)]
mean_bc_corr = breaking_channels_corr[breaking_channels_corr != 0].mean()
mean_corr = correlation_matrix[correlation_matrix != 0].mean()
print(mean_bc_corr, mean_corr)

# %% probe
# pyright: ignore[reportMissingImports]
import shap  # noqa  # pyright: ignore[reportMissingImports]


def pretty_feature_value(values, sort=True, f=True):
    assert len(values) == 32 * 3 * 2, f"Expected 192 values, got {len(values)}"
    sorted_pos = values.argsort()[::-1] if sort else range(len(values))
    fmt = "{:.1f}" if f else "{:.1}"
    return [f"L{idx // 64}{'HC'[(idx % 64) // 32]}{idx % 32}: {fmt.format(values[idx])}" for idx in sorted_pos]


feature_names = [f"L{idx // 64}{'HC'[(idx % 64) // 32]}{idx % 32}" for idx in range(32 * 3 * 2)]
down_probe = probe.coef_[2]
top_down_channels = [
    f"L{idx // 64}{'HC'[(idx % 64) // 32]}{idx % 32}: {down_probe[idx]:.1f}" for idx in np.abs(down_probe).argsort()[::-1]
]
print(top_down_channels)

probe_keys = [f"features_extractor.cell_list.{layer}.hook_{hook_type}" for layer in range(3) for hook_type in ["h", "c"]]
assert toy_out.probe_outs is not None and toy_out2.probe_outs is not None
probe_preds = np.concatenate([toy_out.probe_outs[0], toy_out2.probe_outs[0]], axis=0).reshape(
    -1, *toy_out.probe_outs[0].shape[2:]
)
probe_preds = probe_preds[..., 2]  # down probe
probe_inputs = np.concatenate([toy_cache[key] for key in probe_keys], axis=1)
probe_inputs = np.transpose(probe_inputs, (0, 2, 3, 1))
# calculate fraction of variance explained by each channel in the probe
X = probe_inputs.reshape(-1, probe_inputs.shape[-1])
explainer = shap.Explainer(probe, X)
shap_values = explainer(X)
shap.summary_plot(shap_values[..., 2], X, feature_names=feature_names)  # down probe

indep_shap_values = down_probe[None] * (X - X.mean(0))
shap.summary_plot(indep_shap_values, X, feature_names=feature_names)

shap.plots.bar(shap_values)


# %% L0H17
size = 10
walls = [(i, 0) for i in range(size)]
walls += [(i, size - 1) for i in range(size)]
walls += [(0, i) for i in range(1, size - 1)]
walls += [(size - 1, i) for i in range(1, size - 1)]

targets = [(i, 9 - i) for i in range(1, 9)]
fwd_hooks = []
toy_out = play_level(
    envs,
    model,
    reset_opts=dict(walls=walls, boxes=targets, targets=[], player=()),
    thinking_steps=0,
    fwd_hooks=fwd_hooks,
    max_steps=1,
)
toy_cache = join_cache_across_steps([toy_out.cache])
toy_cache = {
    k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, *v.shape[-3:])
    for k, v in toy_cache.items()
    if len(v.shape) == 6
}
toy_obs = toy_out.obs.squeeze(1)

toy_out = play_level(
    envs,
    model,
    reset_opts=dict(walls=walls, boxes=[], targets=[(8, 8)], player=[(i, 9 - i) for i in range(1, 9)]),
    thinking_steps=0,
    fwd_hooks=fwd_hooks,
    max_steps=1,
)
toy_cache = join_cache_across_steps([toy_out.cache])
toy_cache = {
    k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, *v.shape[-3:])
    for k, v in toy_cache.items()
    if len(v.shape) == 6
}
toy_obs = toy_out.obs.squeeze(1)
plot_layer, plot_channel = 0, 17
show_ticks = False
keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "C", "I", "J", "F", "O"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, plot_channel] for key in keys], axis=1)
if not show_ticks:
    toy_all_channels_for_lc = toy_all_channels_for_lc[::3]
# repeat obs 3 along first dimension
fig = plotly_feature_vis(
    toy_all_channels_for_lc,
    toy_obs,
    feature_labels=[k.rsplit(".")[-1] for k in keys],
    show=True,
)
# %% L1H15
size = 10
walls = [(i, 0) for i in range(size)]
walls += [(i, size - 1) for i in range(size)]
walls += [(0, i) for i in range(1, size - 1)]
walls += [(size - 1, i) for i in range(1, size - 1)]

targets = [(i, 8) for i in range(1, 9, 2)]
boxes = [(i, 5) for i in range(1, 5, 2)] + [(i, 4) for i in range(5, 9, 2)]
targets, boxes = targets[:1] + targets[-1:], boxes[:1] + boxes[-1:]
# targets, boxes = targets[-1:], boxes[-1:]
fwd_hooks = []
toy_out = play_level(
    envs,
    model,
    reset_opts=dict(walls=walls, boxes=boxes, targets=targets, player=(1, 1)),
    thinking_steps=0,
    fwd_hooks=fwd_hooks,
    max_steps=1,
)
toy_cache = join_cache_across_steps([toy_out.cache])
toy_cache = {
    k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, *v.shape[-3:])
    for k, v in toy_cache.items()
    if len(v.shape) == 6
}
toy_obs = toy_out.obs.squeeze(1)

# toy_out = play_level(
#     envs,
#     model,
#     reset_opts=dict(walls=walls, boxes=[], targets=[(8, 8)], player=[(i, 9 - i) for i in range(1, 9)]),
#     thinking_steps=0,
#     fwd_hooks=fwd_hooks,
#     max_steps=1,
# )
# toy_cache = join_cache_across_steps([toy_out.cache])
# toy_cache = {
#     k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, *v.shape[-3:])
#     for k, v in toy_cache.items()
#     if len(v.shape) == 6
# }
# toy_obs = toy_out.obs.squeeze(1)
plot_layer, plot_channel = 1, 15
show_ticks = False
keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "C", "I", "J", "F", "O"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, plot_channel] for key in keys], axis=1)
if not show_ticks:
    toy_all_channels_for_lc = toy_all_channels_for_lc[::3]
# repeat obs 3 along first dimension
fig = plotly_feature_vis(
    toy_all_channels_for_lc,
    toy_obs,
    feature_labels=[k.rsplit(".")[-1] for k in keys],
    show=True,
)

# %%
