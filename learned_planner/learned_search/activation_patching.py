"""Activation patching experiment script.

The `activation_patching` function performs activation patching experiment within the same level from previous timestep to the latest timestep
and returns channels and their attribution scores.
"""
# %%
import dataclasses
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import plotly.io as pio
import torch as th
from cleanba.environments import BoxobanConfig
from tqdm import tqdm

from learned_planner.interp.collect_dataset import join_cache_across_steps
from learned_planner.interp.plot import plotly_feature_vis
from learned_planner.interp.utils import load_jax_model_to_torch, play_level
from learned_planner.notebooks.emacs_plotly_render import set_plotly_renderer
from learned_planner.policies import download_policy_from_huggingface

pio.renderers.default = "notebook"
set_plotly_renderer("emacs")

# %%
MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000"  # DRC(3, 3) 2B checkpoint
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

# try:
#     BOXOBAN_CACHE = Path(__file__).parent.parent.parent / ".sokoban_cache"
# except NameError:
#     BOXOBAN_CACHE = Path(os.getcwd()) / ".sokoban_cache"
BOXOBAN_CACHE = Path("/opt/sokoban_cache")

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
    split="train",
    difficulty="medium",
)
model_cfg, model = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)

orig_state_dict = deepcopy(model.state_dict())


def restore_model():
    model.load_state_dict(orig_state_dict)


# %%

envs = dataclasses.replace(boxo_cfg, num_envs=1).make()
thinking_steps = 0
max_steps = 30
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


fwd_hooks = []
toy_out = play_level(
    envs,
    model,
    reset_opts=dict(walls=walls, boxes=boxes, targets=targets, player=player),
    thinking_steps=thinking_steps,
    fwd_hooks=fwd_hooks,
    max_steps=max_steps,
    hook_steps=list(range(thinking_steps, max_steps)) if thinking_steps > 0 else -1,
)

toy_cache = join_cache_across_steps([toy_out.cache])
toy_cache = {
    k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, *v.shape[-3:])
    for k, v in toy_cache.items()
    if len(v.shape) == 6
}
toy_obs = toy_out.obs.squeeze(1)
toy_obs_rep = np.repeat(toy_obs, 3, axis=0)

# %%

target = toy_cache["features_extractor.cell_list.1.hook_h"][5, 17, 2, 4:8]
value = toy_cache["features_extractor.cell_list.1.hook_h"][8, 17, 2, 4:8]
value_timestep = 2


def skip_key(key, skip_pre_model=True):
    if "fence" in key:
        return True
    if "mlp" in key:
        return True
    if "pool" in key:
        return True
    if skip_pre_model and "pre_model" in key:
        return True
    return False


def activation_patching():
    def sq_wise_intervention_hook(input, hook, sq_i, sq_j, channel, timestep):
        name = hook.name.rsplit(".", 2)[0]
        input[:, channel, sq_i, sq_j] = th.tensor(toy_cache[name][timestep, channel, sq_i, sq_j])
        return input

    obs = toy_obs[value_timestep].unsqueeze(0)
    carry = [
        [
            th.tensor(toy_cache[f"features_extractor.cell_list.{layer}.hook_{h_or_c}"][3 * value_timestep - 1])
            .unsqueeze(0)
            .unsqueeze(0)
            for h_or_c in ["h", "c"]
        ]
        for layer in range(3)
    ]
    eps_start = th.zeros((1,), dtype=th.bool)

    results = {}

    for key in tqdm(toy_cache.keys()):
        if skip_key(key):
            continue
        for src_timestep in range(2):
            for int_step in range(2):
                for channel in range(32):
                    sq_i, sq_j = slice(None), slice(None)
                    # for sq_i in range(10):
                    #     for sq_j in range(10):
                    fwd_hooks = [
                        (
                            key + f".0.{int_step}",
                            partial(
                                sq_wise_intervention_hook,
                                sq_i=sq_i,
                                sq_j=sq_j,
                                channel=channel,
                                timestep=src_timestep,
                            ),
                        )
                    ]
                    with model.input_dependent_hooks_context(
                        obs,
                        carry,
                        eps_start,
                        fwd_hooks=fwd_hooks,
                        bwd_hooks=[],
                    ):
                        with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=[]):
                            _, interv_cache = model.run_with_cache(obs, carry, eps_start, deterministic=True)
                            pred = interv_cache["features_extractor.cell_list.0.hook_h.0.2"][0, 17, 2, 4:8]
                            sq_i = -1 if sq_i == slice(None) else sq_i
                            sq_j = -1 if sq_j == slice(None) else sq_j
                            results[(key, src_timestep, int_step, channel, sq_i, sq_j)] = np.mean((pred.numpy() - target) ** 2)
    return results


results = activation_patching()
# sort results
results = sorted(results.items(), key=lambda x: x[1])

# %%
inp_types, out_types = ["e", "lh", "ch"], ["i", "j", "f", "o"]


def show_conv_ih(layer, out, inp, out_type="o", inp_type="lh"):
    assert inp_type in inp_types
    assert out_type in out_types
    if isinstance(inp, int):
        inp += 32 * inp_types.index(inp_type)
    else:
        inp = slice(32 * inp_types.index(inp_type), 32 * (inp_types.index(inp_type) + 1))
    out_offset = 32 * out_types.index(out_type)
    return model.features_extractor.cell_list[layer].conv_ih.weight.data[out_offset + out, inp]


def top_weights(layer, out, out_type="o", inp_type="lh"):
    top_channels = show_conv_ih(layer, out, None, out_type, inp_type).abs().max(dim=1).values.max(dim=1).values
    return top_channels.argsort(descending=True)


def top_weights_out(layer, inp, out_type="o", inp_type="lh"):
    next_layer = (layer + 1) % 3
    inp = 32 * inp_types.index(inp_type) + inp
    out_idx = out_types.index(out_type)
    top_channels = (
        model.features_extractor.cell_list[next_layer]
        .conv_ih.weight.data[32 * out_idx : 32 * (out_idx + 1), inp]
        .abs()
        .max(dim=1)
        .values.max(dim=1)
        .values
    )
    return top_channels.argsort(descending=True)


# %%

plot_layer, plot_channel = 0, 7

keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "C", "I", "J", "F", "O"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, plot_channel] for key in keys], axis=1)
# repeat obs 3 along first dimension
fig = plotly_feature_vis(toy_all_channels_for_lc, toy_obs_rep, feature_labels=[k.rsplit(".")[-1] for k in keys], show=True)

# %%
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
    (2, 14),
    (2, 17),
]
toy_all_channels = np.stack(
    [toy_cache[f"features_extractor.cell_list.{layer}.hook_h"][:, channel] for layer, channel in future_channels],
    axis=1,
)
fig = plotly_feature_vis(
    toy_all_channels,
    toy_obs_rep,
    feature_labels=[f"L{layer}H{channel}" for layer, channel in future_channels],
    show=True,
)

# %%
