# ruff: noqa
# %%
import concurrent.futures
import dataclasses
import re
from copy import deepcopy
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import torch as th
import tqdm
from cleanba.environments import BoxobanConfig
from stable_baselines3.common.distributions import CategoricalDistribution
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint

from learned_planner import LP_DIR, ON_CLUSTER
from learned_planner.convlstm import ConvLSTMCell
from learned_planner.interp.collect_dataset import (ChannelCoefs, DatasetStore,
                                                    HelperFeatures)
from learned_planner.interp.plot import plotly_feature_vis
from learned_planner.interp.utils import (get_cache_and_probs,
                                          join_cache_across_steps,
                                          load_jax_model_to_torch, pad_level,
                                          parse_level, play_level,
                                          run_fn_with_cache)
from learned_planner.interp.weight_utils import get_conv_weights
from learned_planner.notebooks.emacs_plotly_render import set_plotly_renderer
from learned_planner.policies import download_policy_from_huggingface

set_plotly_renderer("emacs")
pio.renderers.default = "notebook"
th.set_printoptions(sci_mode=False, precision=2)

# cache_path = Path("/training/activations_dataset/hard/0_think_step")
if ON_CLUSTER:
    cache_path = Path("/training/activations_dataset/train_medium/0_think_step")
    N_FILES = 1001
else:
    cache_path = LP_DIR / "drc33_cache/0_think_step"
    N_FILES = 500


def load_data_h():
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:

        def map_fn(i):
            data = pd.read_pickle(cache_path / f"idx_{i}.pkl")
            # return {
            #     "obs": data.obs,
            #     "encoded": data.model_cache["features_extractor.hook_pre_model"],
            #     "actions": data.pred_actions,
            # }
            return [data.model_cache[f"features_extractor.cell_list.{layer}.hook_h"] for layer in range(3)]

        loaded_data = list(tqdm.tqdm(executor.map(map_fn, range(N_FILES)), total=N_FILES))
        layer_wise_h = [th.cat([th.tensor(d[layer]) for d in loaded_data], dim=0) for layer in range(3)]
    return layer_wise_h


# %%
# MODEL_PATH_IN_REPO = "drc11/eue6pax7/cp_2002944000"  # DRC(1, 1) 2B checkpoint
MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000"  # DRC(1, 1) 2B checkpoint
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

num_envs = 1
boxo_cfg = BoxobanConfig(
    cache_path=LP_DIR / "alternative-levels/levels/",
    num_envs=num_envs,
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
def identity(feature: np.ndarray | th.Tensor):
    return feature


def left_shift(feature: np.ndarray | th.Tensor):
    if isinstance(feature, th.Tensor):
        return th.roll(feature, -1, dims=-2)
    else:
        return np.roll(feature, -1, axis=-2)


def right_shift(feature: np.ndarray | th.Tensor):
    if isinstance(feature, th.Tensor):
        return th.roll(feature, 1, dims=-2)
    else:
        return np.roll(feature, 1, axis=-2)


def up_shift(feature: np.ndarray | th.Tensor):
    if isinstance(feature, th.Tensor):
        return th.roll(feature, -1, dims=-3)
    else:
        return np.roll(feature, -1, axis=-3)


def down_shift(feature: np.ndarray | th.Tensor):
    if isinstance(feature, th.Tensor):
        return th.roll(feature, 1, dims=-3)
    else:
        return np.roll(feature, 1, axis=-3)


def up_left_shift(feature: np.ndarray | th.Tensor):
    return up_shift(left_shift(feature))


def up_right_shift(feature: np.ndarray | th.Tensor):
    return up_shift(right_shift(feature))


def down_left_shift(feature: np.ndarray | th.Tensor):
    return down_shift(left_shift(feature))


def down_right_shift(feature: np.ndarray | th.Tensor):
    return down_shift(right_shift(feature))


SHIFT_FNS = [
    identity,
    up_shift,
    down_shift,
    left_shift,
    right_shift,
    up_left_shift,
    up_right_shift,
    down_left_shift,
    down_right_shift,
]

# %%


obs_reference = th.zeros(num_envs, 3, 10, 10, dtype=th.int32)
base_dir = Path("/training") if ON_CLUSTER else LP_DIR
coefs_for_channels = pd.read_pickle(base_dir / "circuit_logs/coefs_for_channels.pkl")
coefs_for_channels[1][25].model.coef_[1] *= 3  # empty squares should be given more weight
# coefs_for_channels[0][5].model.coef_[2:4] *= 0.1

layer_wise_full_coef_matrix = [th.tensor([coefs_for_channels[i][c].model.coef_ for c in range(32)]) for i in range(3)]
layer_wise_intercept = [th.tensor([coefs_for_channels[i][c].model.intercept_ for c in range(32)]) for i in range(3)]
layer_wise_offset_fns = [[globals()[coefs_for_channels[i][c].shift_fn] for c in range(32)] for i in range(3)]
layer_wise_offset_fns[1][5] = lambda x: up_shift(up_shift(left_shift(x)))
# %%


def get_act_from_base_features(layer_idx, pos):
    obs_features = HelperFeatures.feature_from_obs(obs_reference) if feature_from_obs else features_from_play_out[pos, None]  # type: ignore
    assert len(obs_features.shape) == 4 and obs_features.shape[-1] == num_base_features, obs_features.shape
    obs_features_channel_offset = th.stack([layer_wise_offset_fns[layer_idx][c](obs_features) for c in range(32)], dim=1)
    act_from_base_features = th.einsum("bchwn, cn->bchw", obs_features_channel_offset, layer_wise_coef_matrix[layer_idx])
    act_from_base_features += layer_wise_intercept[layer_idx][None, :, None, None]
    return act_from_base_features, obs_features


def apply_conv(x, kernel, center_mean=False):
    if len(x.shape) == 3:
        x = x[:, None]
        kernel = kernel[None]
    assert len(x.shape) == 4 and len(kernel.shape) == 3, (x.shape, kernel.shape)
    ret = th.nn.functional.conv2d(x, kernel[None], padding="same")[:, 0]
    if center_mean:
        ret -= ret.mean(dim=(-1, -2), keepdim=True)
    return ret


def chains(start_square_in_grid: th.Tensor, direction_idx: int, new_sqs_in_chain: int = 2, include_start=True):
    assert len(start_square_in_grid.shape) >= 3 and start_square_in_grid.shape[-1] == 1
    dir_to_fn = [up_shift, down_shift, left_shift, right_shift]
    cur_grid = start_square_in_grid
    final_grid = start_square_in_grid.clone() if include_start else th.zeros_like(start_square_in_grid)
    for _ in range(new_sqs_in_chain):
        cur_grid = dir_to_fn[direction_idx](cur_grid)
        final_grid += cur_grid
    return final_grid


def base_feature_only_forward(
    h_next, skip_input, prev_layer_hidden, cur_h, pos, tick, cell, ijo_scale, ijo_bias, resample_data=None, layer_idx=None
):
    assert layer_idx is not None, "layer_idx must be provided"
    h_copy, obs_feature = get_act_from_base_features(layer_idx, pos)
    return h_copy


def clip(x, min_val, max_val):
    return th.minimum(th.maximum(x, th.tensor(min_val)), th.tensor(max_val))


FEATURE_NAME = ["agent", "floor", "unsolved_boxes", "solved_boxes", "unsolved_targets"]
FEATURE_IDX = {name: idx for idx, name in enumerate(FEATURE_NAME)}


def interpretable_l0h(
    h_next,
    skip_input,
    prev_layer_hidden,
    cur_h,
    pos,
    tick,
    cell: ConvLSTMCell,
    ijo_scale,
    ijo_bias,
    resample_data=None,
):
    layer_idx = 0
    h_copy, obs_feature = get_act_from_base_features(layer_idx, pos)
    targets = obs_feature[..., FEATURE_IDX["unsolved_targets"], None]  # (b, h, w, 1)

    assert h_copy.shape == h_next.shape, (h_copy.shape, h_copy.shape)

    channel_labels_short = {
        1: "near-future-box-down-moves[1sq-up shifted,activates 2 sqs simultaneuously[onsq and onleft]](+0.4),box-up-moves(+0.3),box-left-moves(+0.2)",
        16: "",
        17: "box-right-moves(+0.9)",  # at level start, activates couple of sqs to the right of a box and left of the target
    }

    channel_labels_long = {
        1: """
            The near-future-box-down-moves feature probably dictates which box the agent should go to push down on next.
        """,
        17: """
            Uses encoder to activate 3 squares to the left of targets. If these overextend (e.g., box is on 2nd sq to the left), then conv_ih suppresses the extra activations.
            Gets reinforced through L2H9 (right moves) by positively checking for right moves on the horizontal cells surrounding the square.
        """,
    }

    inputs = {
        # 9: ([], 0),
        16: ([], 0),  # 7,22 right moves | 4,21,26 down moves
        # 12: ([], 0),
        17: ([9], 0),
        # 18: ([], 0),  # hh
    }
    inputs_hh = {
        # 1: [9, 12, 17],
        # 2: [2, 14, 8, 20],
        # 9: [9, 16, 17],
        # 14: [2, 14, 21],
        # 16: [9, 16, 17],
        # 18: [18],
        # 28: [9, 16, 18],
    }
    for c in range(32):
        if c in inputs:
            # if c in [5, 10, 16]:
            #     h_copy[:, c] = 0
            pass
        else:  # do random ablation here
            # h_copy[:, c] = h_next[:, c]
            # h_copy[:, c] *= ijo_scale[0, c]
            # h_copy[:, c] += ijo_bias[0, c]
            # if resample_data is not None:
            #     resample_idx = np.random.choice(resample_data.shape[0], 1)
            #     h_copy[:, c] = resample_data[resample_idx, c]
            pass

    for c_out, (c_in_list, c_bias) in inputs.items():
        i, j, o = th.zeros_like(h_copy[:, 0]), th.zeros_like(h_copy[:, 0]), th.zeros_like(h_copy[:, 0])

        # encoder
        i += apply_conv(skip_input, get_conv_weights(layer_idx, c_out, None, cell, "i", "e", ih=True))
        j += apply_conv(skip_input, get_conv_weights(layer_idx, c_out, None, cell, "j", "e", ih=True))
        o += apply_conv(skip_input, get_conv_weights(layer_idx, c_out, None, cell, "o", "e", ih=True))

        # i *= ijo_scale[0, c_out]
        # j *= ijo_scale[1, c_out]
        # o *= ijo_scale[2, c_out]

        i += cell.conv_ih.bias[c_out]
        j += cell.conv_ih.bias[32 + c_out]
        o += cell.conv_ih.bias[32 * 3 + c_out]

        # i += ijo_bias[0, c_out]
        # j += ijo_bias[1, c_out]
        # o += ijo_bias[2, c_out]

        # if c_out == 17:
        #     h_copy[:, c_out] += th.tanh(i) * th.sigmoid(j) * th.tanh(o) + c_bias
        #     continue

        for c_in in c_in_list:
            i += apply_conv(prev_layer_hidden[:, c_in], get_conv_weights(layer_idx, c_out, c_in, cell, "i", ih=True))
            j += apply_conv(prev_layer_hidden[:, c_in], get_conv_weights(layer_idx, c_out, c_in, cell, "j", ih=True))
            o += apply_conv(prev_layer_hidden[:, c_in], get_conv_weights(layer_idx, c_out, c_in, cell, "o", ih=True))

        c_in_list_hh = inputs_hh.get(c_out, [])
        if c_out not in c_in_list_hh:
            c_in_list_hh.append(c_out)
        for c_in in c_in_list_hh:
            i += apply_conv(cur_h[:, c_in], get_conv_weights(layer_idx, c_out, c_in, cell, "i", ih=False))
            j += apply_conv(cur_h[:, c_in], get_conv_weights(layer_idx, c_out, c_in, cell, "j", ih=False))
            o += apply_conv(cur_h[:, c_in], get_conv_weights(layer_idx, c_out, c_in, cell, "o", ih=False))

        h_copy[:, c_out] += th.tanh(i) * th.sigmoid(j) * th.tanh(o) + c_bias

    # h_copy[:, 17] += 0.2 * up_shift(chains(targets, 2, 3))[..., 0]
    # h_copy[:, 17] = clip(h_copy[:, 17], -1, 1)

    return h_copy


SKIP_L1 = []
SKIP_L2 = []


def interpretable_l1h(
    h_next,
    skip_input,
    prev_layer_hidden,
    cur_h,
    pos,
    tick,
    cell: ConvLSTMCell,
    ijo_scale,
    ijo_bias,
    resample_data=None,
):
    layer_idx = 1
    h_copy, obs_feature = get_act_from_base_features(layer_idx, pos)

    assert h_copy.shape == h_next.shape, (h_copy.shape, h_copy.shape)
    # 17
    channel_labels_short = {
        0: "agent-to-box-pos(-0.2),box-down-moves(+0.15),box-right-moves(+0.05)",
        5: "agent-near-future-up-moves(+0.3),box-down-moves(+0.2),empty-region-below-target-at-level-start(-0.3)",
        6: "unclear (something related to the first box-down-push)",
        8: "box-near-future-down-moves(-0.4),agent-down-moves(+0.3),box-near-future-up-moves(+0.25)",
        13: "box-right-moves(+0.75),agent-future-pos(+0.02)",  # at level start, activates couple of sqs towards the right of a box and left of the target
        17: "box-down-moves(-0.75),box-turns(+0.15)",  # at level start, activates couple of sqs below a box and above the target. (search-y) Multiple lines before one line gets picked
        18: "agent-down-moves(+0.6)",  # including box-down-moves(upshifted) as well
        19: "box-down-moves(-0.25)",
        21: "box-right-moves(-0.5)",  # at level start, activates couple of sqs to the right of a box and left of the target but they disappear very quickly (within 3 ticks)
        25: "all-possible-paths-leading-to-targets(-0.4),agent-near-future-pos(-0.07),walls-and-out-of-plan-sqs(+0.1)",
        28: "agent-right-moves-that-are-not-box-pushes(+0.3),box-up-moves-sometimes-unclear(-0.1)",
        29: "agent-near-future-up-moves(+0.5)",
    }

    channel_labels_long = {
        0: """
            O: copies box-down-moves from L0H2 (on same sq) & L0H28 (from one sq above). L0H28 also surrounds the box-down strip with negative activations.
        """,
    }

    inputs = {
        0: [2, 28, 16, 9, 1],
        5: [18],
        6: [],
        8: [9, 16],  # hh
        13: [17, 31, 16, 9],
        17: [9, 26, 2, 28, 14],  # can replace i with -1
        18: [2, 28, 10, 5, 16, 1],
        19: [2, 9, 28, 5],
        21: [17, 9, 1, 31, 27, 16],  # box-right moves. i,j.
        # 21: list(range(32)),
        25: [],
        28: [1],  # hh
        29: [9, 5, 18],
    }
    # if pos == 0 and tick == 0:
    #     print(list(inputs.keys()))
    inputs_hh = {
        0: [0, 18, 21],
        13: [13, 21],
        17: [10, 25],
        18: [18, 19, 0, 25],
        19: [21, 19, 18, 0],
        21: [25, 28, 18, 21, 19],
        29: [29, 5, 25],
        5: [5, 28, 29],
        28: [28, 19],
    }
    for c in range(32):
        if c in SKIP_L1:
            h_copy[:, c] = h_next[:, c]
            continue
        if c in inputs:
            # h_copy[:, c] = 0
            pass
        else:
            # h_copy[:, c] = h_next[:, c]
            # h_copy[:, c] *= ijo_scale[0, c]
            # h_copy[:, c] += ijo_bias[0, c]
            if resample_data is not None:
                resample_idx = np.random.choice(resample_data.shape[0], 1)
                h_copy[:, c] = resample_data[resample_idx, c]
            pass

    for c_idx, (c_out, c_in_list) in enumerate(inputs.items()):
        if c_out in SKIP_L1:
            continue
        i, j, o = th.zeros_like(h_copy[:, 0]), th.zeros_like(h_copy[:, 0]), th.zeros_like(h_copy[:, 0])

        for c_in in c_in_list:
            i += apply_conv(prev_layer_hidden[:, c_in], get_conv_weights(layer_idx, c_out, c_in, cell, "i", ih=True))
            j += apply_conv(prev_layer_hidden[:, c_in], get_conv_weights(layer_idx, c_out, c_in, cell, "j", ih=True))
            o += apply_conv(prev_layer_hidden[:, c_in], get_conv_weights(layer_idx, c_out, c_in, cell, "o", ih=True))

        c_in_list_hh = inputs_hh.get(c_out, [])
        # if c_out not in c_in_list_hh:
        #     print(c_out, c_in_list_hh)
        # c_in_list_hh.append(c_out)
        for c_in in c_in_list_hh:
            i += apply_conv(cur_h[:, c_in], get_conv_weights(layer_idx, c_out, c_in, cell, "i", ih=False))
            j += apply_conv(cur_h[:, c_in], get_conv_weights(layer_idx, c_out, c_in, cell, "j", ih=False))
            o += apply_conv(cur_h[:, c_in], get_conv_weights(layer_idx, c_out, c_in, cell, "o", ih=False))

        # i *= ijo_scale[0, c_out]
        # j *= ijo_scale[1, c_out]
        # o *= ijo_scale[2, c_out]

        i += cell.conv_ih.bias[c_out]
        j += cell.conv_ih.bias[32 + c_out]
        o += cell.conv_ih.bias[32 * 3 + c_out]

        # i += ijo_bias[0, c_out]
        # j += ijo_bias[1, c_out]
        # o += ijo_bias[2, c_out]

        h_copy[:, c_out] += th.tanh(i) * th.sigmoid(j) * th.tanh(o)

    return h_copy


l2_interp_cache = []


def interpretable_l2h(
    h_next,
    skip_input,
    prev_layer_hidden,
    cur_h,
    pos,
    tick,
    cell: ConvLSTMCell,
    ijo_scale,
    ijo_bias,
    resample_data=None,
):
    layer_idx = 2
    h_copy, obs_feature = get_act_from_base_features(layer_idx, pos)
    agent = obs_feature[..., 0, None]  # (b, h, w, 1)
    assert h_copy.shape == h_next.shape, (h_copy.shape, h_copy.shape)  # (b, c, h, w)

    pool = cell.pool_and_project(cur_h)

    action_channels = [29, 8, 27, 3]
    near_future_action_channels = [28, 4, 23, 26]
    near_future_shift_fns = [up_shift, left_shift, left_shift, identity]
    modifying_channels = action_channels + near_future_action_channels
    for c in range(32):
        if c in SKIP_L2:
            h_copy[:, c] = h_next[:, c]
            continue
        if c in action_channels:
            pass
        elif c in near_future_action_channels:
            h_copy[:, c] = 0
            # pass
        else:
            # h_copy[:, c] *= ijo_scale[0, c]
            # h_copy[:, c] += ijo_bias[0, c]
            # h_copy[:, c] = h_next[:, c]
            # h_copy[:, c] = 0
            pass
            # if resample_data is not None:
            #     resample_idx = np.random.choice(resample_data.shape[0], 1)
            #     h_copy[:, c] = resample_data[resample_idx, c]

            # interpolate code
            # h_copy[:, c] *= 1 - interpolate_alpha[c]
            # h_copy[:, c] += interpolate_alpha[c] * h_next[:, c]

    if tick != 2 or NO_ACTION_HARDCODE:
        for c_idx, c in enumerate(action_channels):
            if c in SKIP_L2:
                continue
            nc = near_future_action_channels[c_idx]
            i = apply_conv(pool[:, nc], get_conv_weights(layer_idx, c, nc, cell, "i", "ch", ih=True))
            j = apply_conv(pool[:, nc], get_conv_weights(layer_idx, c, nc, cell, "j", "ch", ih=True))
            o = apply_conv(pool[:, nc], get_conv_weights(layer_idx, c, nc, cell, "o", "ch", ih=True))

            i *= ijo_scale[0, c]
            j *= ijo_scale[1, c]
            o *= ijo_scale[2, c]

            i += cell.conv_ih.bias[c]
            j += cell.conv_ih.bias[32 + c]
            o += cell.conv_ih.bias[32 * 3 + c]

            i += ijo_bias[0, c]
            j += ijo_bias[1, c]
            o += ijo_bias[2, c]

            h_copy[:, c] += th.tanh(i) * th.sigmoid(j) * th.tanh(o)
    else:  # predict the action with the maximum value on the agent square from the near future action channels
        agent_pos = th.where(agent[0, ..., 0] == 1)
        agent_pos = [agent_pos[0][0].item(), agent_pos[1][0].item()]
        shifted_agent_pos = [get_shifted_agent_pos(agent_pos[0], agent_pos[1], shift_fn) for shift_fn in near_future_shift_fns]

        max_action = np.argmax([cur_h[0, near_future_action_channels[i], y, x] for i, (y, x) in enumerate(shifted_agent_pos)])
        h_copy[:, action_channels[max_action]] = -1 if max_action == 2 else 1

    if tick == 0 and 27 not in SKIP_L2:
        tick_0_l2h27 = (-0.5 * agent) + (0.5 * left_shift(agent)) + (0.5 * up_shift(agent)) + (-0.5 * up_left_shift(agent))
        h_copy[:, 27] += tick_0_l2h27[..., 0]

    # near-future suppression mechanism
    near_future_prev_layer_connections = [[0, 25, 29, 5], [0, 8, 17, 18, 19], [0, 18, 19, 6, 29, 25], [0, 21, 28, 13]]

    # near_future_j_or_o = ["j", "o", "j", "j"]
    l2_interp_cache.append([[], [], [], []])
    for c_idx, c1 in enumerate(near_future_action_channels):
        if c1 in SKIP_L2:
            continue
        i, j, o = th.zeros_like(h_copy[:, c1]), th.zeros_like(h_copy[:, c1]), th.zeros_like(h_copy[:, c1])

        for c2 in near_future_prev_layer_connections[c_idx]:
            # for c2 in range(32):
            i += apply_conv(prev_layer_hidden[:, c2], get_conv_weights(layer_idx, c1, c2, cell, "i", ih=True))
            j += apply_conv(prev_layer_hidden[:, c2], get_conv_weights(layer_idx, c1, c2, cell, "j", ih=True))
            o += apply_conv(prev_layer_hidden[:, c2], get_conv_weights(layer_idx, c1, c2, cell, "o", ih=True))
            # if 3 * pos + tick == 7 and c_idx == 2:
            #     print("j", tick, pos, c1, c2, j[0, 3, 1])
        for c2 in near_future_action_channels + [27]:
            # if c2 == c1:
            #     continue
            i += apply_conv(cur_h[:, c2], get_conv_weights(layer_idx, c1, c2, cell, "i", ih=False))
            j += apply_conv(cur_h[:, c2], get_conv_weights(layer_idx, c1, c2, cell, "j", ih=False))
            o += apply_conv(cur_h[:, c2], get_conv_weights(layer_idx, c1, c2, cell, "o", ih=False))

        i *= ijo_scale[0, c1]
        j *= ijo_scale[1, c1]
        o *= ijo_scale[2, c1]

        i += cell.conv_ih.bias[c1]
        j += cell.conv_ih.bias[32 + c1]
        o += cell.conv_ih.bias[32 * 3 + c1]

        i += ijo_bias[0, c1]
        j += ijo_bias[1, c1]
        o += ijo_bias[2, c1]

        i = th.tanh(i)
        j = th.sigmoid(j)
        o = th.tanh(o)
        l2_interp_cache[-1][1].append(i)
        l2_interp_cache[-1][2].append(j)
        l2_interp_cache[-1][3].append(o)
        h_copy[:, c1] += i * j * o
        h_copy[:, c1] = h_copy[:, c1] * chains(near_future_shift_fns[c_idx](agent), c_idx, 3)[..., 0]
        l2_interp_cache[-1][0].append(h_copy[:, c1])

    # if pos == 11 and tick == 2:
    #     print("after near future action channels")
    #     print(h_copy[0, 2])

    h_copy[:, 9] = h_next[:, 9]

    return h_copy


def get_shifted_agent_pos(y, x, shift_fn):
    if shift_fn == identity:
        return y, x
    elif shift_fn == up_shift:
        return y - 1, x
    elif shift_fn == down_shift:
        return y + 1, x
    elif shift_fn == left_shift:
        return y, x - 1
    elif shift_fn == right_shift:
        return y, x + 1
    else:
        raise ValueError(f"Invalid shift function: {shift_fn}")


interp_forwards = [interpretable_l0h, interpretable_l1h, interpretable_l2h]


model.features_extractor.cell_list[0].interpretable_forward = None
model.features_extractor.cell_list[1].interpretable_forward = None
model.features_extractor.cell_list[2].interpretable_forward = None

# model.features_extractor.cell_list[0].interpretable_forward = interpretable_l0h
# model.features_extractor.cell_list[1].interpretable_forward = interpretable_l1h
# model.features_extractor.cell_list[2].interpretable_forward = interpretable_l2h

# %% PLAY LEVEL SETUP

play_toy = False
two_levels = True
level_reset_opt = {"level_file_idx": 8, "level_idx": 2}
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

# mean_done = 0

combined_probe, combined_intercepts = th.load(LP_DIR / "learned_planner/notebooks/action_l2_probe.pt", weights_only=True)
aggregation_weight, aggregation_bias = th.load(LP_DIR / "learned_planner/notebooks/aggregation.pt", weights_only=True)


def bigger_levels_get_distribution(self, obs, carry, episode_starts, use_interpretable_forward=False):
    _, new_carry = model._recurrent_extract_features(obs, carry, episode_starts)
    new_h, new_c = new_carry[2]
    probe_in = th.cat([new_h, new_c], dim=2).squeeze(0)
    # Get the channels which represent actions
    # actions_per_location = new_h[:, [29, 8, 27, 3], :, :].squeeze(0)
    actions_per_location = th.einsum("nchw,oc->nohw", probe_in, combined_probe) + combined_intercepts[None, :, None, None]

    # Aggregate
    num_action1 = actions_per_location.mean((2, 3))
    num_action2 = actions_per_location.max(dim=2, keepdim=False).values.max(dim=2, keepdim=False).values
    num_action3 = (actions_per_location > 0).float().mean((2, 3))
    actions = (
        num_action1 * aggregation_weight[0]
        + num_action2 * aggregation_weight[1]
        + num_action3 * aggregation_weight[2]
        + aggregation_bias
    )
    return CategoricalDistribution(actions.shape[-1]).proba_distribution(action_logits=actions), new_carry


def bigger_recurrent_initial_state(self, dim_room, N, device=None):
    return [(th.zeros([N, 1, 32, *dim_room], device=device), th.zeros([N, 1, 32, *dim_room], device=device)) for _ in range(3)]


toy_reset_opt = dict(walls=walls, boxes=boxes, targets=targets, player=player)
reset_opts = toy_reset_opt if play_toy else level_reset_opt
dim_room = (20, 20)
level_rep = pad_level(
    """
####################
#      ##  ####  ###
#   $ @##  ####  ###
#  ######  ####  ###
#  ######  ####  ###
#        #         #
#        #         #
#######  ####  ##  #
#######  ####  ##  #
###  ##            #
###  ##            #
#              ##  #
#              ##  #
#####  ####  ####  #
#####  ####  ####  #
#            ####  #
#            #### .#
####################
""",
    *dim_room,
)
reset_opts = parse_level(level_rep)


if getattr(model.get_distribution, "__name__", None) == "get_distribution":
    old_get_distribution = model.get_distribution
model.get_distribution = partial(bigger_levels_get_distribution, model)  # type: ignore

envs = dataclasses.replace(boxo_cfg, dim_room=dim_room).make()
if getattr(model.recurrent_initial_state, "__name__", None) == "recurrent_initial_state":
    old_recurrent_initial_state = model.recurrent_initial_state
model.recurrent_initial_state = partial(bigger_recurrent_initial_state, model, dim_room)


def process_cache(cache):
    cache = join_cache_across_steps([cache])
    shape = cache["features_extractor.cell_list.0.hook_h"].shape
    if len(shape) == 6:
        cache = {
            k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, *v.shape[-3:])
            for k, v in cache.items()
            if len(v.shape) == len(shape)
        }
    elif len(shape) == 5:
        cache = {k: v.squeeze(1).reshape(-1, *v.shape[-3:]) for k, v in cache.items() if len(v.shape) == len(shape)}
    return cache


def process_play_out(play_out):
    cache = process_cache(play_out.cache)
    obs = play_out.obs.squeeze(1)
    return obs, cache


play_levels_kwargs = dict(
    env=envs,
    policy_th=model,
    reset_opts=reset_opts,
    thinking_steps=thinking_steps,
    # fwd_hooks=fwd_hooks,
    max_steps=max_steps,
    # hook_steps=list(range(thinking_steps, max_steps)) if thinking_steps > 0 else -1,
    internal_steps=True,
)

# %% Actual play (without any interpretable forward)
toy_out = play_level(**play_levels_kwargs)  # type: ignore
toy_obs, toy_cache = process_play_out(toy_out)
baseline_probs = toy_out.logits
print("Total len:", len(toy_obs), toy_cache["features_extractor.cell_list.0.hook_h"].shape[0] // 3)

# %% Search algorithm


def search_algo(obs):
    assert len(obs.shape) == 3 and obs.shape[0] == 3
    box_idxs = DatasetStore.get_box_position_per_step(obs, variable_boxes=True, return_map=False)
    box_directions_map = th.zeros(4, *obs.shape[-2:])
    for i in range(box_idxs.shape[0]):
        box_idx = box_idxs[i]
        wall_on_up = 3 + th.where((obs[:, : box_idx[0], box_idx[1]] == 0).all(dim=0))[0][-1]
        wall_on_down = box_idx[0] - 1 + th.where((obs[:, box_idx[0] + 1 :, box_idx[1]] == 0).all(dim=0))[0][0]
        wall_on_left = 3 + th.where((obs[:, box_idx[0], : box_idx[1]] == 0).all(dim=0))[0][-1]
        wall_on_right = box_idx[1] - 1 + th.where((obs[:, box_idx[0], box_idx[1] + 1 :] == 0).all(dim=0))[0][0]

        box_directions_map[0, wall_on_up : box_idx[0] + 1, box_idx[1]] = 1
        box_directions_map[1, box_idx[0] : wall_on_down, box_idx[1]] = 1
        box_directions_map[2, box_idx[0], wall_on_left : box_idx[1] + 1] = 1
        box_directions_map[3, box_idx[0], box_idx[1] : wall_on_right] = 1

    box_directions_map = expand_search(box_directions_map, obs)
    fig = plotly_feature_vis(
        box_directions_map[None].numpy(),
        obs[None],
        # feature_labels=[f"L{layer}{hcijfo[:5].upper()}{batch_no * batch_size + i}" for i in range(batch_size)],
        # height=800,
    )
    fig.show()
    box_directions_map = expand_search(box_directions_map, obs)
    fig = plotly_feature_vis(
        box_directions_map[None].numpy(),
        obs[None],
        # feature_labels=[f"L{layer}{hcijfo[:5].upper()}{batch_no * batch_size + i}" for i in range(batch_size)],
        # height=800,
    )
    fig.show()
    box_directions_map = expand_search(box_directions_map, obs)
    fig = plotly_feature_vis(
        box_directions_map[None].numpy(),
        obs[None],
        # feature_labels=[f"L{layer}{hcijfo[:5].upper()}{batch_no * batch_size + i}" for i in range(batch_size)],
        # height=800,
    )
    fig.show()
    box_directions_map = expand_search(box_directions_map, obs)
    fig = plotly_feature_vis(
        box_directions_map[None].numpy(),
        obs[None],
        # feature_labels=[f"L{layer}{hcijfo[:5].upper()}{batch_no * batch_size + i}" for i in range(batch_size)],
        # height=800,
    )
    fig.show()
    box_directions_map = expand_search(box_directions_map, obs)
    fig = plotly_feature_vis(
        box_directions_map[None].numpy(),
        obs[None],
        # feature_labels=[f"L{layer}{hcijfo[:5].upper()}{batch_no * batch_size + i}" for i in range(batch_size)],
        # height=800,
    )
    fig.show()


def expand_search(box_directions_map, obs):
    new_box_directions_map = box_directions_map.clone()

    # up expand
    up_acts_idxs = th.where(box_directions_map[0] > 0)
    # sort from down to up
    up_acts_idxs = [up_acts_idxs[0].flip(0), up_acts_idxs[1].flip(0)]
    chains = []
    for y, x in zip(*up_acts_idxs):
        chain_idx = None
        for idx, chain in enumerate(chains):
            if chain[-1] == (y + 1, x):
                chain.append((y, x))
                chain_idx = idx
                break
        if chain_idx is None:
            chains.append([(y, x)])

    up_sqs = [(chain[-1][0] - 1, chain[-1][1]) for chain in chains]

    for y, x in up_sqs:
        wall_on_left = 3 + th.where((obs[:, y, :x] == 0).all(dim=0))[0][-1]
        wall_on_right = x - 1 + th.where((obs[:, y, x + 1 :] == 0).all(dim=0))[0][0]
        new_box_directions_map[2, y, wall_on_left : x + 1] = 1
        new_box_directions_map[3, y, x:wall_on_right] = 1

    # down expand
    down_acts_idxs = th.where(box_directions_map[1] > 0)
    chains = []
    for y, x in zip(*down_acts_idxs):
        chain_idx = None
        for idx, chain in enumerate(chains):
            if chain[-1] == (y - 1, x):
                chain.append((y, x))
                chain_idx = idx
                break
        if chain_idx is None:
            chains.append([(y, x)])

    down_sqs = [(chain[-1][0] + 1, chain[-1][1]) for chain in chains]

    for y, x in down_sqs:
        wall_on_left = 3 + th.where((obs[:, y, :x] == 0).all(dim=0))[0][-1]
        wall_on_right = x - 1 + th.where((obs[:, y, x + 1 :] == 0).all(dim=0))[0][0]
        new_box_directions_map[2, y, wall_on_left : x + 1] = 1
        new_box_directions_map[3, y, x:wall_on_right] = 1

    # left expand
    left_acts_idxs = th.where(box_directions_map[2] > 0)
    # sort from right to left
    left_acts_idxs = sorted(zip(*left_acts_idxs), key=lambda x: -x[1])
    chains = []
    for y, x in left_acts_idxs:
        chain_idx = None
        for idx, chain in enumerate(chains):
            if chain[-1] == (y, x + 1):
                chain.append((y, x))
                chain_idx = idx
                break
        if chain_idx is None:
            chains.append([(y, x)])

    left_sqs = [(chain[-1][0], chain[-1][1] - 1) for chain in chains]

    for y, x in left_sqs:
        wall_on_up = 3 + th.where((obs[:, :y, x] == 0).all(dim=0))[0][-1]
        wall_on_down = y - 1 + th.where((obs[:, y + 1 :, x] == 0).all(dim=0))[0][0]
        new_box_directions_map[0, wall_on_up : y + 1, x] = 1
        new_box_directions_map[1, y:wall_on_down, x] = 1

    # right expand
    right_acts_idxs = th.where(box_directions_map[3] > 0)
    # sort from left to right
    right_acts_idxs = sorted(zip(*right_acts_idxs), key=lambda x: x[1])
    chains = []
    for y, x in right_acts_idxs:
        chain_idx = None
        for idx, chain in enumerate(chains):
            if chain[-1] == (y, x - 1):
                chain.append((y, x))
                chain_idx = idx
                break
        if chain_idx is None:
            chains.append([(y, x)])
    print(chains)
    # right_sqs = [(y[-1], x[-1] - 1) for y, x in chains]
    # right_sqs = [(chain[-1][0], chain[-1][1] + 1) for chain in chains]

    for chain in chains:
        for y, x in chain[::-1]:
            box_x = x + 1
            wall_on_up = 3 + th.where((obs[:, :y, box_x] == 0).all(dim=0))[0][-1]
            wall_on_down = y - 1 + th.where((obs[:, y + 1 :, box_x] == 0).all(dim=0))[0][0]
            new_box_directions_map[0, wall_on_up : y + 1, box_x] = 1
            new_box_directions_map[1, y:wall_on_down, box_x] = 1
            if wall_on_up < y + 1 or y < wall_on_down:
                break
            else:
                new_box_directions_map[3, y, x] = 0

    return new_box_directions_map


search_algo(toy_obs[0])


# %% Initialize scale and bias
init_ijo_scale = [th.nn.Parameter(th.ones(3, 32)) for layer_idx in range(3)]
init_ijo_bias = [th.nn.Parameter(th.zeros(3, 32)) for layer_idx in range(3)]

# %% Play with interpretable forward on the original trajectory (off policy)
resample = False  # whether to resample ablate channels not part of the circuit. Uncomment the resample_data line in interpretable_forward fns
# whether to use only the base features as h. If True, connections from previous layers and tick are ignored
base_features_only = False
# whether to use maxpooling circuit or using action predictions directly from the near future action channels. See interpretable_l2h above.
NO_ACTION_HARDCODE = True
num_base_features, feature_from_obs = 5, False  # set 17 for using future direction & action features as well
layer_wise_coef_matrix = [coef_matrix[:, :num_base_features] for coef_matrix in layer_wise_full_coef_matrix]

features_from_play_out = HelperFeatures.from_play_output(toy_out).to_tensor()
features_from_play_out = features_from_play_out[..., :num_base_features]

# skip modifying some channels. Uses original h values for those channels.
SKIP_L1 = [21]
# SKIP_L2 = [28, 4, 23, 26]
SKIP_L2 = []
# layers where interpretable forward is applied
interp_layers = [2]

resample_data = load_data_h() if resample else [None] * 3

for layer_idx in range(3):
    if layer_idx in interp_layers:
        if not base_features_only:
            model.features_extractor.cell_list[layer_idx].interpretable_forward = partial(
                interp_forwards[layer_idx],
                ijo_scale=init_ijo_scale[layer_idx],
                ijo_bias=init_ijo_bias[layer_idx],
                # ijo_scale=ijo_scale[layer_idx], # train code in the last cell
                # ijo_bias=ijo_bias[layer_idx],
                resample_data=resample_data,
            )
        else:
            model.features_extractor.cell_list[layer_idx].interpretable_forward = partial(
                base_feature_only_forward,
                layer_idx=layer_idx,
                ijo_scale=init_ijo_scale[layer_idx],
                ijo_bias=init_ijo_bias[layer_idx],
            )
    else:
        model.features_extractor.cell_list[layer_idx].interpretable_forward = None

cache_with_all_fts, interp_probs = get_cache_and_probs(
    toy_obs.unsqueeze(1),
    model,
    fwd_hooks=fwd_hooks,
    hook_steps=-1,
    use_interpretable_forward=True,
    # use_action_channels=True,
)

same_actions = (interp_probs.argmax(-1) == baseline_probs.argmax(-1)).float().mean().item()
print("% of same actions:", same_actions * 100)
print("Actions", interp_probs.argmax(-1).flatten())
cache_with_all_fts = process_cache(cache_with_all_fts)
print("MSE logits:", ((interp_probs - baseline_probs) ** 2).sum(-1).mean().item())


def norm_mse(true, pred):
    return ((true - pred) ** 2).mean(axis=(0, 2, 3)) / (
        ((true - true.mean(axis=(0, 2, 3), keepdims=True)) ** 2).mean(axis=(0, 2, 3))
    )


def find_bias(true, pred):
    return (
        (true - pred)
        .mean(axis=(0, 2, 3), keepdims=True)
        .round(
            3,
        )
    )


def print_mse(interp_layer, interp_channels):
    interp_h = cache_with_all_fts[f"features_extractor.cell_list.{interp_layer}.hook_interpretable_forward"][
        :, interp_channels
    ]
    input_h = cache_with_all_fts[f"features_extractor.cell_list.{interp_layer}.hook_h"][:, interp_channels]
    orig_h = toy_cache[f"features_extractor.cell_list.{interp_layer}.hook_h"][:, interp_channels]

    print("MSE h:", norm_mse(input_h, interp_h))
    print("MSE h orig:", norm_mse(orig_h, interp_h))
    bias_term = find_bias(orig_h, interp_h)
    print("MSE h orig post bias:", norm_mse(orig_h, interp_h + bias_term))
    print("bias term:", [b for b in bias_term.flatten()])


interp_layer = 0
interp_channels = [1, 2, 5, 9, 10, 14, 16, 17, 18, 26, 28, 31]  # layer 0
try:
    print_mse(interp_layer, interp_channels)
except KeyError:
    print(f"No cache for layer {interp_layer}")
interp_layer = 1
interp_channels = [0, 5, 6, 8, 13, 17, 18, 19, 21, 25, 28, 29]  # layer 1
try:
    print_mse(interp_layer, interp_channels)
except KeyError:
    print(f"No cache for layer {interp_layer}")
interp_layer = 2
interp_channels = [29, 8, 27, 3] + [28, 4, 23, 26]  # layer 2
try:
    print_mse(interp_layer, interp_channels)
except KeyError:
    print(f"No cache for layer {interp_layer}")


# %% Find channels where setting the original h values gives the best action accuracy
for c in range(32):
    print("Channel", c)
    SKIP_L1 = [21]
    SKIP_L2 = [c] + [29, 8, 27, 3] + [28, 4, 23, 26] + [1, 6, 30]
    cache_with_all_fts, interp_probs = get_cache_and_probs(
        toy_obs.unsqueeze(1), model, fwd_hooks=fwd_hooks, hook_steps=-1, use_interpretable_forward=True
    )
    # l2_interp_cache = np.array(l2_interp_cache).squeeze(3)

    print("MSE logits:", ((interp_probs - baseline_probs) ** 2).sum(-1).mean().item())
    same_actions = (interp_probs.argmax(-1) == baseline_probs.argmax(-1)).float().mean().item()
    print("% of same actions:", same_actions * 100)


# %% Play with interpretable forward (on policy)
# Note that on policy can only use base features which can be extracted from obs and not the direction features.
num_base_features, feature_from_obs = 5, True
layer_wise_coef_matrix = [coef_matrix[:, :num_base_features] for coef_matrix in layer_wise_full_coef_matrix]

obs_reference = th.zeros(num_envs, 3, 10, 10, dtype=th.int32)

model.features_extractor.cell_list[1].interpretable_forward = interpretable_l1h

toy_out_base_fts = play_level(**play_levels_kwargs, use_interpretable_forward=True, obs_reference=obs_reference)  # type: ignore
interp_probs = toy_out_base_fts.logits
print("Total len:", len(toy_out_base_fts.obs))
print(
    "% of same actions:", (interp_probs.argmax(-1)[: len(baseline_probs)] == baseline_probs.argmax(-1)).float().mean().item()
)
try:
    first_diff = (interp_probs[: len(baseline_probs)].argmax(-1) != baseline_probs.argmax(-1)).nonzero(as_tuple=True)[0][0]
except IndexError:
    first_diff = len(interp_probs)
print(
    f"MSE logits (until {first_diff}):", ((interp_probs[:first_diff] - baseline_probs[:first_diff]) ** 2).sum(-1).mean().item()
)

# %% Visualize function


def visualize(obs, cache, layer, batch_size, hcijfo, show_ticks):
    toy_all_channels = cache[f"features_extractor.cell_list.{layer}.hook_{hcijfo}"][
        :, batch_no * batch_size : (batch_no + 1) * batch_size
    ]
    if show_ticks:
        if 3 * len(obs) == len(toy_all_channels):
            obs = np.repeat(obs, 3, axis=0)
    else:
        toy_all_channels = toy_all_channels[tick::reps]

    # toy_obs_to_plot = np.concatenate([obs[:3], obs[66:69]], axis=0)
    # toy_all_channels_to_plot = np.concatenate([toy_all_channels[:3], toy_all_channels[66:69]], axis=0)

    fig = plotly_feature_vis(
        toy_all_channels,
        obs,
        feature_labels=[f"L{layer}{hcijfo[:5].upper()}{batch_no * batch_size + i}" for i in range(batch_size)],
        height=800,
    )
    fig.show()


# %%
plot_obs, plot_cache = process_play_out(toy_out_base_fts)
layer, batch_no = 1, 0
batch_size = 7 + 3 * 8
hcijfo = "interpretable_forward"
show_ticks = True
tick = reps - 1

visualize(plot_obs, plot_cache, layer, batch_size, hcijfo, show_ticks)

# %%
layer, batch_no = 0, 0
batch_size = 7 + 3 * 8
hcijfo = "interpretable_forward"
# hcijfo = "h"
show_ticks = True
tick = reps - 1

visualize(toy_obs, cache_with_all_fts, layer, batch_size, hcijfo, show_ticks)

# %%
layer, batch_no = 1, 0
visualize(toy_obs, toy_cache, layer, batch_size, "h", show_ticks)

# %%

plot_layer, plot_channel = 2, 1
# tick = reps - 1
tick = 0
show_ticks = True
keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "I", "J", "O"]]

# toy_all_channels_for_lc = np.stack([toy_cache[key][:, plot_channel] for key in keys], axis=1)
toy_all_channels_for_lc = l2_interp_cache[:, :, plot_channel]
if not show_ticks:
    toy_all_channels_for_lc = toy_all_channels_for_lc[tick::3]

# repeat obs 3 along first dimension
fig = plotly_feature_vis(
    toy_all_channels_for_lc,
    toy_obs.numpy().repeat(3, axis=0),
    feature_labels=[k.rsplit(".")[-1] for k in keys],
)
fig.show()

# %%


# %%
data = [th.tensor(toy_cache[f"features_extractor.cell_list.{layer}.hook_h"]) for layer in range(3)]
data = [th.cat([th.zeros_like(d[0, None]), d]) for d in data]
h_next = [d[1:] for d in data]
cur_state = [d[:-1] for d in data]
# prev_layer_hidden = [d.clone() for d in data]
prev_layer_hidden = [data[-1][:-1]] + [d[1:] for d in data[:-1]]

gt_logits = th.tensor(baseline_probs)
gt_probs = toy_out.act_dist
encoder_output = th.tensor(toy_cache["features_extractor.hook_pre_model"])

# %% Learn ijo scale and bias


# minimizes MSE of the h values
def learn_ijo_scale_bias(lr=5e-3):
    ijo_scale = [th.nn.Parameter(th.ones(3, 32), requires_grad=True) for _ in range(3)]
    ijo_bias = [th.nn.Parameter(th.zeros(3, 32), requires_grad=True) for _ in range(3)]

    optimizer = th.optim.Adam(ijo_scale + ijo_bias, lr=lr)
    mse_loss = th.nn.MSELoss(reduction="sum")

    num_epochs = 50

    data_size = len(h_next[0])
    assert data_size % 3 == 0, data_size
    episode_len = data_size // 3
    assert all(data_size == len(d) for d in h_next)
    assert all(data_size == len(d) for d in prev_layer_hidden)
    assert all(data_size == len(d) for d in cur_state)

    losses = []
    for layer_idx, interp_fn in enumerate(interp_forwards):
        if layer_idx != 2:
            continue
        losses.append([])
        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()
            total_loss = 0
            for pos in range(episode_len):
                for tick in range(3):
                    idx = pos * 3 + tick
                    interp_h = interp_fn(
                        h_next[layer_idx][idx, None],
                        encoder_output[pos, None],
                        prev_layer_hidden[layer_idx][idx, None],
                        cur_state[layer_idx][idx, None],
                        pos,
                        tick,
                        cell=model.features_extractor.cell_list[layer_idx],
                        ijo_scale=ijo_scale[layer_idx],
                        ijo_bias=ijo_bias[layer_idx],
                    )
                    loss = mse_loss(interp_h, h_next[layer_idx][idx, None])
                    total_loss += loss
                # print(loss)
            total_loss /= data_size
            total_loss.backward()
            optimizer.step()
            losses[-1].append(total_loss.item())

        plt.plot(losses[-1])
        plt.show()
        # break

    return ijo_scale, ijo_bias, losses


def l2h_to_action_logits(l2h):
    assert len(l2h.shape) == 4
    mlp_acts = model.mlp_extractor.policy_net(l2h.view(1, 3200))
    policy_logits = model.action_net(mlp_acts)
    return policy_logits


# minimizes KL divergence of the action logits
def learn_ijo_scale_bias_with_logits(lr=1e-1):
    ijo_scale = [th.nn.Parameter(th.ones(3, 32), requires_grad=True) for _ in range(3)]
    ijo_bias = [th.nn.Parameter(th.zeros(3, 32), requires_grad=True) for _ in range(3)]

    optimizer = th.optim.Adam(ijo_scale + ijo_bias, lr=lr)
    # mse_loss = th.nn.MSELoss(reduction="sum")
    kl_loss = th.nn.KLDivLoss(reduction="sum")

    num_epochs = 50

    data_size = len(h_next[0])
    assert data_size % 3 == 0, data_size
    episode_len = data_size // 3
    assert all(data_size == len(d) for d in h_next)
    assert all(data_size == len(d) for d in prev_layer_hidden)
    assert all(data_size == len(d) for d in cur_state)

    assert len(gt_logits) == episode_len

    losses = []

    for epoch in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        total_loss = 0
        layer_inp = th.zeros_like(cur_state[0][0, None], requires_grad=True)
        for pos in range(episode_len):
            for tick in range(3):
                for layer_idx, interp_fn in enumerate(interp_forwards):
                    idx = pos * 3 + tick
                    layer_inp = interp_fn(
                        h_next[layer_idx][idx, None],
                        encoder_output[pos, None],
                        layer_inp,
                        cur_state[layer_idx][idx, None],
                        pos,
                        tick,
                        cell=model.features_extractor.cell_list[layer_idx],
                        ijo_scale=ijo_scale[layer_idx],
                        ijo_bias=ijo_bias[layer_idx],
                    )
            logits = l2h_to_action_logits(layer_inp)
            # assert logits.shape == gt_logits[pos].shape, (logits.shape, gt_logits[pos].shape)
            # loss = mse_loss(layer_inp, h_next[layer_idx][pos*3+tick, None])
            # loss = mse_loss(logits, gt_logits[pos])
            loss = kl_loss(th.nn.functional.log_softmax(logits, -1), gt_probs[pos])
            total_loss += loss
        total_loss /= episode_len
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())
    plt.plot(losses)
    plt.show()
    return ijo_scale, ijo_bias, losses


ijo_scale, ijo_bias, lossses = learn_ijo_scale_bias()
