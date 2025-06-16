"""Activation transfer from long to short-term channels when two actions are to be performed at the same square."""

# %%
import concurrent.futures
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.io as pio
import torch as th
from cleanba.environments import BoxobanConfig
from matplotlib import pyplot as plt
from scipy.stats import bootstrap
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from learned_planner import BOXOBAN_CACHE, LP_DIR, ON_CLUSTER
from learned_planner.interp.channel_group import get_group_channels, layer_groups
from learned_planner.interp.collect_dataset import DatasetStore
from learned_planner.interp.offset_fns import apply_inv_offset_lc, offset_yx
from learned_planner.interp.plot import apply_style, plotly_feature_vis
from learned_planner.interp.render_svg import tiny_world_rgb_to_txt
from learned_planner.interp.utils import load_jax_model_to_torch, parse_level, play_level
from learned_planner.interp.weight_utils import find_ijfo_contribution, get_conv_weights, visualize_top_conv_inputs
from learned_planner.policies import download_policy_from_huggingface

# set_plotly_renderer("emacs")
pio.renderers.default = "notebook"
th.set_printoptions(sci_mode=False, precision=2)

# %%

MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000"  # DRC(3, 3) 2B checkpoint
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

# boxes_direction_probe_file = Path(
#     "/training/TrainProbeConfig/05-probe-boxes-future-direction/wandb/run-20240813_184417-vb6474rg/local-files/probe_l-all_x-all_y-all_c-all.pkl"
# )

boxo_cfg = BoxobanConfig(
    cache_path=BOXOBAN_CACHE,
    num_envs=1,
    max_episode_steps=200,
    min_episode_steps=200,
    asynchronous=False,
    tinyworld_obs=True,
    split=None,
    difficulty="hard",
)
model_cfg, model = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)

orig_state_dict = deepcopy(model.state_dict())


envs = boxo_cfg.make()


def restore_model():
    model.load_state_dict(orig_state_dict)


# %% Load activations and obs of hard levels solved
if ON_CLUSTER:
    cache_path = Path("/training/activations_dataset/train_medium/0_think_step")
    N_FILES = 2001
else:
    cache_path = LP_DIR / "drc33_cache/0_think_step"
    N_FILES = 500


def load_data_h(n_files=N_FILES):
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:

        def map_fn(i):
            data: DatasetStore = pd.read_pickle(cache_path / f"idx_{i}.pkl")
            box_movement_map, box_timesteps = data.boxes_future_direction_map(multioutput=False, return_timestep_map=True)
            agent_movement_map, agent_timesteps = data.agents_future_direction_map(multioutput=False, return_timestep_map=True)

            h = [data.model_cache[f"features_extractor.cell_list.{layer}.hook_h"] for layer in range(3)]
            return data.obs, h, box_movement_map, box_timesteps, agent_movement_map, agent_timesteps

        loaded_data = list(tqdm(executor.map(map_fn, range(n_files)), total=n_files))
        eps_lens = [len(d[0]) for d in loaded_data]
        obs = pad_sequence([th.tensor(d[0]) for d in loaded_data])
        layer_wise_h = [pad_sequence([th.tensor(d[1][layer]) for d in loaded_data]) for layer in range(3)]
        box_moves = [d[2] for d in loaded_data]
        box_timesteps = [d[3] for d in loaded_data]
        agent_moves = [d[4] for d in loaded_data]
        agent_timesteps = [d[5] for d in loaded_data]
    return obs, layer_wise_h, eps_lens, box_moves, box_timesteps, agent_moves, agent_timesteps


obs, layer_wise_h, eps_lens, box_moves, box_timesteps, agent_moves, agent_timesteps = load_data_h()
# %%


@dataclass
class Datapoint:
    env_idx: int
    sq_y: int
    sq_x: int
    prev_start: int
    new_start: int
    new_end: int
    prev_action: int
    new_action: int


def plot_planning_horizon(direction, start_idx=0, one_datapoint=False, box=True, smaller=False) -> list[Datapoint]:
    if smaller:
        apply_style(figsize=(1.82, 1.3), font=8)
    else:
        apply_style(figsize=(2.7, 1.5), font=8)
    groups = get_group_channels(group_names="box" if box else "agent", return_dict=True)
    box_or_agent_moves = box_moves if box else agent_moves
    box_or_agent_timesteps = box_timesteps if box else agent_timesteps
    dir_group = groups[direction]

    long_term_channels, long_term_hs = [], []
    short_term_channels, short_term_hs = [], []
    for lc_dict in dir_group:
        l, c = lc_dict["layer"], lc_dict["idx"]
        acts = layer_wise_h[l][:, :, c] * lc_dict["sign"]
        acts = apply_inv_offset_lc(acts, l, c, True)
        if lc_dict.get("long-term", None):
            long_term_channels.append((l, c))
            long_term_hs.append(acts)
        else:
            short_term_channels.append((l, c))
            short_term_hs.append(acts)
    if len(long_term_hs) == 0:
        print("No long term channels for", ("B" if box else "A"), "direction", ["up", "down", "left", "right"][direction])
        return []
    print(("B" if box else "A") + " Long term channels:", ", ".join([f"L{l}H{c}" for l, c in long_term_channels]))
    print(("B" if box else "A") + " Short term channels:", ", ".join([f"L{l}H{c}" for l, c in short_term_channels]))

    long_term_h = th.stack(long_term_hs, 0).mean(0)
    short_term_h = th.stack(short_term_hs, 0).mean(0)

    full_size = 240 * 3
    mid_point = full_size // 2

    samples_lt = [[] for _ in range(full_size)]
    samples_st = [[] for _ in range(full_size)]
    datapoints = []
    found = False
    for env_idx in range(start_idx, N_FILES):
        total_ticks = eps_lens[env_idx] * 3
        env_long_term_h = long_term_h[:total_ticks, env_idx]
        env_short_term_h = short_term_h[:total_ticks, env_idx]

        for sq_y in range(1, 9):
            for sq_x in range(1, 9):
                if box_or_agent_moves[env_idx][0, sq_y, sq_x] == -1:
                    continue
                uniq_steps = th.unique(box_or_agent_timesteps[env_idx][:, sq_y, sq_x], sorted=True)
                uniq_steps = uniq_steps[1:] if uniq_steps[0] == -1 else uniq_steps

                if len(uniq_steps) < 2:
                    continue
                prev_start = 0
                for step_idx in range(len(uniq_steps) - 1):
                    start_step, end_step = uniq_steps[step_idx] + 1, uniq_steps[step_idx + 1]
                    start_step, end_step = start_step.item(), end_step.item()

                    prev_action = box_or_agent_moves[env_idx][start_step - 1, sq_y, sq_x].item()
                    new_action = box_or_agent_moves[env_idx][start_step, sq_y, sq_x].item()
                    if prev_action == new_action:
                        continue

                    if box_or_agent_moves[env_idx][start_step, sq_y, sq_x] != direction:
                        continue
                    lt_act = env_long_term_h[prev_start * 3 : end_step * 3, sq_y, sq_x]
                    st_act = env_short_term_h[prev_start * 3 : end_step * 3, sq_y, sq_x]
                    partition = (start_step - prev_start) * 3

                    # Update for left part (from prev_start to start_step)
                    for j in range(partition):
                        idx = mid_point - partition + j
                        samples_lt[idx].append(lt_act[j].item())
                        samples_st[idx].append(st_act[j].item())

                    # Update for right part (from start_step to end_step)
                    len_right = (end_step - start_step) * 3
                    for j in range(len_right):
                        idx = mid_point + j
                        samples_lt[idx].append(lt_act[partition + j].item())
                        samples_st[idx].append(st_act[partition + j].item())

                    datapoints.append(
                        Datapoint(env_idx, sq_y, sq_x, prev_start, start_step, end_step, int(prev_action), int(new_action))
                    )
                    found = True
                    if one_datapoint:
                        print(
                            f"Found at env {env_idx}, sq({sq_y}, {sq_x}), start {prev_start}, mid {start_step}, end {end_step}"
                        )
                        break
                    prev_start = start_step
                if one_datapoint and found:
                    break
            if one_datapoint and found:
                break
        if one_datapoint and found:
            break

    # Now, define the region in which to plot (here using mid_point-100 to mid_point+100)
    width = 100
    start_idx = mid_point - width
    end_idx = mid_point + width
    x = np.arange(-width, width)

    # For each time bin in our plotting window, compute the mean and use bootstrap for the 95% CI.
    bin_avg_lt = []
    lower_err_lt = []
    upper_err_lt = []
    bin_avg_st = []
    lower_err_st = []
    upper_err_st = []

    for idx in range(start_idx, end_idx):
        # Long term activations and bootstrap CI
        a_lt = np.array(samples_lt[idx])
        if a_lt.size > 0:
            mean_lt = a_lt.mean()
            bin_avg_lt.append(mean_lt)
            if not one_datapoint:
                bs_res_lt = bootstrap((a_lt,), np.mean, vectorized=False, n_resamples=1000, confidence_level=0.95)
                lo_lt, hi_lt = bs_res_lt.confidence_interval.low, bs_res_lt.confidence_interval.high

                lower_err_lt.append(lo_lt)
                upper_err_lt.append(hi_lt)
        else:
            bin_avg_lt.append(np.nan)
            lower_err_lt.append(0.0)
            upper_err_lt.append(0.0)

        # Short term activations and bootstrap CI
        a_st = np.array(samples_st[idx])
        if a_st.size > 0:
            mean_st = a_st.mean()
            bin_avg_st.append(mean_st)
            if not one_datapoint:
                bs_res_st = bootstrap((a_st,), np.mean, vectorized=False, n_resamples=1000, confidence_level=0.95)
                lo_st, hi_st = bs_res_st.confidence_interval.low, bs_res_st.confidence_interval.high

                lower_err_st.append(lo_st)
                upper_err_st.append(hi_st)
        else:
            bin_avg_st.append(np.nan)
            lower_err_st.append(0.0)
            upper_err_st.append(0.0)

    # Convert lists to numpy arrays for plotting.
    bin_avg_lt = np.array(bin_avg_lt)
    bin_avg_st = np.array(bin_avg_st)

    action_str = ["Up", "Down", "Left", "Right"][direction]
    # plt.figure(figsize=(2.4, 1.8))
    plt.plot(x / 3, bin_avg_lt, label="Long term" if smaller else action_str + " long term")
    plt.plot(x / 3, bin_avg_st, label="Short term" if smaller else action_str + " short term")
    if not one_datapoint:
        plt.fill_between(x / 3, lower_err_lt, upper_err_lt, alpha=0.3)
        plt.fill_between(x / 3, lower_err_st, upper_err_st, alpha=0.3)

    # plt.axvline(0, color="r", linestyle="--")
    plt.legend(loc="lower right", handlelength=0.8 if smaller else 1.2)
    # plt.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.25), handlelength=1.0, columnspacing=0.7)
    plt.grid(True)
    # ax = plt.gca()
    # ax.set_xticks(x_major, minor=False)
    # ax.set_xticks(x, minor=True)
    # ax.set_xticklabels(np.concatenate(np.arange(np.arange(1, (width+1)//3), minor=False)
    kwargs = {"labelpad": 1} if smaller else {}
    plt.xlabel("Centered time steps $(t)$", **kwargs)
    plt.ylabel("Avg activation" + " (down)" if smaller else "", **kwargs)

    filename = f"planning_horizon_{'box' if box else 'agent'}_{action_str}.pdf"
    filename = filename.replace(".pdf", "_smaller.pdf")
    if ON_CLUSTER:
        # plt.savefig(f"/training/new_plots/{filename}", bbox_inches="tight")
        plt.savefig(f"/training/new_plots/{filename}")
    else:
        plt.savefig(f"{LP_DIR}/plots/{filename}")
    plt.show()
    return datapoints


# for direction in range(4):
#     plot_planning_horizon(direction, box=True)

# %%
ret = plot_planning_horizon(1, box=True, smaller=True)

# %%
# datapoints = plot_planning_horizon(1, start_idx=28, one_datapoint=True)
datapoints = plot_planning_horizon(1, box=True)
# orig_len = len(datapoints)
# %% Check contribution on datapoints of right -> down flips

datapoints = plot_planning_horizon(1)
orig_len = len(datapoints)

datapoints = [dp for dp in datapoints if dp.prev_action == 3 and dp.new_action == 1]
print(len(datapoints), orig_len, len(datapoints) / orig_len)

unique_envs = list(set([dp.env_idx for dp in datapoints]))
print(len(unique_envs))


l0h2 = False
if l0h2:
    layer_idx = 0
    c_out = 2
    prev_layer_idx = 0
    ih = False
    prev_tick = True
else:
    # layer_idx = 1
    # c_out = 17
    # prev_layer_idx = 0
    # ih = True
    # prev_tick = False
    layer_idx = 1
    c_out = 17
    prev_layer_idx = 0
    ih = True
    prev_tick = False
# contribution_channels = [2, 10, 14, 9, 29, 20, 28]
contribution_channels = [2, 9, 14]
# contribution_channels = [9, 17, 10]
# contribution_channels = list(range(32))


layer_h = layer_wise_h[prev_layer_idx][:, unique_envs]
cur_state = th.cat([th.zeros_like(layer_h[0, None]), layer_h])
cur_state = cur_state[:-1] if prev_tick else cur_state[1:]
cur_state_reshape = cur_state.reshape(-1, *cur_state.shape[2:])

# j, total_j = find_ijfo_contribution(cur_state_reshape, list(range(32)))
# assert th.isclose(j.sum(-3), total_j, rtol=1e-4, atol=1e-4).all(), (j.sum(-3)[0,4,4], total_j[0,4,4])

j, total_j = find_ijfo_contribution(cur_state_reshape, contribution_channels, layer_idx, c_out, model, ih=ih)
# i, j, f, o = map(lambda x: x.reshape(*cur_state.shape[:2], *x.shape[1:]), (i, j, f, o))
j, total_j = map(lambda x: x.reshape(*cur_state.shape[:2], *x.shape[1:]), (j, total_j))
full_size = 240 * 3
mid_point = full_size // 2

centered_j = th.zeros((len(datapoints), full_size, len(contribution_channels), 4))  # 4 for ijfo
centered_total_j = th.zeros((len(datapoints), full_size, 4))

centered_h = th.zeros((len(datapoints), full_size))

for d_idx, datapoint in enumerate(datapoints):
    # if datapoint.env_idx != 28 or datapoint.sq_y != 4 or datapoint.sq_x != 3:
    #     continue
    # print(f"Datapoint {d_idx}")
    uniq_env_idx = unique_envs.index(datapoint.env_idx)
    sq_y, sq_x = datapoint.sq_y, datapoint.sq_x
    offset_y, offset_x = offset_yx(sq_y, sq_x, [c_out], layer_idx)
    offset_y, offset_x = offset_y.item(), offset_x.item()
    prev_start, new_start, new_end = datapoint.prev_start, datapoint.new_start, datapoint.new_end
    prev_start, new_start, new_end = prev_start * 3, new_start * 3, new_end * 3

    centered_j[d_idx, mid_point - (new_start - prev_start) : mid_point] = j[
        prev_start:new_start, uniq_env_idx, :, offset_y, offset_x
    ]
    centered_j[d_idx, mid_point : mid_point + (new_end - new_start)] = j[
        new_start:new_end, uniq_env_idx, :, offset_y, offset_x
    ]

    centered_total_j[d_idx, mid_point - (new_start - prev_start) : mid_point] = total_j[
        prev_start:new_start, uniq_env_idx, offset_y, offset_x
    ]
    centered_total_j[d_idx, mid_point : mid_point + (new_end - new_start)] = total_j[
        new_start:new_end, uniq_env_idx, offset_y, offset_x
    ]

    centered_h[d_idx, mid_point - (new_start - prev_start) : mid_point] = cur_state[
        prev_start:new_start, uniq_env_idx, c_out, offset_y, offset_x
    ]
    centered_h[d_idx, mid_point : mid_point + (new_end - new_start)] = cur_state[
        new_start:new_end, uniq_env_idx, c_out, offset_y, offset_x
    ]

centered_j = centered_j.numpy()
centered_total_j = centered_total_j.numpy()
centered_h = centered_h.numpy()

start_idx = mid_point - 50
end_idx = mid_point + 50
x = np.arange(-50, 50)

# %%

bin_avg_j = []
lower_err_j = []
upper_err_j = []

bin_avg_h = []

bin_avg_total_j = []
lower_err_total_j = []
upper_err_total_j = []
indices = []

data_slice = slice(None)
# data_slice = slice(9, 10)

for idx_i, idx in enumerate(range(start_idx, end_idx)):
    a_j = np.array(centered_j[data_slice, idx])
    a_total_j = np.array(centered_total_j[data_slice, idx])
    a_j = a_j[np.all(a_j != 0, axis=(1, 2))]  # remove zeroes
    a_total_j = a_total_j[np.all(a_total_j != 0, axis=1)]  # remove zeroes

    a_h = np.array(centered_h[data_slice, idx])
    a_h = a_h[a_h != 0]  # remove zeroes

    if a_j.size > 0:
        mean_j = a_j.mean(0)
        bin_avg_j.append(mean_j)
        mean_total_j = a_total_j.mean(0)
        bin_avg_total_j.append(mean_total_j)

        bin_avg_h.append(a_h.mean(0))

        indices.append(x[idx_i])

        try:
            bs_res_j = bootstrap((a_j,), np.mean, vectorized=False, n_resamples=1000, confidence_level=0.95)
            lo_j, hi_j = bs_res_j.confidence_interval.low, bs_res_j.confidence_interval.high
            lower_err_j.append(lo_j)
            upper_err_j.append(hi_j)

            bs_res_total_j = bootstrap((a_total_j,), np.mean, vectorized=False, n_resamples=1000, confidence_level=0.95)
            lo_total_j, hi_total_j = bs_res_total_j.confidence_interval.low, bs_res_total_j.confidence_interval.high
            lower_err_total_j.append(lo_total_j)
            upper_err_total_j.append(hi_total_j)

        except:  # noqa
            print("Bootstrap failed for idx", idx_i)
            continue

bin_avg_j = np.array(bin_avg_j)
bin_avg_total_j = np.array(bin_avg_total_j)

bin_avg_h = np.array(bin_avg_h)

# %%
ijfo_indices = [1, 3]
plt.rcParams["font.family"] = "serif"
apply_style(figsize=(5, 1.8), font=9)
fig, axs = plt.subplots(1, len(ijfo_indices) + 1, sharey=True)
labels = ["H2: short-term ↓", "H9: short-term →", "H14: long-term ↓", "Sum"]
for idx_in_ijfo_indices, ijfo_idx in enumerate(ijfo_indices):
    for i in range(len(contribution_channels)):
        axs[idx_in_ijfo_indices].plot(
            indices, bin_avg_j[:, i, ijfo_idx], label=labels[i], linestyle="--"
        )  # f"H{contribution_channels[i]}"
        # if len(lower_err_j) > 0:
        #     plt.fill_between(indices, np.array(lower_err_j)[:, i], np.array(upper_err_j)[:, i], alpha=0.3)

    # axs[idx_in_ijfo_indices].plot(indices, bin_avg_total_j[:, ijfo_idx], label="Total", color="gray", linestyle="--")
    total = bin_avg_j[:, :, ijfo_idx].sum(1)
    # total = th.tensor(bin_avg_total_j[:, ijfo_idx])
    # total = th.tensor(bin_avg_j[:, :, ijfo_idx].sum(1))
    # total = th.sigmoid(total) if ijfo_idx == 1 else th.tanh(total)
    # total = total.numpy()
    axs[idx_in_ijfo_indices].plot(indices, total, label="Total", color="black", linewidth=0.8)
    # if len(lower_err_total_j) > 0:
    #     plt.fill_between(indices, np.array(lower_err_total_j), np.array(upper_err_total_j), alpha=0.3)
    axs[idx_in_ijfo_indices].set_xlabel("Centered time steps $(t)$\n " + " $" + ["i", "j", "f", "o"][ijfo_idx] + "$")
    axs[idx_in_ijfo_indices].grid(True)

h = th.tensor(bin_avg_total_j[:, 1]).sigmoid() * th.tensor(bin_avg_total_j[:, 3]).tanh()
# h = th.tensor(bin_avg_j[..., 1].sum(1)).sigmoid() * th.tensor(bin_avg_j[..., 3].sum(1)).tanh()
# axs[-1].plot(indices, bin_avg_h, color="black")
axs[-1].plot(indices, h.numpy(), color="black")
axs[-1].grid(True)
axs[-1].set_xlabel("Centered time steps $(t)$\n  $h \\approx \\sigma(j) \cdot$ tanh$(o)$")

# reduce gap between legend columns
fig.legend(labels, ncol=4, loc="upper center", bbox_to_anchor=(0.54, 1.18), handlelength=1.7, columnspacing=0.7)

axs[0].set_ylabel("Avg activation (L1H17)")
# axs[0].set_xlabel("Time steps $(t)$\n(a) J")
if ON_CLUSTER:
    plt.savefig("/training/new_plots/planning_horizon_contribution.pdf", bbox_inches="tight")
else:
    plt.savefig(f"{LP_DIR}/plots_new/planning_horizon_contribution.pdf", bbox_inches="tight")
plt.show()


# %%
env_idx = 28
direction = 1
env_len = eps_lens[env_idx]
box_groups = get_group_channels("box", return_dict=True)
plot_channels = [(d["layer"], d["idx"]) for d in box_groups[direction]]
feature_acts = np.stack(
    [apply_inv_offset_lc(layer_wise_h[layer][: 3 * env_len, env_idx, idx], layer, idx, True) for layer, idx in plot_channels],
    # [layer_wise_h[layer][: 3 * env_len, env_idx, idx] for layer, idx in plot_channels],
    1,
)
plot_obs = obs[:env_len, env_idx].numpy()
plot_obs = np.repeat(plot_obs, 3, 0)
feature_labels = [f"L{layer}H{idx}" for layer, idx in plot_channels]
plotly_feature_vis(
    feature_acts,
    plot_obs,
    feature_labels=feature_labels,
    common_channel_norm=True,
).show()

# %% all groups visualized

env_idx = 10
env_len = eps_lens[env_idx]


def get_feature_acts(
    box=True,
    agent=False,
    short_term=True,
    long_term=False,
    directions_re="up|down|left|right",
) -> tuple[np.ndarray, list[str]]:
    feature_acts = []
    feature_labels = []
    re_str = rf"^[{'A' if agent else ''}{'B' if box else ''}] ({directions_re})$"

    def fetch(channel_dict):
        long_term_option = channel_dict.get("long-term", None)
        if long_term and long_term_option:
            return True
        if short_term and (not long_term_option):
            return True
        return False

    for key, groups in layer_groups.items():
        if not re.match(re_str, key):
            continue

        plot_channels = [(d["layer"], d["idx"]) for d in groups if fetch(d)]
        feature_acts += [
            apply_inv_offset_lc(layer_wise_h[layer][: 3 * env_len, env_idx, idx], layer, idx, True)
            for layer, idx in plot_channels
        ]
        feature_labels += [f"{key} L{layer}H{idx}" for layer, idx in plot_channels]
    return th.stack(feature_acts, 1).numpy(), feature_labels


feature_acts, feature_labels = get_feature_acts(box=True, agent=False, short_term=True, long_term=True)
plot_obs = obs[:env_len, env_idx].numpy()
plot_obs = np.repeat(plot_obs, 3, 0)
plotly_feature_vis(
    feature_acts,
    plot_obs,
    feature_labels=feature_labels,
    common_channel_norm=True,
    height=800,
).show()

# %% Check kernels between long-term and short-term channels


short_term = (1, 17)
long_term = (0, 14)

get_conv_weights(1, 17, 14, model, "i", ih=True)
get_conv_weights(1, 17, 14, model, "j", ih=True)
get_conv_weights(1, 17, 14, model, "o", ih=True)

#
get_conv_weights(1, 17, 13, model, "o", ih=True)
get_conv_weights(1, 17, 13, model, "i", ih=True)
get_conv_weights(1, 17, 13, model, "j", ih=True)


get_conv_weights(0, 14, 13, model, "i", ih=True)

# %%

env_idx = 28
env_len = eps_lens[env_idx]
start_obs = obs[0, env_idx]
level_rep = tiny_world_rgb_to_txt(start_obs.permute(1, 2, 0).numpy())
print(level_rep)

reset_opts = parse_level(level_rep)
play_out = play_level(
    envs,
    model,
    reset_opts,
    max_steps=env_len,
    internal_steps=True,
)
play_cache = play_out.cache
play_obs = play_out.obs.squeeze(1).numpy()
play_cache = {k: v.squeeze(1) for k, v in play_cache.items() if len(v.shape) == 5}
# %%
# plot_layer, plot_channel = 2, 17
plot_layer, plot_channel = 0, 6
# tick = reps - 1
tick = 0
show_ticks = True
# keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "C", "I", "J", "F", "O"]]
keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "I", "J", "O"]]

play_all_channels_for_lc = np.stack([play_cache[key][:, plot_channel] for key in keys], axis=1)
# if not show_ticks:
#     play_all_channels_for_lc = play_all_channels_for_lc[tick::3]

# repeat obs 3 along first dimension
fig = plotly_feature_vis(
    play_all_channels_for_lc,
    np.repeat(play_obs, 3, 0),
    feature_labels=[k.rsplit(".")[-1] for k in keys],
)
fig.show()

# %% Visualize
plot_layer, plot_channel, ih, ijfo, inp_types = 0, 2, False, "i", "lh"


def ijfo_idx(ijfo):
    return ["i", "j", "f", "o"].index(ijfo)


play_all_channels_for_lc, top_channels, values = visualize_top_conv_inputs(
    plot_layer,
    plot_channel,
    out_type=ijfo,
    model=model,
    cache=play_cache,
    ih=ih,
    num_channels=6 + 1 * 8,
    inp_types=inp_types,
    top_channel_sum=True,
)
plot_channel = 32 * ijfo_idx(ijfo) + plot_channel
play_all_channels_for_lc = play_all_channels_for_lc.numpy()
fig = plotly_feature_vis(
    play_all_channels_for_lc,
    np.repeat(play_obs, 3, 0),
    feature_labels=[f"{c}: {v:.2f}" for c, v in zip(top_channels, values)],  # + ["ih" if ih else "hh"],
    common_channel_norm=True,
)
fig.show()
