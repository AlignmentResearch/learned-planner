"""Demonstration that network prefers paths closer to target than shorter length paths."""

# %%
import dataclasses
from copy import deepcopy
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
import torch as th
from cleanba.environments import BoxobanConfig

from learned_planner import LP_DIR
from learned_planner.interp.channel_group import layer_groups
from learned_planner.interp.offset_fns import offset_yx
from learned_planner.interp.plot import plot_group, plotly_feature_vis
from learned_planner.interp.utils import join_cache_across_steps, load_jax_model_to_torch, play_level
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


# %%
envs = dataclasses.replace(boxo_cfg, num_envs=1).make()
thinking_steps = 0
max_steps = 120

size = 10
walls = [(i, 0) for i in range(size)]
walls += [(i, size - 1) for i in range(size)]
walls += [(0, i) for i in range(1, size - 1)]
walls += [(size - 1, i) for i in range(1, size - 1)]

walls += [(y, x) for y in range(4, 8) for x in range(3, 7)]

boxes = [(3, 6)]
targets = [(8, 6)]
player = (1, 5)
toy_reset_opt = dict(walls=walls, boxes=boxes, targets=targets, player=player)
obs = envs.reset(options=toy_reset_opt)[0][0]


@dataclass
class Path:
    up: list[tuple[int, int]] = field(default_factory=lambda: [])
    down: list[tuple[int, int]] = field(default_factory=lambda: [])
    left: list[tuple[int, int]] = field(default_factory=lambda: [])
    right: list[tuple[int, int]] = field(default_factory=lambda: [])
    is_agent_path: bool = False

    def get_group_dict(self):
        prefix = "A " if self.is_agent_path else "B "
        return {prefix + "up": self.up, prefix + "down": self.down, prefix + "left": self.left, prefix + "right": self.right}

    def box_to_agent_moves(self):
        assert not self.is_agent_path
        up = [(y + 1, x) for y, x in self.up]
        down = [(y - 1, x) for y, x in self.down]
        left = [(y, x + 1) for y, x in self.left]
        right = [(y, x - 1) for y, x in self.right]
        return Path(up=up, down=down, left=left, right=right, is_agent_path=True)

    @classmethod
    def combine(cls, paths: list["Path"]):
        new_path = Path()
        for path in paths:
            new_path.up += path.up
            new_path.down += path.down
            new_path.left += path.left
            new_path.right += path.right
        return new_path


def prepare_state(state: list[tuple[th.Tensor, th.Tensor]], path: Path):
    assert state[0][0].ndim == 4
    group_dict = {**path.get_group_dict(), **(path.box_to_agent_moves().get_group_dict())}
    for group_name, sqs in group_dict.items():
        if not sqs:
            continue
        channel_dicts = layer_groups[group_name]
        for channel_dict in channel_dicts:
            desc = channel_dict["description"].lower()
            if "nfa" in desc or "mpa" in desc:
                continue
            l, c, sign = channel_dict["layer"], channel_dict["idx"], channel_dict["sign"]
            for y, x in sqs:
                offset_y, offset_x = tuple(map(lambda x: x.item(), offset_yx(y, x, [c], l)))
                state[l][0][:, c, offset_y, offset_x] = sign
    return state


def plot_observation_with_arrows(
    obs: np.ndarray,
    path_obj: Path | list[Path],
    base_arrow_color="blue",
    agent_arrow_color="green",
    fig_size=(4, 4),
    show_ticks=False,
):
    """
    Plots an RGB observation with arrows indicating paths.

    Args:
        obs (np.ndarray): The observation image. Expected to be in (H, W, C) or (C, H, W) format.
                          If (C, H, W), it will be transposed to (H, W, C).
        path_obj (Path): A Path object containing lists of coordinates for different arrow directions.
        base_arrow_color (str): Color for non-agent path arrows.
        agent_arrow_color (str): Color for agent path arrows.
        fig_size (tuple): Size of the matplotlib figure.
    """
    if isinstance(obs, th.Tensor):
        obs = obs.numpy()
    if obs.ndim != 3:
        raise ValueError(f"obs must be a 3D array (H,W,C or C,H,W), got {obs.ndim} dimensions")

    # Transpose if channels-first (e.g., 3, H, W)
    if obs.shape[0] == 3 and obs.shape[1] != 3 and obs.shape[2] != 3:
        img_data = np.transpose(obs, (1, 2, 0))
    elif obs.shape[-1] == 3:  # Already H, W, C
        img_data = obs.copy()  # Work on a copy
    else:
        raise ValueError(f"Observation shape {obs.shape} not supported. Expected (C,H,W) or (H,W,C) with C=3.")

    # Normalize image data for display if it's float and not in [0,1] or int not in [0,255]
    if img_data.dtype in [np.float32, np.float64, float]:
        if img_data.min() < 0.0 or img_data.max() > 1.0:
            if img_data.max() > 1.0:  # Potentially 0-255 range
                img_data = img_data / 255.0
            img_data = np.clip(img_data, 0.0, 1.0)
    elif img_data.dtype not in [np.uint8]:
        print(
            f"Warning: Image data type is {img_data.dtype}. Attempting to plot as is. "
            "Consider normalizing to [0,1] float or [0,255] uint8."
        )

    plt.figure(figsize=fig_size)
    plt.imshow(img_data)
    ax = plt.gca()

    arrow_params = {"head_width": 0.3, "head_length": 0.2, "length_includes_head": True, "linewidth": 1.5}
    arrow_draw_length = 0.5

    if isinstance(path_obj, list):
        path_obj = Path.combine(path_obj)

    current_color = agent_arrow_color if path_obj.is_agent_path else base_arrow_color

    paths_to_draw = {
        "up": path_obj.up,
        "down": path_obj.down,
        "left": path_obj.left,
        "right": path_obj.right,
    }

    for direction, coords_list in paths_to_draw.items():
        for r, c in coords_list:
            if direction == "up":
                dx, dy = 0, -arrow_draw_length
            elif direction == "down":
                dx, dy = 0, arrow_draw_length
            elif direction == "left":
                dx, dy = -arrow_draw_length, 0
            elif direction == "right":
                dx, dy = arrow_draw_length, 0
            else:
                continue

            ax.arrow(c, r, dx, dy, fc=current_color, ec=current_color, **arrow_params)

    height, width = img_data.shape[:2]
    ax.set_xlim([-0.5, width - 0.5])
    ax.set_ylim([height - 0.5, -0.5])
    ax.set_aspect("equal", adjustable="box")

    # plt.grid(True, which="both", color="gray", linestyle="--", linewidth=0.5)
    if show_ticks:
        plt.xticks(np.arange(width), [str(i) for i in np.arange(width)])  # Corrected x-axis ticks
        plt.yticks(np.arange(height), [str(i) for i in np.arange(height)])  # Corrected y-axis ticks
        ax.tick_params(axis="both", which="major", labelsize=8)
    else:
        plt.xticks([])
        plt.yticks([])
    # plt.show()


path1 = Path(left=[(3, x) for x in range(3, 7)], down=[(y, 2) for y in range(3, 8)], right=[(8, 2), (8, 3)])
path2 = Path(right=[(3, 6)], down=[(y, 7) for y in range(3, 7)])

state = model.recurrent_initial_state()
state = prepare_state(state, path1)
state = prepare_state(state, path2)

state = [(h[None], c[None]) for h, c in state]

plot_observation_with_arrows(obs, [path1, path2])
plt.savefig("../../new_plots/plan_closer_to_target_heuristic.pdf")
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
    state=state,
)

toy_cache = toy_out.cache
toy_cache = {k: v.squeeze(1) for k, v in toy_cache.items() if len(v.shape) == 5}
toy_obs = toy_out.obs.squeeze(1)
toy_obs_repeated = toy_obs.repeat_interleave(reps, 0).numpy()
print("Total len:", len(toy_obs), toy_cache["features_extractor.cell_list.0.hook_h"].shape[0] // 3)

if two_levels:
    play_toy = True

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


pio.templates.default = "plotly"

plot_group(toy_cache, toy_obs_repeated)


# %%
restore_model()
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
plot_layer, plot_channel = 2, 26
# tick = reps - 1
tick = 0
show_ticks = True
keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "C", "I", "J", "F", "O"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, plot_channel] for key in keys], axis=1)

# repeat obs 3 along first dimension
fig = plotly_feature_vis(
    toy_all_channels_for_lc,
    toy_obs_repeated,
    feature_labels=[k.rsplit(".")[-1] for k in keys],
    common_channel_norm=True,
)
fig.show()

# %%  batch: use this to visualize all channels of a layer
layer, batch_no = 1, 0
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
    toy_obs_repeated,
    feature_labels=[f"L{layer}{hcijfo.upper()}{batch_no * batch_size + i}" for i in range(batch_size)],
    height=800,
    common_channel_norm=True,
)
fig.show()
# %% conv ih hh for ijo
plot_layer, plot_channel = 2, 6

keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["conv_ih", "conv_hh"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, 32 * ijo + plot_channel] for key in keys for ijo in [0, 1, 3]], axis=1)
# repeat obs 3 along first dimension
fig = plotly_feature_vis(
    toy_all_channels_for_lc,
    toy_obs_repeated,
    feature_labels=[k.rsplit(".")[-1][5:] + "_" + "ijfo"[ijo] for k in keys for ijo in [0, 1, 3]],
)
fig.show()
# %% h/c/i/j/f/o
# plot_layer, plot_channel = 2, 6
plot_layer, plot_channel = 1, 5

keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "C", "I", "J", "F", "O"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, plot_channel] for key in keys], axis=1)
# repeat obs 3 along first dimension
fig = plotly_feature_vis(toy_all_channels_for_lc, toy_obs_repeated, feature_labels=[k.rsplit(".")[-1] for k in keys])
fig.show()
