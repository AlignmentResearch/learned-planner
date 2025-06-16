"""Script to zero-ablate kernels between box-direction channels that shows how one plan gets picked."""

# %%
import dataclasses
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch as th
from cleanba.environments import BoxobanConfig
from matplotlib.patches import Patch  # For custom legend
from plotly.subplots import make_subplots

from learned_planner import BOXOBAN_CACHE, IS_NOTEBOOK, LP_DIR
from learned_planner.interp.channel_group import (
    get_channel_dict,
    get_group_channels,
    get_group_connections,
    standardize_channel,
)
from learned_planner.interp.plot import apply_style, plotly_feature_vis, save_video_from_plotly
from learned_planner.interp.utils import load_jax_model_to_torch, play_level
from learned_planner.interp.weight_utils import find_ijfo_contribution
from learned_planner.policies import download_policy_from_huggingface

# pio.kaleido.scope.mathjax = None  # Disable MathJax to remove the loading message

if IS_NOTEBOOK:
    pio.renderers.default = "notebook"
else:
    pio.renderers.default = "png"
th.set_printoptions(sci_mode=False, precision=2)


# The normalization function remains the same as it correctly maps
# "no signal" (0 or negative) to 0.0 and "max signal" to 1.0.
def normalize_heatmap_simple_positive(heatmap_data):
    """
    Normalizes a heatmap to the [0, 1] range.
    Non-positive values in the input heatmap are treated as "no signal" and map to 0.0 intensity.
    The maximum positive value in the input heatmap maps to 1.0 intensity (full color).
    """
    heatmap_float = heatmap_data.astype(float)
    heatmap_clamped = np.maximum(0, heatmap_float)  # Treat non-positive values as zero signal

    max_val = np.max(heatmap_clamped)

    if max_val == 0:
        # If all values are <= 0, the heatmap is effectively all zeros after clamping
        return np.zeros_like(heatmap_clamped, dtype=float)

    normalized = heatmap_clamped / max_val
    return np.clip(normalized, 0, 1)


def display_combined_heatmaps_white_zero(
    heatmaps,
    peak_colors=None,
    legend_labels=None,
    fig_size=(8, 8),
    title="",
    save_path="",
    return_rgb=False,
):
    """
    Combines multiple heatmaps using multiplicative blending.
    "Zero signal" (non-positive values) in each heatmap maps to white.
    Maximum signal maps to the heatmap's assigned peak color.
    Overlapping colors will mix subtractively (e.g., red and green make yellow-brown/dark).

    Args:
        heatmaps (list of np.ndarray): List of 2D numpy arrays (heatmaps).
        peak_colors (list of RGB/RGBA tuples, optional): Peak colors for each heatmap.
                                                       Defaults to 'tab10'/'tab20' colors.
        legend_labels (list of str, optional): Labels for the legend.
        fig_size (tuple, optional): Figure size.
    """
    if not heatmaps:
        print("Warning: No heatmaps provided.")
        return

    num_heatmaps = len(heatmaps)
    if num_heatmaps == 0:
        print("Warning: The list of heatmaps is empty.")
        return

    heatmap_shape = heatmaps[0].shape
    for i, hm in enumerate(heatmaps):
        if hm.shape != heatmap_shape:
            raise ValueError(
                f"All heatmaps must have the same shape. Heatmap {i} has shape {hm.shape}, expected {heatmap_shape}."
            )

    # Determine peak colors if not provided (same logic as before)
    if peak_colors is None:
        cmap_obj = plt.colormaps.get("tab10")
        peak_colors = [cmap_obj(i) for i in range(num_heatmaps)]
    elif len(peak_colors) < num_heatmaps:
        raise ValueError("The number of provided peak_colors is less than the number of heatmaps.")

    # MODIFICATION: Initialize the combined image to white (all ones)
    combined_rgb_image = np.ones((*heatmap_shape, 3), dtype=float)
    white_rgb = np.array([1.0, 1.0, 1.0])  # Define white color

    for i in range(num_heatmaps):
        current_heatmap = heatmaps[i]

        # 1. Normalize the current heatmap (0 for no signal, 1 for max signal)
        # normalized_heatmap = normalize_heatmap_simple_positive(current_heatmap)
        normalized_heatmap = current_heatmap

        # 2. Get the target peak color (RGB part)
        current_peak_color_rgb = np.array(peak_colors[i][:3])

        # 3. MODIFICATION: Calculate layer color contribution
        #    Interpolates from white (for H_norm=0) to peak_color (for H_norm=1)
        #    H_norm_expanded shape: (H, W, 1)
        #    white_rgb shape: (3,)
        #    current_peak_color_rgb shape: (3,)
        #    Resulting layer_color_value shape: (H, W, 3)
        h_norm_expanded = np.expand_dims(normalized_heatmap, axis=-1)
        layer_color_value = (1 - h_norm_expanded) * white_rgb + h_norm_expanded * current_peak_color_rgb

        # 4. MODIFICATION: Multiplicative blending
        #    Each layer's color acts as a filter on the current combined image.
        combined_rgb_image *= layer_color_value

    # 5. Clip values to ensure they are in the valid [0, 1] range (good practice)
    combined_rgb_image = np.clip(combined_rgb_image, 0, 1)

    if return_rgb:
        return combined_rgb_image

    # 6. Display the result
    fig, ax = plt.subplots(figsize=fig_size)
    ax.imshow(combined_rgb_image)
    # MODIFICATION: Update title to reflect new color logic
    # ax.set_title(f"Combined Visualization ({num_heatmaps} Heatmaps, Zero Signal = White)")
    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    ax.axis("off")  # Optional

    # Legend logic (remains the same, shows peak colors)
    if legend_labels:
        if len(legend_labels) == num_heatmaps:
            legend_elements = [Patch(facecolor=peak_colors[i], label=legend_labels[i]) for i in range(num_heatmaps)]
            if num_heatmaps > 5:
                ax.legend(
                    handles=legend_elements,
                    title="Heatmap Legend",
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0.0,
                )
                fig.subplots_adjust(right=0.75 if num_heatmaps > 10 else 0.7)
            else:
                ax.legend(handles=legend_elements, title="Heatmap Legend", loc="best")
        else:
            print("Warning: Number of legend_labels does not match number of heatmaps. Skipping custom legend.")

    if save_path:
        plt.savefig(save_path)


# %%
# MODEL_PATH_IN_REPO = "drc11/eue6pax7/cp_2002944000"  # DRC(1, 1) 2B checkpoint
MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000"  # DRC(1, 1) 2B checkpoint
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


def get_avg(cache, channels, tick, hook="h"):
    if isinstance(channels[0], tuple):
        assert len(channels[0]) == 2
        channels = [{"layer": c[0], "idx": c[1]} for c in channels]
    avg_channels = np.mean(
        np.stack(
            [
                standardize_channel(
                    cache[f"features_extractor.cell_list.{c_dict['layer']}.hook_{hook}"][tick, c_dict["idx"]],
                    c_dict,
                )
                for c_dict in channels
            ],
            axis=0,
        ),
        axis=0,
    )
    return avg_channels


# %% PLAY TOY LEVEL
envs = dataclasses.replace(boxo_cfg, num_envs=1).make()
play_toy = True
thinking_steps = 0

max_steps = 30
size = 10
walls = [(i, 0) for i in range(size)]
walls += [(i, size - 1) for i in range(size)]
walls += [(0, i) for i in range(1, size - 1)]
walls += [(size - 1, i) for i in range(1, size - 1)]

walls += [(y, x) for y in range(4, 7) for x in range(3, 7)]

boxes = [(3, 2)]
targets = [(8, 8)]
player = (1, 1)


toy_reset_opt = dict(walls=walls, boxes=boxes, targets=targets, player=player)
toy_out = play_level(
    envs,
    model,
    reset_opts=toy_reset_opt,
    max_steps=max_steps,
    internal_steps=True,
)

toy_cache = toy_out.cache
toy_cache = {k: v.squeeze(1) for k, v in toy_cache.items() if len(v.shape) == 5}
toy_obs = toy_out.obs.squeeze(1)

print("Total len:", len(toy_obs), toy_cache["features_extractor.cell_list.0.hook_h"].shape[0] // 3)

# %% GNA visualization


width_px, height_px = apply_style(figsize=(2.7, 1.0), px_use_default=False, px_margin=dict(t=2, b=20, l=1, r=1), font=8)
# pio.templates.default = "plotly_white"

# write a temporary image
fig = go.Figure()
TMP_DIR = "/tmp"
fig.write_image(f"{TMP_DIR}/temp.png", width=width_px, height=height_px)

n_cols = 6
fig = make_subplots(
    rows=2,
    cols=n_cols,
    specs=[
        [{"type": "xy", "colspan": 2, "rowspan": 2}, None, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        [None, None, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
    ],
    # subplot_titles=["Observation", "Step 2", "Step 5", "Step 8"],
    vertical_spacing=0.05,
    horizontal_spacing=0.04,
)
heatmap_args = dict(zmin=-1, zmax=1, colorscale="Viridis", showscale=True, colorbar=dict(thickness=3, xpad=1))

tick = 10
sample_obs = toy_obs[tick // 3]
fig.add_trace(go.Image(z=np.transpose(sample_obs, (1, 2, 0))), row=1, col=1)
ifjo_gate = "f"
gate_idx = ["i", "j", "f", "o"].index(ifjo_gate)
gna_layer = 2
input_h_prev_step = toy_cache[f"features_extractor.cell_list.{gna_layer}.hook_h"][tick - 1, None]
input_h_prev_layer = toy_cache[f"features_extractor.cell_list.{gna_layer - 1}.hook_h"][tick, None]
gna_channels = [4, 26]
for row, gna_channel in enumerate(gna_channels):
    c_dict = {"layer": gna_layer, "idx": gna_channel, "sign": "+"}
    f_acts = toy_cache[f"features_extractor.cell_list.{gna_layer}.hook_f"][tick, gna_channel]
    f_acts = standardize_channel(f_acts, c_dict)
    fig.add_trace(go.Heatmap(z=f_acts[::-1], **heatmap_args), row=row + 1, col=3)

    prev_step_ijfo_acts, total_prev_step_ijfo_acts = find_ijfo_contribution(
        th.tensor(input_h_prev_step), list(range(32)), gna_layer, gna_channel, model, ih=False
    )
    prev_layer_ijfo_acts, total_prev_layer_ijfo_acts = find_ijfo_contribution(
        th.tensor(input_h_prev_layer), list(range(32)), gna_layer, gna_channel, model, ih=True
    )

    agent_acts = prev_step_ijfo_acts[0, [27], ..., gate_idx].sum(dim=0).numpy()
    agent_acts = standardize_channel(agent_acts, c_dict)
    fig.add_trace(go.Heatmap(z=agent_acts[::-1], **heatmap_args), row=row + 1, col=4)

    gna_acts = prev_step_ijfo_acts[0, gna_channels, ..., gate_idx].sum(dim=0).numpy()
    gna_acts = standardize_channel(gna_acts, c_dict)

    # prev_layer_channels = [17, 18, 19]
    # prev_layer_channels = list(range(32))
    prev_layer_channels = [17, 19]
    plan_ijfo = prev_layer_ijfo_acts[0, prev_layer_channels, ..., gate_idx].sum(dim=0).numpy()
    plan_ijfo = standardize_channel(plan_ijfo, c_dict)
    fig.add_trace(go.Heatmap(z=plan_ijfo[::-1], **heatmap_args), row=row + 1, col=5)

    prev_layer_channels = [18]
    plan_ijfo = prev_layer_ijfo_acts[0, prev_layer_channels, ..., gate_idx].sum(dim=0).numpy()
    plan_ijfo = standardize_channel(plan_ijfo, c_dict)
    fig.add_trace(go.Heatmap(z=plan_ijfo[::-1], **heatmap_args), row=row + 1, col=6)


c_dict = {"layer": gna_layer, "idx": gna_channel, "sign": "+"}
gate_idx = ["i", "j", "f", "o"].index(ifjo_gate)
input_h_prev_step = toy_cache[f"features_extractor.cell_list.{gna_layer}.hook_h"][tick - 1, None]
y_offset = -1.7
fig.add_annotation(
    text="Observation<br> ",
    x=0.5,
    y=y_offset,
    xref="x domain",
    yref="y3 domain",
    showarrow=False,
    # font=dict(size=10, color="red"),
)

fig.add_annotation(
    text=r"$f\text{-gate}$",
    x=0.5,
    y=y_offset + 0.3,
    xref="x2 domain",
    yref="y2 domain",
    showarrow=False,
    # font=dict(size=10, color="red"),
)

fig.add_annotation(
    text="Agent<br>(L2H27)",
    x=0.5,
    y=y_offset,
    xref="x3 domain",
    yref="y3 domain",
    showarrow=False,
    # font=dict(size=10, color="red"),
)

fig.add_annotation(
    text="Box-down<br>(L1H17)",
    x=0.5,
    y=y_offset,
    xref="x4 domain",
    yref="y4 domain",
    showarrow=False,
    # font=dict(size=10, color="red"),
)

fig.add_annotation(
    text="Agent-down<br>(L1H18)",
    x=0.5,
    y=y_offset,
    xref="x5 domain",
    yref="y5 domain",
    showarrow=False,
    # font=dict(size=10, color="red"),
)

x_offset = -0.3
fig.add_annotation(
    # text="GNA-down",
    text="L2F4",
    x=x_offset - 0.05,
    y=0.5,
    xref="x2 domain",
    yref="y2 domain",
    showarrow=False,
    textangle=-90,
    # font=dict(size=10, color="red"),
)
fig.add_annotation(
    # text="GNA-right",
    text="L2F26",
    x=x_offset - 0.05,
    y=0.5,
    xref="x2 domain",
    yref="y8 domain",
    showarrow=False,
    textangle=-90,
    # font=dict(size=10, color="red"),
)
fig.add_annotation(
    # text="=",
    text=r"$\underleftarrow{\sigma}$",
    xref="x7 domain",
    yref="y6 domain",
    x=x_offset,
    y=0.5,
    showarrow=False,
    # font=dict(size=13),
)
fig.add_annotation(
    # text="=",
    text=r"$\underleftarrow{\sigma}$",
    xref="x7 domain",
    yref="y2 domain",
    x=x_offset,
    y=0.5,
    showarrow=False,
    # font=dict(size=13),
)
fig.add_annotation(
    text="+",
    xref="x8 domain",
    yref="y6 domain",
    x=x_offset,
    y=0.5,
    showarrow=False,
    # font=dict(size=13),
)
fig.add_annotation(
    text="+",
    xref="x8 domain",
    yref="y2 domain",
    x=x_offset,
    y=0.5,
    showarrow=False,
    # font=dict(size=13),
)

fig.add_annotation(
    text="+",
    xref="x9 domain",
    yref="y6 domain",
    x=x_offset,
    y=0.5,
    showarrow=False,
    # font=dict(size=13),
)
fig.add_annotation(
    text="+",
    xref="x9 domain",
    yref="y2 domain",
    x=x_offset,
    y=0.5,
    showarrow=False,
    # font=dict(size=13),
)


fig.update_xaxes(showticklabels=False, visible=False, ticks="", row=1, col=1)
fig.update_yaxes(showticklabels=False, visible=False, ticks="", row=1, col=1)
for col in range(3, 3 + n_cols):
    fig.update_xaxes(showticklabels=False, visible=False, ticks="", constrain="domain", row=1, col=col)
    fig.update_yaxes(showticklabels=False, visible=False, ticks="", row=1, col=col)
    fig.update_xaxes(showticklabels=False, visible=False, ticks="", constrain="domain", row=2, col=col)
    fig.update_yaxes(showticklabels=False, visible=False, ticks="", row=2, col=col)

# fig.write_image(LP_DIR / "new_plots" / "GNA_f_contribution.pdf", scale=2)
fig.write_image(LP_DIR / "new_plots" / "GNA_f_contribution.svg")
if IS_NOTEBOOK:
    fig.show()

# %% Ablate connections from B down to B right and from B right to B down

group_channels = get_group_channels("box_agent")
misc_plan_channels = get_group_channels("Misc plan", return_dict=True)

print(len(group_channels[1]), len(group_channels[3]))
for c_dict in misc_plan_channels[0]:
    if "down" in c_dict["description"]:
        group_channels[1].append((c_dict["layer"], c_dict["idx"]))
    if "right" in c_dict["description"]:
        group_channels[3].append((c_dict["layer"], c_dict["idx"]))
print(len(group_channels[1]), len(group_channels[3]))
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

abl_out = play_level(
    envs,
    model,
    reset_opts=toy_reset_opt,
    max_steps=max_steps,
    internal_steps=True,
)

abl_cache = abl_out.cache
abl_cache = {k: v.squeeze(1) for k, v in abl_cache.items() if len(v.shape) == 5}
abl_obs = abl_out.obs.squeeze(1)

restore_model()

# %%

width_px, height_px = apply_style(figsize=(2.7, 1.0), px_use_default=False, font=12)
save_video = True

if save_video:
    down_acts = get_avg(toy_cache, [get_channel_dict(1, 17)], slice(None))
    right_acts = get_avg(toy_cache, [get_channel_dict(2, 9)], slice(None))
    all_acts = np.stack([down_acts, right_acts], axis=1)
    toy_obs_repeated = np.repeat(toy_obs, 3, axis=0)
    labels = ["Down (L1H17)", "Right (L2H9)"]
    fig = go.Figure()
    fig = plotly_feature_vis(
        all_acts, toy_obs_repeated, feature_labels=labels, common_channel_norm=True, facet_col_spacing=0.01
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
                    len=0.7,
                ),
            ),
            showlegend=False,
        )
    )
    fig.update_layout(margin=dict(l=10, r=5, t=10, b=0, pad=0))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, ticks="")
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, ticks="")
    # shift = 20
    fig.for_each_annotation(lambda a: a.update(y=0.85, yref="paper"))
    frame_width = 800
    frame_height = 400
    # fig.show()
    save_video_from_plotly(fig, LP_DIR / "new_plots" / "two_paths.mp4", fps=4, demo=False)


if save_video:
    down_acts = get_avg(abl_cache, [get_channel_dict(1, 17)], slice(None))
    right_acts = get_avg(abl_cache, [get_channel_dict(2, 9)], slice(None))
    all_acts = np.stack([down_acts, right_acts], axis=1)
    toy_obs_repeated = np.repeat(abl_obs, 3, axis=0)
    labels = ["Down (L1H17)", "Right (L2H9)"]
    fig = go.Figure()
    fig = plotly_feature_vis(
        all_acts, toy_obs_repeated, feature_labels=labels, common_channel_norm=True, facet_col_spacing=0.01
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
                    len=0.7,
                ),
            ),
            showlegend=False,
        )
    )
    fig.update_layout(margin=dict(l=10, r=5, t=10, b=0, pad=0))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, ticks="")
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, ticks="")
    # shift = 20
    fig.for_each_annotation(lambda a: a.update(y=0.85, yref="paper"))
    frame_width = 800
    frame_height = 400
    # fig.show()

    save_video_from_plotly(fig, LP_DIR / "new_plots" / "two_paths_abl.mp4", fps=4, demo=False)


# %%

box_group_channels = get_group_channels("box", return_dict=True)
# down_channels, right_channels = box_group_channels[1], box_group_channels[3]
down_channels = [get_channel_dict(1, 17)]
right_channels = [get_channel_dict(1, 13)]


def normalized_to_rgb_string(norm_rgb):
    r = int(norm_rgb[0] * 255)
    g = int(norm_rgb[1] * 255)
    b = int(norm_rgb[2] * 255)
    return f"rgb({r},{g},{b})"


def custom_scale(color_rgb_str):
    white_str = "rgb(255, 255, 255)"
    return [[0.0, white_str], [1.0, color_rgb_str]]


# Create a subplot grid: 2 rows x 5 cols with a merged cell covering the first 2 rows and 2 cols.
apply_style(figsize=(2.7, 1.1), px_use_default=False, px_margin=dict(t=0, b=5, l=1, r=1))

color1_rgb_str = normalized_to_rgb_string(plt.colormaps.get("tab10")(0))
color2_rgb_str = normalized_to_rgb_string(plt.colormaps.get("tab10")(1))

custom_scales = [custom_scale(rgb_str) for rgb_str in [color1_rgb_str, color2_rgb_str]]

fig = make_subplots(
    rows=2,
    cols=5,
    specs=[
        [{"type": "xy", "colspan": 2, "rowspan": 2}, None, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        [None, None, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
    ],
    vertical_spacing=0.005,
)
abl_y = -1.6
fig.add_annotation(
    text="Observation",
    x=0.5,
    y=abl_y,
    xref="x domain",
    yref="y2 domain",
    showarrow=False,
)
fig.add_annotation(
    text="Original",
    x=1.12,
    y=0.5,
    xref="x domain",
    yref="y2 domain",
    showarrow=False,
    textangle=-90,
)
fig.add_annotation(
    text="Ablated",
    x=1.12,
    y=0.5,
    xref="x domain",
    yref="y5 domain",
    showarrow=False,
    textangle=-90,
)


img = toy_obs[0].numpy()
img = np.transpose(img, (1, 2, 0))
fig.add_trace(go.Image(z=img), row=1, col=1)
fig.update_xaxes(showticklabels=False, visible=False, ticks="", row=1, col=1)
fig.update_yaxes(showticklabels=False, visible=False, ticks="", row=1, col=1)
heatmap_args = dict(zmin=0, zmax=0.4, colorscale="Viridis", showscale=True, colorbar=dict(thickness=3, xpad=3))
multiplying_factor = 1.0
for idx, tick in enumerate(range(2, 9, 3)):
    baseline_down = get_avg(toy_cache, down_channels, tick) * multiplying_factor
    baseline_right = get_avg(toy_cache, right_channels, tick) * multiplying_factor
    baseline_plan = baseline_down + baseline_right

    abl_down = get_avg(abl_cache, down_channels, tick)
    abl_right = get_avg(abl_cache, right_channels, tick)
    abl_plan = abl_down + abl_right

    col_index = idx + 3

    baseline_rgb = display_combined_heatmaps_white_zero([baseline_down, baseline_right], return_rgb=True)
    abl_rgb = display_combined_heatmaps_white_zero([abl_down, abl_right], return_rgb=True)

    baseline_rgb = baseline_rgb * 255
    abl_rgb = abl_rgb * 255
    # fig.update_layout(
    #     # paper_bgcolor=PLOT_BGCOLOR,
    #     plot_bgcolor="black",
    #     margin=dict(pad=0, r=20, t=50, b=60, l=60),
    #     row=1,
    #     col=col_index,
    # )

    fig.add_trace(go.Image(z=baseline_rgb), row=1, col=idx + 3)
    fig.add_trace(go.Image(z=abl_rgb), row=2, col=idx + 3)

    # fig.add_trace(go.Heatmap(z=baseline_plan[::-1], **heatmap_args), row=1, col=idx + 3)
    # fig.add_trace(go.Heatmap(z=abl_plan[::-1], **heatmap_args), row=2, col=idx + 3)

    # Remove x and y ticks for this subplot
    fig.update_xaxes(showticklabels=False, ticks="", row=1, col=col_index)
    fig.update_yaxes(showticklabels=False, ticks="", row=1, col=col_index)
    fig.update_xaxes(showticklabels=False, ticks="", row=2, col=col_index)
    fig.update_yaxes(showticklabels=False, ticks="", row=2, col=col_index)

    x_domain = fig.layout[f"xaxis{col_index}"].domain
    x_mid = (x_domain[0] + x_domain[1]) / 2

    fig.add_annotation(
        text=f"Step: {idx}",
        x=0.5,
        y=abl_y,
        xref=f"x{col_index - 1} domain",
        yref="y2 domain",
        showarrow=False,
        # xanchor="center",
        # yanchor="bottom",
    )

axis_line_style = dict(showline=True, linewidth=1, linecolor="black", mirror=True)

# Prepare a dictionary to hold all the axis updates
layout_updates = {}

# Iterate through all items in the figure's layout
# (fig.layout is a dict-like object)
for key in fig.layout:
    # Check if the key refers to an xaxis or a yaxis
    if key.startswith("xaxis") or key.startswith("yaxis"):
        # For each axis found, apply the defined line style
        # If the axis object already has other properties, this will merge/update them
        layout_updates[key] = axis_line_style

# Apply all collected updates to the figure's layout
if layout_updates:  # Check if there are any updates to apply
    fig.update_layout(**layout_updates)

fig.add_trace(
    go.Scatter(
        x=[None],  # No visible data points
        y=[None],
        mode="markers",
        marker=dict(
            colorscale=custom_scales[0],
            cmin=0,  # Minimum value of the colorscale
            cmax=1,  # Maximum value of the colorscale
            showscale=True,  # This is crucial to show the colorbar
            colorbar=dict(
                title=dict(text="L1H17", font=dict(size=8), side="top"),
                thickness=5,
                tickvals=[0, 0.5, 1],  # Optional: specify tick values
                xpad=0,
                # x=1.02,
                ypad=0,
                y=0.8,
                len=0.4,
            ),
        ),
        showlegend=False,  # Hide this trace from the legend
    )
)

fig.add_trace(
    go.Scatter(
        x=[None],  # No visible data points
        y=[None],
        mode="markers",
        marker=dict(
            colorscale=custom_scales[1],
            cmin=0,  # Minimum value of the colorscale
            cmax=1,  # Maximum value of the colorscale
            showscale=True,  # This is crucial to show the colorbar
            colorbar=dict(
                title=dict(text="L1H13", font=dict(size=8), side="top"),
                thickness=5,
                titleside="top",
                tickvals=[0, 0.5, 1],  # Optional: specify tick values
                xpad=0,
                ypad=0,
                # x=1.08,
                y=0.3,
                len=0.4,
            ),
        ),
        showlegend=False,  # Hide this trace from the legend
    )
)


fig.write_image(LP_DIR / "new_plots" / "ablated_box_down_right_kernels_all_steps.pdf", scale=2)
if IS_NOTEBOOK:
    fig.show()

# %%

box_group_channels = get_group_channels("box", return_dict=True)
down_channels, right_channels = box_group_channels[1], box_group_channels[3]

# %%
apply_style(figsize=(2.7, 1.2), px_use_default=False, px_margin=dict(t=0, b=0, l=0, r=0))

tick = 9
baseline_down = get_avg(toy_cache, down_channels, tick)
baseline_right = get_avg(toy_cache, right_channels, tick)
baseline_plan = baseline_down + baseline_right

abl_down = get_avg(abl_cache, down_channels, tick)
abl_right = get_avg(abl_cache, right_channels, tick)
abl_plan = abl_down + abl_right

combined = np.stack([baseline_plan, abl_plan], axis=0)[None]

fig = plotly_feature_vis(combined, toy_obs[0:1].numpy(), ["Original", "Ablated"])
fig.for_each_annotation(lambda a: a.update(y=0.0))
# rewmove ticka
fig.update_xaxes(showticklabels=False, ticks="")
fig.update_yaxes(showticklabels=False, ticks="")

# fig.write_image(LP_DIR / "new_plots" / "ablated_box_down_right_kernels_final_step.pdf")
if IS_NOTEBOOK:
    fig.show()

# %%

fig, ax = plt.subplots(figsize=(4, 4))
plt.imshow(np.transpose(toy_obs[0], (1, 2, 0)))
plt.axis("off")
plt.savefig(LP_DIR / "new_plots" / "obs.svg")


box_group_channels = get_group_channels("box", return_dict=True, long_term=False)
# representative_box = [(0, 24), (1, 17), (1, 27), (0, 17)]
representative_box = [(0, 24), (1, 17), (1, 27), (1, 13)]
channel_dicts = [[get_channel_dict(layer, c)] for layer, c in representative_box]

# tick = 7
for tick in range(10):
    baseline_maps = [1.2 * get_avg(toy_cache, cd, tick, hook="h") for cd in channel_dicts]
    # baseline_maps = [get_avg(toy_obs, toy_cache, cd, tick) for cd in box_group_channels]

    baseline_maps = [baseline_maps[1], baseline_maps[3]]

    display_combined_heatmaps_white_zero(
        baseline_maps, fig_size=(4, 4), save_path=f"{LP_DIR}/new_plots/activations_tick{tick}.svg"
    )
    # display_combined_heatmaps_white_zero(baseline_maps, fig_size=(4, 4))
    # plt.show()
    # break
