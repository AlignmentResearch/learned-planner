"""Visualize LPE, TPE, and WTA kernels."""
# %%

import argparse

import numpy as np
import plotly.express as px  # Used for default colorscales if needed
import plotly.io as pio
from cleanba.environments import BoxobanConfig
from gym_sokoban.envs.render_utils import CHANGE_COORDINATES

from learned_planner import BOXOBAN_CACHE, IS_NOTEBOOK, LP_DIR
from learned_planner.interp.channel_group import get_channel_sign, get_group_channels, get_group_connections
from learned_planner.interp.offset_fns import OFFSET_VALUES_LAYER_WISE
from learned_planner.interp.plot import apply_style
from learned_planner.interp.utils import load_jax_model_to_torch
from learned_planner.policies import download_policy_from_huggingface

if not IS_NOTEBOOK:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="new_plots")
    args = parser.parse_args()
    output_path = args.output_path
else:
    output_path = LP_DIR / "new_plots"
pio.kaleido.scope.mathjax = None  # Disable MathJax to remove the loading message

if IS_NOTEBOOK:
    pio.renderers.default = "notebook"
else:
    pio.renderers.default = "png"  # Use static PNG renderer

MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000"  # DRC(3, 3) 2B checkpoint
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

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

ijfo_offsets = np.arange(0, 128, 32)
ijfo_str = "ijfo"


def get_kernel(il, ic, ol, oc, abs=False):
    """Get the kernel for a given input and output channel."""
    hh = il == ol
    inp_offset = 0 if hh else 32
    conv = getattr(model.features_extractor.cell_list[ol], "conv_hh" if hh else "conv_ih")
    kern = conv.weight[ijfo_offsets + oc, inp_offset + ic].detach().numpy()
    if abs:
        kern = np.abs(kern)
    return kern


def plot_kernel(il, ic, ol, oc, invert=False, plot_type="j", show=True, transform_sign=True):
    """Plot the kernel for a given input and output channel."
    Args:
        il (int): input layer
        ic (int): input channel
        ol (int): output layer
        oc (int): output channel
        invert (bool): if True, invert the sign of the kernel"
        plot_type (str): type of plot to show, can be "i", "j", "f", "o", "io", or "all""
        show (bool): if True, show the plot"
        transform_sign (bool): if True, multiply the kernel by the sign of the input and the (if available) output channels.
            Otherwise, the original kernel is used.
    """

    kernel = get_kernel(il, ic, ol, oc)
    inp_sign = get_channel_sign(il, ic)
    if plot_type in ijfo_str:
        kernel = kernel[ijfo_str.index(plot_type)]
        try:
            if transform_sign:
                kernel *= get_channel_sign(ol, oc, gate=plot_type)
        except ValueError:
            pass
    elif plot_type == "io":
        i_sign = get_channel_sign(ol, oc, gate="i") if transform_sign else 1
        o_sign = get_channel_sign(ol, oc, gate="o") if transform_sign else 1
        kernel = (i_sign * kernel[0]) + (o_sign * kernel[3])
        kernel = -kernel if invert else kernel
    elif plot_type == "all":
        fig = px.imshow(
            kernel, facet_col=0, facet_col_wrap=4, title=f"Kernel from ({il}, {ic}) to ({ol}, {oc})", zmax=1, zmin=-1
        )
        if show and IS_NOTEBOOK:
            fig.show()
        return
    else:
        raise ValueError(f"Unknown plot type {plot_type}")
    if transform_sign:
        kernel *= inp_sign
    if show and IS_NOTEBOOK:
        fig = px.imshow(
            kernel,
            title=f"Kernel from ({il}, {ic}) to ({ol}, {oc})",
            zmax=min(1, kernel.max()),
            zmin=max(-1, kernel.min()),
        )
        fig.show()
    return kernel


# %% Extract equivalent convolutional filter

apply_style(figsize=(2.05, 1.1), px_margin=dict(t=0, b=8, l=10, r=0), px_use_default=False, font=6)

group_type = "box"
# group_type = "agent"
box_group = get_group_channels(group_type)
box_connections = get_group_connections(box_group)

separate_forward_and_backward = True

linear_kernels = []
kernel_idxs = []
linear_extension_weight_distr = []
only_self = True
for dir_idx, dir_to_all_dir_grp in enumerate(box_connections):
    kernel = np.zeros((2, 4, 3, 3) if separate_forward_and_backward else (4, 3, 3))
    total = np.zeros((2, 4)) if separate_forward_and_backward else np.zeros(1)
    for group_dir in dir_to_all_dir_grp[dir_idx]:
        (il, ic), (ol, oc) = group_dir
        hh = il == ol
        if only_self and (il != ol or ic != oc):
            continue
        inp_offset = 0 if hh else 32
        conv = getattr(model.features_extractor.cell_list[ol], "conv_hh" if hh else "conv_ih")
        if separate_forward_and_backward:
            kern = np.abs(conv.weight[ijfo_offsets + oc, inp_offset + ic].detach().numpy())
            forward_sq = np.array([1, 1]) + np.array(CHANGE_COORDINATES[dir_idx])
            backward_sq = np.array([1, 1]) - np.array(CHANGE_COORDINATES[dir_idx])
            is_fwd = kern[:, *forward_sq] >= kern[:, *backward_sq]
            back_idx = np.where(~is_fwd)[0]
            fwd_idx = np.where(is_fwd)[0]
            kernel[0, back_idx] += kern[back_idx]
            kernel[1, fwd_idx] += kern[fwd_idx]
            total[0, back_idx] += 1
            total[1, fwd_idx] += 1

            linear_extension_weight_distr.append(kern[fwd_idx][:, *forward_sq])
            linear_extension_weight_distr.append(kern[back_idx][:, *backward_sq])

            kernel_idxs.append((dir_idx, (ol, oc), fwd_idx, back_idx))

        else:
            kernel += np.abs(conv.weight[ijfo_offsets + oc, inp_offset + ic].detach().numpy())
            total += 1

    total[total == 0] = 1  # avoid division by zero
    kernel /= total[..., None, None]
    linear_kernels.append(kernel)

linear_extension_weight_distr = np.concatenate(linear_extension_weight_distr, axis=0)

separate_ijfo = False
udlr_labels = ["Up", "Down", "Left", "Right"]
if separate_ijfo:
    linear_kernels = np.concatenate(linear_kernels, axis=0)

    fig = px.imshow(
        np.abs(linear_kernels),
        facet_col=0,
        facet_col_wrap=4,
    )
    fig.show()
else:
    linear_kernels = np.stack(linear_kernels, axis=0).sum(axis=1 + separate_forward_and_backward)
    if separate_forward_and_backward:
        linear_kernels = linear_kernels.transpose(1, 0, 2, 3).reshape(-1, 3, 3)
    col_labels = udlr_labels
    row_labels = ["Forward", "Backward"] if separate_forward_and_backward else [""]
    fig = px.imshow(
        np.abs(linear_kernels),
        facet_col=0,
        facet_col_wrap=4,
    )

    def update_text(d):
        idx = int(d["text"].split("=")[1])
        if separate_forward_and_backward:
            if idx < 4:
                return d.update(text="")
            else:
                return d.update(text=col_labels[idx - 4], y=d["y"] - 0.56)

    fig.for_each_annotation(update_text)

    # If there are multiple rows, add row labels on the left.
    if len(row_labels) > 1:
        num_rows = len(row_labels)
        for i, row in enumerate(row_labels):
            # Position the row label in normalized coordinates.
            y_val = (1.6 * (-i) + 1.8) / num_rows
            fig.add_annotation(
                dict(
                    text=row,
                    xref="paper",
                    yref="paper",
                    x=-0.07,  # adjust x as needed for spacing
                    y=y_val,
                    showarrow=False,
                    textangle=-90,
                )
            )

    fig.update_xaxes(showticklabels=False, ticks="", visible=False)
    fig.update_yaxes(showticklabels=False, ticks="", visible=False)
    # fig.update_traces(colorbar_thickness=5, selector=dict(type='heatmap'))
    fig.update_layout(
        coloraxis_colorbar=dict(
            thickness=5,
            len=0.9,
            xpad=1,
        ),
    )
    if IS_NOTEBOOK:
        fig.show()

fig.write_image(LP_DIR / args.output_path / f"plan_extension_kernels_{group_type}.svg")

# fig.write_image(LP_DIR / args.output_path / f"plan_extension_kernels_{group_type}.pdf")
# with open(LP_DIR / args.output_path / f"linear_kernels_{group_type}.csv", "w") as f:
#     f.write("direction, (ol, oc), fwd_ijfo_idx, back_ijfo_idx\n")
#     for kernel_idx in kernel_idxs:
#         f.write(f"{kernel_idx[0]}, {kernel_idx[1]}, {kernel_idx[2].tolist()}, {kernel_idx[3].tolist()}\n")
# %%


# %% box to agent copy
apply_style(figsize=(2.7, 0.7), px_margin=dict(l=5, r=0, t=0, b=5), px_use_default=False)
ijfo_str = "ijfo"

il, ic = 1, 17
ol, oc = 1, 18
kernels = [plot_kernel(il, ic, ol, oc, plot_type=c, show=False, transform_sign=False) for c in ijfo_str]
kernels = np.stack(kernels, axis=0)
fig = px.imshow(
    kernels,
    facet_col=0,
    facet_col_wrap=4,
)
fig.for_each_annotation(lambda a: a.update(text=ijfo_str[int(a["text"].split("=")[1])], y=a["y"] - 1.1))
fig.update_xaxes(showticklabels=False, ticks="", visible=False)
fig.update_yaxes(showticklabels=False, ticks="", visible=False)
fig.update_layout(
    coloraxis_colorbar=dict(
        thickness=12,
    ),
)
fig.write_image(LP_DIR / args.output_path / "box_to_agent_down_copy.pdf")
if IS_NOTEBOOK:
    fig.show()

# %% turn kernels at corner
apply_style(figsize=(2.05, 1.1), px_margin=dict(l=0, r=0, t=0, b=6), px_use_default=False, font=6)


box_group = get_group_channels("box_agent")
box_connections = get_group_connections(box_group)

separate_forward_and_backward = True

ijfo_offsets = np.arange(0, 128, 32)

# pio.templates.default = "plotly"
plot_types = ["i", "j", "f", "o"]
# plot_types = ["o"]
print("Plotting kernels for", plot_types)
box_group = get_group_channels("box")
box_connections = get_group_connections(box_group)

flip_dir = [1, 0, 3, 2]
udlr = ["Up", "Down", "Left", "Right"]
kernel_idxs = []
all_kernels = np.zeros((2, 4, 3, 3))
all_texts = np.zeros((2, 4), dtype=object)
layout = [(-1, -1), (-1, 1), (1, 1), (1, -1)]

clockwise = [0, 3, 1, 2]  # up, right, down, left

align_labels = True

for input_dir in range(4):
    for output_dir in range(4):
        if input_dir == output_dir or input_dir == flip_dir[output_dir]:
            continue

        total_offset = np.zeros(2, dtype=int) + CHANGE_COORDINATES[input_dir] - CHANGE_COORDINATES[output_dir]
        group_from_to_dir = box_connections[input_dir][output_dir]

        idx_dim1 = layout.index(tuple(total_offset))
        is_anti = clockwise[clockwise.index(input_dir) - 1] == output_dir
        if align_labels and is_anti:
            idx_dim1 += 2
            idx_dim1 %= 4
        kernel_sum = np.zeros((3, 3))
        total = 0
        for (il, ic), (ol, oc) in group_from_to_dir:
            input_offset = OFFSET_VALUES_LAYER_WISE[il][ic]
            output_offset = OFFSET_VALUES_LAYER_WISE[ol][oc]
            candidate_total_offset = input_offset - output_offset
            kernel = get_kernel(il, ic, ol, oc)
            if np.array_equal(candidate_total_offset, total_offset):
                found = True
                for plot_type in plot_types:
                    kernel = plot_kernel(il, ic, ol, oc, plot_type=plot_type, show=False)
                    kernel_sum += kernel
                    total += 1
                    kernel_idxs.append((input_dir, output_dir, il, ic, ol, oc))
        if total > 0:
            kernel_sum /= total
            all_kernels[int(is_anti), idx_dim1] = kernel_sum
            all_texts[int(is_anti), idx_dim1] = f"{udlr[input_dir]} to {udlr[output_dir]}"
            # fig = px.imshow(kernel_sum, title=f"Kernel from ({il}, {ic}) to ({ol}, {oc})", zmax=1, zmin=-1)
            # fig.show()
        else:
            print("Not found for", udlr[input_dir], "to", udlr[output_dir])

all_kernels = all_kernels[:, [2, 3, 0, 1]]  # permute
all_texts = all_texts[:, [2, 3, 0, 1]]
# plot the kernels in a 2x4 grid with a common color scale
all_kernels = all_kernels.reshape(-1, 3, 3)
all_texts = all_texts.reshape(-1)
fig = px.imshow(
    all_kernels,
    facet_col=0,
    facet_col_wrap=4,
    # zmax=min(1, all_kernels.max()),
    # zmin=max(-1, all_kernels.min()),
    color_continuous_scale="balance_r",
)
fig.update_layout(
    coloraxis_colorbar=dict(
        thickness=5,
        len=0.9,
        xpad=1,
    ),
    coloraxis_cmid=0,
)
fig.for_each_annotation(lambda a: a.update(text=all_texts[int(a["text"].split("=")[1])], y=a["y"] - 0.54))
fig.update_xaxes(showticklabels=False, ticks="", visible=False)
fig.update_yaxes(showticklabels=False, ticks="", visible=False)
if IS_NOTEBOOK:
    fig.show()
fig.write_image(LP_DIR / args.output_path / "turn_kernels.pdf")
fig.write_image(LP_DIR / args.output_path / "turn_kernels.svg")
# with open(LP_DIR / args.output_path / "turn_kernels.csv", "w") as f:
#     f.write("input_dir, output_dir, il, ic, ol, oc\n")
#     for kernel_idx in kernel_idxs:
#         f.write(", ".join([str(x) for x in kernel_idx]) + "\n")

# fig.write_image(LP_DIR / args.output_path / "turn_kernels.svg")

# %%
# il, ic =
# kernel = plot_kernel(il, ic, ol, oc, plot_type="all", show=True)


# %% kernels involved in winner takes all mechanism

apply_style(figsize=(1.35, 1.1), px_margin=dict(l=15, r=1, t=2, b=16), px_use_default=False, font=6)
short_term_box_group = get_group_channels("box", long_term=False)
total_channels = sum([len(g) for g in short_term_box_group])
print("Total channels", total_channels)
short_term_box_connections = get_group_connections(short_term_box_group)
connection_strength = [[0] * 4 for _ in range(4)]

# skip = [(0, 31)]
skip = []

for d1, g1 in enumerate(short_term_box_connections):
    for d2, g2 in enumerate(g1):
        for (il, ic), (ol, oc) in g2:
            # if (il, ic) == (ol, oc):
            #     continue
            if (il, ic) in skip or (ol, oc) in skip:
                continue

            kernel = get_kernel(il, ic, ol, oc)
            input_sign = get_channel_sign(il, ic)
            j_kernel = kernel[1] * input_sign
            connection_strength[d1][d2] += np.sum(j_kernel)
            f_kernel = kernel[2] * input_sign
            connection_strength[d1][d2] += np.sum(f_kernel)
            i_kernel = kernel[0] * input_sign * get_channel_sign(ol, oc, gate="i")
            o_kernel = kernel[3] * input_sign * get_channel_sign(ol, oc, gate="o")
            connection_strength[d1][d2] += np.sum(i_kernel)
            connection_strength[d1][d2] += np.sum(o_kernel)
        connection_strength[d1][d2] /= len(g2) * 4

labels = ["Up", "Down", "Left", "Right"]

fig = px.imshow(
    connection_strength,
    color_continuous_scale="RdBu",
    labels=dict(x="Input direction", y="Output direction"),
    zmax=0.5,
    zmin=-0.5,
)
fig.update_layout(
    coloraxis_colorbar=dict(thickness=5, len=0.9, xpad=1),
)
fig.update_xaxes(showticklabels=True, tickvals=np.arange(len(labels)), ticktext=labels, ticks="")
fig.update_yaxes(showticklabels=True, tickvals=np.arange(len(labels)), ticktext=labels, ticks="", tickangle=-90)
if IS_NOTEBOOK:
    fig.show()
fig.write_image(LP_DIR / args.output_path / "winner_takes_all.pdf")
# fig.write_image(LP_DIR / args.output_path / "winner_takes_all.svg")
