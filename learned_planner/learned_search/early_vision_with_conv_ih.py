"""Combined encoder kernel visualization."""

# %%
import concurrent.futures
from pathlib import Path
from typing import List, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io
import plotly.io as pio
import torch as th
import torch.utils.data
import tqdm
from cleanba.environments import BoxobanConfig
from sklearn.cluster import KMeans
from sklearn.linear_model import OrthogonalMatchingPursuit

from learned_planner import BOXOBAN_CACHE, LP_DIR, ON_CLUSTER
from learned_planner.interp.collect_dataset import DatasetStore  # noqa
from learned_planner.interp.plot import apply_style, plotly_feature_vis
from learned_planner.interp.utils import load_jax_model_to_torch
from learned_planner.notebooks.emacs_plotly_render import set_plotly_renderer
from learned_planner.policies import download_policy_from_huggingface

pio.kaleido.scope.mathjax = None  # Disable MathJax to remove the loading message

set_plotly_renderer("emacs")
pio.renderers.default = "notebook"

# %% Load model
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


# %% Extract equivalent convolutional filter


def extract_feature(obs, layer=0):
    with th.no_grad():
        out = model.vf_features_extractor.pre_model(th.as_tensor(obs))
        out = th.cat([out, th.zeros_like(out), th.zeros_like(out)], dim=1)
        out = model.features_extractor.cell_list[layer].conv_ih(out)
    return out


def get_filters(v: float, layer: int = 0) -> np.ndarray:
    # We use a batch size (=3) equal to the input channel dimension so we can extract all of the components in one go.
    # We're basically multiplying each spatial slice of the convolutional filters by the identity matrix.

    # We use 31 here, as a number large enough to be larger than the combined filter for sure. (The combined filter is
    # 7x7, I just wasn't sure of that yet.)
    collapse_rgb = th.zeros(3, 3, 31, 31)

    # Construct identity matrix multiplied by v (in only the center location)
    collapse_rgb[th.arange(3), th.arange(3), 15, 15] = v
    out = extract_feature(collapse_rgb, layer).detach().cpu().numpy()
    out = np.moveaxis(out, 0, -1)
    return out


# ih_layer = 0
convs = []
biases = []
for ih_layer in [0, 1, 2]:
    # Get bias and conv components
    bias = get_filters(0, ih_layer)
    conv = get_filters(1, ih_layer) - bias
    check = get_filters(2, ih_layer) - bias
    assert np.allclose(conv * 2, check, atol=1e-4)

    # Things found via trial and error:
    #
    # - Values of 11:18
    # - Invert the spatial dimensions because it turns out you get an inverted filter from the procedure above. That is, if
    #   we want the convolution result to be the same after putting this in the filter, we need to invert it.
    conv = conv[:, 10:19, 10:19, :][:, ::-1, ::-1, :].copy()
    bias = bias[:, 5, 5, 1]  # Take arbitrary non-boundary indices for per-channel bias
    convs.append(conv)
    biases.append(bias)

# %%
colors = np.array(
    [
        [0, 0, 0],  # WALL
        [243, 248, 238],  # EMPTY
        [254, 126, 125],  # TARGET
        [254, 95, 56],  # BOX_ON_TARGET
        [142, 121, 56],  # BOX
        [160, 212, 56],  # PLAYER
        [219, 212, 56],  # PLAYER_ON_TARGET
    ],
    dtype=np.float32,
)
color_labels = ["WALL", "EMPTY", "TARGET", "BOX_ON_TARGET", "BOX", "PLAYER", "PLAYER_ON_TARGET"]

colors_normed = colors / 255.0

# %% K-means visualization of raw filters
n_clusters = 10
pixel_dataset = np.reshape(conv, (-1, 3))
pos_pixel_dataset = np.maximum(0, pixel_dataset)
neg_pixel_dataset = -np.minimum(0, pixel_dataset)
pixel_dataset = np.concatenate([pos_pixel_dataset, neg_pixel_dataset], axis=0)
pixel_dataset = pixel_dataset[np.linalg.norm(pixel_dataset, axis=-1) > 1e-5]
pixel_dataset = pixel_dataset / np.linalg.norm(pixel_dataset, axis=-1, keepdims=True)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(pixel_dataset)
cluster_centers = kmeans.cluster_centers_

# Define Sokoban colors for interpretation
# Plot color relationships
result = colors_normed @ cluster_centers.T

# distances = np.linalg.norm(colors_normed[:, None] - cluster_centers[None], axis=-1)
# result = distances
# result = kmeans.transform(colors_normed)

# fig = px.imshow(result)
# fig.update_yaxes(ticktext=color_labels, tickvals=list(range(len(color_labels))))
# fig.update_xaxes(ticktext=[f"Cluster {i}" for i in range(n_clusters)], tickvals=list(range(n_clusters)))
# fig.show()

# %% Best-fit of conv with colors as basis

basis = colors[1:]
# basis = basis - np.mean(basis, axis=0, keepdims=True)
# basis = basis / np.linalg.norm(basis, axis=-1, keepdims=True)

pixel_dataset = np.reshape(conv, (-1, 3))
# pixel_dataset = pixel_dataset - np.mean(pixel_dataset, axis=0, keepdims=True)
# pixel_dataset = pixel_dataset / np.linalg.norm(pixel_dataset, axis=-1, keepdims=True)

reg = OrthogonalMatchingPursuit(n_nonzero_coefs=2, fit_intercept=False).fit(basis.T, pixel_dataset.T)
coeffs = reg.coef_
intercept = reg.intercept_

residual = pixel_dataset - coeffs @ basis
# residual -= intercept[:, None]
print(f"mean residual: {np.mean(np.linalg.norm(residual, axis=-1)):.8f}")
print(
    f"baseline residual: {np.mean(np.linalg.norm(pixel_dataset - np.mean(pixel_dataset, axis=0, keepdims=True), axis=-1)):.8f}"
)

coeffs = coeffs.reshape(*conv.shape[:-1], coeffs.shape[-1])
# get idx in basis of highest coeffs
assert np.all(np.sum(np.abs(coeffs) > 0.0, axis=-1) >= 1)
separate_by_magn = False
if separate_by_magn:
    idx = np.argsort(np.abs(coeffs), axis=-1)
    idx = idx[..., -2:]
else:
    # separate by sign
    pos_idxs = np.argmax(coeffs > 0.0, axis=-1)
    neg_idxs = np.argmax(coeffs < 0.0, axis=-1)
    idx = np.stack([neg_idxs, pos_idxs], axis=-1)

# index into basis
first_color = basis[idx[..., -1]]
first_color_coef = np.take_along_axis(coeffs, idx[..., -1, None], axis=-1)
first_color = first_color_coef * first_color
second_idx = idx[..., -2]
second_color = basis[idx[..., -2]]
# zero out if second coeff is zero
second_color = np.where(np.abs(np.take_along_axis(coeffs, second_idx[..., None], axis=-1)) > 0.0, second_color, 0.0)
second_color_coef = np.take_along_axis(coeffs, second_idx[..., None], axis=-1)
# assert np.all(second_color_coef > 0.0), "second color should always be positive"
# second_color_rel_coef = second_color_coef / first_color_coef
second_color = second_color_coef * second_color

# first_color_norm = np.abs(first_color).max(axis=tuple(range(1, len(first_color.shape))), keepdims=True)
# second_color_norm = np.abs(second_color).max(axis=tuple(range(1, len(second_color.shape))), keepdims=True)
max_norm = np.maximum(np.abs(first_color), np.abs(second_color)).max(
    axis=tuple(range(1, len(first_color.shape))), keepdims=True
)

first_color = first_color * (255 / max_norm)
second_color = second_color * (255 / max_norm)

# first_color = np.abs(first_color)
# second_color = np.abs(second_color)
# %%

# %% Visualize cluster centers

posneg_clusters = np.stack([np.maximum(0, cluster_centers), -np.minimum(0, cluster_centers)], axis=0)
fig = px.imshow(posneg_clusters / posneg_clusters.max())
fig.update_xaxes(ticktext=[f"Cluster {i}" for i in range(n_clusters)], tickvals=list(range(n_clusters)))
fig.update_yaxes(ticktext=["+", "-"], tickvals=[0, 1])
fig.show()

# %% Visualize conv, positive and negative


ijfo_list = ["i", "j", "f", "o"]


def plot_pos_and_neg_og(thing, ijfo="o", save=False):
    apply_style(figsize=(5.4, 5.4), px_margin=dict(t=15, b=10, l=10, r=10), px_use_default=not save)
    ijfo_idx = ijfo_list.index(ijfo)
    ijfo_offset = ijfo_idx * 32
    thing = thing[ijfo_offset : ijfo_offset + 32]
    neg = -np.minimum(0.0, thing)
    pos = np.maximum(0.0, thing)
    norm = np.abs(thing).max(axis=tuple(range(1, len(thing.shape))), keepdims=True)[..., None]
    out = (np.stack([pos, neg], axis=1) * (255 / norm)).reshape((-1, *thing.shape[1:]))
    fig = px.imshow(out, facet_col=0, facet_col_wrap=8, facet_row_spacing=0.02)

    prefix = f"L{ih_layer}{ijfo.upper()}"
    labels = sum(([f"+{prefix}{i}", f"-{prefix}{i}"] for i in range(32)), start=[])
    fig.for_each_annotation(lambda a: a.update(text=labels[int(a.text.split("=")[-1])], yshift=-2))
    fig.update_xaxes(showticklabels=False, ticks="")
    fig.update_yaxes(showticklabels=False, ticks="")
    # fig.update_layout(margin=dict(l=15, r=15, t=15, b=15))
    if save:
        fig.write_image(LP_DIR / f"new_plots/{prefix}_og_conv_filters.pdf")
        fig.write_image(LP_DIR / f"new_plots/{prefix}_og_conv_filters.png")
    else:
        fig.update_layout(height=1050)
    return fig


fig = plot_pos_and_neg_og(conv, "o")
height = 1000

fig.show()


def plot_pos_and_neg(thing, ijfo="o", save=False):
    apply_style(figsize=(5.4, 5.4), px_margin=dict(t=15, b=10, l=10, r=10), px_use_default=not save)
    ijfo_idx = ijfo_list.index(ijfo)
    ijfo_offset = ijfo_idx * 32
    thing = thing[ijfo_offset : ijfo_offset + 32]
    pos = first_color[ijfo_offset : ijfo_offset + 32]
    neg = second_color[ijfo_offset : ijfo_offset + 32]
    # print(pos.shape)
    # pos_norm = np.abs(pos).max(axis=tuple(range(1, len(pos.shape))), keepdims=True)
    # neg_norm = np.abs(neg).max(axis=tuple(range(1, len(neg.shape))), keepdims=True)
    # print(pos_norm.shape)
    # pos = pos * (255 / pos_norm)
    # print(pos.shape)
    # neg = neg * (255 / neg_norm) * (second_color_rel_coef[channel_slice])
    out = (np.stack([pos, neg], axis=1)).reshape((-1, *thing.shape[1:]))
    fig = px.imshow(np.abs(out), facet_col=0, facet_col_wrap=8, facet_row_spacing=0.02)

    # fig.update_layout(height=1000)
    prefix = f"L{ih_layer}{ijfo.upper()}"
    labels = sum(([f"+{prefix}{i}", f"-{prefix}{i}"] for i in range(32)), start=[])
    fig.for_each_annotation(lambda a: a.update(text=labels[int(a.text.split("=")[-1])]))
    fig.update_xaxes(showticklabels=False, ticks="")
    fig.update_yaxes(showticklabels=False, ticks="")
    fig.update_layout(margin=dict(l=15, r=15, t=15, b=15))
    # update fig such that the gap between alternating columns is 0.001
    if save:
        fig.write_image(LP_DIR / f"new_plots/{prefix}_basis_conv_filters.pdf")
        fig.write_image(LP_DIR / f"new_plots/{prefix}_basis_conv_filters.png")
    else:
        fig.update_layout(height=1050)
    return fig


# Usage example:
# plot_pos_and_neg_manual(conv, slice(96, None)).show()
# plot_pos_and_neg(conv, "i").show()
# plot_pos_and_neg(conv, "j").show()
# plot_pos_and_neg(conv, "f").show()
# fig = plot_pos_and_neg(conv, "j")

height = 1000

fig.show()


# %%

# %% plot L0 UDLR kernels
apply_style(figsize=(5.4, 0.9), px_margin=None, font=10)

udlr_layers = np.array([0, 0, 0, 0])
udlr_channels = np.array([24, 20, 23, 17])
# assert ih_layer == 0


def plot_udlr(thing, udlr_channels, inp_type="o", save=False):
    ijfo_offset = ["in", "j", "f", "o"].index(inp_type)
    ijfo_offset *= 32
    offset_udlr_channels = udlr_channels + ijfo_offset
    thing = thing[offset_udlr_channels]

    pos = first_color[offset_udlr_channels]
    neg = second_color[offset_udlr_channels]

    out = (np.stack([pos, neg], axis=1)).reshape((-1, *thing.shape[1:]))
    fig = px.imshow(np.abs(out), facet_col=0, facet_col_wrap=8)

    # fig.update_layout(height=1000)
    udlr_labels = ["up", "down", "left", "right"]
    labels = sum(
        (
            [f"+L{l}{inp_type.upper()}{i} ({udlr_lab})", f"-L{l}{inp_type.upper()}{i} ({udlr_lab})"]
            for l, i, udlr_lab in zip(udlr_layers, udlr_channels, udlr_labels)
        ),
        start=[],
    )
    fig.for_each_annotation(
        lambda a: a.update(
            text=labels[int(a.text.split("=")[-1])],
            # borderpad=0,
            yshift=-10,  # Adjust this value as needed
        )
    )
    fig.update_xaxes(showticklabels=False, ticks="")
    fig.update_yaxes(showticklabels=False, ticks="")
    fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=5),
        # If you're using subplots, adjust spacing:
        # grid=dict(
        #     rows=1,
        #     columns=2,
        #     roworder="bottom to top",
        #     xgap=0.05,
        #     ygap=0.05,
        # )
    )
    if save:
        fig.write_image(LP_DIR / "new_plots/udlr_filters.pdf")
    return fig


def plot_udlr_og(things, udlr_layers, udlr_channels, inp_type="o", save=False):
    apply_style(figsize=(5.4, 0.9), px_margin=dict(l=5, r=5, t=5, b=5), font=9, px_use_default=not save)
    ijfo_offset = ["i", "j", "f", "o"].index(inp_type)
    ijfo_offset *= 32
    offset_udlr_channels = udlr_channels + ijfo_offset
    thing = np.stack([things[l][c] for l, c in zip(udlr_layers, offset_udlr_channels)])
    # thing = thing[offset_udlr_channels]
    thing *= np.array([1, -1, 1, 1])[:, None, None, None]

    neg = -np.minimum(0.0, thing)
    pos = np.maximum(0.0, thing)
    norm = np.abs(thing).max(axis=tuple(range(1, len(thing.shape))), keepdims=True)[..., None]
    out = (np.stack([pos, neg], axis=1) * (255 / norm)).reshape((-1, *thing.shape[1:]))
    fig = px.imshow(out, facet_col=0, facet_col_wrap=8)
    # fig.update_layout(height=1000)
    udlr_labels = ["up", "down", "left", "right"]
    labels = sum(
        (
            [f"+L{l}{inp_type.upper()}{i} ({udlr_lab})", f"-L{l}{inp_type.upper()}{i} ({udlr_lab})"]
            for l, i, udlr_lab in zip(udlr_layers, udlr_channels, udlr_labels)
        ),
        start=[],
    )
    fig.for_each_annotation(
        lambda a: a.update(
            text=labels[int(a.text.split("=")[-1])],
            yshift=-80,
        )
    )
    fig.update_xaxes(showticklabels=False, ticks="")
    fig.update_yaxes(showticklabels=False, ticks="")
    fig.update_layout(
        margin=dict(l=5, r=5, t=5, b=5),
        # If you're using subplots, adjust spacing:
        # grid=dict(
        #     rows=1,
        #     columns=2,
        #     roworder="bottom to top",
        #     xgap=0.05,
        #     ygap=0.05,
        # )
    )
    if save:
        fig.write_image(LP_DIR / "new_plots/udlr_filters_og.pdf")

    return fig


plot_udlr_og(convs, udlr_layers, udlr_channels, inp_type="o", save=True).show()


# %%

convs_33 = model.features_extractor.pre_model[0].weight.detach().clone().cpu().moveaxis(1, -1).numpy()
plot_pos_and_neg(convs_33)

# %% 4. Compute eigen-convolutional filters (colors are datapoints)

conv_flat = np.reshape(conv, (conv.shape[0], -1))
conv_mean = np.zeros(())
conv_zmean = conv_flat - conv_mean
conv_cov = (conv_zmean @ conv_zmean.T) / conv_zmean.shape[1]
vals, vecs = np.linalg.eigh(conv_cov)
assert np.allclose((vecs * vals) @ vecs.T, conv_cov)

# Find eigenconvolutions from data instead (did not work as well)
# data_mean = np.mean(hook_h, axis=(0, 2, 3), keepdims=True)
# data_zmean = hook_h - data_mean
# data_cov = np.einsum("nchw,ndhw->cd", data_zmean, data_zmean) / (np.size(data_zmean) / data_zmean.shape[1])
# vals, vecs = np.linalg.eigh(data_cov)

conv_eig = np.einsum("cd,ca->da", vecs, conv_flat)
conv_eig = np.reshape(conv_eig, conv.shape)
plot_pos_and_neg(conv_eig[96:]).show()

# %% Load activations and obs of hard levels solved

# cache_path = Path("/training/activations_dataset/hard/0_think_step")
if ON_CLUSTER:
    cache_path = Path("/training/activations_dataset/hard/0_think_step")
    N_FILES = 1001
else:
    cache_path = LP_DIR / "drc33_cache/0_think_step"
    N_FILES = 500

with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:

    def map_fn(i):
        data = pd.read_pickle(cache_path / f"idx_{i}.pkl")
        return {
            "obs": data.obs,
            "encoded": data.model_cache["features_extractor.hook_pre_model"],
            "actions": data.pred_actions,
        }

    loaded_data = list(tqdm.tqdm(executor.map(map_fn, range(N_FILES)), total=N_FILES))

# %%

hook_h = np.concatenate([d["encoded"] for d in loaded_data])
obs = np.concatenate([d["obs"] for d in loaded_data])
obs_which_level = np.concatenate([np.ones(len(d["obs"]), dtype=np.int32) * i for i, d in enumerate(loaded_data)])

# %% Test expanded version of the NN convolutions
pad_shape = (3, 5, 3, 5)
new_features = th.nn.Sequential(
    th.nn.ZeroPad2d(pad_shape),
    th.nn.Conv2d(3, 128, 9, 1, padding=0),
)
with th.no_grad():
    new_features[1].weight.copy_(th.as_tensor(np.moveaxis(conv, -1, 1)))
    new_features[1].bias.copy_(th.as_tensor(bias))
    # original = model.vf_features_extractor.pre_model(th.as_tensor(obs[:3]) / 255).numpy()
    original = extract_feature(th.as_tensor(obs[:3]) / 255, ih_layer).numpy()
    new = new_features(th.as_tensor(obs[:3]) / 255).numpy()
    assert np.allclose(original[:, :, 2:-3, 2:-3], new[:, :, 2:-3, 2:-3], atol=1e-3)

# px.imshow((original - new), facet_col=0, facet_col_wrap=4, animation_frame=1)
conv_artifacts = (original - new).mean(0)
with th.no_grad():
    # original = model.vf_features_extractor.pre_model(th.as_tensor(obs[5:8]) / 255).numpy()
    original = extract_feature(th.as_tensor(obs[5:8]) / 255, ih_layer).numpy()
    new = new_features(th.as_tensor(obs[5:8]) / 255).numpy()
    # assert np.allclose(original, new + conv_artifacts, atol=1e-3)
    assert np.allclose(original[:, :, 2:-3, 2:-3], new[:, :, 2:-3, 2:-3], atol=1e-3)

# fig = px.imshow(conv_artifacts[96:], facet_col=0, facet_col_wrap=8, facet_col_spacing=0.001, facet_row_spacing=0.002)
# fig.update_layout({"height": 820})
# fig

# %% Find datapoints that excite each eigenconv


pad_obs = th.nn.functional.pad(th.as_tensor(obs / 255.0, dtype=th.float32), pad_shape)
# conv_eig_obs = th.nn.functional.conv2d(
#     pad_obs, th.as_tensor(conv_eig).moveaxis(-1, 1), stride=1, padding=0, bias=th.as_tensor(bias @ vecs)
# ).detach()

conv_to_use = conv
bias_to_use = bias

conv_eig_obs = th.nn.functional.conv2d(
    pad_obs,
    th.as_tensor(conv_to_use).moveaxis(-1, 1),
    stride=1,
    padding=0,
    bias=th.tensor(bias_to_use),
).detach()


# eigened_hook_h = np.einsum("nchw,cd->ndhw", hook_h[:10], vecs)
# assert np.allclose(eigened_hook_h[:, :, 1:-2, 1:-2], conv_eig_obs[:, :, 1:-2, 1:-2], atol=1e-3)

# %%

all_plot = []
for channel_index in range(96, 128):
    channel_to_check = conv_eig_obs[:, channel_index, :, :]

    # Find indices of top 100 activations
    top_k_flat = th.topk(channel_to_check.abs().flatten(), k=100)
    top_20_indices = np.unravel_index(top_k_flat.indices.numpy(), channel_to_check.shape)
    _, top_20_levels_unique_indices = np.unique(obs_which_level[top_20_indices[0]], return_index=True)
    top_20_indices = tuple(top_20_indices[i][top_20_levels_unique_indices] for i in range(3))

    padded = th.nn.functional.pad(th.as_tensor(obs / 255.0)[top_20_indices[0]], (8, 8, 8, 8)).moveaxis(1, -1)
    # padded_activations = th.nn.functional.pad(th.as_tensor(conv_eig_obs)[top_20_indices[0]], (8, 8, 8, 8)).moveaxis(1, -1)
    patches = []
    for i, (x, y) in enumerate(zip(top_20_indices[1], top_20_indices[2])):
        patches.append(padded[i, 8 + (x - 4) + 1 : 8 + x + 4 + 2, 8 + (y - 4) + 1 : 8 + y + 4 + 2, :])
        assert th.all((obs / 255.0)[top_20_indices[0][i], :, x, y] == patches[-1][3, 3])
        # patches.append(padded_activations[i, 8 + (x - 3) : 8 + x + 3 + 1, 8 + (y - 3) : 8 + y + 3 + 1, channel_index])

    patches = np.stack(patches)

    # Create figure showing the top 20 activations
    # thing = conv_to_use[None, channel_index, ..., :]
    # norm_conv_eig = np.abs(thing).max()
    # pos = np.maximum(0, thing) / norm_conv_eig
    # neg = np.maximum(0, -thing) / norm_conv_eig

    first_dom = first_color[None, channel_index] / 255.0
    second_dom = second_color[None, channel_index] / 255.0

    pos = np.zeros_like(first_dom)
    neg = np.zeros_like(first_dom)
    extra_pos = np.zeros_like(first_dom)
    extra_neg = np.zeros_like(first_dom)

    pos[first_dom > 0] = first_dom[first_dom > 0]
    pos[(first_dom < 0) & (second_dom > 0)] = second_dom[(first_dom < 0) & (second_dom > 0)]

    # # neg is negatives
    neg[first_dom < 0] = first_dom[first_dom < 0]
    neg[(first_dom > 0) & (second_dom < 0)] = second_dom[(first_dom > 0) & (second_dom < 0)]
    neg = np.abs(neg)

    # # # extra_pos is extra positives
    # extra_pos[(first_dom > 0) & (second_dom > 0)] = second_dom[(first_dom > 0) & (second_dom > 0)]

    # # # extra_neg is extra negatives
    # extra_neg[(first_dom < 0) & (second_dom < 0)] = second_dom[(first_dom < 0) & (second_dom < 0)]
    # extra_neg = np.abs(extra_neg)

    # to_plot = np.concatenate([pos, neg, extra_pos, extra_neg, patches[:11]], axis=0)
    to_plot = np.concatenate([pos, neg, patches[:13]], axis=0)
    if len(to_plot) < 15:
        to_plot = np.concatenate([to_plot] + [np.zeros_like(to_plot[:1])] * (15 - len(to_plot)))

    all_plot.append(to_plot)


fig = px.imshow(
    np.stack(all_plot),
    animation_frame=0,
    facet_col=1,
    facet_col_wrap=5,
    title=f"Top {len(all_plot)} from different levels",
)
fig.update_layout(height=1000)
fig.show()

# %% Evaluate predictions for each amount of removed eigenvectors


class LevelDataset(torch.utils.data.Dataset):
    def __init__(self, loaded_data: Sequence[Mapping[str, np.ndarray]]):
        super().__init__()
        # Sort from longest to shortest to minimize wasted space
        self.loaded_data = list(sorted(loaded_data, key=lambda x: -len(x["obs"])))

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        return dict(
            obs=th.as_tensor(data["obs"]),
            actions=th.as_tensor(data["actions"]),
            level_idx=idx,
        )


def collate_level_data(batch):
    max_len = max(len(d["obs"]) for d in batch)
    obs = th.zeros((max_len, len(batch), *batch[0]["obs"].shape[1:]), dtype=th.float32)
    actions = th.zeros((max_len, len(batch), 3), dtype=th.long).sub_(1)
    mask = th.zeros((max_len, len(batch)), dtype=th.bool)
    for i, item in enumerate(batch):
        obs[: len(item["obs"]), i, ...] = item["obs"]
        mask[: len(item["obs"]), i, ...] = True
        actions[: len(item["obs"]), i, ...] = item["actions"].view((-1, 3))
    return dict(obs=obs, mask=mask, actions=actions)


# Create dataset and dataloader
level_dataset = LevelDataset(loaded_data)


def logits_for(model, level_dataset, batch_size=32, num_workers: int = 0):
    level_loader = torch.utils.data.DataLoader(
        level_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_level_data, num_workers=num_workers
    )

    @th.no_grad()
    def map_fn(batch):
        observations = batch["obs"].cuda()
        catdist, _ = model.get_distribution(
            observations,
            model.recurrent_initial_state(observations.shape[1], device=observations.device),
            th.zeros(observations.shape[:2], dtype=th.bool, device=observations.device),
        )
        return catdist.distribution.logits.detach().cpu()

    return list(map(map_fn, level_loader))


model.cuda()
# Get baseline predictions
baseline_logits = logits_for(model, level_dataset, batch_size=128, num_workers=2)
baseline_logprobs = [th.nn.functional.log_softmax(logit, dim=-1) for logit in baseline_logits]

# %%
level_loader = torch.utils.data.DataLoader(
    level_dataset, batch_size=baseline_logprobs[0].shape[1], shuffle=False, collate_fn=collate_level_data, num_workers=2
)
level_acc = []
for baseline, batch in zip(baseline_logprobs, level_loader):
    mask = batch["mask"]
    actions = batch["actions"]
    correct = baseline.argmax(-1) == actions[..., 2]
    level_acc.append((correct * mask).sum(0) / mask.sum(0))
accuracy_level = th.cat(level_acc).mean()


print(f"Average proportion of level-actions which are equal: {accuracy_level * 100:.4f}%")


# %%

loaded_data = []

model = model.cuda()
conv_artifacts = th.as_tensor(conv_artifacts, device=model.device)
batch_size = baseline_logprobs[0].shape[1]
# Test different numbers of preserved eigenvectors
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor, tqdm.tqdm(total=32 * 8) as pbar:

    def layers_until_fn(n_clusters: List[int]) -> dict:  # pyright: ignore[reportRedeclaration]
        eigenvectors = th.as_tensor(vecs[:, np.array(n_clusters)], dtype=th.float32, device=model.device)
        to_sub = th.as_tensor(conv_artifacts.cpu() + bias[:, None, None]).to(eigenvectors)

        def ablation_hook(x, hook):
            # Project to eigenspace and back, keeping only top i components
            projected = th.einsum("tnchw,cd->tndhw", x - to_sub, eigenvectors)
            preserved = th.einsum("tndhw,cd->tnchw", projected, eigenvectors) + to_sub
            return preserved

        model.did_setup_input_dependent_hooks = True  # type: ignore
        with th.no_grad(), model.hooks(fwd_hooks=[("features_extractor.hook_pre_model", ablation_hook)]):
            # Run model with ablation hook
            ablated_logits = logits_for(model, level_dataset, batch_size=batch_size, num_workers=2)

        ablated_logprobs = [th.nn.functional.log_softmax(logit, dim=-1) for logit in ablated_logits]

        level_loader = torch.utils.data.DataLoader(
            level_dataset, batch_size=ablated_logprobs[0].shape[1], shuffle=False, collate_fn=collate_level_data, num_workers=2
        )

        def map_fn(args):
            baseline, ablated, batch = args
            mask = batch["mask"]
            correct = baseline.argmax(-1) == ablated.argmax(-1)
            level_acc = (correct * mask).sum(0) / mask.sum(0)
            level_kl = (mask * th.nn.functional.kl_div(ablated, baseline, reduction="none", log_target=True).sum(-1)).sum(
                0
            ) / mask.sum(0)
            pbar.update(1)
            return level_acc, level_kl

        level_acc, level_kl = zip(*list(executor.map(map_fn, zip(baseline_logprobs, ablated_logprobs, level_loader))))

        level_acc = th.cat(level_acc)
        level_kl = th.cat(level_kl)
        pbar.update(1)
        return dict(removed_eigenvectors=n_clusters, accuracy=level_acc.mean(), kl_divergence=level_kl.mean())

    # steps = range(31, -1, -1)
    steps: List[List[int]] = [
        [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
        [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    ]
    loaded_data = list(map(layers_until_fn, steps))


# Convert results to DataFrame for plotting
results_df = pd.DataFrame(loaded_data)

# Create plot
fig = px.line(
    results_df,
    x="removed_eigenvectors",
    y=["accuracy", "kl_divergence"],
    title="Model Performance vs Number of Removed Eigenvectors",
)
fig.update_layout(xaxis_title="Number of Removed Eigenvectors", yaxis_title="Metric Value", hovermode="x unified")
fig.show()


# %% Degradation with kmeans


loaded_data = []

batch_size = baseline_logprobs[0].shape[1]
model_cfg, model = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)
model.features_extractor.pre_model = th.nn.Sequential(
    th.nn.ZeroPad2d((2, 4, 2, 4)),
    th.nn.Conv2d(3, 32, 7, 1, padding=0),
)
model.cuda()


# with th.no_grad():
#     conv0 = th.as_tensor(model.features_extractor.pre_model[0].weight).cpu()
#     # pixels0 = conv0.moveaxis(1, -1).clone().reshape(-1, 3)
#     conv2 = th.as_tensor(model.features_extractor.pre_model[2].weight).cpu()
#     # pixels2 = conv2.moveaxis(0, -1).clone().reshape(-1, 32)
#     pixels = th.cat([conv0.view(-1), conv2.view(-1)]).contiguous()

one_steps = [1000, 500, 200, 100, 50, 20, 10, 5, 2][:1]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor, tqdm.tqdm(total=len(steps) * 8) as pbar:
    # pixel_dataset = np.reshape(conv, (-1, 3))
    # pixel_dataset = pixels.detach().cpu().numpy()

    def layers_until_fn(n_clusters: int) -> dict:  # pyright: ignore[reportRedeclaration]
        # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        # _ = kmeans.fit_predict(pixel_dataset[:, None])
        # for layer_i in [0, 2]:
        #     with th.no_grad():
        #         param = model.features_extractor.pre_model[layer_i].weight
        #         arr = param.detach().cpu().numpy().ravel()
        #         new_conv = kmeans.cluster_centers_[kmeans.predict(arr[:, None])].reshape(param.shape)
        #         param.copy_(th.as_tensor(new_conv))

        with th.no_grad():
            model.features_extractor.pre_model[1].weight.copy_(th.as_tensor(conv).moveaxis(-1, 1))
            model.features_extractor.pre_model[1].bias.copy_(th.as_tensor(bias))

        ablated_logits = logits_for(model, level_dataset, batch_size=batch_size, num_workers=2)

        ablated_logprobs = [th.nn.functional.log_softmax(logit, dim=-1) for logit in ablated_logits]

        level_loader = torch.utils.data.DataLoader(
            level_dataset, batch_size=ablated_logprobs[0].shape[1], shuffle=False, collate_fn=collate_level_data, num_workers=2
        )

        def map_fn(args):
            baseline, ablated, batch = args
            mask = batch["mask"]
            correct = baseline.argmax(-1) == ablated.argmax(-1)
            level_acc = (correct * mask).sum(0) / mask.sum(0)
            level_kl = (mask * th.nn.functional.kl_div(ablated, baseline, reduction="none", log_target=True).sum(-1)).sum(
                0
            ) / mask.sum(0)
            pbar.update(1)
            return level_acc, level_kl

        level_acc, level_kl = zip(*list(executor.map(map_fn, zip(baseline_logprobs, ablated_logprobs, level_loader))))

        level_acc = th.cat(level_acc)
        level_kl = th.cat(level_kl)
        pbar.update(1)
        return dict(preserved_eigenvectors=n_clusters, accuracy=level_acc.mean(), kl_divergence=level_kl.mean())

    loaded_data = list(map(layers_until_fn, one_steps))


# Convert results to DataFrame for plotting
results_df = pd.DataFrame(loaded_data)

# Create plot
fig = px.line(
    results_df,
    x="preserved_eigenvectors",
    y=["accuracy", "kl_divergence"],
    title="Model Performance vs Number of Preserved Eigenvectors",
)
fig.update_layout(xaxis_title="Number of Preserved Eigenvectors", yaxis_title="Metric Value", hovermode="x unified")
fig.show()


# %% Show conv_eigs over time


conv_eig_thing = th.nn.Sequential(
    th.nn.ZeroPad2d((2, 4, 2, 4)),
    th.nn.Conv2d(3, 32, 7, 1, padding=0, bias=False),
)
with th.no_grad():
    conv_eig_thing[1].weight.copy_(th.as_tensor(conv_eig).moveaxis(-1, 1))

    obs_one = level_dataset[10]["obs"]
    activations = conv_eig_thing(obs_one / 255)


fig = plotly_feature_vis(activations.numpy(), obs_one)
fig


# %%

video_path = Path("ob-jupyter") / "conv_eig_videos"
video_path.mkdir(parents=True, exist_ok=True)


def process_level(i):
    obs_one = level_dataset[i]["obs"]
    with th.no_grad():
        activations = conv_eig_thing(obs_one / 255)
    fig = plotly_feature_vis(activations.numpy(), obs_one)
    export_as_video(fig, video_path / f"level_{i}.mp4")


def export_as_video(fig: go.Figure, fpath: Path, size=(960, 720)) -> None:
    # Remove controls
    fig.layout.pop("updatemenus")  # type: ignore
    fig.layout.pop("sliders")  # type: ignore
    writer = cv2.VideoWriter(str(fpath), cv2.VideoWriter.fourcc("M", "P", "4", "V"), fps=5, frameSize=size)
    for frame in tqdm.tqdm(fig.frames):  # type: ignore
        for image, frame_image in zip(fig.data, frame.data):  # type: ignore
            image.source = frame_image.source  # type: ignore
        raster = plotly.io.to_image(fig, width=size[0], height=size[1])
        cv2_image = cv2.imdecode(np.frombuffer(raster, np.uint8), cv2.IMREAD_COLOR)
        writer.write(cv2_image)
    writer.release()


list(tqdm.tqdm(map(process_level, range(20)), total=20))
