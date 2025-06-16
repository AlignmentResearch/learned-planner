"""Figure for demonstrating plan stopping signals."""

# %%
import dataclasses
from copy import deepcopy

import numpy as np
import plotly.io as pio
import torch as th
from cleanba.environments import BoxobanConfig
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from learned_planner import BOXOBAN_CACHE, IS_NOTEBOOK, LP_DIR, ON_CLUSTER
from learned_planner.interp.offset_fns import apply_inv_offset_lc
from learned_planner.interp.plot import apply_style
from learned_planner.interp.utils import load_jax_model_to_torch, play_level
from learned_planner.interp.weight_utils import find_ijfo_contribution
from learned_planner.policies import download_policy_from_huggingface

if IS_NOTEBOOK:
    pio.renderers.default = "notebook"
th.set_printoptions(sci_mode=False, precision=2)


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


# %%
envs = dataclasses.replace(boxo_cfg, num_envs=1).make()
thinking_steps = 0
max_steps = 50
seq_len = 10

# %% PLAY TOY LEVEL
play_toy = True
two_levels = True
level_reset_opt = {"level_file_idx": 8, "level_idx": 2}
thinking_steps = 0

fwd_hooks = []

max_steps = 30
size = 10
walls = [(i, 0) for i in range(size)]
walls += [(i, size - 1) for i in range(size)]
walls += [(0, i) for i in range(1, size - 1)]
walls += [(size - 1, i) for i in range(1, size - 1)]

boxes = [(5, 2)]
targets = [(5, 6)]
player = (1, 2)


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
toy_obs = toy_obs.repeat_interleave(reps, 0).numpy()

print("Total len:", toy_obs.shape[0], toy_cache["features_extractor.cell_list.0.hook_h"].shape[0])
# %%

# ijos = ["i", "j", "o"]
ijfo = "i"
ijfo_offset = ["i", "j", "f", "o"].index(ijfo) * 32
layer_idx = 1
channel = 13
tick = 4

total_o = (
    toy_cache[f"features_extractor.cell_list.{layer_idx}.hook_conv_ih"][tick, ijfo_offset:]
    # + toy_cache[f"features_extractor.cell_list.{layer_idx}.hook_conv_hh"][tick, ijfo_offset:]
)
total_o = total_o[channel]
total_o = apply_inv_offset_lc(total_o, layer_idx, channel, last_dim_grid=True)

encoder_output = toy_cache[f"features_extractor.cell_list.{layer_idx}.hook_layer_input"][tick : tick + 1]
_, encoder_contribution = find_ijfo_contribution(
    th.tensor(encoder_output), range(32), layer_idx, channel, model, ih=True, inp_type="enc"
)
encoder_o = encoder_contribution.numpy()[0, :, :, ijfo_offset]
encoder_o = apply_inv_offset_lc(encoder_o, layer_idx, channel, last_dim_grid=True)

pooled_output = toy_cache[f"features_extractor.cell_list.{layer_idx}.hook_pool_project"][tick : tick + 1]
_, pooled_contribution = find_ijfo_contribution(
    th.tensor(pooled_output), range(32), layer_idx, channel, model, ih=True, inp_type="ch"
)
pooled_o = pooled_contribution.numpy()[0, :, :, ijfo_offset]
pooled_o = apply_inv_offset_lc(pooled_o, layer_idx, channel, last_dim_grid=True)

# encoder_o = pooled_o + encoder_o

inp_channels = [16, 17]
target_channels = [4, 6, 3, 26, 23, 15]
layer_inp = toy_cache[f"features_extractor.cell_list.{layer_idx}.hook_prev_layer_hidden"][tick : tick + 1]
inp_contributions, total_layer_inp = find_ijfo_contribution(
    th.tensor(layer_inp), range(32), layer_idx, channel, model, ih=True, inp_type="lh"
)
inp_contributions = apply_inv_offset_lc(inp_contributions.numpy()[0, ..., ijfo_offset], layer_idx, channel, last_dim_grid=True)
inp_right_contributions = inp_contributions[inp_channels].sum(axis=0)
inp_target_contributions = inp_contributions[target_channels].sum(axis=0)
total_o = encoder_contribution + total_layer_inp
total_o = total_o.numpy()[0, :, :, ijfo_offset]
total_o = apply_inv_offset_lc(total_o, layer_idx, channel, last_dim_grid=True)

global_min = min(total_o.min(), encoder_o.min(), inp_right_contributions.min(), inp_target_contributions.min())
global_max = max(total_o.max(), encoder_o.max(), inp_right_contributions.max(), inp_target_contributions.max())
heatmap_args = dict(zmin=global_min, zmax=global_max, colorscale="Viridis", showscale=True, colorbar=dict(thickness=5, xpad=5))

apply_style(figsize=(2.7, 0.6), px_margin=dict(t=0, b=5, l=2, r=0), px_use_default=False, font=10)
n_cols = 5
fig = make_subplots(
    rows=1,
    cols=n_cols,
    specs=[[{"type": "xy"}] * n_cols],
    horizontal_spacing=0.04,
)
fig.add_trace(go.Image(z=np.transpose(toy_obs[tick], (1, 2, 0))), row=1, col=1)
fig.add_trace(go.Heatmap(z=total_o[::-1], **heatmap_args), row=1, col=2)
fig.add_trace(go.Heatmap(z=inp_right_contributions[::-1], **heatmap_args), row=1, col=3)
fig.add_trace(go.Heatmap(z=encoder_o[::-1], **heatmap_args), row=1, col=4)
fig.add_trace(go.Heatmap(z=inp_target_contributions[::-1], **heatmap_args), row=1, col=5)

# Add equal and plus signs as annotations
fig.add_annotation(
    text="=",
    xref="x2 domain",
    yref="y2 domain",
    x=1.24,
    y=0.5,
    showarrow=False,
    font=dict(size=13),
)

fig.add_annotation(
    text="+",
    xref="x3 domain",
    yref="y3 domain",
    x=1.24,
    y=0.5,
    showarrow=False,
    font=dict(size=13),
)

fig.add_annotation(
    text="+",
    xref="x4 domain",
    yref="y4 domain",
    x=1.24,
    y=0.5,
    showarrow=False,
    font=dict(size=13),
)
y_offset = -0.12
fig.add_annotation(
    text="Observation",
    xref="x domain",
    yref="y2 domain",
    x=0.6,
    y=y_offset,
    showarrow=False,
)

fig.add_annotation(
    # text=f"L{layer_idx}{ijfo.upper()}{channel} (total)",
    text="L1O13",
    xref="x2 domain",
    yref="y2 domain",
    x=0.5,
    y=y_offset,
    showarrow=False,
)

fig.add_annotation(
    text="L0H17",
    xref="x3 domain",
    yref="y3 domain",
    x=0.5,
    y=y_offset,
    showarrow=False,
)

fig.add_annotation(
    text="Encoder",
    xref="x4 domain",
    yref="y4 domain",
    x=0.5,
    y=y_offset,
    showarrow=False,
)

fig.add_annotation(
    text="Target from L0",
    xref="x5 domain",
    yref="y5 domain",
    x=0.6,
    y=y_offset,
    showarrow=False,
)

for i in range(1, n_cols + 1):
    fig.update_xaxes(showticklabels=False, visible=False, ticks="", constrain="domain", row=1, col=i)
    fig.update_yaxes(showticklabels=False, visible=False, ticks="", scaleanchor="x", row=1, col=i)

# fig = px.imshow(encoder_o)
if ON_CLUSTER:
    fig.write_image(LP_DIR / "new_plots" / "plan_stopping.svg")
else:
    fig.write_image(LP_DIR / "new_plots" / "plan_stopping.pdf")
if IS_NOTEBOOK:
    fig.show()

# %%
