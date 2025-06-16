"""Backtrack mechanism case study and causal intervention on wall square."""

# %%
import dataclasses
from copy import deepcopy
from functools import partial

import cairosvg
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import sklearn.linear_model
import torch as th
from cleanba.environments import BoxobanConfig
from plotly.subplots import make_subplots
from stable_baselines3.common.distributions import CategoricalDistribution
from transformer_lens.hook_points import HookPoint

from learned_planner import BOXOBAN_CACHE, IS_NOTEBOOK, LP_DIR, ON_CLUSTER
from learned_planner.interp.channel_group import (
    get_channel_dict,
    get_group_channels,
    get_group_connections,
    split_by_layer,
)
from learned_planner.interp.offset_fns import apply_inv_offset_lc, offset_yx
from learned_planner.interp.plot import apply_style, plotly_feature_vis, save_video_from_plotly
from learned_planner.interp.train_probes import TrainOn
from learned_planner.interp.utils import load_jax_model_to_torch, load_probe, pad_level, parse_level, play_level
from learned_planner.policies import download_policy_from_huggingface

try:
    pio.kaleido.scope.mathjax = None  # Disable MathJax to remove the loading message
except AttributeError:
    pass
if IS_NOTEBOOK:
    pio.renderers.default = "notebook"
else:
    pio.renderers.default = "png"
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


# %% Print model's hook points
def recursive_children(model):
    for c in model.children():
        yield c
        yield from recursive_children(c)


[c.name for c in recursive_children(model) if isinstance(c, HookPoint)]

# %%
# probe = pd.read_pickle("probe-future-directions.pkl")
probe, _ = load_probe("probes/best/boxes_future_direction_map_l-all.pkl")
try:
    new_coef = np.concatenate([probe.estimators_[i].coef_ for i in range(len(probe.estimators_))], 0)
    new_intercept = np.concatenate([probe.estimators_[i].intercept_ for i in range(len(probe.estimators_))], 0)

    probe = sklearn.linear_model.LogisticRegression()
    probe.coef_ = new_coef
    probe.intercept_ = new_intercept
    probe.classes_ = np.arange(new_coef.shape[0])
    probe.n_features_in_ = new_coef.shape[1]
except AttributeError:
    print("Skipping constructing the probe as it is already the right kind")

probe_info = TrainOn(dataset_name="boxes_future_direction")


# %%

combined_probe, combined_intercepts = th.load(LP_DIR / "learned_planner/notebooks/action_l2_probe.pt", weights_only=True)
aggregation_weight, aggregation_bias = th.load(LP_DIR / "learned_planner/notebooks/aggregation.pt", weights_only=True)


def bigger_levels_get_distribution(self, obs, carry, episode_starts, feature_extractor_kwargs=None):
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


# %%
def nx_level(level: str, n: int = 2) -> str:
    """
    Scale a level to nx the size
    """
    out = []
    for line in level.strip().split("\n"):
        for _ in range(n - 1):
            line1 = "".join((n * c if c in " #" else " " * n) for c in line)
            out.append(line1)
        line2 = "".join((n * c if c in " #" else (" " * (n - 1)) + f"{c}") for c in line)
        out.append(line2)
    return "\n".join(out)


# print(
#     nx_level(
#         """
# ###########
# # $@# ## ##
# # ### ## ##
# #     #   #
# #### ## # #
# ## #      #
# #       # #
# ### ## ## #
# #      ##.#
# ###########
# """,
#         n=2,
#     )
# )

level_map = nx_level(
    """
###########
# $@# ## ##
# ### ## ##
#    #    #
#### #### #
#### #### #
## #      #
#       # #
### ## ## #
#      ##.#
###########
""",
    n=2,
)
# level_map = """
# ######################
# ######################
# ##      ##  ####  ####
# ##   $ @##  ####  ####
# ##  ######  ####  ####
# ##  ######  ####  ####
# ##         #        ##
# ##         #        ##
# ########  ########  ##
# ########  ########  ##
# ####  ##            ##
# ####  ##            ##
# ##              ##  ##
# ##              ##  ##
# ######  ####  ####  ##
# ######  ####  ####  ##
# ##            ####  ##
# ##            #### .##
# ######################
# ######################
# """

# original
# level_map = """
# ####################
# #      ##  ####  ###
# #   $ @##  ####  ###
# #  ######  ####  ###
# #  ######  ####  ###
# #        #         #
# #        #         #
# #######  ####  ##  #
# #######  ####  ##  #
# ###  ##            #
# ###  ##            #
# #              ##  #
# #              ##  #
# #####  ####  ####  #
# #####  ####  ####  #
# #            ####  #
# #            #### .#
# ####################
# """

# used in paper
level_map = """
####################
####################
#      ##  ####  ###
#   $ @##  ####  ###
#  ######  ####  ###
#  ######  ####  ###
#        #         #
#        #         #
#######  ########  #
#######  ########  #
###  ##            #
###  ##            #
#              ##  #
#              ##  #
#####  ####  ####  #
#####  ####  ####  #
#            ####  #
#            #### .#
####################
####################
"""


# level_map = """
# ############################
# #         ###  ######   ####
# #         ###  ######   ####
# #     $  @###  ######   ####
# #   #########  ######   ####
# #   #########  ######   ####
# #   #########  ######   ####
# #              ###         #
# #              ###         #
# ##########  ############   #
# ##########  ############   #
# ##########  ############   #
# ####   ###                 #
# ####   ###                 #
# ####   ###                 #
# #                    ###   #
# #                    ###   #
# #                    ###   #
# #######  ######   ######   #
# #######  ######   ######   #
# #######  ######   ######   #
# #                 ###### . #
# ############################
# ############################
# ############################
# """
# level_map = """
# ##############################
# #         ###   ######   #####
# #         ###   ######   #####
# #     $  @###   ######   #####
# #   #########   ######   #####
# #   #########   ######   #####
# #   #########   ######   #####
# #               ###         ##
# #               ###         ##
# ##########   ############   ##
# ##########   ############   ##
# ##########   ############   ##
# ####   ###                  ##
# ####   ###                  ##
# ####   ###                  ##
# #                     ###   ##
# #                     ###   ##
# #                     ###   ##
# #######   ######   ######   ##
# #######   ######   ######   ##
# #######   ######   ######   ##
# #                  ######   ##
# #                  ######   ##
# #                  ###### . ##
# ##############################
# ##############################
# ##############################
# ##############################
# ##############################
# """

# level_map = """
# ##################################
# #                                #
# #   @                            #
# #                                #
# #                                #
# #                                #
# #             $                  #
# #                                #
# #                                #
# #                                #
# #                                #
# #                                #
# #                                #
# #                                #
# #                                #
# #                       .        #
# #                                #
# #                                #
# #                                #
# #                                #
# ##################################
# """

level_map = level_map.strip()

# %%
restore_model()
# %%
internal_steps = False


def flatten(l):
    return [flatten(item) for sublist in l for item in sublist] if isinstance(l, list) else l


def play_big_level(
    level_map: str,
    max_steps: int = 150,
    fwd_hooks: list = [],
    return_cache=True,
    names_filter=None,
    use_probe=False,
):
    thinking_steps = 0
    max_steps = thinking_steps + max_steps
    level_map = level_map.strip()
    dim_room = max(len(level_map.split("\n")), len(level_map.split("\n")[0]))
    dim_room = (dim_room, dim_room)

    model.get_distribution = partial(bigger_levels_get_distribution, model)  # type: ignore

    envs = dataclasses.replace(boxo_cfg, dim_room=dim_room).make()
    model.recurrent_initial_state = partial(bigger_recurrent_initial_state, model, dim_room)

    # model.get_distribution = old_get_distribution
    level_rep = pad_level(
        level_map,
        *dim_room,
    )
    reset_opts = parse_level(level_rep)
    toy_out = play_level(
        envs,
        model,
        # reset_opts=dict(walls=walls, boxes=boxes, targets=targets, player=player),
        reset_opts=reset_opts,
        # reset_opts=level_reset_opt,
        thinking_steps=thinking_steps,
        fwd_hooks=fwd_hooks,
        max_steps=max_steps,
        hook_steps=list(range(thinking_steps, max_steps)) if thinking_steps > 0 else -1,
        probes=[probe] if use_probe else [],
        probe_train_ons=[probe_info] if use_probe else [],
        probe_logits=use_probe,
        internal_steps=internal_steps,
        return_cache=return_cache,
        names_filter=names_filter,
    )

    toy_cache = toy_out.cache
    if return_cache:
        toy_cache = {k: v.squeeze(1) for k, v in toy_cache.items() if len(v.shape) == 5}
    toy_obs = toy_out.obs.squeeze(1).numpy()
    # return toy_cache, toy_obs, toy_out
    return toy_out, toy_obs, toy_cache


# %%

toy_out, toy_obs, toy_cache = play_big_level(level_map, max_steps=150, fwd_hooks=[])

print("Len:", len(toy_obs))
# repeats = 3 if internal_steps else 1
# probe_outs = toy_out.probe_outs[0]
# probe_outs = np.moveaxis(np.reshape(probe_outs, (len(probe_outs) * repeats, *probe_outs.shape[-3:])), -1, 1)
# # assert probe_outs.shape == (45, 5, 10, 10)
# fig = plotly_feature_vis(probe_outs, np.repeat(toy_obs, repeats, 0), "alternative box dirs probe").show()

# %%

box_right_channels = get_group_channels("box", return_dict=True)[3]
layer_wise_right = split_by_layer([box_right_channels])

int_y_start, int_y_end = 6, 8
int_x_start, int_x_end = 8, 12


def abs_hook(inp, hook):
    layer = int(hook.name.split(".")[2])
    right_channels = layer_wise_right[layer]
    for c_dict in right_channels:
        idx = c_dict["idx"]
        offset = offset_yx(0, 0, [idx], layer)
        offset_y, offset_x = offset[0][0], offset[1][0]
        inp[:, c_dict["idx"], offset_y + int_y_start : offset_y + int_y_end, offset_x + int_x_start : offset_x + int_x_end] = (
            inp[
                :, c_dict["idx"], offset_y + int_y_start : offset_y + int_y_end, offset_x + int_x_start : offset_x + int_x_end
            ].abs()
            * c_dict["sign"]
        )
    return inp


fwd_hooks = [
    (
        f"features_extractor.cell_list.{layer}.{hook_type}.{pos}.{int_step}",
        abs_hook,
    )
    for layer in [0, 1, 2]
    for pos in [0]
    for int_step in [0, 1, 2]
    for hook_type in ["hook_h"]
]

abl_out, abl_obs, abl_cache = play_big_level(level_map, max_steps=150, fwd_hooks=fwd_hooks)

print("Len:", len(abl_obs))


# %% Actual plot

box_group_channels = get_group_channels("box", return_dict=True)


def standardize_channel(channel_value, channel_info: tuple[int, int] | dict):
    """Standardize the channel value based on its sign and index."""
    assert len(channel_value.shape) >= 2, f"Invalid channel value shape: {channel_value.shape}"
    if isinstance(channel_info, tuple):
        l, c = channel_info
        channel_dict = get_channel_dict(l, c)
    else:
        channel_dict = channel_info
    channel_value = apply_inv_offset_lc(channel_value, channel_dict["layer"], channel_dict["idx"], last_dim_grid=True)
    sign = channel_dict["sign"]
    if isinstance(sign, str):
        assert sign in ["+", "-"], f"Invalid sign: {sign}"
        sign = 1 if sign == "+" else -1
    elif not isinstance(sign, int):
        raise ValueError(f"Invalid sign type: {type(sign)}")
    return channel_value * sign


def get_avg(cache, channels, tick, hook_type="h"):
    avg_channels = np.mean(
        np.stack(
            [
                standardize_channel(
                    cache[f"features_extractor.cell_list.{c_dict['layer']}.hook_{hook_type}"][tick, c_dict["idx"]],
                    c_dict,
                )
                for c_dict in channels
            ],
            axis=0,
        ),
        axis=0,
    )
    return avg_channels


# %%
save_video = False
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
    # if IS_NOTEBOOK:
    #     fig.show()

    save_video_from_plotly(fig, "backtrack.mp4", fps=4, demo=False)

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

    save_video_from_plotly(fig, "backtrack_abl.mp4", fps=4, demo=False)


# %%
# Create a subplot grid: 2 rows x 5 cols with a merged cell covering the first 2 rows and 2 cols.
apply_style(figsize=(5, 2), px_use_default=False, px_margin=dict(t=1, b=20, l=1, r=1), font=8)

fig = make_subplots(
    rows=2,
    cols=5,
    specs=[
        [{"type": "xy", "colspan": 2, "rowspan": 2}, None, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        [None, None, {"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
    ],
    # subplot_titles=["Observation", "Step 2", "Step 5", "Step 8"],
    vertical_spacing=0.12,
)

y = -0.28
fig.add_annotation(
    text="(a) Observation",
    x=0.5,
    y=y + 0.1,
    xref="x domain",
    yref="y5 domain",
    showarrow=False,
)
x = 0.95
fig.add_annotation(
    text="D1",
    x=x,
    y=0.44,
    xref="x domain",
    yref="y domain",
    showarrow=False,
)
fig.add_annotation(
    text="D2",
    x=x,
    y=0.64,
    xref="x domain",
    yref="y domain",
    showarrow=False,
)
fig.add_annotation(
    text="D3",
    x=x - 0.4,
    y=0.64,
    xref="x domain",
    yref="y domain",
    showarrow=False,
)
img = toy_obs[0]
img = np.transpose(img, (1, 2, 0))
fig.add_trace(go.Image(z=img), row=1, col=1)
fig.update_xaxes(showticklabels=False, visible=False, constrain="domain", ticks="", row=1, col=1)
fig.update_yaxes(showticklabels=False, visible=False, constrain="domain", ticks="", row=1, col=1)

tick = 3
all_acts = []
for dir_grp in box_group_channels:
    avg_act = get_avg(toy_cache, dir_grp, tick)
    all_acts.append(avg_act)
all_acts = np.stack(all_acts, axis=0).sum(axis=0)
fig.add_trace(go.Heatmap(z=all_acts[::-1], colorscale="Viridis", showscale=False), row=1, col=3)

fig.add_annotation(
    text=f"(b) All box channels<br>step {tick // 3} tick {tick % 3}",
    x=0.5,
    y=y,
    xref="x2 domain",
    yref="y3 domain",
    showarrow=False,
    # xanchor="center",
    # yanchor="bottom",
)

tick = 16
right_l2h9 = [c_dict for c_dict in box_group_channels[3] if c_dict["layer"] == 2 and c_dict["idx"] == 9]
all_acts = get_avg(toy_cache, right_l2h9, tick)
fig.add_trace(go.Heatmap(z=all_acts[::-1], colorscale="Viridis", showscale=False), row=1, col=4)
fig.add_annotation(
    text=f"(c) L2H9 (right)<br>step {tick // 3} tick {tick % 3}",
    x=0.5,
    y=y,
    xref="x3 domain",
    yref="y3 domain",
    showarrow=False,
    # xanchor="center",
    # yanchor="bottom",
)

right_l2h9 = [c_dict for c_dict in box_group_channels[3] if c_dict["layer"] == 2 and c_dict["idx"] == 9]
all_acts = get_avg(toy_cache, right_l2h9, tick, hook_type="i")
fig.add_trace(go.Heatmap(z=all_acts[::-1], colorscale="Viridis", showscale=False), row=1, col=5)
fig.add_annotation(
    text=f"(d) L2I9 (right)<br>step {tick // 3} tick {tick % 3}",
    x=0.5,
    y=y,
    xref="x4 domain",
    yref="y3 domain",
    showarrow=False,
    # xanchor="center",
    # yanchor="bottom",
)

right_l2h9 = [c_dict for c_dict in box_group_channels[3] if c_dict["layer"] == 2 and c_dict["idx"] == 9]
all_acts = get_avg(toy_cache, right_l2h9, tick, hook_type="o")
fig.add_trace(go.Heatmap(z=all_acts[::-1], colorscale="Viridis", showscale=False), row=2, col=3)
fig.add_annotation(
    text=f"(e) L2O9 (right)<br>step {tick // 3} tick {tick % 3}",
    x=0.5,
    y=y,
    xref="x5 domain",
    yref="y5 domain",
    showarrow=False,
    # xanchor="center",
    # yanchor="bottom",
)

tick = 28
right_l2h9 = [c_dict for c_dict in box_group_channels[3] if c_dict["layer"] == 2 and c_dict["idx"] == 9]
all_acts = get_avg(toy_cache, right_l2h9, tick)
fig.add_trace(go.Heatmap(z=all_acts[::-1], colorscale="Viridis", showscale=False), row=2, col=4)
fig.add_annotation(
    text=f"(f) L2H9 (right)<br>step {tick // 3} tick {tick % 3}",
    x=0.5,
    y=y,
    xref="x6 domain",
    yref="y5 domain",
    showarrow=False,
    # xanchor="center",
    # yanchor="bottom",
)

right_l2h9 = [c_dict for c_dict in box_group_channels[3] if c_dict["layer"] == 2 and c_dict["idx"] == 9]
all_acts = get_avg(abl_cache, right_l2h9, tick)
fig.add_trace(go.Heatmap(z=all_acts[::-1], colorscale="Viridis", showscale=False), row=2, col=5)
fig.add_annotation(
    text=f"(g) L2H9 (right)<br>Abl. step {tick // 3} tick {tick % 3}",
    x=0.5,
    y=y,
    xref="x7 domain",
    yref="y5 domain",
    showarrow=False,
    # xanchor="center",
    # yanchor="bottom",
)

for row in range(2):
    for col in range(3, 6):
        fig.update_xaxes(showticklabels=False, visible=False, constrain="domain", ticks="", row=row + 1, col=col)
        fig.update_yaxes(showticklabels=False, visible=False, constrain="domain", ticks="", row=row + 1, col=col)
if IS_NOTEBOOK:
    fig.show()
if ON_CLUSTER:
    fig.write_image("/training/new_plots/backtrack.svg")
    # cairosvg.svg2pdf(url="/training/new_plots/backtrack.svg", write_to="/training/new_plots/backtrack.pdf")
else:
    fig.write_image(LP_DIR / "new_plots" / "backtrack.pdf")

# %%
restore_model()

group_channels = get_group_channels("box_agent")

group_connections = get_group_connections(group_channels)

# ablate connections from B down to B right


def steroid_network(factor):
    restore_model()
    for layer in range(3):
        model.features_extractor.cell_list[layer].conv_ih.weight.data[:, 32:64] *= factor
        model.features_extractor.cell_list[layer].conv_hh.weight.data *= factor

    # for g1 in group_connections:
    #     for g2 in g1:
    #         for inplc, outlc in g2:
    #             inpl, inc = inplc
    #             outl, outc = outlc
    #             outc_ijfo = [idx * 32 + outc for idx in range(4)]
    #             if inpl == outl:
    #                 model.features_extractor.cell_list[outl].conv_hh.weight.data[outc_ijfo, inc] *= factor
    #             else:
    #                 model.features_extractor.cell_list[outl].conv_ih.weight.data[outc_ijfo, inc + 32] *= factor


# %%
orig_solved, steroid_solved = [], []

level_map = """
##########
# $@# # ##
# ### # ##
#    #   #
#### ### #
## #     #
#      # #
### # ## #
#     ##.#
##########
"""  # this level is not solvable but bigger levels with Nx level will be with N>=2

# restore_model()
# for n in range(2, 6):
#     toy_out, _, _ = play_big_level(nx_level(level_map, n), max_steps=60 * n, fwd_hooks=[], return_cache=False)
#     print("Size:", toy_out.obs.shape[-1], "& N:", n, "& Solved:", toy_out.solved.item())
#     orig_solved.append(toy_out.solved.item())

steroid_network(factor=1.20)
for n in range(2, 6):
    toy_out, _, _ = play_big_level(nx_level(level_map, n), max_steps=60 * n, fwd_hooks=[], return_cache=False)
    print("Size:", toy_out.obs.shape[-1], "& N:", n, "& Solved:", toy_out.solved.item())
    steroid_solved.append(toy_out.solved.item())

# steroid_network(factor=1.3)
# for n in range(2, 6):
#     toy_out, _, _ = play_big_level(nx_level(level_map, n), max_steps=60 * n, fwd_hooks=[], return_cache=False)
#     print("Size:", toy_out.obs.shape[-1], "& N:", n, "& Solved:", toy_out.solved.item())
#     steroid_solved.append(toy_out.solved.item())

print("Orig solved:", orig_solved)
print("Steroid solved:", steroid_solved)

# %%
n = 4
names_filter = [f"features_extractor.cell_list.{layer}.hook_h.0.{int_step}" for layer in [0, 1, 2] for int_step in range(3)]
restore_model()
toy_out, toy_obs, toy_cache = play_big_level(
    nx_level(level_map, n),
    max_steps=60 * n,
    fwd_hooks=[],
    return_cache=True,
    names_filter=names_filter,
    use_probe=False,
)

steroid_network(factor=1.21)
steered_out, steered_obs, steered_cache = play_big_level(
    nx_level(level_map, n),
    max_steps=60 * n,
    fwd_hooks=[],
    return_cache=True,
    names_filter=names_filter,
    use_probe=False,
)
print("solved:", steered_out.solved.item())

chaotic_factor = 1.4
steroid_network(factor=chaotic_factor)
chaotic_out, chaotic_obs, chaotic_cache = play_big_level(
    nx_level(level_map, n),
    max_steps=60 * n,
    fwd_hooks=[],
    return_cache=True,
    names_filter=names_filter,
    use_probe=False,
)
print("solved:", chaotic_out.solved.item())

# %%
apply_style(figsize=(5, 1.5), px_margin=dict(t=5, b=12, l=5, r=5), px_use_default=False)

# Visualize the sum of box channels for the original and steered models at tick 3
tick, tick_more = 150, 300
orig_acts = np.stack([get_avg(toy_cache, dir_grp, tick) for dir_grp in box_group_channels], axis=0).sum(axis=0)
steered_acts = np.stack([get_avg(steered_cache, dir_grp, tick) for dir_grp in box_group_channels], axis=0).sum(axis=0)

steered_acts_more = np.stack([get_avg(steered_cache, dir_grp, tick_more) for dir_grp in box_group_channels], axis=0).sum(
    axis=0
)
n_cols = 3
fig = make_subplots(rows=1, cols=n_cols, specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]], horizontal_spacing=0.04)
fig.add_trace(go.Heatmap(z=orig_acts[::-1], colorscale="Viridis", showscale=False), row=1, col=1)
fig.add_trace(go.Heatmap(z=steered_acts[::-1], colorscale="Viridis", showscale=False), row=1, col=2)
fig.add_trace(go.Heatmap(z=steered_acts_more[::-1], colorscale="Viridis", showscale=False), row=1, col=3)
y = -0.1
fig.add_annotation(
    text=f"(a) Original net at step {tick // 3} tick {tick % 3}",
    x=0.5,
    y=y,
    xref="x domain",
    yref="y domain",
    showarrow=False,
)
fig.add_annotation(
    text=f"(b) Steered net at step {tick // 3} tick {tick % 3}",
    x=0.5,
    y=y,
    xref="x2 domain",
    yref="y2 domain",
    showarrow=False,
)
fig.add_annotation(
    text=f"(c) Steered net at step {tick_more // 3} tick {tick_more % 3}",
    x=0.5,
    y=y,
    xref="x3 domain",
    yref="y3 domain",
    showarrow=False,
)

obs_size = toy_obs.shape[-1]

for col in range(1, n_cols + 1):
    fig.update_xaxes(showticklabels=False, constrain="domain", ticks="", visible=False, row=1, col=col)
    fig.update_yaxes(showticklabels=False, constrain="domain", ticks="", visible=False, row=1, col=col)

if ON_CLUSTER:
    # fig.write_image(f"/training/new_plots/steered_net_on_{obs_size}x{obs_size}.pdf")
    svg_path = f"/training/new_plots/steered_net_on_{obs_size}x{obs_size}.svg"
    fig.write_image(svg_path)
    cairosvg.svg2pdf(url=svg_path, write_to=svg_path.replace(".svg", ".pdf"))
else:
    fig.write_image(LP_DIR / "new_plots" / f"steered_net_on_{obs_size}x{obs_size}.pdf")
if IS_NOTEBOOK:
    fig.show()

# %%

apply_style(figsize=(1.0, 1.0), px_margin=dict(t=0, b=0, l=0, r=0), px_use_default=False)
# pio.templates.default = "plotly"
# tick_more = 450
chaotic_acts = np.stack([get_avg(chaotic_cache, dir_grp, tick_more) for dir_grp in box_group_channels], axis=0).sum(axis=0)

fig = make_subplots(rows=1, cols=1, specs=[[{"type": "xy"}]], horizontal_spacing=0.04)
fig.add_trace(go.Heatmap(z=chaotic_acts[::-1], colorscale="Viridis", showscale=False), row=1, col=1)
fig.update_xaxes(showticklabels=False, constrain="domain", ticks="", visible=False, row=1, col=1)
fig.update_yaxes(showticklabels=False, constrain="domain", ticks="", visible=False, row=1, col=1)

if ON_CLUSTER:
    # fig.write_image(f"/training/new_plots/steered_net_on_{obs_size}x{obs_size}.pdf")
    svg_path = f"/training/new_plots/steered{chaotic_factor}_net_on_{obs_size}x{obs_size}.svg"
    fig.write_image(svg_path)
    cairosvg.svg2pdf(url=svg_path, write_to=svg_path.replace(".svg", ".pdf"))
else:
    fig.write_image(LP_DIR / "new_plots" / f"steered{chaotic_factor}_net_on_{obs_size}x{obs_size}.pdf")
if IS_NOTEBOOK:
    fig.show()


# %%

# %% Useful for debugging

# pio.templates.default = "plotly"
# toy_obs = toy_out.obs.squeeze(1).numpy()
# toy_obs = np.repeat(toy_obs, 3, 0)
# plotly_feature_vis(
#     np.zeros((len(toy_obs), 0, *toy_obs.shape[2:])),
#     toy_obs,
# )

# # %% Visualize features grouped by type

# pio.templates.default = "plotly"
# layer_values = {}
# ablated = True
# for k, v in (steered_cache if ablated else toy_cache).items():
#     if m := re.match("^.*([0-9]+)\\.hook_([h])$", k):
#         layer_values[int(m.group(1))] = v[:100]

# desired_groups = ["B up", "B down", "B left", "B right"]

# channels = []
# labels = []

# for group in desired_groups:
#     for layer in layer_groups[group]:
#         channels.append(layer_values[layer["layer"]][:, layer["idx"], :, :])
#         labels.append(f"{group} L{layer['layer']}H{layer['idx']}")

# channels = np.stack(channels, 1)

# fig = plotly_feature_vis(channels, np.repeat(steered_obs if ablated else toy_obs, 3, 0)[:100], feature_labels=labels)
# fig.update_layout(height=800)
# if IS_NOTEBOOK:
#     fig.show()

# # %%
# names_filter = [f"features_extractor.cell_list.{layer}.hook_h.0.{int_step}" for layer in [0, 1, 2] for int_step in range(3)]
# steroid_network(factor=1.22)
# for n in range(8, 9):
#     toy_out, toy_obs, toy_cache = play_big_level(
#         nx_level(level_map, n), max_steps=70 * n, fwd_hooks=[], return_cache=True, names_filter=names_filter
#     )
#     print("Size:", toy_out.obs.shape[-1], "& N:", n, "& Solved:", toy_out.solved.item())
