# %%
import dataclasses
import re
from copy import deepcopy
from functools import partial

import numpy as np
import plotly.io as pio
import sklearn.linear_model
import torch as th
from cleanba.environments import BoxobanConfig
from stable_baselines3.common.distributions import CategoricalDistribution
from transformer_lens.hook_points import HookPoint

from learned_planner import BOXOBAN_CACHE, LP_DIR
from learned_planner.interp.channel_group import get_group_channels, get_group_connections, layer_groups
from learned_planner.interp.collect_dataset import join_cache_across_steps
from learned_planner.interp.plot import plotly_feature_vis
from learned_planner.interp.train_probes import TrainOn
from learned_planner.interp.utils import load_jax_model_to_torch, load_probe, pad_level, parse_level, play_level
from learned_planner.interp.weight_utils import visualize_top_conv_inputs
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
        if all(l == "#" for l in line):
            out.append((line * (n))[: -2 * (n - 1)])
            continue
        for _ in range(n - 1):
            line1 = "#" + "".join((n * c if c in " #" else " " * n) for c in line[1:-1]) + "#"
            out.append(line1)
        line2 = "#" + "".join((n * c if c in " #" else (" " * (n - 1)) + f"{c}") for c in line[1:-1]) + "#"
        out.append(line2)
    return "\n".join(out)


print(
    nx_level(
        """
###########
# $@# ## ##
# ### ## ##
#     #   #
#### ## # #
## #      #
#       # #
### ## ## #
#      ##.#
###########
""",
        n=4,
    )
)

level_map = nx_level(
    """
###########
# $@# ## ##
# ### ## ##
#     #   #
#### #### #
## #      #
#       # #
### ## ## #
#      ##.#
###########
""",
    n=4,
)

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

# print_hook = True


# def strengthen_act_hook(inp, hook):
#     global print_hook
#     if print_hook:
#         print("strengthen hook applied")
#         print_hook = False
#     # inp[inp > 0.5] = 1
#     # inp[inp < -0.5] = -1
#     inp[inp > 0.45] += 0.3
#     inp[inp < -0.45] -= 0.3
#     inp = inp.clamp(-0.9, 0.9)
#     return inp


# fwd_hooks = [
#     (
#         f"features_extractor.cell_list.{layer}.{hook_type}.{pos}.{int_step}",
#         strengthen_act_hook,
#     )
#     for layer in [0, 1, 2]
#     for pos in [0]
#     for int_step in [0, 1, 2]
#     for hook_type in ["hook_h"]
# ]


def is_connected(lc1, lc2):
    """Checks if lc1 is fed as input to lc2"""
    l1, c1 = lc1
    l2, c2 = lc2

    return l1 == l2 or ((l1 + 1) % 3 == l2)


layer_values = {}

# for k, v in toy_cache.items():
#     if m := re.match("^.*([0-9]+)\\.hook_([h])$", k):
#         layer_values[int(m.group(1))] = v

desired_groups_box = ["B up", "B down", "B left", "B right"]
box_channels = get_group_channels(desired_groups_box)

desired_groups_agent = ["A up", "A down", "A left", "A right"]
agent_channels = get_group_channels(desired_groups_agent)
# combine box and agent channels
group_channels = [b + a for b, a in zip(box_channels, agent_channels)]
group_channels += get_group_channels(["Misc plan", "T", "No label"])
# inp,out
group_connections = get_group_connections(group_channels)
restore_model()
# ablate connections from B down to B right
factor = 1.2
for g1 in group_connections:
    for g2 in g1:
        for inplc, outlc in g2:
            inpl, inc = inplc
            outl, outc = outlc
            outc_ijfo = [idx * 32 + outc for idx in range(4)]
            if inpl == outl:
                model.features_extractor.cell_list[outl].conv_hh.weight.data[outc_ijfo, inc] *= factor
            else:
                model.features_extractor.cell_list[outl].conv_ih.weight.data[outc_ijfo, inc + 32] *= factor

# for layer in range(3):
#     model.features_extractor.cell_list[layer].conv_ih.weight.data[:, 32:64] *= factor
#     # model.features_extractor.cell_list[layer].conv_ih.weight.data[:, 32:64] *= factor
#     model.features_extractor.cell_list[layer].conv_hh.weight.data *= factor


# %%
restore_model()
# %%


def flatten(l):
    return [flatten(item) for sublist in l for item in sublist] if isinstance(l, list) else l


thinking_steps = 0
fwd_hooks = []
max_steps = thinking_steps + 150
dim_room = max(len(level_map.split("\n")), len(level_map.split("\n")[0])) + 1
dim_room = (dim_room, dim_room)
internal_steps = False

if getattr(model.get_distribution, "__name__", None) == "get_distribution":
    old_get_distribution = model.get_distribution
model.get_distribution = partial(bigger_levels_get_distribution, model)  # type: ignore

envs = dataclasses.replace(boxo_cfg, dim_room=dim_room).make()
if getattr(model.recurrent_initial_state, "__name__", None) == "recurrent_initial_state":
    old_recurrent_initial_state = model.recurrent_initial_state
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
    probes=[probe],
    probe_train_ons=[probe_info],
    probe_logits=True,
    internal_steps=internal_steps,
)

toy_cache = toy_out.cache
toy_cache = {k: v.squeeze(1) for k, v in toy_cache.items() if len(v.shape) == 5}
toy_obs = toy_out.obs.squeeze(1)
toy_obs = toy_out.obs.squeeze(1).numpy()


repeats = 3 if internal_steps else 1
probe_outs = toy_out.probe_outs[0]
probe_outs = np.moveaxis(np.reshape(probe_outs, (len(probe_outs) * repeats, *probe_outs.shape[-3:])), -1, 1)
# assert probe_outs.shape == (45, 5, 10, 10)
fig = plotly_feature_vis(probe_outs, np.repeat(toy_obs, repeats, 0), "alternative box dirs probe").show()

# %% Visualize all channels


for k, v in toy_cache.items():
    if m := re.match("^.*hook_([h])$", k):
        fig = plotly_feature_vis(v, np.repeat(toy_obs, 3, 0), k, m.group(1).upper())
        fig.update_layout(height=800)
        fig.show()

# %% Visualize features grouped by type

layer_values = {}

for k, v in toy_cache.items():
    if m := re.match("^.*([0-9]+)\\.hook_([h])$", k):
        layer_values[int(m.group(1))] = v

desired_groups = ["B up", "B down", "B left", "B right"]

channels = []
labels = []

for group in desired_groups:
    for layer in layer_groups[group]:
        channels.append(layer_values[layer["layer"]][:, layer["idx"], :, :])
        labels.append(f"{group} L{layer['layer']}H{layer['idx']}")

channels = np.stack(channels, 1)

fig = plotly_feature_vis(channels, np.repeat(toy_obs, 3, 0), feature_labels=labels)
fig.update_layout(height=800)
fig.show()


# %%
plot_layer, plot_channel = 0, 2
# plot_layer, plot_channel = 0, 17
# tick = reps - 1
tick = 0
show_ticks = True
# keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "C", "I", "J", "F", "O"]]
keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "J", "O"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, plot_channel] for key in keys], axis=1)
# if not show_ticks:
#     toy_all_channels_for_lc = toy_all_channels_for_lc[tick::3]

# repeat obs 3 along first dimension
fig = plotly_feature_vis(
    toy_all_channels_for_lc,
    np.repeat(toy_obs, 3, 0),
    feature_labels=[k.rsplit(".")[-1] for k in keys],
)
fig.show()

# %% conv ih and hh

plot_layer, plot_channel = 1, 17

keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["conv_ih", "conv_hh"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, 32 * ijo + plot_channel] for key in keys for ijo in [0, 1, 3]], axis=1)
# repeat obs 3 along first dimension
fig = plotly_feature_vis(
    toy_all_channels_for_lc,
    np.repeat(toy_obs, 3, 0),
    feature_labels=[k.rsplit(".")[-1][5:] + "_" + "ijfo"[ijo] for k in keys for ijo in [0, 1, 3]],
)
fig.show()

# %% Visualize

plot_layer, plot_channel, ih, ijfo, inp_types = 2, 9, False, "o", "lh"


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
    np.repeat(toy_obs, 3, 0),
    feature_labels=[f"{c}: {v:.2f}" for c, v in zip(top_channels, values)],  # + ["ih" if ih else "hh"],
    common_channel_norm=True,
)
fig.show()

# %% Activation patching


def abs_hook(inp, hook):
    inp[:, 17, 4:6, 7:11] = inp[:, 17, 4:6, 7:11].abs()
    return inp


def abs_hook_nine(inp, hook):
    inp[:, 9, 4:6, 7:11] = inp[:, 9, 4:6, 7:11].abs()
    return inp


def abs_hook_l1(inp, hook):
    inp[:, 13, 5:6, 7:11] = inp[:, 13, 5:6, 7:11].abs()
    return inp


fwd_hooks = [
    (
        f"features_extractor.cell_list.{layer}.{hook_type}.{pos}.{int_step}",
        abs_hook,
    )
    for layer in [0]
    for pos in [0]
    for int_step in [0, 1, 2]
    for hook_type in ["hook_h"]
]
fwd_hooks += [
    (
        f"features_extractor.cell_list.{layer}.{hook_type}.{pos}.{int_step}",
        abs_hook_nine,
    )
    for layer in [2]
    for pos in [0]
    for int_step in [0, 1, 2]
    for hook_type in ["hook_h"]
]

fwd_hooks += [
    (
        f"features_extractor.cell_list.{layer}.{hook_type}.{pos}.{int_step}",
        abs_hook_l1,
    )
    for layer in [1]
    for pos in [0]
    for int_step in [0, 1, 2]
    for hook_type in ["hook_h"]
]

toy_out = play_level(
    envs,
    model,
    reset_opts=reset_opts,
    thinking_steps=thinking_steps,
    fwd_hooks=fwd_hooks,
    max_steps=max_steps,
    hook_steps=list(range(4, 100)),
    probes=[probe],
    probe_train_ons=[probe_info],
    probe_logits=True,
    internal_steps=internal_steps,
)
toy_cache = join_cache_across_steps([toy_out.cache])
toy_cache = {
    k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, *v.shape[-3:])
    for k, v in toy_cache.items()
    if len(v.shape) == 6
}
toy_obs = toy_out.obs.squeeze(1)
toy_obs = toy_out.obs.squeeze(1).numpy()


repeats = 3 if internal_steps else 1
probe_outs = toy_out.probe_outs[0]
probe_outs = np.moveaxis(np.reshape(probe_outs, (len(probe_outs) * repeats, *probe_outs.shape[-3:])), -1, 1)
# assert probe_outs.shape == (45, 5, 10, 10)
fig = plotly_feature_vis(probe_outs, np.repeat(toy_obs, repeats, 0), "alternative box dirs probe").show()
