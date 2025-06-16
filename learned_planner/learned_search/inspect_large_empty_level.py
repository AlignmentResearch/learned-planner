"""Try out large empty levels where agent has to push a single box across one dimension."""

# %%
import dataclasses
from copy import deepcopy
from functools import partial

import numpy as np
import plotly.io as pio
import sklearn.linear_model
import torch as th
from cleanba.environments import BoxobanConfig
from stable_baselines3.common.distributions import CategoricalDistribution
from transformer_lens.hook_points import HookPoint

from learned_planner import LP_DIR
from learned_planner.interp.channel_group import get_group_channels, get_group_connections
from learned_planner.interp.plot import plotly_feature_vis
from learned_planner.interp.train_probes import TrainOn
from learned_planner.interp.utils import load_jax_model_to_torch, load_probe, play_level
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
    cache_path=LP_DIR / "alternative-levels/levels",
    num_envs=1,
    max_episode_steps=200,
    min_episode_steps=200,
    asynchronous=False,
    tinyworld_obs=True,
    split="train",
    difficulty="unfiltered",
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


# %% multiply weights

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

for layer in range(3):
    model.features_extractor.cell_list[layer].conv_ih.weight.data[:, 32:64] *= factor
    # model.features_extractor.cell_list[layer].conv_ih.weight.data[:, 32:64] *= factor
    model.features_extractor.cell_list[layer].conv_hh.weight.data *= factor


# %% restore model
restore_model()
# %%


def flatten(l):
    return [flatten(item) for sublist in l for item in sublist] if isinstance(l, list) else l


thinking_steps = 12
fwd_hooks = []
max_steps = thinking_steps + 50
# dim_room = max(len(level_map.split("\n")), len(level_map.split("\n")[0])) + 1
dim_room = 15
dim_room = (dim_room, dim_room)
internal_steps = False

if getattr(model.get_distribution, "__name__", None) == "get_distribution":
    old_get_distribution = model.get_distribution
model.get_distribution = partial(bigger_levels_get_distribution, model)  # type: ignore

envs = dataclasses.replace(boxo_cfg, dim_room=dim_room).make()
if getattr(model.recurrent_initial_state, "__name__", None) == "recurrent_initial_state":
    old_recurrent_initial_state = model.recurrent_initial_state
model.recurrent_initial_state = partial(bigger_recurrent_initial_state, model, dim_room)


size = dim_room[0]
walls = [(i, 0) for i in range(size)]
walls += [(i, size - 1) for i in range(size)]
walls += [(0, i) for i in range(1, size - 1)]
walls += [(size - 1, i) for i in range(1, size - 1)]

player = (1, 1)
y = 7
# boxes = [(y, 2)]
# targets = [(y, size - 2)]

boxes = [(2, y)]
targets = [(size - 2, y)]


reset_opts = dict(walls=walls, boxes=boxes, targets=targets, player=player)

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
