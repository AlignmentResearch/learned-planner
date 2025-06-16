# %%
import dataclasses
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import plotly.io as pio
import torch as th
from cleanba.environments import BoxobanConfig
from transformer_lens.hook_points import HookPoint

from learned_planner.interp.collect_dataset import join_cache_across_steps
from learned_planner.interp.plot import plotly_feature_vis
from learned_planner.interp.utils import load_jax_model_to_torch, play_level
from learned_planner.notebooks.emacs_plotly_render import set_plotly_renderer
from learned_planner.policies import download_policy_from_huggingface

pio.renderers.default = "notebook"

set_plotly_renderer("emacs")

# %%
MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000"  # DRC(3, 3) 2B checkpoint
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

# try:
#     BOXOBAN_CACHE = Path(__file__).parent.parent.parent / ".sokoban_cache"
# except NameError:
#     BOXOBAN_CACHE = Path(os.getcwd()) / ".sokoban_cache"
BOXOBAN_CACHE = Path("/opt/sokoban_cache")

boxes_direction_probe_file = Path(
    "/training/TrainProbeConfig/05-probe-boxes-future-direction/wandb/run-20240813_184417-vb6474rg/local-files/probe_l-all_x-all_y-all_c-all.pkl"
)

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

# %%


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

envs = dataclasses.replace(boxo_cfg, num_envs=512).make()
clean_obs = run_policy_reset(seq_len, envs, model)
corrupted_obs = run_policy_reset(seq_len, envs, model)

# %%

zero_carry = model.recurrent_initial_state(envs.num_envs)
eps_start = th.zeros((seq_len, envs.num_envs), dtype=th.bool)
eps_start[0, :] = True

(clean_actions, clean_values, clean_log_probs, _), clean_cache = model.run_with_cache(clean_obs, zero_carry, eps_start)
# Create corrupted activations which are other random levels. The hope is that, over a large data set, they will change
# the output enough times and in different enough ways that we can correctly attribute to things in the latest layer
_, corrupted_cache = model.run_with_cache(corrupted_obs, zero_carry, eps_start)

# %%
key_pattern = [rf".*hook_{v}\.\d\.\d$" for v in ["h", "c", "i", "j", "f", "o"]]

# %%


def interpolate_rnn_inputs(alpha, am1_tuple, a_tuple):
    am1, _, _ = am1_tuple
    a, _, _ = a_tuple

    return ((1 - alpha) * am1 + alpha * a, *am1_tuple[1:])


def add_attributions(
    attributions: Dict[str, th.Tensor],
    loss_fn,
    model,
    clean_inputs: Any,
    corrupted_inputs: Any,
    clean_cache: Dict[str, th.Tensor],
    corrupted_minus_clean_cache: Dict[str, th.Tensor],
    *,
    ablate_at_every_hook: bool = False,
    n_gradients_to_integrate: int = 5,
) -> Dict[str, th.Tensor]:
    """Attributes the output to every parameter in `clean_cache`. The attribution strength is added go `attributions`
    and returned. The `attributions` parameter is useful for computing the attribution in several minibatches.

    """
    assert n_gradients_to_integrate >= 1

    def set_corrupted_hook(inputs: th.Tensor, hook: HookPoint):
        nonlocal alpha
        if ablate_at_every_hook:
            desired = clean_cache[str(hook.name)] + alpha * corrupted_minus_clean_cache[str(hook.name)]
            return inputs + (desired - inputs).detach()
        return None

    def save_gradient_hook(grad: th.Tensor, hook: HookPoint):
        nonlocal attributions
        attr = (grad.detach() * corrupted_minus_clean_cache[str(hook.name)]).sum(0).detach() / n_gradients_to_integrate
        try:
            attributions[str(hook.name)].add_(attr)
        except KeyError:
            attributions[str(hook.name)] = attr

    keys = clean_cache.keys()
    keys = [fk for fk in keys if any(re.match(k, fk) for k in key_pattern)]

    fwd_hooks = [(k, set_corrupted_hook) for k in keys]
    bwd_hooks = [(k, save_gradient_hook) for k in keys]
    with model.input_dependent_hooks_context(*clean_inputs, fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
        with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
            for k in range(n_gradients_to_integrate):
                alpha = k / max(1, n_gradients_to_integrate - 1)

                # Check that computation is right
                assert 0.0 <= alpha <= 1.0
                if k == 0:
                    assert alpha == 0.0
                if n_gradients_to_integrate > 1 and alpha == n_gradients_to_integrate - 1:
                    assert alpha == 1.0

                loss_fn(interpolate_rnn_inputs(alpha, clean_inputs, corrupted_inputs)).backward()
                model.zero_grad()
    return attributions


def loss_fn(inputs):
    obs, init_carry, eps_start = inputs
    dist, _carry = model.get_distribution(obs, init_carry, eps_start)
    logits = dist.distribution.log_prob(clean_actions)
    bw_tensor = logits.sum()
    return bw_tensor


# TODO: modify this call so caches only contain keys that we want to look at (hook_h, hook_c, hook_ijfo, etc.)
attributions = add_attributions(
    {},
    loss_fn,
    model,
    clean_inputs=(clean_obs, zero_carry, eps_start),
    corrupted_inputs=(corrupted_obs, zero_carry, eps_start),
    clean_cache={k: v.detach() for k, v in clean_cache.items()},
    corrupted_minus_clean_cache={k: (v - corrupted_cache[k]).detach() for k, v in clean_cache.items()},
    ablate_at_every_hook=False,
    n_gradients_to_integrate=5,
)

# %%
mean_attributions = {}
mean_attributions_channels = {}
do_abs = True


def map_key(key):
    layer = int(key.split(".")[2])
    layer_type = key.split(".")[3][5:].upper()
    return f"L{layer}{layer_type}"


for k, v in attributions.items():
    if k.rsplit(".", 2)[0] in mean_attributions:
        mean_attributions[k.rsplit(".", 2)[0]] += v.abs() if do_abs else v
    else:
        mean_attributions[k.rsplit(".", 2)[0]] = v.abs() if do_abs else v
for k, v in mean_attributions.items():
    for c in range(v.shape[0]):
        mean_attributions_channels[map_key(k) + f"{c}"] = v[c].mean().item()

mean_attributions_channels = sorted(mean_attributions_channels.items(), key=lambda x: x[1], reverse=True)
# %%

envs = dataclasses.replace(boxo_cfg, num_envs=1).make()

out = play_level(
    envs,
    model,
    reset_opts=dict(level_file_idx=0, level_idx=0),
    thinking_steps=6,
)

cache = join_cache_across_steps([out.cache])
cache = {k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, 32, 10, 10) for k, v in cache.items() if len(v.shape) == 6}


def map_to_normal_key(short_key):
    return f"features_extractor.cell_list.{int(short_key[1])}.hook_{short_key[2].lower()}"


# %%
batch_no = 0

short_keys = [k for k, v in mean_attributions_channels[batch_no * 15 : (batch_no + 1) * 15]]

all_channels = np.stack([cache[map_to_normal_key(short_key)][:, int(short_key[3:])] for short_key in short_keys], axis=1)
obs = out.obs.squeeze(1)
# repeat obs 3 along first dimension
obs = obs.repeat_interleave(3, 0).numpy()
fig = plotly_feature_vis(all_channels, obs, feature_labels=short_keys, show=True)

# %%
plot_layer, plot_channel = 0, 23

keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "C", "I", "J", "F", "O"]]

all_channels = np.stack([cache[key][:, plot_channel] for key in keys], axis=1)
obs = out.obs.squeeze(1)
# repeat obs 3 along first dimension
obs = obs.repeat_interleave(3, 0).numpy()
fig = plotly_feature_vis(all_channels, obs, feature_labels=[k.rsplit(".")[-1] for k in keys], show=True)


# %%
envs = dataclasses.replace(boxo_cfg, num_envs=1).make()

size = 10
walls = [(i, 0) for i in range(size)]
walls += [(i, size - 1) for i in range(size)]
walls += [(0, i) for i in range(1, size - 1)]
walls += [(size - 1, i) for i in range(1, size - 1)]

walls += [(y, x) for y in range(4, 6) for x in range(3, 6)]

boxes = [(3, 2)]
targets = [(8, 8)]
player = (1, 1)


def prune_hook(inputs: th.Tensor, hook: HookPoint):
    return th.zeros_like(inputs)


fwd_hooks = [
    (f"features_extractor.cell_list.{layer}.{hook_name}.{pos}.{int_pos}", prune_hook)
    for pos in range(1)
    for int_pos in range(1)
    for layer in range(3)
    for hook_name in ["hook_input_h", "hook_input_c"]
]
toy_out = play_level(
    envs,
    model,
    reset_opts=dict(walls=walls, boxes=boxes, targets=targets, player=player),
    thinking_steps=0,
    fwd_hooks=fwd_hooks,
)

toy_cache = join_cache_across_steps([toy_out.cache])
toy_cache = {
    k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, 32, 10, 10) for k, v in toy_cache.items() if len(v.shape) == 6
}
toy_obs = toy_out.obs.squeeze(1)


boxes = [(7, 7)]
targets = [(1, 1)]
player = (8, 8)
toy_out2 = play_level(
    envs,
    model,
    reset_opts=dict(walls=walls, boxes=boxes, targets=targets, player=player),
    thinking_steps=0,
    fwd_hooks=fwd_hooks,
)
toy_cache2 = join_cache_across_steps([toy_out2.cache])
toy_cache2 = {
    k: np.transpose(v.squeeze(2), (1, 0, 2, 3, 4)).reshape(-1, 32, 10, 10) for k, v in toy_cache2.items() if len(v.shape) == 6
}
toy_obs2 = toy_out2.obs.squeeze(1)

toy_cache = {k: np.concatenate([v, toy_cache2[k]], axis=0) for k, v in toy_cache.items()}
toy_obs_non_rep = th.cat([toy_obs, toy_obs2], dim=0)
toy_obs = toy_obs_non_rep.repeat_interleave(3, 0).numpy()


# %%
batch_no = 2
short_keys = [k for k, v in mean_attributions_channels[batch_no * 15 : (batch_no + 1) * 15]]
toy_all_channels = np.stack(
    [toy_cache[map_to_normal_key(short_key)][:, int(short_key[3:])] for short_key in short_keys], axis=1
)
fig = plotly_feature_vis(toy_all_channels, toy_obs, feature_labels=short_keys, show=True)

# %%
plot_layer, plot_channel = 1, 25

keys = [f"features_extractor.cell_list.{plot_layer}.hook_{k.lower()}" for k in ["H", "C", "I", "J", "F", "O"]]

toy_all_channels_for_lc = np.stack([toy_cache[key][:, plot_channel] for key in keys], axis=1)
# repeat obs 3 along first dimension
fig = plotly_feature_vis(toy_all_channels_for_lc, toy_obs, feature_labels=[k.rsplit(".")[-1] for k in keys], show=True)

# %%
pio.renderers.default = "notebook"
batch_no = 0
full_keys = []
toy_all_channels = np.stack([toy_cache["features_extractor.hook_pre_model"][:, i] for i in range(31)], axis=1)
fig = plotly_feature_vis(toy_all_channels, toy_obs_non_rep, feature_labels=[f"{i}" for i in range(31)], show=True)
