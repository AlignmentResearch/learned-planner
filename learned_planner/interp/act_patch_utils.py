import itertools
from functools import partial
from typing import Callable

import numpy as np
import torch as th

from learned_planner.interp.render_svg import BOX, BOX_ON_TARGET, FLOOR, PLAYER, PLAYER_ON_TARGET, TARGET, WALL
from learned_planner.interp.utils import run_fn_with_cache

tile_dict = {
    "wall": WALL,
    "target": TARGET,
    "box_on_target": BOX_ON_TARGET,
    "box": BOX,
    "player": PLAYER,
    "player_on_target": PLAYER_ON_TARGET,
    "floor": FLOOR,
}


def skip_cache_key(key, skip_pre_model=True):
    if "fence" in key:
        return True
    if "mlp" in key:
        return True
    if "pool" in key:
        return True
    if skip_pre_model and "pre_model" in key:
        return True
    return False


def get_obs(level_file_idx, level_idx, envs):
    reset_opts = {"level_file_idx": level_file_idx, "level_idx": level_idx}
    obs, _ = envs.reset(options=reset_opts)
    return obs[0]


def corrupt_obs(obs: np.ndarray, y: int, x: int, tile: str) -> np.ndarray:
    """Corrupts all observations at position (y, x) with the given tile.

    Args:
        obs (np.ndarray): The observations to corrupt. Shape: (N, 3, 10, 10).
        y (int): The y-coordinate of the position to corrupt.
        x (int): The x-coordinate of the position to corrupt.
        tile (str): The tile to corrupt the position with.
            Can be one of: "wall", "target", "box_on_target", "box", "player", "player_on_target", "floor".

    Returns:
        np.ndarray: The corrupted observations.
    """
    obs = obs.copy()
    obs[:, :, y, x] = tile_dict[tile]
    return obs


def to_tensor(x: np.ndarray | th.Tensor) -> th.Tensor:
    if isinstance(x, np.ndarray):
        return th.tensor(x)
    return x


def sq_wise_intervention_hook(
    inp: th.Tensor,
    hook,
    sq_y: list[int] | slice,
    sq_x: list[int] | slice,
    channel: int | list[int],
    timestep: int,
    cache: dict[str, np.ndarray],
    cross_squares: bool = True,
    cross_channels: bool = True,
):
    # name = hook.name.rsplit(".", 2)[0]
    name = hook.name
    ret = inp.clone()
    if cross_squares:
        assert isinstance(sq_y, list) and isinstance(sq_x, list)
        for y, x in itertools.product(sq_y, sq_x):
            ret[..., channel, y, x] = to_tensor(cache[name][..., timestep, channel, y, x])
    else:
        if isinstance(channel, int) or not cross_channels:
            ret[..., channel, sq_y, sq_x] = to_tensor(cache[name][..., timestep, channel, sq_y, sq_x])
        else:
            for ch in channel:
                ret[..., ch, sq_y, sq_x] = to_tensor(cache[name][..., timestep, ch, sq_y, sq_x])
    return ret


def get_cache_and_probs(obs, model, fwd_hooks=None, hook_steps=-1, names_filter=None):
    if isinstance(obs, np.ndarray):
        obs = th.tensor(obs)
    if len(obs.shape) == 3:
        obs = obs.unsqueeze(0).unsqueeze(0)
    elif len(obs.shape) == 4:
        obs = obs.unsqueeze(0)
    num_envs = obs.shape[1]
    zero_carry = model.recurrent_initial_state(num_envs)
    eps_start = th.zeros((1, num_envs), dtype=th.bool)

    # (actions, values, log_probs, _), cache = model.run_with_cache(obs, zero_carry, eps_start)
    with th.no_grad():
        (distribution, state), cache = run_fn_with_cache(
            model,
            "get_distribution",
            obs,
            zero_carry,
            eps_start,
            # return_repeats=False,
            fwd_hooks=fwd_hooks if (hook_steps == -1) else None,
            names_filter=names_filter,
        )
        log_probs = distribution.distribution.logits
    return cache, log_probs


def mse_loss_fn(patched_log_probs: th.Tensor, clean_log_probs: th.Tensor, corrupted_log_probs: th.Tensor):
    return th.mean((patched_log_probs - clean_log_probs) ** 2) / th.mean((corrupted_log_probs - clean_log_probs) ** 2)


def activation_patching(
    corrupted_input: np.ndarray | th.Tensor,
    corrupted_log_probs: th.Tensor,
    clean_log_probs: th.Tensor,
    model,
    fwd_hooks,
):
    cache, log_probs = get_cache_and_probs(corrupted_input, model, fwd_hooks=fwd_hooks)
    loss = mse_loss_fn(log_probs, clean_log_probs, corrupted_log_probs)
    return loss, log_probs, cache


def activation_patching_sq_wise(
    hook_channel_sq_list: list[tuple[str, int | list[int], list[int] | slice, list[int] | slice, int]],
    corrupted_input: np.ndarray | th.Tensor,
    corrupted_log_probs: th.Tensor,
    clean_log_probs: th.Tensor,
    clean_cache: dict[str, np.ndarray],
    model,
    cross_squares: bool = True,
    cross_channels: bool = True,
    patching_hook: Callable = sq_wise_intervention_hook,
):
    fwd_hooks = [
        (
            key if "pre_model" in key else key + f".0.{int_step}",
            partial(
                patching_hook,
                sq_y=sq_y,
                sq_x=sq_x,
                channel=channel,
                timestep=0,
                cache=clean_cache,
                cross_squares=cross_squares,
                cross_channels=cross_channels,
            ),
        )
        for key, channel, sq_y, sq_x, int_step in hook_channel_sq_list
    ]

    cache, log_probs = get_cache_and_probs(corrupted_input, model, fwd_hooks=fwd_hooks)
    loss = mse_loss_fn(log_probs, clean_log_probs, corrupted_log_probs)
    return loss, log_probs, cache
