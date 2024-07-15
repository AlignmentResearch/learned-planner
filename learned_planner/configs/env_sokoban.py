import dataclasses
from pathlib import Path

from learned_planner.configs.misc import DEFAULT_TRAINING, random_seed
from learned_planner.environments import EnvpoolSokobanVecEnvConfig

DEFAULT_BOXOBAN_CACHE: Path = DEFAULT_TRAINING / ".sokoban_cache"


def envpool_sokoban(
    boxoban_cache: Path = DEFAULT_BOXOBAN_CACHE,
    max_episode_steps: int = 120,
    n_envs: int = 2,
    n_eval_episodes: int = 2,
    *,
    seed: int = random_seed(),
) -> tuple[EnvpoolSokobanVecEnvConfig, EnvpoolSokobanVecEnvConfig]:
    train_cfg = EnvpoolSokobanVecEnvConfig(
        n_envs=n_envs,
        seed=seed,
        px_scale=4,
        reward_finished=10.0,
        reward_box=1.0,
        reward_step=-0.1,
        cache_path=boxoban_cache,
        split="train",
        difficulty="unfiltered",
        max_episode_steps=max_episode_steps,
        min_episode_steps=max_episode_steps * 3 // 4,
    )
    valid_cfg = dataclasses.replace(
        train_cfg,
        split="valid",
        load_sequentially=True,
        n_levels_to_load=n_eval_episodes,
    )
    return train_cfg, valid_cfg


def envpool_sokoban_103(
    boxoban_cache: Path = DEFAULT_BOXOBAN_CACHE,
    max_episode_steps: int = 120,
    n_envs: int = 2,
    n_eval_episodes: int = 2,
    *,
    seed: int = random_seed(),
) -> tuple[EnvpoolSokobanVecEnvConfig, EnvpoolSokobanVecEnvConfig]:
    train_cfg = EnvpoolSokobanVecEnvConfig(
        n_envs=n_envs,
        seed=seed,
        px_scale=4,
        reward_finished=10.0,
        reward_box=1.0,
        reward_step=0.0,
        cache_path=boxoban_cache,
        split="train",
        difficulty="unfiltered",
        max_episode_steps=max_episode_steps,
        min_episode_steps=max_episode_steps // 2,
    )
    valid_cfg = dataclasses.replace(
        train_cfg,
        split="valid",
        load_sequentially=True,
        n_levels_to_load=n_eval_episodes,
    )
    return train_cfg, valid_cfg
