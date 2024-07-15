import dataclasses
from pathlib import Path

from learned_planner.configs.command_config import WandbCommandConfig
from learned_planner.configs.env_sokoban import envpool_sokoban_103
from learned_planner.configs.misc import DEFAULT_TRAINING, random_seed
from learned_planner.configs.train_drc import DeviceLiteral, drc_1_1, recurrent_ppo_103
from learned_planner.evaluate import EvaluateConfig


def eval_command(
    device: DeviceLiteral,
    training_mount: Path,
    n_envs: int = 256,
    max_episode_steps: int = 120,
    *,
    seed: int = random_seed(),
):
    return WandbCommandConfig(
        base_save_prefix=training_mount,
        cmd=EvaluateConfig(
            policy=drc_1_1(),
            total_timesteps=int(3e10),
            alg=recurrent_ppo_103(device=device),
            eval_env=dataclasses.replace(
                envpool_sokoban_103(
                    training_mount / ".sokoban_cache",
                    n_envs=n_envs,
                    max_episode_steps=max_episode_steps,
                )[1],  # [1] use the validation split
                reward_step=-0.1,
                difficulty="medium",
                n_envs_to_render=min(64, n_envs),
            ),
            device=device,
            record_video=True,
            n_eval_episodes=5120,
            n_steps=max_episode_steps,
            n_eval_steps=240,
            load_path=None,
            seed=seed,
            eval_steps_to_think=[0, 2, 4, 8, 12, 16, 24],
        ),
    )


# fmt: off
def eval_local(): return eval_command("cpu", Path("."))
def eval_cluster(): return eval_command("cuda", DEFAULT_TRAINING)
# fmt: on
