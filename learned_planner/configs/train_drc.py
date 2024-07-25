import dataclasses
from pathlib import Path
from typing import Literal

from stable_baselines3.common.recurrent.buffers import SamplingType

from learned_planner.activation_fns import ReLUConfig
from learned_planner.configs.command_config import WandbCommandConfig
from learned_planner.configs.env_sokoban import envpool_sokoban, envpool_sokoban_103
from learned_planner.configs.misc import DEFAULT_TRAINING, random_seed
from learned_planner.convlstm import CompileConfig, ConvConfig, ConvLSTMCellConfig, ConvLSTMOptions
from learned_planner.optimizers import AdamOptimizerConfig, PolynomialLRSchedule
from learned_planner.policies import ConvLSTMPolicyConfig, NetArchConfig
from learned_planner.train import RecurrentPPOConfig, TrainConfig


def conv_lstm_policy(
    n_recurrent_layers: int = 3, repeats_per_step: int = 3, rnn_hidden_channels: int = 32, mlp_hidden: int = 256
) -> ConvLSTMPolicyConfig:
    n_non_recurrent_layers = 2
    return ConvLSTMPolicyConfig(
        features_extractor=ConvLSTMOptions(
            compile=CompileConfig(),
            embed=[ConvConfig(features=rnn_hidden_channels, kernel_size=3) for _ in range(n_non_recurrent_layers)],
            recurrent=ConvLSTMCellConfig(ConvConfig(features=rnn_hidden_channels, kernel_size=3)),
            n_recurrent=n_recurrent_layers,
            repeats_per_step=repeats_per_step,
            pre_model_nonlin=ReLUConfig(),
        ),
        net_arch=NetArchConfig([mlp_hidden], [mlp_hidden]),
    )


# fmt: off
def drc_3_3(): return conv_lstm_policy(3, 3, rnn_hidden_channels=32, mlp_hidden=256)
def drc_1_1(): return conv_lstm_policy(1, 1, rnn_hidden_channels=32, mlp_hidden=256)
# fmt: on

DeviceLiteral = Literal["cuda", "cpu"]


def adam_optimizer(device: DeviceLiteral = "cuda", lr: float = 2.5e-4):
    return AdamOptimizerConfig(
        lr=PolynomialLRSchedule(lr=lr, power=1, baseline=lr * 0.1),
        eps=1e-4,
        fused=(device == "cuda"),
    )


def recurrent_ppo(device: DeviceLiteral = "cuda", lr_base: float = 2.5e-4, batch_envs: int = 256, batch_time: int = 20):
    return RecurrentPPOConfig(
        optimizer=adam_optimizer(device=device, lr=lr_base * (batch_envs / 32)),
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.1,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=1.0,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=0.01,
        batch_time=batch_time,
        batch_envs=batch_envs,
        max_grad_norm=2.0,
    )


def recurrent_ppo_103(device: DeviceLiteral = "cuda", lr_base: float = 2.5e-4, batch_envs: int = 256, batch_time: int = 20):
    return RecurrentPPOConfig(
        optimizer=adam_optimizer(device=device, lr=lr_base * (batch_envs / 32)),
        n_epochs=4,
        gamma=0.97,
        gae_lambda=0.95,
        clip_range=0.06,
        clip_range_vf=1e-3,
        normalize_advantage=False,
        ent_coef=0.01,
        vf_coef=1.0,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=0.02,
        batch_time=batch_time,
        batch_envs=batch_envs,
        max_grad_norm=0.5,
        sampling_type=SamplingType.CLASSIC,
    )


def train_command(
    device: DeviceLiteral,
    training_mount: Path,
    n_envs: int = 512,
    max_episode_steps: int = 120,
    *,
    seed: int = random_seed(),
):
    train_cfg, eval_cfg = envpool_sokoban(
        training_mount / ".sokoban_cache",
        max_episode_steps=max_episode_steps,
        n_envs=n_envs,
        n_eval_episodes=n_envs,
    )
    return WandbCommandConfig(
        base_save_prefix=training_mount,
        cmd=TrainConfig(
            policy=drc_3_3(),
            total_timesteps=int(3e10),
            checkpoint_freq=600_000,
            alg=recurrent_ppo(device=device),
            env=train_cfg,
            eval_env=eval_cfg,
            device=device,
            record_video=False,
            n_eval_episodes=n_envs,
            n_steps=max_episode_steps,
            n_eval_steps=240,
            load_path=None,
            seed=seed,
        ),
    )


def train_command_103(
    device: DeviceLiteral,
    training_mount: Path,
    n_envs: int = 512,
    max_episode_steps: int = 120,
    *,
    seed: int = random_seed(),
):
    train_env, eval_env = envpool_sokoban_103(
        training_mount / "sokoban_cache2", max_episode_steps=max_episode_steps, n_envs=n_envs, n_eval_episodes=n_envs
    )
    return WandbCommandConfig(
        base_save_prefix=training_mount,
        cmd=TrainConfig(
            policy=drc_1_1(),
            total_timesteps=int(3e10),
            checkpoint_freq=300_000,
            alg=recurrent_ppo_103(device=device),
            env=train_env,
            eval_env=eval_env,
            device=device,
            record_video=False,
            n_eval_episodes=n_envs,
            n_steps=max_episode_steps,
            n_eval_steps=240,
            load_path=None,
            seed=seed,
        ),
    )


def train_command_114(
    device: DeviceLiteral,
    training_mount: Path,
    n_envs: int = 512,
    max_episode_steps: int = 120,
    *,
    seed: int = random_seed(),
):
    train_env, eval_env = envpool_sokoban_103(
        training_mount / ".sokoban_cache", max_episode_steps=max_episode_steps, n_envs=n_envs, n_eval_episodes=n_envs
    )
    train_env.reward_step = -0.1
    eval_env.seed = 1234
    return WandbCommandConfig(
        base_save_prefix=training_mount,
        cmd=TrainConfig(
            policy=drc_1_1(),
            total_timesteps=int(1.5e9),
            checkpoint_freq=100_000,
            alg=dataclasses.replace(
                recurrent_ppo_103(device=device, batch_envs=256),
                clip_range_vf=0.1,
                optimizer=AdamOptimizerConfig(
                    lr=PolynomialLRSchedule(lr=2e-3, baseline=2e-4),
                    betas=(0.9, 0.95),
                    amsgrad=True,
                    fused=(device == "cuda"),
                    eps=1e-4,
                ),
            ),
            env=train_env,
            eval_env=eval_env,
            device=device,
            record_video=False,
            n_eval_episodes=n_envs,
            n_steps=max_episode_steps,
            n_eval_steps=240,
            load_path=None,
            seed=seed,
        ),
    )


# fmt: off
def train_local(): return train_command("cpu", Path("."))
def train_cluster(): return train_command("cuda", DEFAULT_TRAINING)
def train_local_103(): return train_command_103("cpu", Path("."))
def train_cluster_103(): return train_command_103("cuda", DEFAULT_TRAINING)
def train_local_114(): return train_command_114("cpu", Path("."))
def train_cluster_114(): return train_command_114("cuda", DEFAULT_TRAINING)

# fmt: on
