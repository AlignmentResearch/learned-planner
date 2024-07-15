import abc
import contextlib
import copy
import dataclasses
import logging
import math
import random
import zipfile
from pathlib import Path
from typing import ClassVar, Generic, Literal, Optional, Sequence, Tuple, TypeVar

import torch
from stable_baselines3 import PPO, RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.recurrent.buffers import SamplingType
from stable_baselines3.common.recurrent.policies import (
    BaseRecurrentActorCriticPolicy,
    RecurrentActorCriticPolicy,
)
from stable_baselines3.common.save_util import json_to_data
from stable_baselines3.common.type_aliases import check_cast, non_null
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecTransposeImage

from learned_planner.common import (
    PrefixEvalCallback,
    WandbVecVideoRecorder,
    catch_different_env_types_warning,
)
from learned_planner.convlstm import ConvLSTMOptions
from learned_planner.environments import (
    BaseSokobanEnvConfig,
    EnvConfig,
    EnvpoolVecEnvConfig,
    NNRewardVecEnv,
    NNRewardVecEnvConfig,
)
from learned_planner.optimizers import (
    AdamOptimizerConfig,
    BaseLRSchedule,
    BaseOptimizerConfig,
    FlatLRSchedule,
)
from learned_planner.policies import (
    BasePolicyConfig,
    BaseRecurrentPolicyConfig,
    BatchedRewardPolicyConfig,
    RewardToyModel,
)
from learned_planner.wandb_logger import configure_logger

DEFAULT_N_ENVS = 2

log = logging.getLogger(__name__)

PPOOrRecurrentPPOT = TypeVar("PPOOrRecurrentPPOT", PPO, RecurrentPPO)


@dataclasses.dataclass
class GenericPPOConfig(abc.ABC, Generic[PPOOrRecurrentPPOT]):
    _alg_class: ClassVar[type[PPO] | type[RecurrentPPO]]

    optimizer: BaseOptimizerConfig = dataclasses.field(default_factory=AdamOptimizerConfig)
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float | BaseLRSchedule = dataclasses.field(default_factory=lambda: FlatLRSchedule(0.2))
    clip_range_vf: float | BaseLRSchedule | None = None
    normalize_advantage: bool = True
    ent_coef: float | BaseLRSchedule = 0.01
    vf_coef: float | BaseLRSchedule = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1  # If -1, only sample at beginning of rollout.
    target_kl: Optional[float] = None

    def _make(
        self,
        alg_class: type[PPOOrRecurrentPPOT],
        policy: type[ActorCriticPolicy] | type[RecurrentActorCriticPolicy] | str,
        env: VecEnv,
        n_steps: int,
        seed: int,
        device: torch.device,
        policy_kwargs: dict | None,
        # accommodate variable args: batch_size or (batch_time, batch_envs)
        **kwargs,
    ) -> PPOOrRecurrentPPOT:
        policy_kwargs = (policy_kwargs or {}).copy()
        policy_kwargs.update(self.optimizer.policy_kwargs())

        return alg_class(
            policy=policy,  # type: ignore
            env=env,
            learning_rate=self.optimizer.lr,
            n_steps=n_steps,
            **kwargs,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            clip_range_vf=self.clip_range_vf,
            normalize_advantage=self.normalize_advantage,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            use_sde=self.use_sde,
            sde_sample_freq=self.sde_sample_freq,
            target_kl=self.target_kl,
            tensorboard_log=None,
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
        )

    @abc.abstractmethod
    def make(
        self,
        policy: type[ActorCriticPolicy] | type[RecurrentActorCriticPolicy] | str,
        env: VecEnv,
        n_steps: int,
        seed: int,
        device: torch.device,
        policy_kwargs: dict | None,
    ) -> PPOOrRecurrentPPOT:
        ...


@dataclasses.dataclass
class PPOConfig(GenericPPOConfig[PPO]):
    _alg_class = PPO
    batch_size: int = 128

    def make(
        self,
        policy: type[ActorCriticPolicy] | type[RecurrentActorCriticPolicy] | str,
        env: VecEnv,
        n_steps: int,
        seed: int,
        device: torch.device,
        policy_kwargs: dict | None,
        # Avoid adding kwargs on purpose -- these are all the args that PPOConfig accepts.
    ) -> PPO:
        return self._make(
            PPO,
            policy=policy,
            env=env,
            n_steps=n_steps,
            seed=seed,
            device=device,
            policy_kwargs=policy_kwargs,
            batch_size=self.batch_size,
        )


@dataclasses.dataclass
class RecurrentPPOConfig(GenericPPOConfig[RecurrentPPO]):
    _alg_class = RecurrentPPO
    batch_time: Optional[int] = 20
    batch_envs: int = 128
    sampling_type: SamplingType = SamplingType.CLASSIC

    def make(
        self,
        policy: type[ActorCriticPolicy] | type[RecurrentActorCriticPolicy] | str,
        env: VecEnv,
        n_steps: int,
        seed: int,
        device: torch.device,
        policy_kwargs: dict | None,
        # Avoid adding kwargs on purpose -- these are all the args that RecurrentPPOConfig accepts.
    ) -> RecurrentPPO:
        return self._make(
            RecurrentPPO,
            policy=policy,
            env=env,
            n_steps=n_steps,
            seed=seed,
            device=device,
            policy_kwargs=policy_kwargs,
            batch_time=self.batch_time,
            batch_envs=self.batch_envs,
            sampling_type=self.sampling_type,
        )


@dataclasses.dataclass
class ABCCommandConfig(abc.ABC):
    device: Literal["cuda", "cpu", "auto"] = "auto"  # Device to use
    seed: int = random.randint(0, 2**31 - 1)  # Random seed (default: random)

    @property
    def th_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @abc.abstractmethod
    def run(self, run_dir: Path):
        ...


@dataclasses.dataclass
class BaseCommandConfig(ABCCommandConfig):
    """Common arguments to train and evaluate."""

    policy: BasePolicyConfig = dataclasses.field(default_factory=BatchedRewardPolicyConfig)
    total_timesteps: int = 10000000000  # Total amount of time to train for
    alg: GenericPPOConfig = dataclasses.field(default_factory=PPOConfig)
    env: EnvConfig = dataclasses.field(default_factory=NNRewardVecEnvConfig)
    eval_env: Optional[EnvConfig] = None  # Evaluation environment. If None, use the same as env
    hidden_state_init: str = "random"  # Hidden state initialization method
    fully_observable: bool = False  # Fully observable environment
    verbose: int = 0  # Verbosity level
    record_video: bool = True  # Record video
    n_eval_episodes: int = DEFAULT_N_ENVS  # Number of evaluation episodes
    n_steps: int = 40  # Number of steps
    n_eval_steps: int = 80  # Number of evaluation steps
    load_path: Path | None = None  # Path to load from
    reward_fn_index: int = 1  # Reward function index
    eval_steps_to_think: list[int] = dataclasses.field(
        default_factory=lambda: [0]
    )  # Number of steps to think used in evaluation
    test_that_eval_split_is_validation: bool = False

    def __post_init__(self):
        if self.eval_env is None:
            self.eval_env = copy.deepcopy(self.env)
        assert isinstance(self.eval_env, type(self.env))
        if isinstance(self.eval_env, EnvpoolVecEnvConfig):
            self.eval_env = dataclasses.replace(
                self.eval_env,
                max_episode_steps=self.n_eval_steps,
                min_episode_steps=self.n_eval_steps,
            )
        self.eval_env.n_envs = min(self.eval_env.n_envs, self.n_eval_episodes)
        assert 0 in self.eval_steps_to_think, "0 should be in eval_steps_to_think"

    @abc.abstractmethod
    def run(self, run_dir: Path):
        ...


@dataclasses.dataclass
class TrainConfig(BaseCommandConfig):
    checkpoint_freq: int = 100000

    def run(self, run_dir: Path):
        return train(self, run_dir)


def make_with_updated_model(reward_fn: torch.nn.Module):
    def eq(a, b):
        return torch.equal(a.cpu(), b.cpu())

    @contextlib.contextmanager
    def with_updated_model(pec: PrefixEvalCallback):
        fex = pec.model.policy.features_extractor
        if isinstance(fex, RewardToyModel):
            if all(eq(a, b) for a, b in zip(fex.reward_fn.parameters(), reward_fn.parameters())):
                log.debug(f"For pec={pec.prefix} parameters are the same, not updating")
                yield
            else:
                orig_reward_fn = copy.deepcopy(fex.reward_fn)
                fex.reset_reward(reward_fn)
                assert all(eq(a, b) for a, b in zip(fex.reward_fn.parameters(), reward_fn.parameters()))
                assert not all(eq(a, b) for a, b in zip(fex.reward_fn.parameters(), orig_reward_fn.parameters()))

                log.debug(f"For pec={pec.prefix} parameters are different")
                yield

                fex.reset_reward(orig_reward_fn)
                assert not all(eq(a, b) for a, b in zip(fex.reward_fn.parameters(), reward_fn.parameters()))
                assert all(eq(a, b) for a, b in zip(fex.reward_fn.parameters(), orig_reward_fn.parameters()))
        else:
            yield

    return with_updated_model


def create_vec_env_and_eval_callbacks(
    args: BaseCommandConfig, run_dir: Path, eval_freq: int, save_model: bool = True
) -> Tuple[VecEnv, list[PrefixEvalCallback]]:
    eval_callbacks: list[PrefixEvalCallback] = []
    if isinstance(args.env, NNRewardVecEnvConfig):
        reward_model_path = Path(__file__).parent / "notebooks"
        reward_fn = torch.load(reward_model_path / "hard_reward.pt", map_location=args.th_device)
        reward_fn_2 = torch.load(reward_model_path / "hard_reward_2.pt", map_location=args.th_device)

        env_kwargs = dict(
            device=args.th_device,
            hidden_state_init=args.hidden_state_init,
            fully_observable=args.fully_observable,
            # Not setting this causes weird jagged loss curves, probably because the rollout buffer includes several
            # truncated parts of episodes
        )
        assert args.reward_fn_index in [1, 2]
        vec_env = NNRewardVecEnv(
            **env_kwargs,  # type: ignore
            split="train",
            reward_fn=(reward_fn, reward_fn_2)[args.reward_fn_index - 1],
            max_episode_steps=args.n_steps,
        )

        eval_env_kwargs = dict(
            **env_kwargs,
            max_episode_steps=args.n_eval_steps,
        )

        for name, rfn in [("reward_fn", reward_fn), ("reward_fn_2", reward_fn_2)]:
            for is_test in [True, False]:
                split = "test" if is_test else "train"
                eval_env = NNRewardVecEnv(**eval_env_kwargs, split=split, reward_fn=rfn)  # type: ignore

                log_dir_key = f"videos/{str(split)}/{name}"
                base_log_dir = run_dir / log_dir_key

                if args.record_video:
                    eval_env = WandbVecVideoRecorder(eval_env, section_name=log_dir_key)
                eval_callbacks.append(
                    PrefixEvalCallback(
                        eval_env,
                        base_log_path=base_log_dir,
                        prefix=log_dir_key,
                        with_updated_model=make_with_updated_model(rfn),
                        eval_freq=eval_freq,
                        n_eval_episodes=args.n_eval_episodes,
                        log_path=None,
                        best_model_save_path=str(base_log_dir) if save_model else None,
                        deterministic=False,
                        render=False,
                        verbose=args.verbose,
                        warn=False,  # Don't care if env not wrapped in a Monitor/VecMonitor
                    )
                )
    else:
        log_dir_key = "train"
        base_log_dir = run_dir / "train"

        if isinstance(args.env, EnvpoolVecEnvConfig):
            assert isinstance(args.eval_env, EnvpoolVecEnvConfig)
            vec_env = args.env.make(device=args.th_device)
        else:
            env_cfg = check_cast(BaseSokobanEnvConfig, args.env)
            env_cfg = dataclasses.replace(env_cfg, max_episode_steps=args.n_steps)
            if env_cfg.n_envs > 1:
                vec_env = VecTransposeImage(SubprocVecEnv([env_cfg.make] * env_cfg.n_envs))
            else:
                vec_env = VecTransposeImage(DummyVecEnv([env_cfg.make]))

        if isinstance(args.eval_env, EnvpoolVecEnvConfig):
            eval_env = args.eval_env.make(device=args.th_device)
            if args.record_video:
                eval_env = WandbVecVideoRecorder(eval_env, section_name=log_dir_key)
        else:
            eval_env_config = check_cast(BaseSokobanEnvConfig, args.eval_env)
            if eval_env_config.n_envs > 1:
                eval_env = SubprocVecEnv([eval_env_config.make] * eval_env_config.n_envs)
            else:
                eval_env = DummyVecEnv([eval_env_config.make])
            if args.record_video:
                eval_env = WandbVecVideoRecorder(eval_env, section_name=log_dir_key)
            eval_env = VecTransposeImage(eval_env)

        logger = None
        for n_steps in args.eval_steps_to_think:
            eval_callbacks.append(
                PrefixEvalCallback(
                    eval_env=eval_env,
                    base_log_path=base_log_dir,
                    prefix=f"{log_dir_key}/{n_steps:02d}_steps",
                    eval_freq=eval_freq,
                    n_eval_episodes=args.n_eval_episodes,
                    log_path=None,
                    best_model_save_path=str(base_log_dir) if (save_model and n_steps == 0) else None,
                    deterministic=False,
                    render=False,
                    verbose=args.verbose,
                    warn=False,
                    n_steps_to_think=n_steps,
                    logger=logger,
                )
            )
            logger = eval_callbacks[-1].logger

    return vec_env, eval_callbacks


def make_model(
    args: BaseCommandConfig, run_dir: Path, vec_env: VecEnv, eval_callbacks: Sequence[PrefixEvalCallback]
) -> BaseAlgorithm:
    alg_class = args.alg._alg_class

    if isinstance(args.policy, BaseRecurrentPolicyConfig) and (not issubclass(alg_class, RecurrentPPO)):
        raise ValueError(f"Recurrent policy {args.policy} requires RecurrentPPO algorithm")

    if not isinstance(args.policy, BaseRecurrentPolicyConfig) and alg_class is not PPO:
        raise ValueError(f"Recurrent policy {args.policy} requires PPO algorithm")

    policy, policy_kwargs = args.policy.policy_and_kwargs(vec_env)

    assert isinstance(policy, str) or issubclass(policy, ActorCriticPolicy)
    if issubclass(alg_class, RecurrentPPO):
        assert isinstance(policy, str) or issubclass(policy, BaseRecurrentActorCriticPolicy)

    model = args.alg.make(
        policy=policy,
        env=vec_env,
        seed=args.seed,
        device=args.th_device,
        n_steps=args.n_steps,
        policy_kwargs=policy_kwargs,
    )

    if isinstance(args.policy.features_extractor, ConvLSTMOptions):
        if args.policy.features_extractor.fancy_init:
            # Initialize in a fancy way the parts of the policy which are not inside the ConvLSTMFeaturesExtractor

            for p in model.policy.mlp_extractor.parameters():
                # Types Pytorch bug. Gain is an int but it should be a float.
                if p.ndim == 1:
                    torch.nn.init.normal_(p, std=torch.nn.init.calculate_gain("tanh"))  # type: ignore
                else:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(p)
                    torch.nn.init.orthogonal_(p, gain=torch.nn.init.calculate_gain("tanh") / math.sqrt(fan_in))  # type: ignore

            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(model.policy.action_net.weight)
            # Parameter variance should be 1/fan_in^2, because there's a softmax afterwards
            torch.nn.init.orthogonal_(model.policy.action_net.weight, gain=1.0 / fan_in)

            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(model.policy.value_net.weight)
            # A stddev of 1.0 for the output seems about right for the value scales we use (-.1 per step)
            torch.nn.init.orthogonal_(model.policy.value_net.weight, gain=1.0 / math.sqrt(fan_in))  # type: ignore

    if args.load_path is not None:
        model.set_parameters(args.load_path, exact_match=True, device=args.device)

        with zipfile.ZipFile(args.load_path) as model_zip:
            json_data = model_zip.read("data").decode()
            saved_model_attrs = json_to_data(json_data, custom_objects=None)

            # Copy some important attributes from the old model
            for attr in ["num_timesteps", "_num_timesteps_at_start", "_last_obs", "_n_updates"]:
                setattr(model, attr, saved_model_attrs[attr])

            for optional_attr in ["_last_lstm_states"]:
                if optional_attr in saved_model_attrs:
                    setattr(model, optional_attr, saved_model_attrs[optional_attr])

    postprocess_model_and_envs(args, run_dir, model, vec_env, eval_callbacks)
    return model


def postprocess_model_and_envs(
    args: BaseCommandConfig, run_dir: Path, model: BaseAlgorithm, vec_env: VecEnv, eval_callbacks: Sequence[PrefixEvalCallback]
):
    sb3_logger = configure_logger(run_dir, "train_process", verbose=args.verbose)
    model.set_logger(sb3_logger)

    if (comp := args.policy.features_extractor.compile) is not None:
        model.policy.features_extractor = comp(model.policy.features_extractor)  # type: ignore[assignment]

    assert model.policy is not None
    if isinstance(model.policy.features_extractor, RewardToyModel):
        log.debug("re-setting the reward_fn")
        env = check_cast(NNRewardVecEnv, vec_env.unwrapped)
        # Reset reward to the correct parameters -- they get randomized by `alg_class`
        model.policy.features_extractor.reset_reward(env.reward_fn)

        # Set the policies so that the rendering can show where they query the reward.
        env = check_cast(NNRewardVecEnv, vec_env.unwrapped)
        env.policy_features = model.policy.features_extractor

        for callback in eval_callbacks:
            env = check_cast(NNRewardVecEnv, callback.eval_env.unwrapped)
            env.policy_features = model.policy.features_extractor


def init_learning_staggered_environment(args: BaseCommandConfig, model: BaseAlgorithm, *, max_episode_step_multiple: int = 50):
    """Init model learning, and stagger the episodes so that they don't all start at the same time."""
    model.learn(total_timesteps=0, reset_num_timesteps=False)
    vec_env = non_null(model.env)

    env_action = torch.as_tensor(vec_env.action_space.sample())
    vec_env_action = env_action.expand((args.env.n_envs, *env_action.shape))
    for _ in range(args.env.max_episode_steps * max_episode_step_multiple):
        model._last_obs, _reward, model._last_episode_starts, _info = vec_env.step(vec_env_action)


def train(args: TrainConfig, run_dir: Path):
    """Trains a policy that uses algorithm `args.alg` and policy `args.policy` in the environment `args.env`."""
    vec_env, eval_callbacks = create_vec_env_and_eval_callbacks(args, run_dir, eval_freq=args.checkpoint_freq)
    model = make_model(args, run_dir, vec_env, eval_callbacks)

    if args.test_that_eval_split_is_validation:
        assert eval_callbacks[0].eval_env.cfg.split == "valid"

    cp_callback = CheckpointCallback(args.checkpoint_freq, str(run_dir), verbose=1)

    with catch_different_env_types_warning(vec_env, eval_callbacks):
        init_learning_staggered_environment(args, model)
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[
                cp_callback,
                *eval_callbacks,
            ],
            reset_num_timesteps=False,
        )
    model.logger.close()
