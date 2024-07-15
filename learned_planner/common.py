import contextlib
import logging
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.type_aliases import non_null
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, unwrap_vec_wrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn

from learned_planner.environments import EnvpoolSokobanVecEnvConfig, EnvpoolVecEnv
from learned_planner.wandb_logger import WandBOutputFormat, configure_logger

T = TypeVar("T")

log = logging.getLogger(__name__)


class WandbVecVideoRecorder(VecEnvWrapper):
    """
    Saves video of the environment it wraps to Wandb
    """

    section_name: str
    frames: list[np.ndarray]
    fps: int

    def __init__(
        self,
        venv: VecEnv,
        section_name: str,
        fps: int = 30,
        observation_space: Optional[gym.spaces.Space] = None,
        action_space: Optional[gym.spaces.Space] = None,
    ):
        super().__init__(venv, observation_space, action_space)
        self.section_name = section_name
        self.frames = []
        self.fps = fps

    def reset(self) -> VecEnvObs:
        self.save_and_reset_video()
        return self.venv.reset()

    def save_and_reset_video(self) -> None:
        if self.frames:
            # Only record when we've done more than 1 step
            self.frames.append(non_null(self.venv.render()).detach().cpu().numpy())
            data = np.moveaxis(np.asarray(self.frames).astype(np.uint8), 3, 1)
            assert data.shape[1] == 3, f"Expected 3 color channels, got {data.shape[1]}"
            wandb.log({self.section_name: wandb.Video(data_or_path=data, fps=self.fps, format="mp4")})
            log.debug(f"Sent video to section {self.section_name} with {len(self.frames)} frames.")

        self.frames.clear()

    def step_wait(self) -> VecEnvStepReturn:
        out = self.venv.step_wait()
        self.frames.append(non_null(self.venv.render()).detach().cpu().numpy())
        return out


@contextlib.contextmanager
def catch_different_env_types_warning(vec_env: VecEnv, eval_callbacks: Sequence[EvalCallback]):
    with warnings.catch_warnings():
        # Issue in mpl_toolkits and sphinxcontrib
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="Deprecated call to `pkg_resources.declare_namespace",
            module="pkg_resources",
        )
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message="pkg_resources is deprecated as an API",
            module="imageio_ffmpeg",
        )

        def _does_wrapper_contain_wandb_video_vec_env(wrapper: VecEnv) -> bool:
            """
            Whether the wrapper contains a WandbVecVideoRecorder, which is immediately wrapping an env of the same
            type as `vec_env`.
            """
            video_recorder = unwrap_vec_wrapper(wrapper, WandbVecVideoRecorder)
            if video_recorder is None:
                return False
            if not isinstance(vec_env, type(video_recorder.venv)):
                return False
            return True

        if all(_does_wrapper_contain_wandb_video_vec_env(cb.eval_env) for cb in eval_callbacks):
            # This warning is not relevant if the eval_envs are just wrapping the training env
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="Training and eval env are not of the same type",
                module="stable_baselines3",
            )

        yield


class PrefixEvalCallback(EvalCallback):
    """
    Like EvalCallback, but:
    - logs everything to wandb under `base_log_path`
    - runs the eval_env inside the `with_updated_model` context manager
    - makes sure to call .reset() after an evaluation step if `eval_env` is a WandbVecVideoRecorder, so the
        evaluation video gets saved.
    - Maybe gives the model steps to think before evaluating
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        base_log_path: Path,
        prefix: str,
        with_updated_model: Optional[Callable[["PrefixEvalCallback"], ContextManager]] = None,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        logger: Optional[Logger] = None,
        n_steps_to_think: int = 0,
    ):
        super().__init__(
            eval_env,
            callback_on_new_best,
            callback_after_eval,
            n_eval_episodes,
            eval_freq,
            log_path,
            best_model_save_path,
            deterministic,
            render,
            verbose,
            warn,
        )
        self.n_steps_to_think = n_steps_to_think
        self.verbose = verbose
        self.base_log_path = base_log_path
        self.prefix = prefix
        self.with_updated_model = with_updated_model or (lambda _: contextlib.nullcontext())
        self._logger = logger

    @property
    def logger(self):
        if self._logger is None:
            self._logger = configure_logger(self.base_log_path, self.prefix, verbose=self.verbose)
        return self._logger

    def _on_step(self) -> bool:
        # Update the prefix in case common logger is used across multiple callbacks
        for output_format in self.logger.output_formats:
            if hasattr(output_format, "prefix"):
                assert isinstance(output_format, WandBOutputFormat)
                output_format.prefix = self.prefix
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            with self.with_updated_model(self):
                out = super()._on_step()
            # Make sure to save the video
            if (eval_env := unwrap_vec_wrapper(self.eval_env, WandbVecVideoRecorder)) is not None:
                eval_env.save_and_reset_video()
            return out

        return super()._on_step()

    def _on_training_end(self) -> None:
        self.logger.close()
        self.training_env.close()
        self.eval_env.close()
        return super()._on_training_end()

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        if locals_["done"]:
            if isinstance(env := self.eval_env.unwrapped, EnvpoolVecEnv) and isinstance(
                cfg := env.cfg, EnvpoolSokobanVecEnvConfig
            ):
                last_step_reward: float = cfg.reward_step + cfg.reward_box + cfg.reward_finished

                # Check that this steps' reward is equal to the reward of the last step if the episode is successful.
                # This is NOT the same as the cumulative reward of the episode.
                maybe_is_success = bool(locals_["reward"] == last_step_reward)
                self._is_success_buffer.append(maybe_is_success)
            else:
                super()._log_success_callback(locals_, globals_)

    def _evaluate_policy(self) -> Union[tuple[float, float], tuple[list[float], list[int]]]:
        print(f"Calling evalute_policy with {self.n_steps_to_think=}")
        if (
            isinstance(env := self.eval_env.unwrapped, EnvpoolVecEnv)
            and isinstance(cfg := env.cfg, EnvpoolSokobanVecEnvConfig)
            and cfg.load_sequentially
        ):
            # need to re-create the env to load from the start
            self.eval_env = cfg.make(device=env.device)
        return evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=self._log_success_callback,
            n_steps_to_think=self.n_steps_to_think,
        )
