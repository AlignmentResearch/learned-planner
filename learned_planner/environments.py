"""
Custom Gym environments.
"""

import abc
import dataclasses
import os
import random
import warnings
from functools import partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import gym_sokoban  # noqa: F401  # type: ignore[import]
import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.spaces import Box, Discrete
from stable_baselines3.common.type_aliases import check_cast
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
    tile_images,
)
from stable_baselines3.common.vec_env.util import obs_as_tensor
from typing_extensions import Self

from learned_planner.make_gym_env import make_env
from learned_planner.policies import RewardToyModel

if TYPE_CHECKING:
    import envpool
    import envpool.python.protocol

    EnvPoolProtocol = envpool.python.protocol.EnvPool
else:
    try:
        # Hide envpool imports so that we can launch experiments from a MacOS machine, which does not support envpool.
        # Actually trying to use envpool will of course raise an error.

        import envpool
        import envpool.python.protocol

        EnvPoolProtocol = envpool.python.protocol.EnvPool
    except ImportError:
        EnvPoolProtocol = None

DEFAULT_N_ENVS = 2


@dataclasses.dataclass
class EnvConfig(abc.ABC):
    max_episode_steps: int
    n_envs: int = DEFAULT_N_ENVS  # Number of environments
    n_envs_to_render: int = min(64, DEFAULT_N_ENVS)  # Number of environments to render

    def __post_init__(self):
        assert self.n_envs_to_render <= self.n_envs


@dataclasses.dataclass
class NNRewardVecEnvConfig(EnvConfig):
    max_episode_steps: int = 100


class BasicVecEnv(VecEnv):
    _device: th.device

    def __init__(self, num_envs: int, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, device: th.device):
        super().__init__(num_envs, observation_space, action_space)
        self._device = device

    @property
    def device(self) -> th.device:
        return self._device

    def close(self) -> None:
        pass  # No need to do anything

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return [False] * len(list(self._get_indices(indices)))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        n_idx = len(list(self._get_indices(indices)))
        return [getattr(self, attr_name)] * n_idx

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        raise RuntimeError(
            "{self.__class__.__name__} doesn't contain any `gym.Env` inside, so calling `env_method` doesn't make sense."
        )


class NNRewardVecEnv(BasicVecEnv):
    """
    A vectorized environment that uses a neural network to compute rewards.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "has_terminal_obs": False}
    reward_range = (-float("inf"), float("inf"))
    policy_features: Optional[RewardToyModel]

    def __init__(
        self,
        num_envs: int,
        reward_fn: th.nn.Module,
        obs_dim: int = 2,
        hidden_dim: int = 8,
        device: th.device = th.device("cpu"),
        max_episode_steps: int = 100,
        split: Literal["train", "test", "everything"] = "train",
        hidden_state_init: Literal["random", "zeros"] = "random",
        fully_observable: bool = False,
        num_envs_to_render: int = 64,
    ):
        observation_space = gym.spaces.Dict(
            {
                "obs": Box(low=-1.0, high=1.0, shape=(obs_dim,)),
                **({} if fully_observable else {"hidden": Box(low=-4.0, high=4.0, shape=(hidden_dim,))}),
            }
        )
        action_space = Discrete(2 * obs_dim + 1)  # go +/- in each dimension and stay still
        self.render_mode = "rgb_array"
        super().__init__(num_envs=num_envs, observation_space=observation_space, action_space=action_space, device=device)

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.split = split
        self.fully_observable = fully_observable
        self.hidden_state_init = hidden_state_init
        self.state = th.zeros((num_envs, obs_dim + hidden_dim), device=self.device)
        self.num_envs_to_render = num_envs_to_render
        if self.num_envs_to_render > self.num_envs:
            raise ValueError(f"{num_envs_to_render=} must be less than or equal to {num_envs=}")

        self.action_key = th.zeros((int(action_space.n), obs_dim), device=self.device)
        for i in range(obs_dim):
            self.action_key[2 * i, i] = -0.05
            self.action_key[2 * i + 1, i] = 0.05
        self.action_key[2 * obs_dim, :] = 0.0

        self.waiting = False

        self._informations = [{} for _ in range(self.num_envs)]
        self._DONES_ZEROS = th.zeros((self.num_envs,), dtype=th.bool, device=self.device)

        self.reward_fn = reward_fn.to(self.device)
        self.max_episode_steps = max_episode_steps
        self.seed()
        self.reward_background = th.zeros((self.num_envs, 101, 101, 3), dtype=th.uint8)
        self.reward_background_up_to_date = False
        self._time_step = 0
        self.policy_features = None

    def make_reward_background(self):
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap("viridis")
        i_s, j_s = th.meshgrid(*[th.linspace(-1.0, 1.0, 101)] * 2, indexing="ij")
        state = th.zeros((self.num_envs_to_render, 101, 101, self.obs_dim + self.hidden_dim), device=self.device)
        state[:, :, :, : self.obs_dim] = th.stack([i_s, j_s], dim=-1)
        state[:, :, :, self.obs_dim :] = self.state[: self.num_envs_to_render, None, None, self.obs_dim :]

        with th.no_grad():
            rewards = self.reward_fn(state).squeeze(-1).cpu().numpy()
        rewards_color: th.Tensor = th.as_tensor(cmap((rewards - rewards.min()) / (rewards.max() - rewards.min())))
        self.reward_background = rewards_color[:, :, :, :3].mul_(255).to(dtype=th.uint8, device="cpu")

        self.reward_background_up_to_date = True

    def observation(self) -> Dict[str, th.Tensor]:
        if self.fully_observable:
            return {"obs": self.state}
        return {"obs": self.state[:, : self.obs_dim], "hidden": self.state[:, self.obs_dim :]}

    N_TEST_DIMS = 3

    def reset(self) -> Dict[str, th.Tensor]:
        self.reward_background_up_to_date = False
        self._time_step = 0

        # Reset state to random values between 0 and 1
        th.rand(
            self.state.size(),
            generator=self.generator,
            dtype=self.state.dtype,
            device=self.state.device,
            out=self.state,
        )

        mul = th.ones((self.obs_dim + self.hidden_dim), device=self.state.device, dtype=self.state.dtype)
        add = th.zeros((self.obs_dim + self.hidden_dim), device=self.state.device, dtype=self.state.dtype)

        if self.hidden_state_init == "zeros":
            mul[:] = 0.0
        elif self.hidden_state_init == "random":
            mul[: self.obs_dim] = 2.0
            add[: self.obs_dim] = -1.0
            mul[self.obs_dim + self.N_TEST_DIMS :] = 8.0
            add[self.obs_dim + self.N_TEST_DIMS :] = -4.0

            if self.split == "test":
                # Test mode: states in the orthant of larger than [1.5, 1.5, 1.5, -4, -4, ...]
                mul[self.obs_dim : self.obs_dim + self.N_TEST_DIMS] = 4.0 - 1.5
                add[self.obs_dim : self.obs_dim + self.N_TEST_DIMS] = 1.5
            elif self.split == "train":
                # Train mode: states in the many orthant all numbers are less than [1.5, 1.5, 1.5, 4, 4, ...]
                mul[self.obs_dim : self.obs_dim + self.N_TEST_DIMS] = 1.5 - (-4.0)
                add[self.obs_dim : self.obs_dim + self.N_TEST_DIMS] = -4.0
            elif self.split == "both":
                mul[self.obs_dim : self.obs_dim + self.N_TEST_DIMS] = 8.0
                add[self.obs_dim : self.obs_dim + self.N_TEST_DIMS] = -4.0
            else:
                raise ValueError(f"Unknown split {self.split}")
        else:
            raise ValueError(f"Unknown hidden_state_init {self.hidden_state_init}")

        self.state.mul_(mul).add_(add)  # Modify state

        return self.observation()

    def step_async(self, actions: th.Tensor) -> None:
        assert not self.waiting, "This code assumes that step_async and step_wait are called alternately"
        new_state = self.state[:, : self.obs_dim] + self.action_key[actions]
        th.clip(new_state, -1.0, 1.0, out=self.state[:, : self.obs_dim])
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        assert self.waiting, "Need to call step_async() before step_wait()"
        self.waiting = False
        with th.no_grad():
            rewards: th.Tensor = self.reward_fn(self.state).squeeze(1)
        self._time_step += 1
        if self._time_step >= self.max_episode_steps:
            dones = ~self._DONES_ZEROS
            observation = self.reset()  # VecEnvs reset themselves at the end of the episode
            # Here we should set _informations[i]["terminal_observation"] for many envs. But that's expensive and kind
            # of useless so we're not going to.
            # TODO: vectorize the infos
        else:
            dones = self._DONES_ZEROS
            observation = self.observation()
        return (observation, rewards, dones, self._informations)

    def seed(self, seed: Optional[int] = None) -> list[Union[None, int]]:
        self.generator = th.Generator(device=self.device)
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)
        return [seed] * self.num_envs

    def render(self, mode: Optional[str] = None) -> Optional[th.Tensor]:
        if not self.reward_background_up_to_date:
            self.make_reward_background()
        img = self.reward_background.clone()
        observation = self.observation()["obs"][: self.num_envs_to_render]
        y, x = [(50 * (1 + z)).to(dtype=th.int64, device="cpu") for z in observation.T]

        if self.policy_features is not None:
            with th.no_grad():
                # Because we're using the last input, this is going to be delayed by 1 step. It should still be
                # fine for qualitative observations.
                # TODO: synchronize `reward_locations` with `observation`, by delaying the latter.
                reward_locations: th.Tensor = self.policy_features.last_reward_fn_input
                if reward_locations.shape[0] != self.num_envs:
                    reward_locations = reward_locations.view((-1, self.num_envs, *reward_locations.shape[1:]))[0]

                assert reward_locations.shape[0] == self.num_envs
                assert reward_locations.shape[2] == (self.obs_dim + self.hidden_dim)
                reward_locations = reward_locations[: self.num_envs_to_render].moveaxis(1, 0)
                reward_locations = (50 * (1 + reward_locations)).to(th.int64)

                # Paint a 5x5 square centered on (y, x)
                arange = th.arange(self.num_envs_to_render)
                RED = th.tensor([255, 0, 0], dtype=th.uint8)
                BLUE = th.tensor([0, 0, 255], dtype=th.uint8)
                for offset_y in range(-2, 3):
                    for offset_x in range(-2, 3):
                        y_paint = th.clip(y + offset_y, 0, img.shape[1] - 1)
                        x_paint = th.clip(x + offset_x, 0, img.shape[2] - 1)
                        img[arange, y_paint, x_paint, :] = RED

                        for reward_location in reward_locations:
                            y_paint = th.clip(reward_location[:, 0] + offset_y, 0, img.shape[1] - 1)
                            x_paint = th.clip(reward_location[:, 1] + offset_x, 0, img.shape[2] - 1)
                            img[arange, y_paint, x_paint, :] = BLUE

        tiled_img: th.Tensor = tile_images(img)

        if mode is None or mode == "rgb_array":
            return tiled_img
        elif mode == "human":
            import cv2

            cv2.imshow("vecenv", tiled_img.numpy()[:, :, ::-1])  # type: ignore
            cv2.waitKey(1)  # type: ignore
            return tiled_img
        else:
            raise NotImplementedError(f"Render mode {mode} not supported by this VecEnv")


@dataclasses.dataclass
class EnvpoolVecEnvConfig(EnvConfig):
    max_episode_steps: int = 120
    num_threads: int = 0
    thread_affinity_offset: int = -1
    seed: int = 42
    max_num_players: int = 1

    px_scale: int = 4  # How much does each pixel get scaled when rendering

    @abc.abstractproperty
    def env_id(self) -> str:
        ...

    def make(self: Self, device: th.device) -> "EnvpoolVecEnv[Self]":
        return EnvpoolVecEnv(self, device=device)


EnvpoolVecEnvConfigT = TypeVar("EnvpoolVecEnvConfigT", bound=EnvpoolVecEnvConfig)


class EnvpoolVecEnv(BasicVecEnv, Generic[EnvpoolVecEnvConfigT]):
    """
    Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.

    :param env: The envpool object.
    """

    cfg: EnvpoolVecEnvConfigT
    num_envs_to_render: int
    env: EnvPoolProtocol
    _last_obs_np: np.ndarray
    _informations: list[dict]
    _real_info: dict

    def __init__(
        self,
        cfg: EnvpoolVecEnvConfigT,
        device: th.device = th.device("cpu"),
        **env_kwargs,
    ):
        env_id: str = cfg.env_id
        dummy_spec = envpool.make_spec(env_id)
        special_kwargs = dict(
            num_envs=cfg.n_envs,
            batch_size=cfg.n_envs,
        )
        SPECIAL_KEYS = {"base_path", "gym_reset_return_info"}
        env_kwargs = {k: getattr(cfg, k) for k in dummy_spec._config_keys if not (k in special_kwargs or k in SPECIAL_KEYS)}

        self.cfg = cfg
        self.num_envs_to_render = cfg.n_envs_to_render
        self.env = envpool.make_gymnasium(env_id, **special_kwargs, **env_kwargs)

        with warnings.catch_warnings():
            # Envpool envs don't have `render_mode` attribute, because they don't render.
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message="The `render_mode` attribute is not defined in your environment. It will be set to None.",
                module="stable_baselines3",
            )
            assert isinstance(self.env.observation_space, gym.spaces.Space)
            assert isinstance(self.env.action_space, gym.spaces.Space)
            super().__init__(
                num_envs=cfg.n_envs,
                observation_space=self.env.observation_space,
                action_space=self.env.action_space,
                device=device,
            )
        obs, _ = check_cast(tuple, self.env.reset())
        self._last_obs_np = obs[: self.num_envs_to_render]

        # Envpool sensibly returns a dict-of-lists, whereas VecEnv expects a list-of-dicts.
        #
        # This weird double-dict structure is so we only have to update one pointer for every iteration. each
        # `self._informations[i]["real_info"]` points to the same dict (which basically behaves as a pointer), which in
        # turn points to the original list-of-dicts.
        self._real_info = {"real_info": None}
        self._informations = [{"env": i, "real_info": self._real_info} for i in range(self.num_envs)]

    def step_async(self, actions: th.Tensor) -> None:
        self.env.send(actions.detach().cpu().numpy())

    def reset(self) -> VecEnvObs:
        obs, _ = check_cast(tuple, self.env.reset())
        self._last_obs_np = obs[: self.num_envs_to_render]
        return obs_as_tensor(obs, self.device)

    def seed(self, seed: Optional[int] = None) -> Sequence[int]:
        MAX_SEED = 2**31 - 1 - self.num_envs
        if seed is None:
            seed = random.randint(0, MAX_SEED)
        assert seed <= MAX_SEED, "Seed may not fit in a C++ int"
        self.env = envpool.make_gymnasium(self.cfg.env_id, **self.env.config)
        return range(seed, seed + self.num_envs)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, terminated, truncated, info_dict = check_cast(tuple, self.env.recv())
        self._last_obs_np = obs[: self.num_envs_to_render]
        self._real_info["real_info"] = info_dict  # mutate a structure inside `self._informations`
        return (
            obs_as_tensor(obs, self.device),
            th.as_tensor(rewards, device=self.device),
            th.as_tensor(terminated | truncated, device=self.device),
            self._informations,  # list-of-dicts that points to `info_dict` on every individual dict.
        )

    def render(self, mode: Optional[str] = None) -> Optional[th.Tensor]:
        imgs = th.as_tensor(self._last_obs_np, device="cpu")
        imgs = imgs.moveaxis(1, -1)
        # Expand resolution by 4x so the mp4 looks better
        imgs = imgs.repeat_interleave(self.cfg.px_scale, 2)
        imgs = imgs.repeat_interleave(self.cfg.px_scale, 1)
        return tile_images(imgs)


@dataclasses.dataclass
class EnvpoolSokobanVecEnvConfig(EnvpoolVecEnvConfig):
    """Sokoban in Envpool.

    - `min_episode_steps` contains the minimum length of en episode. In the underlying Envpool library, actual episodes
      get reset somewhere between `min_episode_steps` and `max_episode_steps`, both inclusive.

      We want to break up the temporal correlation of episodes. If they all begin and start at the same time, only the
      first few steps may have any meaningful rewards, and the rest will be kind of random. Learning will be very poor.
      For this reason, we have episodes with random lengths.
    """

    reward_finished: float = 10.0  # Reward for completing a level
    reward_box: float = 1.0  # Reward for putting a box on target
    reward_step: float = -0.1  # Reward for completing a step
    verbose: int = 0  # Verbosity level [0-2]
    min_episode_steps: int = 0  # The minimum length of an episode.
    load_sequentially: bool = False
    n_levels_to_load: int = -1  # -1 means "all levels". Used only when `load_sequentially` is True.

    # Not present in _SokobanEnvSpec
    cache_path: Path = Path(__file__).parent.parent / ".sokoban_cache"
    split: Literal["train", "valid", "test", None] = "train"
    difficulty: Literal["unfiltered", "medium", "hard"] = "unfiltered"

    def __post_init__(self):
        if self.difficulty == "hard":
            assert self.split is None
        else:
            assert self.split is not None
        assert self.min_episode_steps >= 0
        assert self.min_episode_steps <= self.max_episode_steps, f"{self.min_episode_steps=} {self.max_episode_steps=}"
        if not self.load_sequentially:
            assert self.n_levels_to_load == -1, "`n_levels_to_load` must be -1 when `load_sequentially` is False"

    @property
    def env_id(self) -> str:
        return "Sokoban-v0"

    @property
    def dim_room(self) -> int:
        return 10

    @property
    def levels_dir(self) -> str:
        levels_dir = self.cache_path / "boxoban-levels-master" / self.difficulty
        if self.difficulty == "hard":
            assert self.split is None
        else:
            assert self.split is not None
            levels_dir = levels_dir / self.split

        not_end_txt = [s for s in os.listdir(levels_dir) if not (levels_dir / s).is_dir() and not s.endswith(".txt")]
        if len(not_end_txt) > 0:
            raise ValueError(f"{levels_dir=} does not exist or some of its files don't end in .txt: {not_end_txt}")
        return str(levels_dir)


@dataclasses.dataclass
class BaseSokobanEnvConfig(EnvConfig):
    max_episode_steps: int = 120  # default value from gym_sokoban
    min_episode_steps: int = 0
    tinyworld_obs: bool = False
    tinyworld_render: bool = False
    render_mode: str = "rgb_8x8"  # can be "rgb_array" or "rgb_8x8"
    terminate_on_first_box: bool = False

    reward_finished: float = 10.0  # Reward for completing a level
    reward_box: float = 1.0  # Reward for putting a box on target
    reward_step: float = -0.1  # Reward for completing a step

    seed: int = dataclasses.field(default_factory=lambda: random.randint(0, 2**31 - 1))
    reset: bool = False

    def env_kwargs(self) -> dict[str, Any]:
        return dict(
            tinyworld_obs=self.tinyworld_obs,
            tinyworld_render=self.tinyworld_render,
            render_mode=self.render_mode,
            # Sokoban env uses `max_steps` internally
            max_steps=self.max_episode_steps,
            # Passing `max_episode_steps` to Gymnasium makes it add a TimeLimitWrapper
            max_episode_steps=self.max_episode_steps,
            min_episode_steps=self.min_episode_steps,
            terminate_on_first_box=self.terminate_on_first_box,
            reset_seed=self.seed,
            reset=self.reset,
        )

    def env_reward_kwargs(self):
        return dict(
            reward_finished=self.reward_finished,
            reward_box_on_target=self.reward_box,
            penalty_box_off_target=-self.reward_box,
            penalty_for_step=self.reward_step,
        )

    @abc.abstractproperty
    def make(self) -> Callable[[], gym.Env]:
        ...


@dataclasses.dataclass
class SokobanConfig(BaseSokobanEnvConfig):
    "Procedurally-generated Sokoban"

    name: ClassVar[str] = "Sokoban-v2"

    dim_room: Optional[tuple[int, int]] = None
    num_boxes: int = 4
    num_gen_steps: Optional[int] = None

    @property
    def make(self) -> Callable[[], gym.Env]:
        kwargs = self.env_kwargs()
        for k in ["dim_room", "num_boxes", "num_gen_steps"]:
            if (a := getattr(self, k)) is not None:
                kwargs[k] = a
        make_fn = partial(
            make_env,
            self.name,
            **kwargs,
            **self.env_reward_kwargs(),
        )
        return make_fn


@dataclasses.dataclass
class BoxobanConfig(BaseSokobanEnvConfig):
    "Sokoban levels from the Boxoban data set"

    name: ClassVar[str] = "Boxoban-Val-v1"  # Any Boxoban-*-* name will work

    cache_path: Path = Path(__file__).parent.parent / ".sokoban_cache"
    split: Literal["train", "valid", "test", None] = "train"
    difficulty: Literal["unfiltered", "medium", "hard"] = "unfiltered"

    @property
    def make(self) -> Callable[[], gym.Env]:
        if self.difficulty == "hard":
            if self.split is not None:
                raise ValueError("`hard` levels have no splits")
        elif self.difficulty == "medium":
            if self.split == "test":
                raise ValueError("`medium` levels don't have a `test` split")

        make_fn = partial(
            make_env,
            self.name,
            cache_path=self.cache_path,
            split=self.split,
            difficulty=self.difficulty,
            **self.env_kwargs(),
            **self.env_reward_kwargs(),
        )
        return make_fn


@dataclasses.dataclass
class FixedBoxobanConfig(BoxobanConfig):
    name: ClassVar[str] = "FixedBoxoban-Val-v1"
