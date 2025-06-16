"""
Policies that use the reward function.
"""
import abc
import copy
import dataclasses
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

import gymnasium as gym
import huggingface_hub
import stable_baselines3.common.distributions
import torch
import torch.nn as nn
from farconf import to_dict
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.recurrent.policies import (
    BaseRecurrentActorCriticPolicy,
    RecurrentFeaturesExtractorActorCriticPolicy,
)
from stable_baselines3.common.recurrent.torch_layers import RecurrentFeaturesExtractor, RecurrentState
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule, check_cast, non_null
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.ppo import MlpPolicy
from torchvision.models.resnet import BasicBlock
from transformer_lens.hook_points import HookPoint

from learned_planner.activation_fns import ActivationFnConfig, ReLUConfig
from learned_planner.convlstm import BaseFeaturesExtractorConfig, ConvLSTMOptions, NCHWtoNHWC

log = logging.getLogger(__name__)


@dataclasses.dataclass
class NetArchConfig:
    pi: Sequence[int]
    vf: Sequence[int]


@dataclasses.dataclass
class FlattenFeaturesExtractorConfig(BaseFeaturesExtractorConfig):
    def kwargs(self, vec_env: VecEnv) -> dict[str, Any]:
        return dict(features_extractor_class=FlattenExtractor)


@dataclasses.dataclass
class BasePolicyConfig(abc.ABC):
    policy: type[BasePolicy] | str
    features_extractor: BaseFeaturesExtractorConfig = dataclasses.field(default_factory=FlattenFeaturesExtractorConfig)
    net_arch: Optional[NetArchConfig] = None

    def policy_and_kwargs(self, vec_env: VecEnv) -> tuple[type[BasePolicy] | str, dict[str, Any]]:
        kwargs = self.features_extractor.kwargs(vec_env)
        if self.net_arch is not None:
            kwargs["net_arch"] = to_dict(self.net_arch)
        return self.policy, kwargs


@dataclasses.dataclass
class MlpPolicyConfig(BasePolicyConfig):
    policy: type[BasePolicy] | str = "MlpPolicy"
    features_extractor: BaseFeaturesExtractorConfig = dataclasses.field(default_factory=FlattenFeaturesExtractorConfig)
    is_drc: bool = False


@dataclasses.dataclass
class ResNetExtractorConfig(BaseFeaturesExtractorConfig):
    hidden_channels: int = 64
    layers: int = 9
    kernel_size: int = 3

    def kwargs(self, vec_env: VecEnv) -> dict[str, Any]:
        return dict(features_extractor_class=ResNetExtractor, features_extractor_kwargs=dict(cfg=self))

    def make(self, channels: int):
        return nn.Sequential(
            nn.Conv2d(channels, self.hidden_channels, kernel_size=self.kernel_size, stride=1, padding="same"),
            *(BasicBlock(self.hidden_channels, self.hidden_channels) for i in range(self.layers)),
        )


class ResNetExtractor(FlattenExtractor):
    cfg: ResNetExtractorConfig

    def __init__(self, observation_space: gym.Space, cfg: ResNetExtractorConfig) -> None:
        extractor = nn.Sequential(cfg.make(channels=non_null(observation_space.shape)[-3]), nn.Flatten())

        in_space = check_cast(gym.spaces.Box, observation_space)
        example_input_shape = in_space.shape
        with torch.no_grad():
            example_input = torch.zeros((1, *example_input_shape))
            example_output = extractor(example_input)
            n_features = example_output.numel()

        low = in_space.low.flat[0]
        high = in_space.high.flat[0]

        super().__init__(gym.spaces.Box(low, high, (n_features,)))
        self.cfg = cfg
        self.flatten = extractor


class GuezResidualBlock(nn.Module):
    """
    A resnet block with a relu, conv, and residual connection.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the kernel for the convolutional layer.
        """
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same")

        self.hook_relu = HookPoint()
        self.hook_conv = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(x)
        out = self.hook_relu(out)
        out = self.conv(out)
        out = self.hook_conv(out)
        return out + residual


class GuezConvSequence(nn.Module):
    """
    A sequence of conv,resnet,resnet layers used in Guez et. al. (2019)'s ResNet architecture.
    """

    def __init__(self, out_channels: int, inp_channels: int, kernel_size: int = 3, stride: int = 1, is_input: bool = False):
        """
        Args:
            out_channels: Number of output channels for the conv layer.
            kernel_size: Size of the kernel for the conv layer.
            stride: Stride for the conv layer.
            is_input: Whether this is the first layer in the sequence (for input normalization).
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=inp_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding="same",
        )
        self.resnet0 = GuezResidualBlock(out_channels, out_channels, kernel_size)
        self.resnet1 = GuezResidualBlock(out_channels, out_channels, kernel_size)

        self.hook_lead_conv = HookPoint()
        self.hook_resnet0 = HookPoint()
        self.hook_resnet1 = HookPoint()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the conv and resnet layers.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the conv and resnet layers.
        """
        x = self.conv(x)
        x = self.hook_lead_conv(x)
        x = self.resnet0(x)
        x = self.hook_resnet0(x)
        x = self.resnet1(x)
        x = self.hook_resnet1(x)
        return x


@dataclasses.dataclass
class GuezResNetExtractorConfig(BaseFeaturesExtractorConfig):
    channels: tuple[int, ...] = (32, 32, 64, 64, 64, 64, 64, 64, 64)
    strides: tuple[int, ...] = (1,) * 9
    kernel_sizes: tuple[int, ...] = (4,) * 9

    @property
    def num_layers(self) -> int:
        return len(self.channels)

    def kwargs(self, vec_env: VecEnv) -> dict[str, Any]:
        return dict(features_extractor_class=ResNetExtractor, features_extractor_kwargs=dict(cfg=self))

    def make(self, channels: int):
        assert (
            len(self.channels) == len(self.strides) == len(self.kernel_sizes)
        ), f"{len(self.channels)=}, {len(self.strides)=}, {len(self.kernel_sizes)=} must be equal."
        num_layers = len(self.channels)
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        f"guez_conv_sequence_{i}",
                        GuezConvSequence(
                            self.channels[i],
                            self.channels[i - 1] if i > 0 else channels,
                            kernel_size=self.kernel_sizes[i],
                            stride=self.strides[i],
                        ),
                    )
                    for i in range(num_layers)
                ]
                + [("relu", nn.ReLU()), ("to_nhwc", NCHWtoNHWC())]
            )
        )


@dataclasses.dataclass
class BaseRecurrentPolicyConfig(BasePolicyConfig):
    policy: ClassVar[type[BaseRecurrentActorCriticPolicy]]  # type: ignore[override]


class RewardToyModel:
    """
    Mixin for policies that use the reward function in their forward pass. It also stores the last input to the reward,
    so it can be used to render where the reward was queried.

    This is used for creating a 'toy model' of a mesa-optimizer.
    """

    reward_fn: nn.Module
    last_reward_fn_input: torch.Tensor

    def _register_reward_fn_input_hook(self, reward_fn: nn.Module, reward_batch: int):
        reward_fn.register_forward_pre_hook(self._store_last_input_hook)
        self.reward_batch = reward_batch
        self.last_reward_fn_input = torch.zeros(()).expand((int(1e12), reward_batch, 10))

    def _store_last_input_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        log.debug("Called _store_last_input_hook")
        (self.last_reward_fn_input,) = inputs

    def reset_reward(self, reward_fn: nn.Module):
        """Make the model use ``reward_fn`` as its internal reward function."""
        self.reward_fn.requires_grad_(False)
        self.reward_fn.load_state_dict(reward_fn.state_dict())
        for p in self.reward_fn.parameters():
            assert not p.requires_grad


class RewardNNFeaturesExtractor(BaseFeaturesExtractor, RewardToyModel):
    """
    A FeaturesExtractor (NN before a policy) which contains a query to `reward_fn` in its forward pass.
    """

    reward_fn: nn.Module

    def __init__(
        self,
        observation_space: gym.Space,
        activation_fn: Type[nn.Module],
        reward_fn: Optional[nn.Module] = None,
        reward_batch_size: int = 5,
        hidden_size: int = 64,
    ):
        super().__init__(observation_space, features_dim=hidden_size)

        assert isinstance(observation_space, gym.spaces.Dict)
        self.residual_dim = get_flattened_obs_dim(observation_space["obs"])
        self.reward_batch_size = reward_batch_size
        if reward_fn is None:
            reward_fn = torch.load(Path(__file__).resolve().parent / "notebooks/hard_reward.pt", map_location="cpu")
        else:
            reward_fn = copy.deepcopy(reward_fn)
        self.reward_fn = non_null(reward_fn)

        self._register_reward_fn_input_hook(self.reward_fn, self.reward_batch_size)

        self.hidden_size = hidden_size
        self.activation_fn = activation_fn

        self.pre_reward_fn = nn.Sequential(
            nn.Linear(self.residual_dim, hidden_size),
            activation_fn(),
            nn.Linear(self.hidden_size, self.residual_dim * self.reward_batch_size),
        )
        # self.mid_reward_fn = nn.Sequential(
        #     nn.Linear(residual_dim, hidden_size),
        #     activation_fn(),
        #     nn.Linear(hidden_size, residual_dim),
        # )
        self.mid_reward_fn = nn.Identity()
        self.post_reward_fn = nn.Sequential(
            nn.Linear((self.residual_dim + 1) * self.reward_batch_size, self.hidden_size),
            activation_fn(),
            nn.Linear(self.hidden_size, self.features_dim),
        )

    def pre_reward_forward(self, observation: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor = self.pre_reward_fn(observation).view(
            observation.shape[0], self.reward_batch_size, self.residual_dim
        )
        pre_reward = observation.view(*observation.shape[:-1], 1, self.residual_dim) + residual
        return pre_reward

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        observation = inputs["obs"]
        pre_reward = self.pre_reward_forward(observation)

        non_observation = (
            inputs["hidden"].unsqueeze(-2).expand(*observation.shape[:-1], self.reward_batch_size, -1).to(pre_reward.device)
        )
        rewards = self.reward_fn(torch.cat([pre_reward, non_observation], dim=-1))
        assert rewards.shape == (*observation.shape[:-1], self.reward_batch_size, 1)
        trunk = self.mid_reward_fn(pre_reward)
        assert trunk.shape == (*observation.shape[:-1], self.reward_batch_size, self.residual_dim)

        after_mid_reward = torch.cat([trunk, rewards], dim=-1).view(
            *observation.shape[:-1], self.reward_batch_size * (self.residual_dim + 1)
        )
        post_reward = self.post_reward_fn(after_mid_reward)
        assert post_reward.shape == (*observation.shape[:-1], self.features_dim)
        return post_reward


@dataclasses.dataclass
class BaseRewardNNFeaturesExtractorConfig(BaseFeaturesExtractorConfig):
    features_extractor_class: ClassVar[type[BaseFeaturesExtractor]]

    reward_batch_size: int = 5
    hidden_size: int = 64

    def kwargs(self, vec_env: VecEnv) -> dict[str, Any]:
        return dict(
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs(vec_env),
        )

    def features_extractor_kwargs(self, vec_env: VecEnv) -> dict[str, Any]:
        return dict(
            reward_fn=vec_env.unwrapped.reward_fn,  # type: ignore
            reward_batch_size=self.reward_batch_size,
            hidden_size=self.hidden_size,
        )


@dataclasses.dataclass
class RewardNNFeaturesExtractorConfig(BaseRewardNNFeaturesExtractorConfig):
    features_extractor_class: ClassVar[type[BaseFeaturesExtractor]] = RewardNNFeaturesExtractor
    activation: ActivationFnConfig = dataclasses.field(default_factory=ReLUConfig)

    def features_extractor_kwargs(self, vec_env: VecEnv) -> dict[str, Any]:
        # Using super() here gives an error that I don't understand:
        #   TypeError: __class__ set to <class 'learned_planner.policies.RewardNNFeaturesExtractorConfig'> defining
        #     'RewardNNFeaturesExtractorConfig' as <class 'learned_planner.policies.RewardNNFeaturesExtractorConfig'>
        super_kwargs = BaseRewardNNFeaturesExtractorConfig.features_extractor_kwargs(self, vec_env)
        return dict(activation_fn=self.activation.fn, **super_kwargs)


class IdentityDictFeaturesExtractor(BaseFeaturesExtractor):
    """Passes the observation throuth as-is. Breaks types."""

    def __init__(self, observation_space: gym.Space):
        assert isinstance(observation_space, gym.spaces.Dict)
        dict_features_dim = {k: get_flattened_obs_dim(observation_space[k]) for k in observation_space.keys()}
        super().__init__(observation_space, features_dim=sum(dict_features_dim.values()))
        self.dict_features_dim = dict_features_dim

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return inputs  # type: ignore


RewardGRUFeaturesExtractorState = Tuple[torch.Tensor, ...]


class RewardGRUFeaturesExtractor(
    RecurrentFeaturesExtractor[Dict[str, torch.Tensor], RewardGRUFeaturesExtractorState], RewardToyModel
):
    """Behaves like a multi-layer GRU, but contains a reward_fn evaluated at every step and layer."""

    def __init__(
        self,
        observation_space: gym.Space,
        reward_fn: nn.Module,
        reward_batch_size: int,
        hidden_size: int,
        num_layers: int = 2,
    ):
        super().__init__(observation_space, features_dim=hidden_size)
        assert isinstance(observation_space, gym.spaces.Dict)

        self.input_size = get_flattened_obs_dim(observation_space["obs"])
        self.hidden_obs_size = get_flattened_obs_dim(observation_space["hidden"])
        self.reward_batch_size = reward_batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        if self.num_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.to_residual_fn = nn.Linear(self.input_size, self.hidden_size)
        self.first_lstm = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1)

        self.lstms = torch.nn.ModuleList(
            [
                nn.GRU(
                    input_size=self.hidden_size + self.reward_batch_size,
                    hidden_size=self.hidden_size,
                    num_layers=1,
                )
                for _ in range(self.num_layers - 1)
            ]
        )
        self.to_reward_fns = torch.nn.ModuleList(
            [nn.Linear(self.hidden_size, self.reward_batch_size * self.input_size) for _ in range(self.num_layers - 1)]
        )
        self.reward_fn = copy.deepcopy(reward_fn)
        self._register_reward_fn_input_hook(self.reward_fn, self.reward_batch_size)

    def recurrent_initial_state(
        self, n_envs: Optional[int] = None, *, device: Optional[torch.device | str] = None
    ) -> RewardGRUFeaturesExtractorState:
        if n_envs is None:
            shape = (1, self.hidden_size)
        else:
            shape = (1, n_envs, self.hidden_size)
        return tuple(torch.zeros(shape, device=device) for _ in range(self.num_layers))

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        state: RewardGRUFeaturesExtractorState,
        episode_starts: torch.Tensor,
    ) -> Tuple[torch.Tensor, RewardGRUFeaturesExtractorState]:
        new_states: List[torch.Tensor] = list(state)
        obs = observations["obs"]
        hidden_obs = observations["hidden"]

        seq_len__batch_size = episode_starts.shape

        obs = obs.view(*seq_len__batch_size, self.input_size)
        hidden_obs = hidden_obs.view(*seq_len__batch_size, 1, self.hidden_obs_size).expand(
            *seq_len__batch_size, self.reward_batch_size, self.hidden_obs_size
        )

        trunk = self.to_residual_fn(obs)

        residual, new_states[0] = self._process_sequence(self.first_lstm, trunk, state[0], episode_starts)
        trunk = trunk + residual

        for i, (rnn, to_reward_fn) in enumerate(zip(self.lstms, self.to_reward_fns)):
            assert isinstance(rnn, torch.nn.RNNBase)

            reward_in = to_reward_fn(trunk).view(*seq_len__batch_size, self.reward_batch_size, self.input_size)
            rewards = self.reward_fn(torch.cat([reward_in, hidden_obs], dim=-1)).squeeze(-1)
            rnn_in = torch.cat([trunk, rewards], dim=-1)
            residual, new_states[i + 1] = self._process_sequence(rnn, rnn_in, state[i + 1], episode_starts)
            trunk = trunk + residual

        return trunk, tuple(new_states)


@dataclasses.dataclass
class RewardGRUFeaturesExtractorConfig(BaseRewardNNFeaturesExtractorConfig):
    features_extractor_class = RewardGRUFeaturesExtractor
    num_layers: int = 2

    def features_extractor_kwargs(self, vec_env: VecEnv) -> dict[str, Any]:
        # Using super() here gives an error that I don't understand:
        #   TypeError: __class__ set to <class 'learned_planner.policies.RewardGRUFeaturesExtractorConfig'> defining
        #     'RewardGRUFeaturesExtractorConfig' as <class 'learned_planner.policies.RewardGRUFeaturesExtractorConfig'>
        super_kwargs = BaseRewardNNFeaturesExtractorConfig.features_extractor_kwargs(self, vec_env)
        return dict(num_layers=self.num_layers, **super_kwargs)


V2RewardGRUFeaturesExtractorState = Tuple[torch.Tensor, torch.Tensor]


class V2RewardGRUFeaturesExtractor(
    RecurrentFeaturesExtractor[Dict[str, torch.Tensor], V2RewardGRUFeaturesExtractorState], RewardToyModel
):
    """A two-layer GRU which queries the reward model `reward_fn` in between the two layers."""

    def __init__(
        self,
        observation_space: gym.Space,
        reward_fn: Optional[nn.Module] = None,
        reward_batch_size: int = 5,
        hidden_size: int = 64,
    ):
        super().__init__(observation_space, features_dim=hidden_size)
        assert isinstance(observation_space, gym.spaces.Dict)

        self.input_size = get_flattened_obs_dim(observation_space["obs"])
        self.hidden_obs_size = get_flattened_obs_dim(observation_space["hidden"])
        self.reward_batch_size = reward_batch_size
        self.hidden_size = hidden_size

        if reward_fn is None:
            reward_fn = torch.load(Path(__file__).resolve().parent / "notebooks/hard_reward.pt", map_location="cpu")
        else:
            reward_fn = copy.deepcopy(reward_fn)
        self.reward_fn = non_null(reward_fn)
        self._register_reward_fn_input_hook(self.reward_fn, self.reward_batch_size)

        self.pre_reward_rnn = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=1)
        self.pre_reward_linear = nn.Linear(self.hidden_size, self.input_size * self.reward_batch_size)

        self.post_reward_rnn = nn.GRU(
            input_size=self.reward_batch_size * (self.input_size + 1), hidden_size=self.hidden_size, num_layers=1
        )
        self.post_reward_linear = nn.Linear(self.hidden_size, self.features_dim)

    def forward(
        self, observations: Dict[str, torch.Tensor], state: V2RewardGRUFeaturesExtractorState, episode_starts: torch.Tensor
    ) -> Tuple[torch.Tensor, V2RewardGRUFeaturesExtractorState]:
        new_states: List[torch.Tensor] = list(state)
        obs = observations["obs"]
        hidden_obs = observations["hidden"]

        seq_len__batch_size = episode_starts.shape

        obs = obs.view(*seq_len__batch_size, self.input_size)
        pre_reward_pre_linear, new_states[0] = self._process_sequence(self.pre_reward_rnn, obs, state[0], episode_starts)
        pre_reward = self.pre_reward_linear(pre_reward_pre_linear)
        residual = pre_reward.view(*seq_len__batch_size, self.reward_batch_size, self.input_size)
        trunk = obs.view(*seq_len__batch_size, 1, self.input_size)

        trunk = trunk + residual

        hidden_obs = hidden_obs.view(*seq_len__batch_size, 1, self.hidden_obs_size).expand(
            *seq_len__batch_size, self.reward_batch_size, self.hidden_obs_size
        )
        rewards = self.reward_fn(torch.cat([trunk, hidden_obs], dim=-1))

        after_mid_reward = torch.cat([trunk, rewards], dim=-1).view(
            *seq_len__batch_size, self.reward_batch_size * (self.input_size + 1)
        )
        post_reward, new_states[1] = self._process_sequence(self.post_reward_rnn, after_mid_reward, state[1], episode_starts)
        out_features = self.post_reward_linear(post_reward)

        assert len(new_states) == 2
        return out_features, (new_states[0], new_states[1])

    def recurrent_initial_state(
        self, n_envs: Optional[int] = None, *, device: Optional[torch.device | str] = None
    ) -> V2RewardGRUFeaturesExtractorState:
        if n_envs is None:
            shape = (1, self.hidden_size)
        else:
            shape = (1, n_envs, self.hidden_size)
        return (torch.zeros(shape, device=device), torch.zeros(shape, device=device))


@dataclasses.dataclass
class V2RewardGRUFeaturesExtractorConfig(BaseRewardNNFeaturesExtractorConfig):
    features_extractor_class: ClassVar[type[BaseFeaturesExtractor]] = V2RewardGRUFeaturesExtractor


class BatchedRewardPolicy(ActorCriticPolicy):
    _default_features_extractor_class: ClassVar[Type[BaseFeaturesExtractor]] = RewardNNFeaturesExtractor

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = BaseFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if features_extractor_class is BaseFeaturesExtractor:
            features_extractor_class = self._default_features_extractor_class
        assert issubclass(features_extractor_class, BaseFeaturesExtractor)
        net_arch = []
        features_extractor_kwargs = {} if features_extractor_kwargs is None else features_extractor_kwargs.copy()
        features_extractor_kwargs.update(dict(activation_fn=activation_fn))
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


@dataclasses.dataclass
class BatchedRewardPolicyConfig(BasePolicyConfig):
    policy: type[BasePolicy] | str = BatchedRewardPolicy
    features_extractor: BaseFeaturesExtractorConfig = dataclasses.field(default_factory=RewardNNFeaturesExtractorConfig)

    def policy_and_kwargs(self, vec_env: VecEnv) -> tuple[type[BasePolicy] | str, dict[str, Any]]:
        assert self.net_arch is None
        return BasePolicyConfig.policy_and_kwargs(self, vec_env)


class RecurrentBatchedRewardPolicy(RecurrentFeaturesExtractorActorCriticPolicy[RecurrentState]):
    _default_features_extractor_class: ClassVar[Type[RecurrentFeaturesExtractor]] = RewardGRUFeaturesExtractor

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[
            RecurrentFeaturesExtractor[Dict[str, torch.Tensor], RecurrentState]
        ] = RecurrentFeaturesExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if features_extractor_class is RecurrentFeaturesExtractor:
            features_extractor_class = self._default_features_extractor_class
        assert issubclass(features_extractor_class, RecurrentFeaturesExtractor)
        net_arch = []
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )


@dataclasses.dataclass
class RecurrentBatchedRewardPolicyConfig(BaseRecurrentPolicyConfig):
    policy: ClassVar[type[BaseRecurrentActorCriticPolicy]] = RecurrentBatchedRewardPolicy
    features_extractor: BaseFeaturesExtractorConfig = dataclasses.field(default_factory=RewardGRUFeaturesExtractorConfig)

    def policy_and_kwargs(self, vec_env: VecEnv) -> tuple[type[BasePolicy] | str, dict[str, Any]]:
        assert self.net_arch is None
        return BasePolicyConfig.policy_and_kwargs(self, vec_env)


class V2RecurrentBatchedRewardPolicy(RecurrentBatchedRewardPolicy):
    _default_features_extractor_class = V2RewardGRUFeaturesExtractor


@dataclasses.dataclass
class V2RecurrentBatchedRewardPolicyConfig(RecurrentBatchedRewardPolicyConfig):
    policy: ClassVar[type[BaseRecurrentActorCriticPolicy]] = V2RecurrentBatchedRewardPolicy
    features_extractor: BaseFeaturesExtractorConfig = dataclasses.field(default_factory=V2RewardGRUFeaturesExtractorConfig)


class ObsExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        construct_wrapped: Callable[[gym.Space], BaseFeaturesExtractor],
        **kwargs,
    ) -> None:
        key_obs_space = observation_space["obs"]
        wrapped = construct_wrapped(observation_space, **kwargs)
        super().__init__(key_obs_space, wrapped.features_dim)
        self.wrapped = wrapped

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.wrapped(observations["obs"])


@dataclasses.dataclass
class ObsExtractorConfig(BaseFeaturesExtractorConfig):
    features_extractor_class = ObsExtractor

    def kwargs(self, vec_env: VecEnv) -> dict[str, Any]:
        return {}


class ObsMlpPolicy(MlpPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.Space,
        lr_schedule: Schedule,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        **kwargs,
    ):
        super().__init__(
            observation_space=gym.spaces.Dict({"obs": observation_space["obs"]}),
            action_space=action_space,
            features_extractor_class=cast(
                Type[ObsExtractor], lambda x, **kwargs: ObsExtractor(x, features_extractor_class, **kwargs)
            ),
            lr_schedule=lr_schedule,
            **kwargs,
        )

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        return super().predict_values({"obs": obs["obs"]})

    def get_distribution(  # type: ignore[override]
        self, obs: Dict[str, torch.Tensor]
    ) -> stable_baselines3.common.distributions.Distribution:
        return super().get_distribution({"obs": obs["obs"]})

    def extract_features(  # type: ignore[override]
        self, obs: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return super().extract_features({"obs": obs["obs"]})

    def predict(  # type: ignore[override]
        self,
        observation: Dict[str, torch.Tensor],
        state: Optional[Tuple[torch.Tensor, ...]] = None,
        episode_start: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        return super().predict({"obs": observation["obs"]}, state, episode_start, deterministic)


@dataclasses.dataclass
class ObsMlpPolicyConfig(BasePolicyConfig):
    policy = ObsMlpPolicy
    features_extractor: BaseFeaturesExtractorConfig = dataclasses.field(default_factory=ObsExtractorConfig)


@dataclasses.dataclass
class ConvLSTMPolicyConfig(BaseRecurrentPolicyConfig):
    policy = RecurrentFeaturesExtractorActorCriticPolicy
    features_extractor: BaseFeaturesExtractorConfig = dataclasses.field(default_factory=ConvLSTMOptions)
    is_drc: bool = True


@dataclasses.dataclass
class CombinedExtractorConfig(BaseFeaturesExtractorConfig):
    def kwargs(self, vec_env: VecEnv) -> dict[str, Any]:
        return {}


@dataclasses.dataclass
class MultiInputPolicyConfig(BasePolicyConfig):
    policy = "MultiInputPolicy"
    features_extractor: BaseFeaturesExtractorConfig = dataclasses.field(default_factory=CombinedExtractorConfig)

    def policy_and_kwargs(self, vec_env: VecEnv) -> tuple[type[BasePolicy] | str, dict[str, Any]]:
        return self.policy, {}


def download_policy_from_huggingface(local_or_hgf_repo_path: str | Path, force_download: bool = False) -> Path:
    local_or_hgf_repo_path = Path(local_or_hgf_repo_path)
    if not local_or_hgf_repo_path.exists():
        try:
            local_path = huggingface_hub.snapshot_download(
                "AlignmentResearch/learned-planner",
                allow_patterns=[str(local_or_hgf_repo_path) + "*"],
                force_download=force_download,
            )
            local_or_hgf_repo_path = Path(local_path) / local_or_hgf_repo_path
            if not local_or_hgf_repo_path.exists():
                raise FileNotFoundError(f"Model {local_or_hgf_repo_path} not found in local cache or on the hub")
        except (huggingface_hub.errors.HFValidationError, FileNotFoundError):
            print("Retrying with force_download=True")
            if not force_download:
                return download_policy_from_huggingface(local_or_hgf_repo_path, force_download=True)
            raise ValueError(f"Model {local_or_hgf_repo_path} not found in local cache or on the hub")

    return local_or_hgf_repo_path
