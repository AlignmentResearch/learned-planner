"""
License: MIT. Copyright (c) 2017 Andrea Palazzi
Original code: https://github.com/ndrplz/ConvLSTM_pytorch
"""
import abc
import dataclasses
import itertools
import math
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, TypeVar

import gymnasium as gym
import torch as th
import torch.nn as nn
from mamba_lens.input_dependent_hooks import InputDependentHookPoint
from stable_baselines3.common.recurrent.torch_layers import RecurrentFeaturesExtractor
from stable_baselines3.common.type_aliases import check_cast
from stable_baselines3.common.vec_env import VecEnv
from transformer_lens.hook_points import HookPoint

from learned_planner.activation_fns import ActivationFnConfig, ReLUConfig

IntOrShape2d = int | tuple[int, int]
ConvLSTMState = List[Tuple[th.Tensor, th.Tensor]]
ConvLSTMCellState = tuple[th.Tensor, th.Tensor]


def _expand_shape_2d(s: IntOrShape2d) -> tuple[int, int]:
    if isinstance(s, int):
        return (s, s)
    return s


T = TypeVar("T")
PaddingType = Literal["same", "valid"]
PaddingModeType = Literal["zeros", "reflect", "replicate", "circular"]


def _extend_for_multilayer(param: T | Sequence[T], length: int) -> list[T]:
    if isinstance(param, Sequence):
        if len(param) != length:
            raise ValueError(f"Asked for list of length {length}, but the output has length {len(param)}.")
        return list(param)
    return [param] * length


class NCHWtoNHWC(nn.Module):
    """Permute tensor from (N, C, H, W) to (N, H, W, C)."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # if x.ndim != 4:
        #     raise ValueError(f"Input tensor must be 4D (NCHW), but got {x.ndim}D")
        return x.moveaxis(-3, -1).contiguous()


class Conv2dSame(th.nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, input: th.Tensor) -> th.Tensor:
        ih, iw = input.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            input = th.nn.functional.pad(input, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return th.nn.functional.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


# conv_layer_s2_same = Conv2dSame(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), groups=1, bias=True)
# out = conv_layer_s2_same(th.zeros(1, 3, 224, 224))


@dataclasses.dataclass
class ConvConfig:
    features: int
    kernel_size: IntOrShape2d
    strides: IntOrShape2d = 1
    padding: PaddingType | IntOrShape2d = "same"
    padding_mode: PaddingModeType = "zeros"
    use_bias: bool = True

    def __post_init__(self):
        if not (isinstance(self.padding, int) or self.padding in ("same", "valid")):
            raise ValueError(f"Invalid {self.padding=}")

    def make_conv(self, **kwargs):
        in_channels = kwargs.pop("in_channels", self.features)
        bias = kwargs.pop("use_bias", self.use_bias)
        padding = self.padding
        if isinstance(padding, str):
            padding = padding.lower()

        strides = self.strides if isinstance(self.strides, tuple) else (self.strides, self.strides)
        strides = th.tensor(strides)
        use_same_conv = padding == "same" and th.any(strides > 1).item()
        conv_cls = Conv2dSame if use_same_conv else nn.Conv2d
        # conv_cls = nn.Conv2d
        return conv_cls(
            in_channels=in_channels,
            out_channels=self.features,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=0 if use_same_conv else self.padding,
            padding_mode=self.padding_mode.lower(),
            bias=bias,
            **kwargs,
        )


@dataclasses.dataclass
class ConvLSTMCellConfig:
    conv: ConvConfig
    pool_and_inject: Literal["horizontal", "vertical", "no"] = "horizontal"
    pool_projection: Literal["full", "per-channel", "max", "mean"] = "full"

    output_activation: Literal["sigmoid", "tanh"] = "sigmoid"
    forget_bias: float = 0.0
    fence_pad: Literal["same", "valid", "no"] = "same"


def make_postfix(pos, recurrent_step):
    return f".{pos}.{recurrent_step}"


class ConvLSTMCell(nn.Module):
    def __init__(
        self,
        cfg: ConvLSTMCellConfig,
        in_channels: int,
        prev_layer_hidden_channels: int,
        recurrent_steps: int,
        fancy_init: bool = False,
    ):
        super(ConvLSTMCell, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.prev_layer_hidden_channels = prev_layer_hidden_channels
        self.hidden_channels = cfg.conv.features
        self.recurrent_steps = recurrent_steps

        def input_dependent_postfixes(input):
            L, B = input.shape[:2]
            for b, pos, step in itertools.product(range(B), range(L), range(self.recurrent_steps)):
                postfix = make_postfix(pos, step)
                yield postfix

        conv_cfg = dataclasses.replace(cfg.conv, features=4 * self.hidden_channels)
        conv_ih_channels = self.in_channels + self.prev_layer_hidden_channels
        if self.cfg.pool_and_inject != "no":
            conv_ih_channels += self.hidden_channels
        self.conv_ih = conv_cfg.make_conv(in_channels=conv_ih_channels)
        self.hook_conv_ih = InputDependentHookPoint(input_dependent_postfixes)

        self.conv_hh = conv_cfg.make_conv(in_channels=self.hidden_channels, use_bias=False)
        self.hook_conv_hh = InputDependentHookPoint(input_dependent_postfixes)

        if cfg.fence_pad != "no":
            self.fence_conv = dataclasses.replace(conv_cfg, use_bias=False, padding=cfg.fence_pad).make_conv(in_channels=1)
            self.hook_fence_conv = InputDependentHookPoint(input_dependent_postfixes)

        if self.cfg.pool_and_inject == "no":
            self.pool_project = None
        elif self.cfg.pool_projection == "full":
            self.pool_project = nn.Linear(2 * self.hidden_channels, self.hidden_channels, bias=False)
            self.hook_pool_project = InputDependentHookPoint(input_dependent_postfixes)
        elif self.cfg.pool_projection == "per-channel":
            self.pool_project = nn.Parameter(th.Tensor(2, self.hidden_channels))
            self.hook_pool_project = InputDependentHookPoint(input_dependent_postfixes)
        else:
            self.pool_project = None

        if fancy_init:
            with th.no_grad():
                # Ensure we're splitting the conv weight by lstm_part correctly
                assert self.conv_ih.weight.size(0) == 4 * self.hidden_channels
                assert self.conv_hh.weight.size(0) == 4 * self.hidden_channels

                # We want to initialize for different nonlinearities in different parts of the conv
                for i, lstm_part in enumerate(["i", "f", "g", "o"]):
                    nonlin = "tanh" if lstm_part == "i" else "sigmoid"
                    nonlin = self.cfg.output_activation if lstm_part == "o" else nonlin
                    nn.init.kaiming_normal_(
                        self.conv_ih.weight[i * self.hidden_channels : (i + 1) * self.hidden_channels, ...],
                        nonlinearity=nonlin,
                    )
                    if self.conv_ih.bias is not None:
                        nn.init.normal_(self.conv_ih.bias, std=nn.init.calculate_gain(nonlin))

                    nn.init.kaiming_normal_(
                        self.conv_hh.weight[i * self.hidden_channels : (i + 1) * self.hidden_channels, ...],
                        nonlinearity=nonlin,
                    )
                    if self.conv_hh.bias is not None:
                        nn.init.normal_(self.conv_hh.bias, std=nn.init.calculate_gain(nonlin))

                # Also initialize pool_project
                pool_weight, pool_bias = None, None
                if isinstance(self.pool_project, nn.Linear):
                    pool_weight = self.pool_project.weight
                    pool_bias = self.pool_project.bias
                elif isinstance(self.pool_project, nn.Parameter):
                    pool_weight = self.pool_project
                if pool_weight is not None:
                    nn.init.kaiming_normal_(pool_weight, nonlinearity="linear")
                if pool_bias is not None:
                    nn.init.normal_(pool_bias, std=nn.init.calculate_gain("linear"))

        self.hook_i = InputDependentHookPoint(input_dependent_postfixes)
        self.hook_j = InputDependentHookPoint(input_dependent_postfixes)
        self.hook_f = InputDependentHookPoint(input_dependent_postfixes)
        self.hook_o = InputDependentHookPoint(input_dependent_postfixes)

        self.hook_input_h = InputDependentHookPoint(input_dependent_postfixes)
        self.hook_input_c = InputDependentHookPoint(input_dependent_postfixes)
        self.hook_h = InputDependentHookPoint(input_dependent_postfixes)
        self.hook_c = InputDependentHookPoint(input_dependent_postfixes)
        self.hook_layer_input = InputDependentHookPoint(input_dependent_postfixes)
        self.hook_prev_layer_hidden = InputDependentHookPoint(input_dependent_postfixes)

        self.interpretable_forward: Optional[Callable] = None  # interpretable forward function
        self.hook_interpretable_forward = InputDependentHookPoint(input_dependent_postfixes)

    def pool_and_project(self, to_pool: th.Tensor) -> th.Tensor:
        B, C, H, W = to_pool.shape
        h_max = to_pool.max(2).values.max(2).values
        h_mean = to_pool.mean(dim=(2, 3))
        if self.cfg.pool_projection == "max":
            pooled_h = h_max
        elif self.cfg.pool_projection == "mean":
            pooled_h = h_mean
        elif self.cfg.pool_projection == "full":
            assert self.pool_project is not None
            assert isinstance(self.pool_project, nn.Linear)
            h_max_and_mean = th.cat([h_max, h_mean], dim=1)
            pooled_h = self.pool_project(h_max_and_mean)

        elif self.cfg.pool_projection == "per-channel":
            assert self.pool_project is not None
            assert isinstance(self.pool_project, nn.Parameter)
            pooled_h = self.pool_project[0] * h_max + self.pool_project[1] * h_mean
        else:
            raise ValueError(f"{self.cfg.pool_projection=}")

        pooled_h_expanded = pooled_h[:, :, None, None].expand(B, C, H, W)
        return pooled_h_expanded

    def forward(
        self,
        skip_input: th.Tensor,
        prev_layer_hidden: th.Tensor,
        cur_state: ConvLSTMCellState,
        pos: int,
        tick: int,
        use_interpretable_forward: bool = False,
    ) -> tuple[th.Tensor, ConvLSTMCellState]:
        h_cur, c_cur = cur_state
        # Have to squeeze the incoming state because we added a n_layers dimension on purpose, to have the batch size be
        # the 2nd dimension
        h_cur = h_cur.squeeze(0)
        c_cur = c_cur.squeeze(0)

        postfix = make_postfix(pos, tick)

        h_cur, c_cur = self.hook_input_h(h_cur, postfix), self.hook_input_c(c_cur, postfix)
        skip_input = self.hook_layer_input(skip_input, postfix)
        prev_layer_hidden = self.hook_prev_layer_hidden(prev_layer_hidden, postfix)

        fence_feature = self.fence_feature(skip_input, postfix)

        if self.pool_project is not None:
            if self.cfg.pool_and_inject == "horizontal":
                to_pool = h_cur
            elif self.cfg.pool_and_inject == "vertical":
                to_pool = prev_layer_hidden
            else:
                raise ValueError(f"Invalid {self.cfg.pool_and_inject=}")
            assert self.cfg.pool_and_inject

            pooled_h = self.pool_and_project(to_pool)
            pooled_h = self.hook_pool_project(pooled_h, postfix)
            combined = th.cat([skip_input, prev_layer_hidden, pooled_h], dim=1)  # concatenate along channel axis
        else:
            assert self.cfg.pool_and_inject == "no"
            combined = th.cat([skip_input, prev_layer_hidden], dim=1)  # concatenate along channel axis

        conv_ih_output = self.conv_ih(combined)
        conv_ih_output = self.hook_conv_ih(conv_ih_output, postfix)
        conv_hh_output = self.conv_hh(h_cur)
        conv_hh_output = self.hook_conv_hh(conv_hh_output, postfix)
        combined_conv = conv_ih_output + conv_hh_output + fence_feature
        # cc_i, cc_f, cc_o, cc_g = th.split(combined_conv, self.hidden_channels, dim=1)
        cc_i, cc_j, cc_f, cc_o = th.split(combined_conv, self.hidden_channels, dim=1)
        i = th.tanh(cc_i)
        j = th.sigmoid(cc_j)
        f = th.sigmoid(cc_f + self.cfg.forget_bias)
        if self.cfg.output_activation == "sigmoid":
            o = th.sigmoid(cc_o)
        elif self.cfg.output_activation == "tanh":
            o = th.tanh(cc_o)
        else:
            raise ValueError(f"Invalid {self.cfg.output_activation=}")
        i, j, f, o = self.hook_i(i, postfix), self.hook_j(j, postfix), self.hook_f(f, postfix), self.hook_o(o, postfix)

        c_next = f * c_cur + i * j
        h_next = o * th.tanh(c_next)

        h_next = self.hook_h(h_next, postfix)
        c_next = self.hook_c(c_next, postfix)

        if use_interpretable_forward:
            if self.interpretable_forward is not None:
                h_next = self.interpretable_forward(h_next, skip_input, prev_layer_hidden, h_cur, pos, tick, self)
                # hook for convenience of fetching updated acts. We should use the above interpretable_forward function to make
                # any live changes to the hidden state.
                h_next = self.hook_interpretable_forward(h_next, postfix)

        # Unsqueeze state on the way out
        return h_next, (h_next.unsqueeze(0), c_next.unsqueeze(0))

    def recurrent_initial_state(
        self, image_size: tuple[int, int], n_envs: Optional[int] = None, *, device: Optional[th.device | str] = None
    ) -> ConvLSTMCellState:
        height, width = image_size

        shape: tuple[int, ...]
        if n_envs is None:
            shape = (1, self.hidden_channels, height, width)
        else:
            shape = (1, n_envs, self.hidden_channels, height, width)
        return (th.zeros(shape, device=device), th.zeros(shape, device=device))

    @staticmethod
    def output_spatial_shape(conv: nn.Conv2d, shape: tuple[int, int]) -> tuple[int, int]:
        """
        Output shape of the ConvLSTMCell, following the formula in the Pytorch docs
        (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
        """

        def _compute_shape(s: tuple[int, int], c: nn.Conv2d, i: int) -> int:
            pad: int
            if isinstance(conv, Conv2dSame):
                return s[i] // c.stride[i]
            if isinstance(c.padding, str):
                if c.padding == "valid":
                    pad = 0
                elif c.padding == "same":
                    return s[i] // c.stride[i]
                else:
                    raise NotImplementedError(f"Don't know how to handle {c.padding=}")
            else:
                pad = c.padding[i]
            base_value = s[i] + 2 * pad - c.dilation[i] * (c.kernel_size[i] - 1) - 1
            return base_value // c.stride[i] + 1

        return (_compute_shape(shape, conv, 0), _compute_shape(shape, conv, 1))

    def fence_feature(self, observation: th.Tensor, postfix) -> th.Tensor:
        """Adds a channel to seq_inputs which is 1 at the boundary and 0 inside"""
        if self.cfg.fence_pad == "no":
            return th.zeros(1, device=observation.device, dtype=observation.dtype)
        h, w = observation.shape[-2:]
        if self.cfg.fence_pad == "valid":
            if isinstance(self.cfg.conv.kernel_size, Sequence):
                h += self.cfg.conv.kernel_size[0] - 1
                w += self.cfg.conv.kernel_size[1] - 1
            else:
                h += self.cfg.conv.kernel_size - 1
                w += self.cfg.conv.kernel_size - 1
        elif self.cfg.fence_pad != "same":
            raise ValueError(f"Invalid {self.cfg.fence_pad=}")

        fence_feature = th.ones((h, w), dtype=observation.dtype, device=observation.device)

        # Paint the interior with zeros
        fence_feature[1:-1, 1:-1] = 0
        fence_feature = self.fence_conv(fence_feature.unsqueeze(0).unsqueeze(0))

        # Expand the batch dimensions
        fence_feature = fence_feature.expand(*observation.shape[:-3], *fence_feature.shape[-3:])
        fence_feature = self.hook_fence_conv(fence_feature, postfix)

        return fence_feature


@dataclasses.dataclass
class CompileConfig:
    options: dict[str, Any] = dataclasses.field(default_factory=lambda: {"triton.cudagraphs": True})
    fullgraph: bool = True
    backend: str = "inductor"

    def __call__(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        return th.compile(fn, options=self.options, fullgraph=self.fullgraph, backend=self.backend)


@dataclasses.dataclass
class BaseFeaturesExtractorConfig(abc.ABC):
    compile: Optional[CompileConfig] = None

    @abc.abstractmethod
    def kwargs(self, vec_env: VecEnv) -> dict[str, Any]:
        ...


@dataclasses.dataclass
class ConvLSTMOptions(BaseFeaturesExtractorConfig):
    embed: Sequence[ConvConfig] = dataclasses.field(default_factory=lambda: [ConvConfig(features=32, kernel_size=1)])
    recurrent: ConvLSTMCellConfig = dataclasses.field(
        default_factory=lambda: ConvLSTMCellConfig(ConvConfig(features=32, kernel_size=3))
    )
    n_recurrent: int = 1
    repeats_per_step: int = 1
    pre_model_nonlin: ActivationFnConfig = dataclasses.field(default_factory=ReLUConfig)
    residual: bool = False
    skip_final: bool = True
    fancy_init: bool = False  # TODO: run different values of this setting in test
    transpose: bool = False  # adds NCHWtoNHWC layer to the end. Useful for converting jax models to pytorch

    def kwargs(self, vec_env: VecEnv) -> dict[str, Any]:
        return dict(features_extractor_class=ConvLSTMFeaturesExtractor, features_extractor_kwargs=dict(cfg=self))


class ConvLSTMFeaturesExtractor(RecurrentFeaturesExtractor[th.Tensor, ConvLSTMState]):
    cell_list: Sequence[ConvLSTMCell]
    pre_model: th.nn.Module
    cfg: ConvLSTMOptions

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        cfg: ConvLSTMOptions,
    ):
        in_channels: int = observation_space.shape[0]

        out_spatial_shape = check_cast(tuple[int, int], observation_space.shape[1:])

        pre_model = []
        for i, conv_cfg in enumerate(cfg.embed):
            if i == 0:
                conv = conv_cfg.make_conv(in_channels=in_channels)
            else:
                conv = conv_cfg.make_conv()
            if cfg.fancy_init:
                nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity=cfg.pre_model_nonlin.name)
                if conv.bias is not None:
                    nn.init.normal_(conv.bias, std=nn.init.calculate_gain(cfg.pre_model_nonlin.name))
            pre_model.append(conv)
            out_spatial_shape = ConvLSTMCell.output_spatial_shape(conv, out_spatial_shape)
            nlin = cfg.pre_model_nonlin.fn()
            pre_model.append(nlin)

        skip_input_dim = in_channels if len(cfg.embed) == 0 else cfg.embed[-1].features
        cell_list: list[ConvLSTMCell] = []
        self.cell_input_shapes = []
        for i in range(cfg.n_recurrent):
            # In the first recurrent layer, there is a skip connection from the last time step's last layer. So the
            # previous hidden channels are just the top layer's amount of hidden channels.
            prev_layer_hidden_channels = cfg.recurrent.conv.features
            cell = ConvLSTMCell(
                cfg=cfg.recurrent,
                in_channels=skip_input_dim,
                prev_layer_hidden_channels=prev_layer_hidden_channels,
                recurrent_steps=cfg.repeats_per_step,
                fancy_init=cfg.fancy_init,
            )
            cell_list.append(cell)
            self.cell_input_shapes.append(out_spatial_shape)
            out_spatial_shape = ConvLSTMCell.output_spatial_shape(cell.conv_ih, out_spatial_shape)

        if len(cell_list) == 0:
            raise ValueError("There must be at least one recurrent layer")
        self.out_shape = (cfg.recurrent.conv.features, *out_spatial_shape)
        super().__init__(observation_space, features_dim=math.prod(self.out_shape))
        self.cfg = cfg

        self.cell_list = nn.ModuleList(cell_list)  # type: ignore
        self.pre_model = nn.Sequential(*pre_model)
        self.hook_pre_model = HookPoint()

        self.transpose_layer = NCHWtoNHWC() if cfg.transpose else nn.Identity()

    def forward(
        self,
        observations: th.Tensor,
        state: ConvLSTMState,
        episode_starts: th.Tensor,
        return_repeats: bool = False,
        use_interpretable_forward: bool = False,
    ) -> Tuple[th.Tensor, ConvLSTMState]:
        if episode_starts.ndim == 1:
            seq_len = 1
            batch_sz = episode_starts.shape[0]
            observations = observations.unsqueeze(0)
            episode_starts = episode_starts.unsqueeze(0)
            squeeze_end = not return_repeats
        else:
            assert episode_starts.ndim == 2
            seq_len, batch_sz = episode_starts.shape
            squeeze_end = False

        observations = observations.view(seq_len * batch_sz, *observations.shape[2:])
        observations = self.pre_model(observations)

        state = list(state)  # Copy the list
        multiplier = self.cfg.repeats_per_step if return_repeats else 1
        out_values: list[th.Tensor] = [None] * seq_len * multiplier  # type: ignore

        seq_inputs = observations.view(seq_len, batch_sz, *observations.shape[1:])
        seq_inputs = self.hook_pre_model(seq_inputs)
        for t in range(seq_len):
            obs_input = seq_inputs[t]

            # Reset state at this time step if necessary
            not_reset_state = ~episode_starts[t, :, None, None, None]
            for d in range(len(state)):
                state[d] = (not_reset_state * state[d][0], not_reset_state * state[d][1])

            for r in range(self.cfg.repeats_per_step):
                prev_layer_hidden_state = state[-1][0].squeeze(0)  # Top-down skip connection from previous time step
                for i, cell in enumerate(self.cell_list):
                    new_state_h, state[i] = cell.forward(
                        obs_input, prev_layer_hidden_state, state[i], t, r, use_interpretable_forward
                    )
                    if self.cfg.residual:
                        prev_layer_hidden_state = new_state_h + prev_layer_hidden_state
                    else:
                        prev_layer_hidden_state = new_state_h
                if not self.cfg.skip_final:  # Pass the residual connection on to the next repetition
                    state[-1] = (prev_layer_hidden_state.unsqueeze(0), state[-1][1])
                if return_repeats:
                    out_values[t * multiplier + r] = state[-1][0] + (obs_input.unsqueeze(0) if self.cfg.skip_final else 0)
                elif r == self.cfg.repeats_per_step - 1:
                    out_values[t] = state[-1][0] + (obs_input.unsqueeze(0) if self.cfg.skip_final else 0)

        if squeeze_end:
            return self.transpose_layer(out_values[0]).view(batch_sz, -1), state

        out = self.transpose_layer(th.cat(out_values, dim=0)).view(seq_len * multiplier, batch_sz, -1)

        return out, state

    def recurrent_initial_state(
        self, n_envs: Optional[int] = None, *, device: Optional[th.device | str] = None
    ) -> ConvLSTMState:
        return [
            cell.recurrent_initial_state(cell_input_shape, n_envs, device=device)
            for (cell_input_shape, cell) in zip(self.cell_input_shapes, self.cell_list)
        ]
