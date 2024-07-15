from typing import Dict, Literal, Optional

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.recurrent.policies import (
    RecurrentFeaturesExtractorActorCriticPolicy,
)
from stable_baselines3.common.type_aliases import check_cast
from stable_baselines3.common.vec_env import VecNormalize

from learned_planner.configs.train_drc import drc_1_1, drc_3_3
from learned_planner.convlstm import ConvConfig, ConvLSTMCell, ConvLSTMCellConfig, ConvLSTMOptions
from learned_planner.policies import ConvLSTMPolicyConfig


@pytest.mark.parametrize("pool_and_inject", ["horizontal", "no"])
def test_equivalent_to_lstm(
    pool_and_inject: Literal["horizontal", "vertical", "no"],
    batch_size=3,
    input_size=5,
    hidden_size=4,
    num_time=3,
    prev_layer_hidden_channels=1,
):
    th.manual_seed(1234)

    cfg = ConvLSTMCellConfig(
        conv=ConvConfig(
            features=hidden_size,
            kernel_size=1,
            strides=1,
            use_bias=True,
        ),
        pool_and_inject=pool_and_inject,
        fence_pad="no",
    )

    conv_cell = ConvLSTMCell(
        cfg=cfg,
        in_channels=input_size,
        prev_layer_hidden_channels=prev_layer_hidden_channels,
        recurrent_steps=1,
    )

    lstm_cell = th.nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=True)

    with th.no_grad():
        if isinstance(conv_cell.pool_project, th.nn.Linear):
            conv_cell.pool_project.weight.zero_()
        elif isinstance(conv_cell.pool_project, th.nn.Parameter):
            conv_cell.pool_project.data.zero_()

        conv_ih_weight = conv_cell.conv_ih.weight[:, :input_size, 0, 0]
        lstm_cell.weight_ih[:hidden_size].copy_(conv_ih_weight[hidden_size : 2 * hidden_size])
        lstm_cell.weight_ih[hidden_size : 2 * hidden_size].copy_(conv_ih_weight[2 * hidden_size : 3 * hidden_size])
        lstm_cell.weight_ih[2 * hidden_size : 3 * hidden_size].copy_(conv_ih_weight[:hidden_size])
        lstm_cell.weight_ih[3 * hidden_size :].copy_(conv_ih_weight[3 * hidden_size :])
        conv_cell.conv_ih.weight[:, input_size:, 0, 0].zero_()

        conv_hh_weight = conv_cell.conv_hh.weight[:, :, 0, 0]
        lstm_cell.weight_hh[:hidden_size].copy_(conv_hh_weight[hidden_size : 2 * hidden_size])
        lstm_cell.weight_hh[hidden_size : 2 * hidden_size].copy_(conv_hh_weight[2 * hidden_size : 3 * hidden_size])
        lstm_cell.weight_hh[2 * hidden_size : 3 * hidden_size].copy_(conv_hh_weight[:hidden_size])
        lstm_cell.weight_hh[3 * hidden_size :].copy_(conv_hh_weight[3 * hidden_size :])

        assert conv_cell.conv_ih.bias is not None
        bias_ih = conv_cell.conv_ih.bias
        lstm_cell.bias_ih[:hidden_size].copy_(bias_ih[hidden_size : 2 * hidden_size])
        lstm_cell.bias_ih[hidden_size : 2 * hidden_size].copy_(bias_ih[2 * hidden_size : 3 * hidden_size])
        lstm_cell.bias_ih[2 * hidden_size : 3 * hidden_size].copy_(bias_ih[:hidden_size])
        lstm_cell.bias_ih[3 * hidden_size :].copy_(bias_ih[3 * hidden_size :])
        lstm_cell.bias_hh.zero_()
    x = th.randn((batch_size, input_size))
    h = th.randn((batch_size, hidden_size))
    c = th.randn((batch_size, hidden_size))

    prev_layer_hidden_channels_tensor = th.zeros((batch_size, prev_layer_hidden_channels, 1, 1))

    for t in range(num_time):
        _, (next_h_conv, next_c_conv) = conv_cell.forward(
            x.unsqueeze(-1).unsqueeze(-1),
            prev_layer_hidden=prev_layer_hidden_channels_tensor,
            cur_state=(h.unsqueeze(-1).unsqueeze(-1), c.unsqueeze(-1).unsqueeze(-1)),
            pos=t,
            step=0,
        )

        next_h, next_c = lstm_cell(x, (h, c))

        assert th.allclose(next_h, next_h_conv.squeeze(-1).squeeze(-1)), f"h failed at {t=}"
        assert th.allclose(next_c, next_c_conv.squeeze(-1).squeeze(-1)), f"c failed at {t=}"

        h, c = next_h, next_c


# TODO: we haven't quite gotten the number of parameters right. But also it's not clear the table in the paper is
# reliable.
@pytest.mark.parametrize("drc, n_params", [(drc_3_3, 2_042_376), (drc_1_1, 1_752_456)])
def test_param_size(drc, n_params):
    cfg = check_cast(ConvLSTMPolicyConfig, drc())
    cfg_convlstm = check_cast(ConvLSTMOptions, cfg.features_extractor)
    cfg_convlstm.recurrent.pool_and_inject = "horizontal"
    cfg_convlstm.compile = None

    cfg_convlstm.embed[0].kernel_size = 8
    cfg_convlstm.embed[0].strides = 4
    cfg_convlstm.embed[0].padding = (3, 3)

    cfg_convlstm.embed[1].kernel_size = 4
    cfg_convlstm.embed[1].strides = 2
    cfg_convlstm.embed[1].padding = (1, 1)

    cls, kwargs = cfg.policy_and_kwargs(None)  # type: ignore
    observation_space = spaces.Box(0.0, 1.0, (3, 80, 80))
    action_space = spaces.Discrete(4)
    policy = cls(observation_space=observation_space, action_space=action_space, lr_schedule=1e-4, **kwargs)  # type: ignore

    x = th.randn((1, 1, 3, 80, 80))
    policy(x, policy.recurrent_initial_state(1), th.zeros(x.shape[:2], dtype=th.bool))
    print(list(n for (n, p) in policy.named_parameters()))

    actual_n_params = sum(p.numel() for p in policy.parameters())
    assert actual_n_params >= n_params
    assert actual_n_params <= n_params * 21 // 20
