from typing import Callable

import numpy as np
import torch as th
from cleanba import cleanba_impala
from cleanba import convlstm as cleanba_convlstm

from learned_planner.activation_fns import IdentityActConfig, ReLUConfig
from learned_planner.convlstm import CompileConfig, ConvConfig, ConvLSTMCellConfig, ConvLSTMOptions
from learned_planner.policies import ConvLSTMPolicyConfig, NetArchConfig


def jax_to_th(x):
    return th.tensor(np.asarray(x))


def conv_args_process(conv_args):
    d = conv_args.__dict__
    lower_keys = ["padding", "padding_mode"]
    skip_keys = ["initialization"]
    ret = {}
    for k, v in d.items():
        if k in skip_keys:
            continue
        if k in lower_keys and isinstance(v, str):
            v = v.lower()
        ret[k] = v
    return ret


def jax_to_torch_cfg(jax_cfg):
    assert isinstance(jax_cfg, cleanba_convlstm.ConvLSTMConfig)
    mlp_hiddens = jax_cfg.mlp_hiddens
    recurrent_conv = ConvConfig(**conv_args_process(jax_cfg.recurrent.conv))
    recurrent_less_conv = dict(jax_cfg.recurrent.__dict__)
    del recurrent_less_conv["conv"]
    recurrent = ConvLSTMCellConfig(recurrent_conv, **recurrent_less_conv)
    return ConvLSTMPolicyConfig(
        features_extractor=ConvLSTMOptions(
            compile=CompileConfig(),
            embed=[ConvConfig(**conv_args_process(jax_embed)) for jax_embed in jax_cfg.embed],
            recurrent=recurrent,
            n_recurrent=jax_cfg.n_recurrent,
            repeats_per_step=jax_cfg.repeats_per_step,
            pre_model_nonlin=ReLUConfig() if jax_cfg.use_relu else IdentityActConfig(),
            skip_final=jax_cfg.skip_final,
            residual=jax_cfg.residual,
        ),
        net_arch=NetArchConfig(mlp_hiddens, mlp_hiddens),
    )


def jax_get(param_name, params):
    param_name = param_name.split(".")
    ret_params = params
    for p in param_name:
        ret_params = ret_params[p]
    return ret_params


def copy_params_from_jax(torch_policy, jax_params, jax_args):
    h, w = 10, 10
    network_params = jax_params["network_params"]

    num_recurrent_layers = jax_args.net.n_recurrent
    num_embed_layers = len(jax_args.net.embed)
    is_pool_and_inject = jax_args.net.recurrent.pool_and_inject != "no"
    num_mlps = len(jax_args.net.mlp_hiddens)
    hidden_channels = jax_args.net.recurrent.conv.features

    # copy embed params
    for i in range(num_embed_layers):
        conv = torch_policy.features_extractor.pre_model[2 * i]
        conv.weight.data.copy_(th.tensor(np.asarray(jax_get(f"conv_list_{i}.kernel", network_params)).transpose(3, 2, 0, 1)))
        conv.bias.data.copy_(th.tensor(np.asarray(jax_get(f"conv_list_{i}.bias", network_params))))

    # copy recurrent conv params
    for i in range(num_recurrent_layers):
        cell_i = torch_policy.features_extractor.cell_list[i]

        for th_key, jax_key in [("conv_ih", "ih"), ("conv_hh", "hh"), ("fence_conv", "fence")]:
            conv = getattr(cell_i, th_key)
            weight = np.asarray(jax_get(f"cell_list_{i}.{jax_key}.kernel", network_params).transpose(3, 2, 0, 1))
            if jax_key == "fence":
                print("fence", weight.shape)
                weight = np.sum(weight, axis=1, keepdims=True)
                print("fence", weight.shape)

            conv.weight.data.copy_(th.tensor(weight))
            try:
                bias = np.asarray(jax_get(f"cell_list_{i}.{jax_key}.bias", network_params))
                conv.bias.data.copy_(th.tensor(bias))
            except KeyError:
                pass

        if is_pool_and_inject:
            weight = np.asarray(jax_get(f"cell_list_{i}.project", network_params))
            cell_i.pool_project.data.copy_(th.tensor(weight))

    # copy actor, critic params
    for i in range(num_mlps):
        mlp_weights = jax_get(f"dense_list_{i}.kernel", network_params).transpose()
        if i == 0:
            mlp_weights = th.tensor(np.asarray(mlp_weights.reshape(mlp_weights.shape[0], h, w, hidden_channels)))
            mlp_weights = mlp_weights.permute(0, 3, 1, 2).reshape(mlp_weights.shape[0], -1)
        else:
            mlp_weights = th.tensor(np.asarray(mlp_weights))
        mlp_bias = np.asarray(jax_get(f"dense_list_{i}.bias", network_params))
        getattr(torch_policy.mlp_extractor.policy_net, f"fc{i}").weight.data.copy_(mlp_weights)
        getattr(torch_policy.mlp_extractor.policy_net, f"fc{i}").bias.data.copy_(th.tensor(mlp_bias))
        getattr(torch_policy.mlp_extractor.value_net, f"fc{i}").weight.data.copy_(mlp_weights)
        getattr(torch_policy.mlp_extractor.value_net, f"fc{i}").bias.data.copy_(th.tensor(mlp_bias))

    th_keys = ["action_net", "value_net"]
    jax_keys = ["actor_params.Output", "critic_params.Output"]

    for th_key, jax_key in zip(th_keys, jax_keys):
        mlp_weights = np.asarray(jax_get(f"{jax_key}.kernel", jax_params).transpose())
        mlp_bias = np.asarray(jax_get(f"{jax_key}.bias", jax_params))
        getattr(torch_policy, th_key).weight.data.copy_(th.tensor(mlp_weights))
        getattr(torch_policy, th_key).bias.data.copy_(th.tensor(mlp_bias))


def load_jax_model_to_torch(path, vec_env):
    _, _, args, state, _ = cleanba_impala.load_train_state(path, vec_env)
    cfg = jax_to_torch_cfg(args.net)
    policy_cls, kwargs = cfg.policy_and_kwargs(vec_env)  # type: ignore
    assert isinstance(policy_cls, Callable)
    policy = policy_cls(
        observation_space=vec_env.single_observation_space,
        action_space=vec_env.action_space,
        activation_fn=th.nn.ReLU,
        lr_schedule=lambda _: 0.0,
        normalize_images=True,
        **kwargs,
    )
    copy_params_from_jax(policy, state.params["params"], args)
    return cfg, policy
