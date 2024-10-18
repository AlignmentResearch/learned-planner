import pathlib
import pickle
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
import torch as th
from cleanba import cleanba_impala
from cleanba import convlstm as cleanba_convlstm
from cleanba.environments import BoxobanConfig, EnvpoolBoxobanConfig, convert_to_cleanba_config
from gymnasium.spaces import MultiDiscrete
from sklearn.multioutput import MultiOutputClassifier

from learned_planner import BOXOBAN_CACHE, ON_CLUSTER
from learned_planner.activation_fns import IdentityActConfig, ReLUConfig
from learned_planner.convlstm import CompileConfig, ConvConfig, ConvLSTMCellConfig, ConvLSTMOptions
from learned_planner.interp.render_svg import BOX, BOX_ON_TARGET, FLOOR, PLAYER, TARGET
from learned_planner.policies import ConvLSTMPolicyConfig, NetArchConfig, download_policy_from_huggingface


@dataclass
class PlayLevelOutput:
    obs: th.Tensor
    acts: th.Tensor
    logits: th.Tensor
    rewards: th.Tensor
    lengths: th.Tensor
    solved: th.Tensor
    cache: dict[str, th.Tensor]
    probe_outs: Optional[list[np.ndarray]] = None
    sae_outs: Optional[th.Tensor] = None


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
                weight = np.sum(weight, axis=1, keepdims=True)

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


def load_jax_model_to_torch(path, env_cfg):
    env_cfg = convert_to_cleanba_config(env_cfg)
    vec_env = env_cfg.make()
    _, _, args, state, _ = cleanba_impala.load_train_state(path, env_cfg)
    cfg = jax_to_torch_cfg(args.net)
    policy_cls, kwargs = cfg.policy_and_kwargs(vec_env)  # type: ignore
    assert isinstance(policy_cls, Callable)
    action_space = vec_env.action_space
    if isinstance(action_space, MultiDiscrete):
        action_space = action_space[0]
    policy = policy_cls(
        observation_space=vec_env.single_observation_space,
        action_space=action_space,
        activation_fn=th.nn.ReLU,
        lr_schedule=lambda _: 0.0,
        normalize_images=True,
        **kwargs,
    )
    policy.eval()
    copy_params_from_jax(policy, state.params["params"], args)
    return cfg, policy


def get_boxoban_cfg(
    num_envs: int = 1,
    episode_steps=80,
    difficulty: str = "medium",
    split: str = "valid",
    load_sequentially_envpool=True,
    use_envpool=False,
    boxoban_cache=BOXOBAN_CACHE,
    **kwargs,
):
    if ON_CLUSTER or use_envpool:
        cfg_cls = EnvpoolBoxobanConfig
        extra_kwargs: dict[str, Any] = dict(load_sequentially=load_sequentially_envpool, **kwargs)
    else:
        cfg_cls = BoxobanConfig
        extra_kwargs: dict[str, Any] = dict(asynchronous=False, tinyworld_obs=True, **kwargs)
    return cfg_cls(
        cache_path=boxoban_cache,
        num_envs=num_envs,
        max_episode_steps=episode_steps,
        min_episode_steps=episode_steps,
        difficulty=difficulty,  # type: ignore
        split=split,  # type: ignore
        **extra_kwargs,
    )


def load_policy(
    local_or_hgf_repo_path: str = "drc33/bkynosqi/cp_2002944000/",
    difficulty: str = "medium",
    split: str = "valid",
):
    model_path = download_policy_from_huggingface(local_or_hgf_repo_path)
    boxo_cfg = get_boxoban_cfg(difficulty=difficulty, split=split)
    cfg, policy_th = load_jax_model_to_torch(model_path, boxo_cfg)
    return cfg, policy_th


def load_probe(path: str | pathlib.Path = "", wandb_id: str = ""):
    if path != "" and wandb_id != "":
        raise ValueError("Cannot specify both probe_path and probe_wandb_id")

    if wandb_id != "":
        command = f"/training/findprobe.sh {wandb_id}"
        path = subprocess.run(command, shell=True, capture_output=True, text=True).stdout
        path = path.strip()

    path = download_policy_from_huggingface(path)  # returns same path if it exists or else tries to download from huggingface
    if not path.exists():
        raise FileNotFoundError(f"Probe file not found at {path}")

    with open(path, "rb") as f:
        probe = pickle.load(f)
    grid_wise = probe.n_features_in_ % 100 != 0
    return probe, grid_wise


def is_probe_multioutput(probe):
    return isinstance(probe, MultiOutputClassifier)


def prepare_cache_values(
    cache: dict[str, th.Tensor],
    layer: int,
    hooks: list[str],
    step: int,
    internal_steps: bool = False,
    is_concatenated_cache: bool = False,
) -> list[list[th.Tensor]]:
    key = "features_extractor.cell_list.{layer}.{hook}.{step}.{internal_step}"
    int_steps = [0, 1, 2] if internal_steps else [2]
    if is_concatenated_cache:
        key = key.replace(".{step}.{internal_step}", "")
        cache_values = [
            th.tensor(cache[key.format(layer=layer, hook=hook)])
            for layer in (range(3) if layer == -1 else [layer])
            for hook in hooks
        ]
        cache_values = [cache_values]
    else:
        cache_values = [
            [
                cache[key.format(layer=layer, step=step, internal_step=int_step, hook=hook)]
                for layer in (range(3) if layer == -1 else [layer])
                for hook in hooks
            ]
            for int_step in int_steps
        ]
    return cache_values


def predict(cache, probe, train_on, step: int, internal_steps: bool = False, is_concatenated_cache=False):
    """Predict the probe on the activations of the policy.

    Args:
        cache (dict): Activations of the policy.
        probe (sklearn.linear_model.LogisticRegression): Probe to run on the activations of the policy.
        train_on (ProbeTrainOn): Configuration to train the probe on.
        step (int): Step at which to run the probe. In most cases, this will be 0 as we don't evaluate simultaneously
            on multiple steps. We sequentially evaluate the policy by interacting with the environment.
        internal_steps (bool): Whether to run the probe on all internal steps or just the last one. Assumes 3 internal steps.

    Returns:
        np.ndarray: Probe predictions.
    """
    cache_values = prepare_cache_values(cache, train_on.layer, train_on.hooks, step, internal_steps, is_concatenated_cache)

    assert all(
        [len(cache_value.shape) == 4 for cache_values_at_a_step in cache_values for cache_value in cache_values_at_a_step]
    )
    # assert len(cache_values.shape) == 4
    s, b, _, h, w = len(cache_values), *cache_values[0][0].shape
    if train_on.grid_wise:
        cache_values = [th.cat(cache_values_at_a_step, dim=1) for cache_values_at_a_step in cache_values]
        stack_cache_values = th.stack(cache_values, dim=0)
        stack_cache_values = stack_cache_values.permute(0, 1, 3, 4, 2)
        stack_cache_values = stack_cache_values.reshape(-1, stack_cache_values.shape[-1]).cpu()
        probe_preds = probe.predict(stack_cache_values)
        if isinstance(probe, MultiOutputClassifier):
            probe_preds = probe_preds.reshape(s, b, h, w, -1)
        else:
            probe_preds = probe_preds.reshape(s, b, h, w)
    else:
        cache_values = [
            th.cat(
                [
                    cache_value_at_a_step.reshape(cache_value_at_a_step.shape[0], -1)
                    for cache_value_at_a_step in cache_values_at_a_step
                ],
                dim=1,
            )
            for cache_values_at_a_step in cache_values
        ]
        stack_cache_values = th.stack(cache_values, dim=0)
        # stack_cache_values = stack_cache_values.reshape(*stack_cache_values.shape[:2], -1)
        stack_cache_values = stack_cache_values.reshape(-1, stack_cache_values.shape[-1]).cpu()
        probe_preds = probe.predict(stack_cache_values)
        probe_preds = probe_preds.reshape(s, b)
    if is_concatenated_cache:
        probe_preds = probe_preds.squeeze()
    return probe_preds


def process_cache_for_sae(cache_tensor, grid_wise: bool = False):
    if len(cache_tensor.shape) == 4:
        if grid_wise:
            if isinstance(cache_tensor, np.ndarray):
                return np.transpose(cache_tensor, (0, 2, 3, 1)).reshape(-1, cache_tensor.shape[1])
            else:
                return cache_tensor.permute(0, 2, 3, 1).reshape(-1, cache_tensor.shape[1])
        else:
            # TODO: check if this is correct since channels should be flattened together
            return cache_tensor.reshape(cache_tensor.shape[0], -1)
    elif len(cache_tensor.shape) == 5:
        if grid_wise:
            if isinstance(cache_tensor, np.ndarray):
                return np.transpose(cache_tensor, (0, 1, 3, 4, 2)).reshape(-1, cache_tensor.shape[2])
            else:
                return cache_tensor.permute(0, 1, 3, 4, 2).reshape(-1, cache_tensor.shape[2])
        else:
            # TODO: check if this is correct
            return cache_tensor.reshape(cache_tensor.shape[0] * cache_tensor.shape[1], -1)


def encode_with_sae(
    sae,
    cache,
    internal_steps=False,
    decode=False,
    is_concatenated_cache=False,
) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
    if is_concatenated_cache:
        original_act = cache[sae.cfg.hook_name]
        assert len(original_act.shape) == 4
        initial_dims = original_act.shape[:1]
        processed_act = process_cache_for_sae(original_act, grid_wise=sae.cfg.grid_wise)
        if isinstance(processed_act, np.ndarray):
            processed_act = th.tensor(processed_act)
    else:
        int_steps = [0, 1, 2] if internal_steps else [2]
        original_act = th.stack([cache[sae.cfg.hook_name + f".0.{i}"] for i in int_steps])
        initial_dims = original_act.shape[:2]
        processed_act = process_cache_for_sae(original_act, grid_wise=sae.cfg.grid_wise)

    sae_feature_activations = sae.encode(processed_act.to(sae.device))  # type: ignore
    sae_feature_reshaped = sae_feature_activations.reshape(*initial_dims, 10, 10, -1)
    if decode:
        sae_out = sae.decode(sae_feature_activations).to(original_act.device)
        return sae_feature_reshaped, sae_out.reshape(*initial_dims, 10, 10, -1)
    return sae_feature_reshaped


def play_level(
    env,
    policy_th,
    reset_opts={},
    probes=[],
    probe_train_ons=[],
    sae=None,
    thinking_steps=0,
    max_steps=120,
    internal_steps=False,
    fwd_hooks=None,
    hook_steps=-1,
    names_filter=None,
):
    """Execute the policy on the environment and the probe on the policy's activations.

    Args:
        env (gymnasium.Env): Environment to play the level in.
        policy_th (torch.nn.Module): Policy to play the level with.
        reset_opts (dict): Options to reset the environment with. Useful for custom-built levels
            or providing the `level_file_idx` and `level_idx` of a level in Boxoban.
        probes (list[sklearn.linear_model.LogisticRegression]): Probes to run on the activations of the policy.
        probe_train_ons (list[ProbeTrainOn]): Correponding configuration of the probe.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Observations, actions, rewards, all probe outputs.
    """
    assert len(probe_train_ons) == len(probes)
    try:
        start_obs = env.reset(options=reset_opts)[0]
    except:  # noqa
        start_obs = env.reset()[0]
    device = policy_th.device
    start_obs = th.tensor(start_obs, device=device)
    all_obs = [start_obs]
    all_acts = []
    all_logits = []
    all_rewards = []
    all_cache = []
    all_sae_outs = []

    all_probe_outs = [[] for _ in probes]
    N = start_obs.shape[0]
    eps_done = th.zeros(N, dtype=th.bool)
    eps_solved = th.zeros(N, dtype=th.bool)
    episode_lengths = th.zeros(N, dtype=th.int32)

    state = policy_th.recurrent_initial_state(N, device=device)
    obs = start_obs
    r, d, t = [0.0], th.tensor([False] * N, dtype=th.bool, device=device), th.tensor([False] * N, dtype=th.bool, device=device)
    for i in range(max_steps):
        (distribution, state), cache = run_fn_with_cache(
            policy_th,
            "get_distribution",
            obs,
            state,
            th.tensor([0.0] * N, dtype=th.bool, device=device),
            # return_repeats=False,
            fwd_hooks=fwd_hooks if (hook_steps == -1) or (i in hook_steps) else None,
            names_filter=names_filter,
        )
        best_act = distribution.get_actions(deterministic=True)
        all_acts.append(best_act)
        all_logits.append(distribution.distribution.logits.detach())
        if i >= thinking_steps:
            obs, r, d, t, _ = env.step(best_act.cpu().numpy())
            d, t = th.tensor(d), th.tensor(t)
            obs = th.tensor(obs, device=device)
            eps_done |= d | t
            episode_lengths[~eps_solved] += 1
            eps_solved |= d

            all_rewards.append(r)
        all_obs.append(obs)
        all_cache.append(cache)

        for pidx, (probe, probe_train_on) in enumerate(zip(probes, probe_train_ons)):
            probe_out = predict(cache, probe, probe_train_on, step=0, internal_steps=internal_steps)
            if N == 1:
                probe_out = probe_out.squeeze(1)
            if not internal_steps:
                probe_out = probe_out.squeeze(0)
            all_probe_outs[pidx].append(probe_out)
        if sae:
            sae_acts = encode_with_sae(sae, cache, internal_steps=internal_steps)
            all_sae_outs.append(sae_acts.squeeze(1) if internal_steps else sae_acts.squeeze(0).squeeze(0))  # type: ignore
        if eps_done.all().item():
            break
    return PlayLevelOutput(
        obs=th.stack(all_obs[:-1]).cpu(),
        acts=th.stack(all_acts),
        logits=th.stack(all_logits),
        rewards=th.tensor(np.array(all_rewards)),
        lengths=episode_lengths,
        solved=eps_solved,
        cache={k: th.stack([cache[k].cpu() for cache in all_cache]) for k in all_cache[0].keys()},
        probe_outs=[np.stack(probe_out) for probe_out in all_probe_outs],
        sae_outs=th.stack(all_sae_outs) if sae else None,
    )


def run_fn_with_cache(
    hooked_model,
    fn_name: str,
    *model_args,
    names_filter=None,
    device=None,
    remove_batch_dim=False,
    incl_bwd=False,
    reset_hooks_end=True,
    clear_contexts=False,
    # pos_slice=None,
    **model_kwargs,
):
    """Combines the run_with_cache functions from MambaLens and TransformerLens to run arbitrary functions
    with cache."""
    model_kwargs = dict(list(model_kwargs.items()))
    fwd_hooks = None
    if "fwd_hooks" in model_kwargs:
        fwd_hooks = model_kwargs["fwd_hooks"]
        del model_kwargs["fwd_hooks"]
    bwd_hooks = None
    if "bwd_hooks" in model_kwargs:
        bwd_hooks = model_kwargs["bwd_hooks"]
        del model_kwargs["bwd_hooks"]
    # need to wrap run_with_cache to setup input_dependent hooks
    setup_all_input_hooks = False

    # turn names_filter into a fwd_hooks for setup input dependent hooks stuff
    if names_filter is None:
        setup_all_input_hooks = True
    else:
        name_fake_hooks = [(name, None) for name in names_filter]
        if fwd_hooks is None:
            fwd_hooks = name_fake_hooks
        else:
            fwd_hooks = fwd_hooks + name_fake_hooks

    with hooked_model.input_dependent_hooks_context(
        *model_args, fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, setup_all_input_hooks=setup_all_input_hooks, **model_kwargs
    ):
        fwd_hooks = [(name, hook) for name, hook in (fwd_hooks if fwd_hooks else []) if hook is not None]
        bwd_hooks = bwd_hooks if bwd_hooks else []
        with hooked_model.hooks(fwd_hooks, bwd_hooks, reset_hooks_end, clear_contexts) as hooked_hooked_model:
            # pos_slice = Slice.unwrap(pos_slice)

            cache_dict, fwd, bwd = hooked_hooked_model.get_caching_hooks(
                names_filter,
                incl_bwd,
                device,
                remove_batch_dim=remove_batch_dim,
                # pos_slice=pos_slice,
            )

            with hooked_hooked_model.hooks(
                fwd_hooks=fwd,
                bwd_hooks=bwd,
                reset_hooks_end=reset_hooks_end,
                clear_contexts=clear_contexts,
            ):
                if fn_name:
                    model_out = getattr(hooked_hooked_model, fn_name)(*model_args, **model_kwargs)
                else:
                    model_out = hooked_hooked_model(*model_args, **model_kwargs)
                if incl_bwd:
                    model_out.backward()

    return model_out, cache_dict


def get_metrics(preds: np.ndarray, labels: np.ndarray, classification: bool, key_prefix: str = ""):
    if classification:
        try:
            negative_label = labels.min()
        except ValueError:
            return {}
        preds = preds[: len(labels)]
        assert len(preds.shape) == len(labels.shape), f"{preds.shape} != {labels.shape}"
        acc = (preds == labels).mean()
        prec = (preds[preds != negative_label] == labels[preds != negative_label]).mean()
        rec = (preds[labels != negative_label] == labels[labels != negative_label]).mean()
        f1 = 2 * prec * rec / (prec + rec)
        metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    else:
        loss = th.nn.functional.mse_loss(th.tensor(preds), th.tensor(labels))
        metrics = {"loss": loss.item()}
    if key_prefix:
        metrics = {f"{key_prefix}/{k}": v for k, v in metrics.items()}
    return metrics


def get_player_pos(obs):
    """Get the position of the player in the observation using the pixel values of the player.

    The pixel values are taken from here:
    https://github.com/AlignmentResearch/gym-sokoban/tree/default/gym_sokoban/envs/render_utils.py#L113-L120

    Args:
        obs (np.ndarray): Observation of the level of shape (10, 10, 3).

    Returns:
        Tuple[int, int]: Position of the player in the observation.
    """
    # assert isinstance(obs, np.ndarray) and obs.shape == (3, 10, 10)
    assert isinstance(obs, np.ndarray) and obs.shape == (10, 10, 3)
    # agent_pos = np.where(((obs[0] == 160) | (obs[0] == 219)) & (obs[1] == 212) & (obs[2] == 56))
    agent_pos = np.where(((obs[..., 0] == 160) | (obs[..., 0] == 219)) & (obs[..., 1] == 212) & (obs[..., 2] == 56))
    assert len(agent_pos[0]) == 1
    return agent_pos[0][0], agent_pos[1][0]


def get_solved_obs(obs):
    """Get the solved observation by taking the observation right before the level is solved.

    Args:
        obs (np.ndarray): Observation of the level of shape (3, 10, 10).

    Returns:
        np.ndarray: Solved observation.
    """
    assert len(obs.shape) == 3
    permuted = False
    if isinstance(obs, th.Tensor):
        obs = obs.clone()
        if obs.shape[2] != 3:
            obs = obs.permute(1, 2, 0)
            permuted = True
        box_on_target = th.tensor(BOX_ON_TARGET)
        floor = th.tensor(FLOOR)
        player = th.tensor(PLAYER)
    else:
        obs = obs.copy()
        if obs.shape[2] != 3:
            obs = np.transpose(obs, (1, 2, 0))
            permuted = True
        box_on_target = BOX_ON_TARGET
        floor = FLOOR
        player = PLAYER
    assert obs.shape[2] == 3, f"Expected RGB image at first or last axis, got shape {obs.shape}"

    obs[(obs == TARGET).all(-1)] = box_on_target  # type: ignore
    box_off_target = (obs == BOX).all(-1)
    assert box_off_target.sum() == 1, f"Expected 1 box off target, got {box_off_target.sum()}"
    obs[(obs == player).all(-1)] = floor  # type: ignore
    obs[box_off_target] = player  # type: ignore

    if permuted:
        if isinstance(obs, th.Tensor):
            obs = obs.permute(2, 0, 1)
        else:
            obs = np.transpose(obs, (2, 0, 1))
    return obs
