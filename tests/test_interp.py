import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch as th
from cleanba import cleanba_impala
from cleanba.environments import BoxobanConfig
from huggingface_hub import snapshot_download

from learned_planner.convlstm import ConvLSTMOptions
from learned_planner.interp.utils import jax_to_th, load_jax_model_to_torch

MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000/"
MODEL_BASE_PATH = pathlib.Path(
    snapshot_download("AlignmentResearch/learned-planner", allow_patterns=[MODEL_PATH_IN_REPO + "*"]),
)
MODEL_PATH = MODEL_BASE_PATH / MODEL_PATH_IN_REPO


@pytest.fixture
def load_jax_and_torch_model(BOXOBAN_CACHE):
    env = BoxobanConfig(
        cache_path=BOXOBAN_CACHE,
        num_envs=1,
        max_episode_steps=120,
        asynchronous=False,
        tinyworld_obs=True,
    ).make()
    jax_policy, carry_t, jax_args, train_state, _ = cleanba_impala.load_train_state(MODEL_PATH, env)
    cfg_th, policy_th = load_jax_model_to_torch(MODEL_PATH, env)
    return (jax_policy, carry_t, jax_args, train_state), (cfg_th, policy_th)


def test_model_conversion(load_jax_and_torch_model):
    (jax_policy, carry_t, _, train_state), (_, policy_th) = load_jax_and_torch_model
    key, k1 = jax.random.split(jax.random.PRNGKey(1234), 2)

    obs = jax.random.randint(k1, (1, 1, 3, 10, 10), 0, 255)
    obs_th = th.tensor(np.asarray(obs), dtype=th.int32)
    carry_th = [
        [jax_to_th(e.h.transpose(0, 3, 1, 2)).unsqueeze(0), jax_to_th(e.c.transpose(0, 3, 1, 2)).unsqueeze(0)] for e in carry_t
    ]
    eps_start = jnp.zeros((1, 1), dtype=jnp.bool_)
    eps_start_th = th.tensor(np.asarray(eps_start))
    _, _, values_jax, _ = jax_policy.apply(  # type: ignore
        train_state.params,
        carry_t,
        obs,
        eps_start,
        method=jax_policy.get_logits_and_value,
    )
    values_th = policy_th.predict_values(obs_th, carry_th, eps_start_th).item()
    assert np.isclose(values_jax.item(), values_th, atol=1e-4)


def test_no_modify_inplace(load_jax_and_torch_model, model=None, device="cpu"):
    _, (cfg, policy) = load_jax_and_torch_model
    # This test compares cached things with outputs from run_with_cache
    # it detects if there was in-place modification changing the output of run_with_hooks after things are stored
    # (a bug that causes issues when doing patching)

    manual_cache = {}

    def save_hook(tensor, hook):
        manual_cache[hook.name] = tensor.detach().clone()
        return tensor.clone()

    seq_len = 4
    obs = th.randint(0, 255, (seq_len, 2, 3, 10, 10))
    carry = policy.recurrent_initial_state(2)
    eps_start = th.zeros((4, 2), dtype=th.bool)
    eps_start[0, :] = True
    model_args = [obs, carry, eps_start]
    fwd_hooks = []
    bwd_hooks = []
    # fwd_hooks=None expands all input_dependent hooks
    # we need this context to get the input dependent hooks
    with policy.input_dependent_hooks_context(
        fwd_hooks=None,
        bwd_hooks=None,
        model_args=model_args,
        setup_all_input_hooks=True,
        model_kwargs={},
    ):
        for name, hp in policy.hook_dict.items():
            fwd_hooks.append((name, save_hook))

    assert isinstance(cfg.features_extractor, ConvLSTMOptions)

    recurrent_hooks, non_recurrent_hooks = 8, 8
    total_hooks = non_recurrent_hooks + recurrent_hooks * cfg.features_extractor.n_recurrent * (
        seq_len * cfg.features_extractor.repeats_per_step + 1
    )
    assert len(fwd_hooks) == total_hooks, f"Expected {total_hooks} hooks, got {len(fwd_hooks)}"

    _ = policy.run_with_hooks(
        *model_args,
        fwd_hooks=fwd_hooks,
        bwd_hooks=bwd_hooks,
    )

    run_with_cache_logit, run_with_cache_cache = policy.run_with_cache(*model_args)

    for k in manual_cache.keys():
        manual_value = manual_cache[k]
        run_with_cache_value = run_with_cache_cache[k]
        did_modify_inplace = th.any(manual_value != run_with_cache_value)
        if did_modify_inplace:
            print("mismatch for key", k, "do you modify it in place?")
            assert not did_modify_inplace
