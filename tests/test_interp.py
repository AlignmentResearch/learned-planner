import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Literal
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch as th
import wandb  # noqa: F401  # pyright: ignore
from cleanba import cleanba_impala
from farconf import update_fns_to_cli
from stable_baselines3.common.type_aliases import check_cast

from learned_planner import LP_DIR, MODEL_PATH_IN_REPO
from learned_planner.__main__ import main
from learned_planner.configs.command_config import WandbCommandConfig
from learned_planner.configs.train_drc import DeviceLiteral
from learned_planner.configs.train_probe import train_probe as train_probe_cfg_fn
from learned_planner.convlstm import ConvLSTMOptions
from learned_planner.environments import BoxobanConfig
from learned_planner.interp import train_probes
from learned_planner.interp.collect_dataset import DatasetStore
from learned_planner.interp.plot import save_video
from learned_planner.interp.utils import jax_to_th, load_jax_model_to_torch
from learned_planner.policies import download_policy_from_huggingface

# when running the test using pytest, the DatasetStore class is not available in the main module
# and pickle.load searches for it in the main module. So we need to set it manually
setattr(sys.modules["__main__"], "DatasetStore", DatasetStore)

MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)


@pytest.fixture
def load_jax_and_torch_model(BOXOBAN_CACHE):
    env_cfg = BoxobanConfig(
        cache_path=BOXOBAN_CACHE,
        n_envs=1,
        n_envs_to_render=0,
        max_episode_steps=120,
        tinyworld_obs=True,
    )
    jax_policy, carry_t, jax_args, train_state, _ = cleanba_impala.load_train_state(MODEL_PATH, env_cfg)
    cfg_th, policy_th = load_jax_model_to_torch(MODEL_PATH, env_cfg)
    return (jax_policy, carry_t, jax_args, train_state), (cfg_th, policy_th)


def test_model_conversion(load_jax_and_torch_model):
    (jax_policy, carry_t, _, train_state), (_, policy_th) = load_jax_and_torch_model
    key, k1 = jax.random.split(jax.random.PRNGKey(1234), 2)

    obs = jax.random.randint(k1, (1, 1, 3, 10, 10), 0, 255)
    obs_th = th.tensor(np.asarray(obs), dtype=th.int32)
    carry_th = [
        [
            jax_to_th(e.h.transpose(0, 3, 1, 2)).unsqueeze(0),
            jax_to_th(e.c.transpose(0, 3, 1, 2)).unsqueeze(0),
        ]
        for e in carry_t
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

    recurrent_hooks, non_recurrent_hooks = 14, 8
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

    assert len(manual_cache) == len(run_with_cache_cache), "manual_cache and run_with_cache_cache have different keys"

    for k in manual_cache.keys():
        manual_value = manual_cache[k]
        run_with_cache_value = run_with_cache_cache[k]
        did_modify_inplace = th.any(manual_value != run_with_cache_value)
        if did_modify_inplace:
            print("mismatch for key", k, "do you modify it in place?")
            assert not did_modify_inplace


@pytest.mark.parametrize("train_fn", [train_probe_cfg_fn])
@pytest.mark.parametrize("device", ["cpu"])
@patch("wandb.log")
def test_train_probe(
    _wandb_log: Mock,
    tmpdir: Path,
    train_fn: Callable[[], WandbCommandConfig],
    device: Literal["cpu", "cuda"],
):
    # save dataset python save_ds.py --dataset_path ./ --labels_type pred_value --num_datapoints 5
    script_path = LP_DIR / "learned_planner/interp/save_ds.py"
    acts_path = Path(__file__).parent / "probes_dataset"
    cmd = f"python {script_path} --dataset_path {acts_path} --labels_type pred_value --num_datapoints 5 --for_test"
    subprocess.run(cmd, shell=True, check=True)
    
    def _update_train_fn(cfg: WandbCommandConfig, device: DeviceLiteral, training_mount: Path) -> WandbCommandConfig:
        cfg.base_save_prefix = training_mount
        cfg.cmd = check_cast(train_probes.TrainProbeConfig, cfg.cmd)
        cfg.cmd.weight_decay = 1e-4
        cfg.cmd.device = device
        cfg.cmd.eval_cache_path = Path(__file__).parent
        cfg.cmd.eval_difficulty = "unfiltered"
        cfg.cmd.batch_size = 5
        cfg.cmd.eval_episodes = 1
        cfg.cmd.dataset_path = Path(__file__).parent / "probes_dataset/pred_value_5_skip5.pt"
        cfg.cmd.train_on.dataset_name = "pred_value"
        cfg.cmd.policy_path = str(MODEL_PATH)
        cfg.cmd.train_on.mean_pool_grid = True
        return cfg

    cli, _ = update_fns_to_cli(
        train_fn,
        partial(
            _update_train_fn,
            device=device,
            training_mount=tmpdir,
        ),
    )
    _, metrics = main(cli, run_dir=tmpdir)  # type: ignore
    key = "train/l-all_c-all_ds-pred_value_mpg-True/loss"
    assert metrics[key] < 0.01, f"train loss should be less than 0.01, got {metrics[key]}"


# Support: pytest -v -k test_probe_datasets -o python_functions=test_probe_datasets[True]
@pytest.mark.parametrize("save", [True, False])
def test_probe_datasets(save: bool):
    dataset_path = Path(__file__).parent / "probes_dataset"
    acts_ds = train_probes.ActivationsDataset(dataset_path, num_data_points=1)
    ds_cache = DatasetStore.load(acts_ds.level_files[0])
    all_probe_names = [
        "agent_in_a_cycle",
        "agents_future_position_map",
        "agents_future_direction_map",
        "boxes_future_position_map",
        "boxes_future_direction_map",
        "next_target",
        "next_box",
    ]
    all_probe_infos = [train_probes.TrainOn(layer=-1, dataset_name=name) for name in all_probe_names]
    all_probe_preds = [getattr(ds_cache, name)().numpy() for name in all_probe_names]

    imgs = ds_cache.obs.numpy()
    box_labels = ds_cache.boxes_label_map().numpy()
    target_labels = ds_cache.target_labels_map().numpy()

    save_video(
        "all_gt_videos.mp4",
        imgs,
        all_probe_preds,
        all_probe_preds,
        all_probe_infos=all_probe_infos,
        box_labels=box_labels,
        target_labels=target_labels,
    )

    assert all(len(probe_pred) == len(imgs) for probe_pred in all_probe_preds)
