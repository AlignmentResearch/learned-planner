import argparse
import multiprocessing
import pathlib
from functools import partial

import numpy as np
import pandas as pd
import torch as th
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm

from learned_planner import BOXOBAN_CACHE, MODEL_PATH_IN_REPO, ON_CLUSTER
from learned_planner.interp.collect_dataset import DatasetStore
from learned_planner.interp.plot import save_video as save_video_fn
from learned_planner.interp.train_probes import TrainOn
from learned_planner.interp.utils import get_boxoban_cfg, load_jax_model_to_torch, load_probe, play_level
from learned_planner.policies import download_policy_from_huggingface

MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)


def patch_in_box_direction(cache, hook, layer, h_or_c, coef_from_my_map, alpha, probe_info, n_segments, per_segment_neurons):
    segment_idx = 2 * layer + h_or_c if probe_info.layer == -1 else h_or_c
    # cache += coef_from_my_map[per_segment_neurons * segment_idx : per_segment_neurons * (segment_idx + 1)].unsqueeze(0) * bd_magnitude
    ci_vector = coef_from_my_map[per_segment_neurons * segment_idx : per_segment_neurons * (segment_idx + 1)].unsqueeze(0)
    dot_product = th.sum(ci_vector * cache, dim=1, keepdim=True)  # (s, b, 10, 10)
    magnitude = th.max(
        th.tensor(0.0), ((alpha / n_segments) - dot_product) / (th.sum(ci_vector**2, dim=1, keepdim=True) + 1e-8)
    )
    cache += magnitude * ci_vector
    return cache


def get_fwd_hooks(hook_fn):
    num_layers = 3 if probe_info.layer == -1 else 1
    hook_h_cs = ["hook_h", "hook_c"]
    fwd_hooks = [
        (
            f"features_extractor.cell_list.{layer}.{h_or_c_name}.{pos}.{int_pos}",
            # partial(patch_in_box_direction, layer=layer, h_or_c=h_or_c_idx),
            partial(hook_fn, layer=layer, h_or_c=h_or_c_idx),
        )
        for pos in range(1)
        for int_pos in range(3)
        for layer in (range(num_layers) if num_layers > 1 else [probe_info.layer])
        for h_or_c_idx, h_or_c_name in enumerate(hook_h_cs)
    ]
    return fwd_hooks


def ci_score_on_a_level(
    level_idx_tuple,
    boxo_cfg,
    probe,
    probe_info,
    coef,
    intercept,
    logits,
    per_segment_neurons,
    n_segments,
    save_video=False,
    timestep_to_fetch_probe_plan=5,
    hook_steps=-1,
    box_direction=True,
):
    boxo_env = boxo_cfg.make()
    _, policy_th = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)
    lfi, li = level_idx_tuple
    reset_opts = {"level_file_idx": lfi, "level_idx": li}
    obs = boxo_env.reset(options=reset_opts)[0]
    obs = th.tensor(obs)

    probes, probe_infos = [probe], [probe_info]
    out = play_level(
        boxo_env,
        policy_th=policy_th,
        reset_opts=reset_opts,
        probes=probes,
        probe_train_ons=probe_infos,
        thinking_steps=0,
        max_steps=80,  # 80 steps to compute the gt
    )

    ds_cache = DatasetStore(None, out.obs.squeeze(1), out.rewards, out.solved, out.acts, th.zeros(len(out.obs)), {})
    if box_direction:
        gt = ds_cache.boxes_future_direction_map(multioutput=multioutput).numpy()
    else:
        gt = ds_cache.agents_future_direction_map(multioutput=multioutput).numpy()
    probe_preds = out.probe_outs[0]

    probe_plan = probe_preds[timestep_to_fetch_probe_plan]
    gt = gt[timestep_to_fetch_probe_plan]

    idx_i, idx_j = np.where((gt != -1) & (gt == probe_plan))

    all_ci_outputs = []

    def flip_direction(direction):
        # up -> down, down -> up, left -> right, right -> left
        return [1, 0, 3, 2][direction]

    for i, j in zip(idx_i, idx_j):
        walls_on_side = ds_cache.get_wall_directions(i, j)
        next_to_wall = walls_on_side.any()
        # if ds_cache.is_next_to_a_wall(i, j):
        #     continue

        direction_idx = probe_plan[i, j]
        assert direction_idx > -1
        # ortho_directions = [2, 3] if direction_idx < 2 else [0, 1]
        ci_directions = [i for i in range(4) if i != direction_idx]
        for ci_direction in ci_directions:
            my_direction_map = -1 * np.ones((10, 10))
            my_direction_map[i, j] = ci_direction

            my_map_as_idx = (th.tensor(my_direction_map) + 1).to(th.int64)
            coef_from_my_map = th.index_select(coef, 0, my_map_as_idx.view(-1)).view(10, 10, -1).permute(2, 0, 1)
            intercept_from_my_map = th.index_select(intercept, 0, my_map_as_idx.view(-1)).view(1, 1, 10, 10)
            # we try three different logit values to pick the best one
            for logit in logits:
                alpha = -intercept_from_my_map + logit

                fwd_hooks = get_fwd_hooks(
                    partial(
                        patch_in_box_direction,
                        coef_from_my_map=coef_from_my_map,
                        alpha=alpha,
                        probe_info=probe_info,
                        n_segments=n_segments,
                        per_segment_neurons=per_segment_neurons,
                    )
                )
                steer_out = play_level(
                    boxo_env,
                    policy_th=policy_th,
                    reset_opts=reset_opts,
                    probes=probes,
                    probe_train_ons=probe_infos,
                    thinking_steps=0,
                    fwd_hooks=fwd_hooks,
                    # hook_steps=range(11),
                    hook_steps=range(hook_steps) if hook_steps >= 0 else -1,
                    max_steps=80,
                )
                steer_ds_cache = DatasetStore(
                    None,
                    steer_out.obs.squeeze(1),
                    steer_out.rewards,
                    steer_out.solved,
                    steer_out.acts,
                    th.zeros(len(steer_out.obs)),
                    {},
                )
                steer_gt = steer_ds_cache.boxes_future_direction_map(multioutput=multioutput).numpy()
                steer_gt = steer_gt[timestep_to_fetch_probe_plan]
                if next_to_wall:
                    box_dir_cond = walls_on_side[ci_direction] or walls_on_side[flip_direction(ci_direction)]
                    agent_dir_cond = walls_on_side[direction_idx]
                    if (box_direction and box_dir_cond) or (not box_direction and agent_dir_cond):
                        # CI to push on or against a wall then the box should not move
                        ci_success = steer_gt[i, j] == -1
                    else:
                        ci_success = steer_gt[i, j] == ci_direction
                else:
                    ci_success = steer_gt[i, j] == ci_direction
                all_ci_outputs.append((lfi, li, i, j, next_to_wall, direction_idx, ci_direction, logit, ci_success))
                if save_video:
                    name = f"automated_cis/{lfi}_{li}_idx_{i}_{j}_steered_{direction_idx}_{ci_direction}_logit_{logit}.mp4"
                    save_video_fn(name, steer_out.obs.squeeze(1), steer_out.probe_outs, all_probe_infos=probe_infos)

    all_ci_outputs = np.array(all_ci_outputs, dtype=int)
    if len(all_ci_outputs) > 0:
        mean_ci_success = all_ci_outputs[:, -1].mean()
        all_ci_outputs = all_ci_outputs.reshape(-1, 9)
    else:
        mean_ci_success = "NA"
    print("Level:", (lfi, li), "Mean ci success:", mean_ci_success, "Total:", len(all_ci_outputs))
    return all_ci_outputs


def get_random_levels(num_levels, total_files):
    if difficulty == "hard":
        total_levels = 3332
    else:
        total_levels = 1000 * total_files
    return [(ri // 1000, ri % 1000) for ri in np.random.randint(total_levels, size=num_levels)]


def get_probe_and_info(args):
    probe, grid_wise = load_probe(args.probe_path, args.probe_wandb_id)
    multioutput = isinstance(probe, MultiOutputClassifier)

    if multioutput:
        raise NotImplementedError

    probe_info = TrainOn(layer=args.layer, grid_wise=grid_wise, dataset_name=args.dataset_name, hooks=args.hooks.split(","))
    return probe, probe_info, multioutput


def get_coef(probe, probe_info, multioutput):
    if multioutput:
        coef = th.tensor([probe.estimators_[i].coef_.squeeze(0) for i in range(len(probe.estimators_))])
        intercept = th.tensor([probe.estimators_[i].intercept_ for i in range(len(probe.estimators_))])
    else:
        coef = th.tensor(probe.coef_)
        intercept = th.tensor(probe.intercept_)
        # zero out null direction as we only want to steer in the 4 directions
        coef[0] = 0
        intercept[0] = 0

    num_layers = 3 if probe_info.layer == -1 else 1
    n_segments = 2 * num_layers
    per_segment_neurons = coef.shape[1] // n_segments

    return coef, intercept, n_segments, per_segment_neurons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe_path",
        type=str,
        default="probes/best/boxes_future_direction_map_l-all.pkl",
        help="Path of the probe on disk or on learned-planner huggingface repo",
    )
    parser.add_argument("--probe_wandb_id", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="boxes_future_direction_map")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--hooks", type=str, default="hook_h,hook_c")
    parser.add_argument("--level", type=int, nargs=2, default=(-1, -1), help="level file index, level index")
    parser.add_argument("--hook_steps", type=int, default=-1, help="hook steps")
    parser.add_argument("--split", type=str, default="valid", help="split")
    parser.add_argument("--difficulty", type=str, default="medium", help="difficulty")
    parser.add_argument("--num_levels", type=int, default=10, help="number of levels to run")
    parser.add_argument("--save_video", action="store_true", help="save video")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--output_base_path", type=str, default="iclr_logs/ci_score/", help="Path to save plots and cache.")
    parser.add_argument("--logits", type=str, default="15,20,25,30", help="logits to try")
    args = parser.parse_args()

    probe, probe_info, multioutput = get_probe_and_info(args)
    coef, intercept, n_segments, per_segment_neurons = get_coef(probe, probe_info, multioutput)

    dataset_name = args.dataset_name
    assert "direction" in dataset_name, "Only direction datasets are supported"
    box_direction = "box" in dataset_name
    split = args.split
    difficulty = args.difficulty
    logits = [int(logit) for logit in args.logits.split(",")]

    boxo_cfg = get_boxoban_cfg(
        difficulty=difficulty,
        split=split if args.split != "None" and args.split != "" else None,
        use_envpool=False,  # envpool doesn't support options on reset
    )

    if ON_CLUSTER:
        args.output_base_path = pathlib.Path("/training/") / args.output_base_path
    args.output_base_path = pathlib.Path(args.output_base_path) / dataset_name / f"{split}_{difficulty}"
    args.output_base_path.mkdir(parents=True, exist_ok=True)

    map_fn = partial(
        ci_score_on_a_level,
        boxo_cfg=boxo_cfg,
        probe=probe,
        probe_info=probe_info,
        coef=coef,
        intercept=intercept,
        logits=logits,
        per_segment_neurons=per_segment_neurons,
        n_segments=n_segments,
        save_video=args.save_video,
        hook_steps=args.hook_steps,
        box_direction=box_direction,
    )

    if all(v >= 0 for v in args.level):
        lfi, li = args.level
        print("Running on level", args.level)
        results = map_fn(args.level)
        file_name = f"ci_results_lfi_{lfi}_li_{li}.csv"
    else:
        np.random.seed(args.seed)

        level_files_dir = BOXOBAN_CACHE / "boxoban-levels-master" / difficulty / split
        num_files = len(list(level_files_dir.glob("*.txt")))

        lfi_li_list = get_random_levels(args.num_levels, num_files)
        if args.num_workers > 1:
            pool = multiprocessing.Pool(args.num_workers)
            results = list(tqdm(pool.imap(map_fn, lfi_li_list), total=len(lfi_li_list)))
            pool.close()
            pool.join()
        else:
            results = [map_fn(lfi_li) for lfi_li in tqdm(lfi_li_list)]
        results = np.concatenate(results)
        file_name = "ci_results.csv"
    df = pd.DataFrame(
        results,
        columns=["lfi", "li", "i", "j", "next_to_wall", "direction_idx", "ci_direction", "logit", "ci_success"],
    )
    split = "None" if split is None else split
    csv_file = args.output_base_path / file_name
    df.to_csv(csv_file, index=False)
    print("Final Mean ci success: ", df["ci_success"].mean(), "Total: ", len(results))
