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


def patch_in_box(
    cache,
    hook,
    layer,
    h_or_c,
    coef,
    intercept,
    alpha,
    positive_square,
    negative_squares,
    probe_info,
    per_segment_neurons,
    n_segments,
):
    segment_idx = 2 * layer + h_or_c if probe_info.layer == -1 else h_or_c
    steering_vector = coef[:, per_segment_neurons * segment_idx : per_segment_neurons * (segment_idx + 1)]
    dot_product = th.sum(steering_vector[..., None, None] * cache, dim=1, keepdim=True)  # (b, c, h, w)
    dot_product += intercept / n_segments
    probe_logit = th.max(th.tensor(0.0), dot_product)
    norm_sq = th.sum(steering_vector**2, dim=1, keepdim=True)

    cache[:, :, negative_squares[:, 0], negative_squares[:, 1]] -= (
        probe_logit[:, :, negative_squares[:, 0], negative_squares[:, 1]] / norm_sq
    ) * steering_vector.unsqueeze(-1)

    magnitude = th.max(
        th.tensor(0.0), (alpha / n_segments - dot_product[:, :, positive_square[0], positive_square[1]]) / norm_sq
    )
    cache[:, :, positive_square[0], positive_square[1]] += magnitude * steering_vector
    return cache


def get_fwd_hooks(hook_fn):
    num_layers = 3 if probe_info.layer == -1 else 1
    hook_h_cs = ["hook_h", "hook_c"]
    fwd_hooks = [
        (
            f"features_extractor.cell_list.{layer}.{h_or_c_name}.{pos}.{int_pos}",
            partial(hook_fn, layer=layer, h_or_c=h_or_c_idx),
        )
        for pos in range(1)
        for int_pos in range(3)
        for layer in (range(num_layers) if num_layers > 1 else [probe_info.layer])
        for h_or_c_idx, h_or_c_name in enumerate(hook_h_cs)
    ]
    return fwd_hooks


def get_next_box(ds_cache):
    all_boxes_at_timesteps = ds_cache.get_box_positions()
    first_different_box_pos = (all_boxes_at_timesteps != all_boxes_at_timesteps[:1]).any((-1, -2)).nonzero()
    if len(first_different_box_pos) > 0:
        first_different_box_pos = first_different_box_pos[0, 0].item()
        next_box = ds_cache.different_positions(
            all_boxes_at_timesteps[0], all_boxes_at_timesteps[first_different_box_pos], successive_positions=True
        )[0].pop()
        next_box = th.tensor(next_box)
    else:
        next_box = th.zeros(2)
    all_boxes = all_boxes_at_timesteps[0]
    return next_box, all_boxes


def get_next_target(ds_cache):
    all_targets = ds_cache.get_target_positions()
    try:
        next_target = ds_cache.next_target(map_to_grid=False)[0]
    except IndexError:
        next_target = th.zeros(2)
    return next_target, all_targets


def get_next_pos(ds_cache, next_box):
    if next_box:
        next_pos, all_pos = get_next_box(ds_cache)
    else:
        next_pos, all_pos = get_next_target(ds_cache)
    return next_pos, all_pos


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
    hook_steps=-1,
    next_box=False,
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
        max_steps=30,
    )

    ds_cache = DatasetStore(None, out.obs.squeeze(1), out.rewards, True, out.acts, th.zeros(len(out.obs)), {})
    next_pos, all_pos = get_next_pos(ds_cache, next_box=next_box)

    all_ci_outputs = []
    for idx in range(len(all_pos)):
        positive_square = all_pos[idx]
        if (positive_square == next_pos).all().item():
            continue

        negative_squares = th.cat([all_pos[:idx], all_pos[idx + 1 :]])

        for logit in logits:
            fwd_hooks = get_fwd_hooks(
                partial(
                    patch_in_box,
                    coef=coef,
                    intercept=intercept,
                    alpha=logit,
                    positive_square=positive_square,
                    negative_squares=negative_squares,
                    probe_info=probe_info,
                    per_segment_neurons=per_segment_neurons,
                    n_segments=n_segments,
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
                hook_steps=range(hook_steps) if hook_steps >= 0 else -1,
                max_steps=30,
            )
            steer_ds_cache = DatasetStore(
                None,
                steer_out.obs.squeeze(1),
                steer_out.rewards,
                True,
                steer_out.acts,
                th.zeros(len(steer_out.obs)),
                {},
            )
            steer_next_pos, _ = get_next_pos(steer_ds_cache, next_box=next_box)
            ci_success = (steer_next_pos == positive_square).all().item()
            all_ci_outputs.append((lfi, li, idx, logit, ci_success))
            print("Level:", (lfi, li), "idx:", idx, "Logit:", logit, "CI success:", ci_success)
            if save_video:
                name = f"automated_cis/{lfi}_{li}_steer_{'box' if next_box else 'target'}_{positive_square}_logit_{logit}.mp4"
                save_video_fn(name, steer_out.obs.squeeze(1), steer_out.probe_outs, all_probe_infos=probe_infos)
    all_ci_outputs = np.array(all_ci_outputs, dtype=int)
    if len(all_ci_outputs) > 0:
        mean_ci_success = all_ci_outputs[:, -1].mean()
        all_ci_outputs = all_ci_outputs.reshape(-1, 5)
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

    num_layers = 3 if probe_info.layer == -1 else 1
    n_segments = 2 * num_layers
    per_segment_neurons = coef.shape[1] // n_segments

    return coef, intercept, n_segments, per_segment_neurons


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probe_path",
        type=str,
        default="probes/best/next_target_l-all.pkl",
        help="Path of the probe on disk or on learned-planner huggingface repo",
    )
    parser.add_argument("--probe_wandb_id", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="next_target")
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
    parser.add_argument("--logits", type=str, default="10,15,20,25,30", help="logits to try")
    args = parser.parse_args()

    probe, probe_info, multioutput = get_probe_and_info(args)
    coef, intercept, n_segments, per_segment_neurons = get_coef(probe, probe_info, multioutput)

    next_box = "box" in args.dataset_name

    split = args.split
    difficulty = args.difficulty
    logits = [int(logit) for logit in args.logits.split(",")]

    boxo_cfg = get_boxoban_cfg(
        difficulty=difficulty,
        split=split if args.split != "None" and args.split != "" else None,
    )
    if ON_CLUSTER:
        args.output_base_path = pathlib.Path("/training/") / args.output_base_path
    args.output_base_path = pathlib.Path(args.output_base_path) /  f"{'box' if next_box else 'target'}/{split}_{difficulty}"
    args.output_base_path.mkdir(parents=True, exist_ok=True)

    map_fn = partial(
        ci_score_on_a_level,
        probe=probe,
        probe_info=probe_info,
        coef=coef,
        intercept=intercept,
        logits=logits,
        per_segment_neurons=per_segment_neurons,
        n_segments=n_segments,
        boxo_cfg=boxo_cfg,
        save_video=args.save_video,
        hook_steps=args.hook_steps,
        next_box=next_box,
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
        columns=["lfi", "li", "box" if next_box else "target", "logit", "ci_success"],
    )
    split = "None" if split is None else split
    csv_file = args.output_base_path / file_name
    df.to_csv(csv_file, index=False)
    print("Final Mean ci success: ", df["ci_success"].mean(), "Total: ", len(results))
