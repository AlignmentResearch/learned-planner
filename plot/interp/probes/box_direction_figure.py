# %%
import argparse
import os
from functools import partial

import numpy as np
import torch as th
from gym_sokoban.envs.sokoban_env import CHANGE_COORDINATES
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import learned_planner.interp.plot  # noqa
from learned_planner import LP_DIR, MODEL_PATH_IN_REPO, ON_CLUSTER
from learned_planner.interp.collect_dataset import DatasetStore
from learned_planner.interp.plot import save_video as save_video_fn
from learned_planner.interp.render_svg import tiny_world_rgb_to_svg
from learned_planner.interp.train_probes import TrainOn
from learned_planner.interp.utils import (
    get_boxoban_cfg,
    is_probe_multioutput,
    load_jax_model_to_torch,
    load_probe,
    play_level,
)
from learned_planner.policies import download_policy_from_huggingface

parser = argparse.ArgumentParser()
parser.add_argument("--fancy", action="store_true")
parser.add_argument("--save_video", action="store_true")
args = parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


set_seed(42)
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

if ON_CLUSTER:
    path, wandb_id = "", "vb6474rg"
    agent_path, agent_wandb_id = "", "dirnsbf3"
else:
    path, wandb_id = LP_DIR / "probes/best/boxes_future_direction_map_l-all.pkl", ""
    agent_path, agent_wandb_id = LP_DIR / "probes/best/agents_future_direction_map_l-all.pkl", ""

box_probe, grid_wise = load_probe(path, wandb_id)
box_probe_info = TrainOn(layer=-1, grid_wise=grid_wise, dataset_name="boxes_future_direction_map")

agent_probe, grid_wise = load_probe(agent_path, agent_wandb_id)
agent_probe_info = TrainOn(layer=-1, grid_wise=grid_wise, dataset_name="agents_future_direction_map")

# probes = [box_probe, agent_probe]
# probe_infos = [box_probe_info, agent_probe_info]
probes = [agent_probe, box_probe]
probe_infos = [agent_probe_info, box_probe_info]
multioutputs = [is_probe_multioutput(agent_probe), is_probe_multioutput(box_probe)]

difficulty = "hard"
split = None

boxo_cfg = get_boxoban_cfg(difficulty=difficulty, split=split)
boxo_env = boxo_cfg.make()
cfg_th, policy_th = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)

thinking_steps = 0
# %%


def plt_obs_with_direction_probe_for_paper(probe_preds, gt_labels, ax, color_scheme=["red", "green", "blue"], vector=False):
    """Helper function to plot the level image with the direction probe predictions."""
    # scale = 96 if args.fancy else 1
    # offset = 0.5 if args.fancy else 0
    assert probe_preds.ndim == 3 and probe_preds.shape[2] == 4
    assert gt_labels.ndim == 3 and gt_labels.shape[2] == 4
    grid = np.arange(10, dtype=float)
    # grid += offset
    max_value = probe_preds.max()
    print("Max prediction value:", max_value)
    if vector:
        grid += 0.5
        probe_preds = probe_preds[::-1]
        gt_labels = gt_labels[::-1]
    # grid = scale * grid
    for dir_idx in range(4):
        probe_preds_dir = probe_preds[..., dir_idx] > 0.0
        gt_labels_dir = gt_labels[..., dir_idx] > 0.0 if gt_labels is not None else None
        delta_i, delta_j = CHANGE_COORDINATES[dir_idx]
        # delta_i, delta_j = scale * delta_i, scale * delta_j
        cmap = ListedColormap(color_scheme[:2])
        color_args = [(gt_labels_dir == probe_preds_dir).astype(int)]
        color_args[0][0, 0] = 0  # to avoid color collapse when preds are correct

        ax.quiver(
            grid,
            grid,
            delta_j * probe_preds_dir,
            -delta_i * probe_preds_dir,
            *color_args,
            cmap=cmap,  # only used when color_args is not empty
            color=color_scheme[2],  # only used when color_args is empty
            scale_units="xy",
            alpha=np.maximum(probe_preds[..., dir_idx] / max_value, 0.2),
            scale=1,
            minshaft=1,
            minlength=0,
            width=0.009,
            headwidth=3,
        )


def plt_vector_obs(img, ax):
    ax.pcolormesh(img[::-1])
    ax.axis("off")


def play_level_and_plot_directions(
    reset_opts: dict,
    plot_box_probe: bool = False,
    thinking_steps: int = 0,
    fwd_hooks=None,
    hook_steps: int | list[int] = -1,
    save_name: str = "fig1.svg",
    save_video: bool = False,
    max_steps: int = 80,
):
    out = play_level(
        boxo_env,
        policy_th=policy_th,
        reset_opts=reset_opts,
        probes=probes,
        probe_train_ons=probe_infos,
        thinking_steps=thinking_steps,
        max_steps=max_steps + thinking_steps,
        fwd_hooks=fwd_hooks,
        hook_steps=hook_steps,
    )
    all_obs = out.obs.squeeze(1)
    ds = DatasetStore(None, all_obs[thinking_steps:], out.rewards, out.solved, out.acts, th.zeros(len(all_obs)), {})
    # ds_name = probe_infos[plot_box_probe].dataset_name
    # gt = getattr(ds, ds_name)(multioutput=multioutputs[plot_box_probe], variable_boxes=True)
    gts = [getattr(ds, probe_infos[i].dataset_name)(multioutput=multioutputs[i], variable_boxes=True) for i in range(2)]
    gt = gts[plot_box_probe]
    plan = out.probe_outs[plot_box_probe]
    plan_onehot = th.nn.functional.one_hot(th.tensor(plan + 1), num_classes=5).numpy()[..., 1:]
    gt_onehot = th.nn.functional.one_hot(gt.to(th.long) + 1, num_classes=5).numpy()[..., 1:]

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    img = all_obs[0].permute(1, 2, 0).numpy()
    plt_vector_obs(img, ax)
    plt_obs_with_direction_probe_for_paper(plan_onehot.sum(axis=0), gt_onehot.sum(axis=0), ax, vector=True)
    os.makedirs("svgs/" + (save_name.rsplit("/", 1)[0] if "/" in save_name else ""), exist_ok=True)
    svg_save_name = save_name.replace(".svg", ("_agent" if plot_box_probe else "_box") + "_directions.svg")
    fig.savefig("svgs/" + svg_save_name, backend="svg", transparent=True)
    # plt.show()
    svg = tiny_world_rgb_to_svg(img)
    with open("svgs/" + svg_save_name.replace(".svg", "_fancy.svg"), "w") as f:
        f.write(svg)

    if save_video:
        name = save_name.replace(".svg", f"{'_fancy' if args.fancy else ''}.mp4")
        save_video_fn(
            name, all_obs, out.probe_outs, [gt.numpy() for gt in gts], all_probe_infos=probe_infos, fancy_sprite=args.fancy
        )


lfi, li = 0, 23
reset_opts = {"level_file_idx": lfi, "level_idx": li}
play_level_and_plot_directions(
    reset_opts,
    save_name=f"fig1_level_{lfi}_{li}.svg",
    save_video=args.save_video,
)

# %% Custom Levels
dim_room = 10
U, D, L, R = 0, 1, 2, 3


def construct_direction_map(directions, start_square):
    # print("start_square", start_square)
    my_direction_map = -1 * np.ones((10, 10))
    next_box = None
    for d in directions:
        if my_direction_map[start_square] == -1:
            my_direction_map[start_square] = d
        if d == U:
            start_square = (start_square[0] - 1, start_square[1])
        elif d == D:
            start_square = (start_square[0] + 1, start_square[1])
        elif d == L:
            start_square = (start_square[0], start_square[1] - 1)
        elif d == R:
            start_square = (start_square[0], start_square[1] + 1)
        if next_box is None:
            next_box = start_square
    return my_direction_map, next_box


def patch_in_box_direction(cache, hook, layer, h_or_c, coef_from_my_map, n_segments, per_segment_neurons, alpha):
    segment_idx = 2 * layer + h_or_c if box_probe_info.layer == -1 else h_or_c
    probeslice = slice(segment_idx * per_segment_neurons, (segment_idx + 1) * per_segment_neurons)
    steering_vector = coef_from_my_map[probeslice].unsqueeze(0)
    dot_product = th.sum(steering_vector * cache, dim=1, keepdim=True)  # (s, b, 10, 10)
    magnitude = th.max(
        th.tensor(0.0), ((alpha / n_segments) - dot_product) / (th.sum(steering_vector**2, dim=1, keepdim=True) + 1e-8)
    )
    cache += magnitude * steering_vector
    return cache


def path_specific_hooks(directions_map, probe, probe_info, logit=15.0):
    coef = th.tensor(probe.coef_)
    intercept = th.tensor(probe.intercept_)
    coef[0] = 0
    intercept[0] = 0
    num_layers = 3 if probe_info.layer == -1 else 1
    n_segments = 2 * num_layers
    per_segment_neurons = coef.shape[1] // n_segments

    directions_map_as_idx = (th.tensor(directions_map) + 1).to(th.int64)
    coef_for_directions_map = th.index_select(coef, 0, directions_map_as_idx.view(-1)).view(10, 10, -1).permute(2, 0, 1)
    intercept_for_directions_map = th.index_select(intercept, 0, directions_map_as_idx.view(-1)).view(1, 1, 10, 10)
    alpha = -intercept_for_directions_map + logit

    hook_h_cs = ["hook_h", "hook_c"]
    fwd_hooks = [
        (
            f"features_extractor.cell_list.{layer}.{h_or_c_name}.{pos}.{int_pos}",
            partial(
                patch_in_box_direction,
                layer=layer,
                h_or_c=h_or_c_idx,
                coef_from_my_map=coef_for_directions_map,
                n_segments=n_segments,
                per_segment_neurons=per_segment_neurons,
                alpha=alpha,
            ),
        )
        for pos in range(1)
        for int_pos in range(3)
        for layer in (range(3) if probe_info.layer == -1 else [probe_info.layer])
        for h_or_c_idx, h_or_c_name in enumerate(hook_h_cs)
    ]
    return fwd_hooks


# %% Box Near Target

walls = [(0, i) for i in range(dim_room)] + [(dim_room - 1, i) for i in range(dim_room)]
walls += [(i, 0) for i in range(1, dim_room - 1)] + [(i, dim_room - 1) for i in range(1, dim_room - 1)]

# middle block
walls += [(y, x) for y in range(4, 8) for x in range(2, 8)]
b_t_x, a_x = 2, 7
boxes = [
    (2, b_t_x),
]
targets = [(1, b_t_x)]
player = (8, a_x)
reset_opts = dict(walls=walls, boxes=boxes, targets=targets, player=player)

play_level_and_plot_directions(
    reset_opts,
    plot_box_probe=True,
    save_name="boxneartarget/no_probe.svg",
    save_video=args.save_video,
    max_steps=30,
)

# %% Box Near Target: Agent direction CI at Step 0


path1 = -1 * np.ones((10, 10))
path2 = -1 * np.ones((10, 10))

for ups in [(8, 1), (7, 1), (6, 1), (5, 1), (4, 1)]:
    path1[ups] = U

for lefts in [(8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7)]:
    path1[lefts] = L

for right in [(8, 7)]:
    path2[right] = R
for up in [(8, 8), (7, 8), (6, 8), (5, 8), (4, 8)]:
    path2[up] = U
for left in [(3, 8), (3, 7), (3, 6), (3, 5), (3, 4), (3, 3)]:
    path2[left] = L


play_level_and_plot_directions(
    reset_opts,
    plot_box_probe=True,
    fwd_hooks=path_specific_hooks(path1, agent_probe, agent_probe_info),
    hook_steps=[0],
    save_name="boxneartarget/adp_at_step0_path1.svg",
    save_video=args.save_video,
    max_steps=30,
)

play_level_and_plot_directions(
    reset_opts,
    plot_box_probe=True,
    fwd_hooks=path_specific_hooks(path2, agent_probe, agent_probe_info),
    hook_steps=[0],
    save_name="boxneartarget/adp_at_step0_path2.svg",
    save_video=args.save_video,
    max_steps=30,
)

# %% Box Far from Target
walls = [(0, i) for i in range(dim_room)] + [(dim_room - 1, i) for i in range(dim_room)]
walls += [(i, 0) for i in range(1, dim_room - 1)] + [(i, dim_room - 1) for i in range(1, dim_room - 1)]

# middle block
walls += [(y, x) for y in range(4, 7) for x in range(3, 7)]
b_t_x, a_x = 7, 8
boxes = [
    (7, b_t_x),
]
targets = [(3, 2)]
player = (8, a_x)
reset_opts = dict(walls=walls, boxes=boxes, targets=targets, player=player)

play_level_and_plot_directions(
    reset_opts,
    save_name="boxfarfromtarget/no_probe.svg",
    save_video=args.save_video,
    max_steps=30,
)


directions1 = [L] * 5 + [U] * 4
directions2 = [U] * 4 + [L] * 5
path1, _ = construct_direction_map(directions1, boxes[0])
path2, _ = construct_direction_map(directions2, boxes[0])

play_level_and_plot_directions(
    reset_opts,
    fwd_hooks=path_specific_hooks(path1, box_probe, box_probe_info),
    hook_steps=[0],
    save_name="boxfarfromtarget/bdp_at_step0_path1.svg",
    save_video=args.save_video,
    max_steps=30,
)

play_level_and_plot_directions(
    reset_opts,
    fwd_hooks=path_specific_hooks(path2, box_probe, box_probe_info),
    hook_steps=[0],
    save_name="boxfarfromtarget/bdp_at_step0_path2.svg",
    save_video=args.save_video,
    max_steps=30,
)
# %% Empty Level Box Far from Target
walls = [(0, i) for i in range(dim_room)] + [(dim_room - 1, i) for i in range(dim_room)]
walls += [(i, 0) for i in range(1, dim_room - 1)] + [(i, dim_room - 1) for i in range(1, dim_room - 1)]

b_t_x, a_x = 2, 7
boxes = [
    (7, 6),
]
targets = [(1, b_t_x)]
player = (8, a_x)
reset_opts = dict(walls=walls, boxes=boxes, targets=targets, player=player)

play_level_and_plot_directions(
    reset_opts,
    save_name="emptylevel_boxfarfromtarget/no_probe.svg",
    save_video=args.save_video,
    max_steps=30,
)

lefts, ups = 4, 6
directions = -1 * np.ones(lefts + ups)
directions[:lefts] = L
directions[lefts:] = U

for rand_idx in range(3):
    rand_directions = np.random.permutation(directions)
    # np.append(directions, U)
    my_direction_map, _ = construct_direction_map(rand_directions, boxes[0])

    play_level_and_plot_directions(
        reset_opts,
        fwd_hooks=path_specific_hooks(my_direction_map, box_probe, box_probe_info, logit=5.0),
        hook_steps=-1,
        save_name=f"emptylevel_boxfarfromtarget/bdp_at_all_steps_randpath{rand_idx}.svg",
        save_video=args.save_video,
        max_steps=30,
    )

for rand_idx in range(3):
    rand_directions = np.random.permutation(directions)
    # np.append(directions, U)
    my_direction_map, _ = construct_direction_map(rand_directions, boxes[0])

    play_level_and_plot_directions(
        reset_opts,
        fwd_hooks=path_specific_hooks(my_direction_map, box_probe, box_probe_info, logit=5.0),
        hook_steps=[0, 1, 2, 3],
        save_name=f"emptylevel_boxfarfromtarget/bdp_at_first4steps_randpath{rand_idx}.svg",
        save_video=args.save_video,
        max_steps=30,
    )


# %% Empty Level Box Near Target
walls = [(0, i) for i in range(dim_room)] + [(dim_room - 1, i) for i in range(dim_room)]
walls += [(i, 0) for i in range(1, dim_room - 1)] + [(i, dim_room - 1) for i in range(1, dim_room - 1)]

b_t_x, a_x = 2, 7
boxes = [
    (2, 2),
]
targets = [(1, 2)]
player = (8, a_x)
reset_opts = dict(walls=walls, boxes=boxes, targets=targets, player=player)

play_level_and_plot_directions(
    reset_opts,
    plot_box_probe=True,
    save_name="emptylevel_boxneartarget/no_probe.svg",
    save_video=args.save_video,
    max_steps=30,
)

lefts, ups = 5, 5
directions = -1 * np.ones(lefts + ups)
directions[:lefts] = L
directions[lefts:] = U

for rand_idx in range(3):
    rand_directions = np.random.permutation(directions)
    np.append(directions, U)
    my_direction_map, _ = construct_direction_map(rand_directions, player)

    play_level_and_plot_directions(
        reset_opts,
        plot_box_probe=True,
        fwd_hooks=path_specific_hooks(my_direction_map, agent_probe, agent_probe_info, logit=20.0),
        hook_steps=-1,
        save_name=f"emptylevel_boxneartarget/adp_at_all_steps_randpath{rand_idx}.svg",
        save_video=args.save_video,
        max_steps=30,
    )

# %%
