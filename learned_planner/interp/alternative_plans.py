import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn.functional as F
from gym_sokoban.envs.sokoban_env import CHANGE_COORDINATES

# %%


@th.compile
def obs_is_box(obs: th.Tensor) -> th.Tensor:
    return (
        ((obs[..., 0, :, :] == 254) | (obs[..., 0, :, :] == 142))
        & ((obs[..., 1, :, :] == 95) | (obs[..., 1, :, :] == 121))
        & (obs[..., 2, :, :] == 56)
    )


def compute_directions(one_ds: "DatasetStore", boxes_are_walls: bool = False) -> th.Tensor:
    """
    Compute directions tensor based on the steps needed to reach the target positions and walls.

    Args:
        one_ds (DatasetStore): Input dataset store object.

    Returns:
        directions (th.Tensor): Tensor containing the directions to reach the target positions.
    """
    th._dynamo.mark_dynamic(one_ds.obs, 0)
    # Compute wall positions
    wall_mask = (one_ds.obs == 0).all(dim=1)
    if boxes_are_walls:
        wall_mask |= obs_is_box(one_ds.obs)
    wall_mask = wall_mask.rename("N", "y", "x")
    th._dynamo.mark_dynamic(wall_mask, 0)

    target_pos = one_ds.get_target_positions()
    th._dynamo.mark_dynamic(target_pos, 0)

    # Compute number of steps needed to reach any position from the boxes' positions.
    steps_to_reach = init_steps_to_reach(one_ds)
    th._dynamo.mark_dynamic(steps_to_reach, 0)
    steps_to_reach = propagate_steps_from_boxes(steps_to_reach, wall_mask)
    th._dynamo.mark_dynamic(steps_to_reach, 0)

    # Compute the directions needed to reach a target, based on the target positions, steps to reach them and walls.
    directions = compute_paths_to_target(steps_to_reach, target_pos, wall_mask)
    return directions


def init_steps_to_reach(one_ds: "DatasetStore") -> th.Tensor:
    SENTINEL_MAX = 999  # Number that is clearly larger than all distances we deal with
    steps_to_reach = th.zeros((one_ds.obs.shape[0], 4, 10, 10), dtype=th.long) + SENTINEL_MAX
    box_positions = one_ds.get_box_positions()
    box_positions_arange = th.arange(len(box_positions))

    for i in range(4):
        steps_to_reach[box_positions_arange, i, box_positions[:, i, 0], box_positions[:, i, 1]] = 0
    return steps_to_reach


@th.compile
def propagate_steps_from_boxes(steps_to_reach: th.Tensor, wall_mask: th.Tensor) -> th.Tensor:
    """
    Compute the number of steps needed to reach any position, starting from the positions of boxes.

    Args:
        one_ds (DatasetStore): Input dataset store object.

    Returns:
        Tuple[th.Tensor, th.Tensor]: A tuple containing the wall mask and steps to reach tensor.
    """
    SENTINEL_MAX = 999  # Number that is clearly larger than all distances we deal with

    wall_mask = wall_mask.rename(None)

    bc_wall_mask = th.broadcast_to(wall_mask[:, None, :, :], steps_to_reach.shape)
    padded_wall_mask = F.pad(wall_mask[:, None, :, :], (1, 1, 1, 1), value=1).bool()

    for _ in range(100):
        next_steps = steps_to_reach + 1
        from_cardinal_directions = th.stack(
            [
                steps_to_reach[:, :, 1:-1, 1:-1],
                next_steps[:, :, 1:-1, :-2] + SENTINEL_MAX * padded_wall_mask[..., 2:-2, :-4],
                next_steps[:, :, 1:-1, 2:] + SENTINEL_MAX * padded_wall_mask[..., 2:-2, 4:],
                next_steps[:, :, :-2, 1:-1] + SENTINEL_MAX * padded_wall_mask[..., :-4, 2:-2],
                next_steps[:, :, 2:, 1:-1] + SENTINEL_MAX * padded_wall_mask[..., 4:, 2:-2],
            ],
            dim=0,
        )

        steps_to_reach[:, :, 1:-1, 1:-1] = th.min(from_cardinal_directions, dim=0).values
        steps_to_reach[bc_wall_mask] = SENTINEL_MAX

    return steps_to_reach.rename("N", "box", "y", "x")


@th.compile
def compute_paths_to_target(steps_to_reach: th.Tensor, target_pos: th.Tensor, wall_mask: th.Tensor) -> th.Tensor:
    """
    Compute the paths that lead to the target based on steps and wall mask.

    Args:
        steps_to_reach (th.Tensor): Tensor representing steps to reach any position.
        target_pos (th.Tensor): Tensor of target positions.
        wall_mask (th.Tensor): Tensor representing the wall mask.

    Returns:
        th.Tensor: Tensor with directions to reach the target positions.
    """
    wall_mask_bc_neg = ~wall_mask
    assert wall_mask_bc_neg.names == ("N", "y", "x")

    directions = th.zeros((4, wall_mask.size(0), 4, 4, 10, 10), dtype=th.bool)
    directions[0, :, :, th.arange(len(target_pos)), target_pos[:, 0], target_pos[:, 1]] = True
    directions = directions.rename("direction", "N", "box", "target", "y", "x")

    steps_to_reach_target = steps_to_reach.rename(None)[:, :, target_pos[:, 0], target_pos[:, 1]]
    max_steps_to_reach_target = int(steps_to_reach_target.max().item())

    """Compute the paths that lead to the target."""
    old_direction_names = directions.names

    for steps in range(max_steps_to_reach_target, -1, -1):
        visited = bitwise_and(
            directions.rename(None).any(dim=0).rename(*directions.names[1:]),
            (steps_to_reach == steps).align_as(directions[0]),
        )

        possibly_coming_from = steps_to_reach == steps - 1
        for n_direction, slices, slices_opposite in [
            (0, np.s_[..., :-2, 1:-1], np.s_[..., 2:, 1:-1]),  # UP
            (1, np.s_[..., 2:, 1:-1], np.s_[..., :-2, 1:-1]),  # DOWN
            (2, np.s_[..., 1:-1, :-2], np.s_[..., 1:-1, 2:]),  # LEFT
            (3, np.s_[..., 1:-1, 2:], np.s_[..., 1:-1, :-2]),  # RIGHT
        ]:
            visited_from = visited[slices]
            comes_from_this_direction = bitwise_and(
                bitwise_and(
                    visited_from,
                    possibly_coming_from[..., 1:-1, 1:-1].align_as(visited_from),
                ),
                wall_mask_bc_neg[slices_opposite].align_as(visited_from),
            )

            mask = comes_from_this_direction.align_as(directions[0])
            directions = directions.rename(None)
            directions[n_direction, ..., 1:-1, 1:-1][mask.rename(None)] = True
            directions = directions.rename(*old_direction_names)

    directions = directions.rename(None)
    directions[0, :, :, th.arange(len(target_pos)), target_pos[:, 0], target_pos[:, 1]] = False
    return directions


def bitwise_and(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    """
    Performs a bitwise AND operation, preserving tensor names.

    Args:
        a (th.Tensor): First tensor.
        b (th.Tensor): Second tensor.

    Returns:
        th.Tensor: The result of the bitwise AND operation.
    """
    assert a.names == b.names, (a.names, b.names)
    return (a.rename(None) & b.rename(None)).rename(*a.names)


def visualize_paths(one_ds: "DatasetStore", directions: th.Tensor) -> plt.Figure:
    """
    Visualize the computed paths using Matplotlib.

    Args:
        one_ds (DatasetStore): Input dataset store object.
        directions (th.Tensor): Tensor containing the direction paths.

    Returns:
        plt.Figure: A Matplotlib figure showing the paths.
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for axes_i in range(axes.shape[0]):
        for axes_j in range(axes.shape[1]):
            ax = axes[axes_i, axes_j]
            direction_mask = directions[:, 0, axes_i, axes_j, :, :]
            plot_directions(ax, one_ds.obs[0], direction_mask)
    return fig


def plot_directions(ax: plt.Axes, obs: th.Tensor, direction_mask: th.Tensor) -> None:
    """
    Helper function to plot directions on the grid.

    Args:
        ax (plt.Axes): Matplotlib axis to plot on.
        obs (th.Tensor): Observation tensor from the dataset.
        direction_mask (th.Tensor): Tensor representing the direction mask.
    """
    assert obs.shape == (3, 10, 10)
    assert direction_mask.shape == (4, 10, 10)

    ax.imshow(obs.permute((1, 2, 0)))
    directions_i, directions_j = np.where(np.any(direction_mask.numpy(), axis=0))
    for i, j in zip(directions_i, directions_j):
        for pred_direction_idx in range(4):
            if direction_mask[pred_direction_idx, i, j]:
                delta_i, delta_j = CHANGE_COORDINATES[pred_direction_idx]
                color = ["red", "green", "blue", "yellow"][pred_direction_idx]
                ax.arrow(j, i, delta_j, delta_i, color=color, head_width=0.2, head_length=0.2, alpha=0.5)


# %%

if __name__ == "__main__":
    from learned_planner.interp.collect_dataset import DatasetStore

    level_files_path = Path("/training/activations_dataset/hard/0_think_step/")
    level_files = [level_files_path / p for p in os.listdir(level_files_path) if p.endswith(".pkl")]
    np.random.shuffle(level_files)  # type: ignore

    one_ds = DatasetStore.load(str(level_files[0]))
    directions = compute_directions(one_ds)
    visualize_paths(one_ds, directions)
