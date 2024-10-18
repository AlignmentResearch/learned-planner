import os
import time

import matplotlib
import matplotlib.cm as cm
import numpy as np
import plotly.express as px
from gym_sokoban.envs.sokoban_env import CHANGE_COORDINATES
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from learned_planner.interp.render_svg import fancy_obs
from learned_planner.interp.utils import get_metrics, get_player_pos


def apply_style():
    style = {
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "legend.fontsize": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (3.25, 2),
        "figure.constrained_layout.use": True,
    }
    matplotlib.rcParams.update(style)


apply_style()


def plt_obs_with_cycle_probe(
    obs,
    probe_pred_prev_timestep,
    probe_pred_curr_timestep,
    gt_curr_timestep,
    last_player_pos,
    show_dot: bool,  # this will be true on internal steps and the first external step
    ax,
    scale: int = 1,
    offset: float = 0,
):
    """Helper function to plot the level image with the cycle probe predictions."""
    if not probe_pred_curr_timestep:
        return "", last_player_pos
    title_prefix = " | In Cycle"
    player_pos = get_player_pos(obs)
    player_pos = (player_pos[1], player_pos[0])
    if gt_curr_timestep is None:
        color = "blue"
    else:
        color = "green" if probe_pred_curr_timestep == gt_curr_timestep else "red"
    if show_dot or (not probe_pred_prev_timestep) or last_player_pos == player_pos:
        ax.plot(*player_pos, color, marker="o")
    else:
        ax.plot(
            [scale * (last_player_pos[0] + offset), scale * (player_pos[0] + offset)],  # type: ignore
            [scale * (last_player_pos[1] + offset), scale * (player_pos[1] + offset)],  # type: ignore
            color=color,
            linewidth=2 * scale,
        )
    return title_prefix, player_pos


def plt_obs_with_position_probe(
    probe_preds, gt_labels, ax, marker="s", s=100, heatmap_color_range=None, scale: int = 1, offset: float = 0
):
    """Helper function to plot the level image with the position probe predictions."""
    if heatmap_color_range is not None:
        if isinstance(ax, matplotlib.axes._axes.Axes):  # type: ignore
            return ax.imshow(probe_preds, cmap="viridis", vmin=heatmap_color_range[0], vmax=heatmap_color_range[1])
        else:
            ax.set_data(probe_preds)
            return None
    positives = np.where(probe_preds == 1)
    if gt_labels is None:
        ax.scatter(scale * (positives[1] + offset), scale * (positives[0] + offset), color="blue", marker=marker, s=s)
    else:
        gt_positives = gt_labels[positives] == 1
        ax.scatter(
            scale * (positives[1][gt_positives] + offset),
            scale * (positives[0][gt_positives] + offset),
            color="green",
            marker=marker,
            s=s,
        )
        ax.scatter(
            scale * (positives[1][~gt_positives] + offset),
            scale * (positives[0][~gt_positives] + offset),
            color="red",
            marker=marker,
            s=s,
        )


def plt_obs_with_direction_probe(
    probe_preds, gt_labels, ax, color_scheme=["red", "green", "blue"], vector: bool = False, scale: int = 1, offset: float = 0
):
    """Helper function to plot the level image with the direction probe predictions."""
    if probe_preds.ndim == 2:
        directions_i, directions_j = np.where(probe_preds != -1)
        head_size = scale * 0.2
        for i, j in zip(directions_i, directions_j):
            pred_direction_idx = probe_preds[i, j]
            delta_i, delta_j = CHANGE_COORDINATES[pred_direction_idx]
            delta_i, delta_j = scale * delta_i, scale * delta_j
            color_idx = 2 if gt_labels is None else (gt_labels[i, j] == pred_direction_idx).astype(int)
            color = color_scheme[color_idx]
            if vector:
                i += 0.5 + offset
                j += 0.5 + offset
                ax.arrow(
                    scale * j, scale * (10 - i), delta_j, -delta_i, color=color, head_width=head_size, head_length=head_size
                )
            else:
                i += offset
                j += offset
                ax.arrow(scale * j, scale * i, delta_j, delta_i, color=color, head_width=head_size, head_length=head_size)
    elif probe_preds.ndim == 3:  # multioutput
        assert probe_preds.shape[2] == 4
        grid = np.arange(10, dtype=float)
        grid += offset
        if vector:
            grid += 0.5
            probe_preds = probe_preds[::-1]
            gt_labels = gt_labels[::-1]
        grid = scale * grid
        for dir_idx in range(4):
            probe_preds_dir = probe_preds[..., dir_idx]
            gt_labels_dir = gt_labels[..., dir_idx] if gt_labels is not None else None
            delta_i, delta_j = CHANGE_COORDINATES[dir_idx]
            delta_i, delta_j = scale * delta_i, scale * delta_j
            if gt_labels_dir is None:
                color_args, cmap = (), None
            else:
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
                scale=1,
                minshaft=1,
                minlength=0,
            )
    else:
        raise ValueError("probe_preds should be 2D or 3D.")


def plt_obs_with_box_labels(the_labels, ax, scale: int = 1, offset: float = 0):
    """Plot the box label as B0 to B3 at the top left of the square."""
    location_i, location_j = np.where(the_labels != -1)

    unique_locations = set(zip(location_i, location_j))
    assert len(unique_locations) == 4, f"Expected 4 unique box label locations, but found {len(unique_locations)}"

    for i, j in unique_locations:
        the_label = the_labels[i, j]
        i, j = i + offset, j + offset
        ax.text(scale * j, scale * i, f"B{the_label}", fontsize=10, color="black", ha="left", va="top")


def plt_obs_with_target_labels(the_labels, ax, scale: int = 1, offset: float = 0):
    """Plot the target label as T0 to T3 at the bottom left of the square."""
    location_i, location_j = np.where(the_labels != -1)

    unique_locations = set(zip(location_i, location_j))
    assert len(unique_locations) == 4, f"Expected 4 unique target label locations, but found {len(unique_locations)}"

    for i, j in unique_locations:
        the_label = the_labels[i, j]
        i, j = i + offset, j + offset
        ax.text(scale * j, scale * i, f"T{the_label}", fontsize=10, color="black", ha="left", va="bottom")


last_player_pos = None


def save_video(
    filename,
    all_obs,
    all_probes_preds=[],
    all_gt_labels=[],
    all_probe_infos=[],
    overlapped=False,
    show_internal_steps_until=0,
    sae_feature_offset=0,
    base_dir="videos",
    box_labels=None,
    target_labels=None,
    remove_ticks=True,
    truncate_title=-1,
    fancy_sprite=False,
):
    """Save the video of the level given by all_obs. Video will be saved in the folder videos_{probe_type}.

    Args:
        filename (str): Name of the video file (with extension).
        all_obs (np.ndarray): observations of the level of shape (num_steps, 3, 10, 10).
        all_probes_preds (Optional[list[np.ndarray]]): list of predictions from multiple probes.
            The np arrays can be of the shape (num_steps,) or (num_steps, 10, 10) depending on the `probe_type`.
            Default is None.
        all_gt_labels (list[np.ndarray]): list of ground truth labels for the probes.
        all_probe_infos (list[ProbeTrainOn]): list of ProbeTrainOn.
        overlapped (bool): Whether to plot the probes on the same image or side-by-side.
        show_internal_steps_until (int): Number of internal steps to show. Default is 0.
        box_labels (np.ndarray): labels of the boxes in the level of shape (num_steps, 10, 10).
        target_labels (np.ndarray): labels of the targets of shape (10, 10).
    """
    if all_probe_infos:
        assert len(all_probes_preds) == len(all_probe_infos)
    max_len = len(all_obs)
    if all_gt_labels:
        assert len(all_gt_labels) == len(all_probes_preds)
    repeats_per_step = all_probes_preds[0].shape[1] if show_internal_steps_until else 1
    global last_player_pos
    last_player_pos = None
    os.makedirs(base_dir, exist_ok=True)
    title_prefix = ""
    scale = 96 if fancy_sprite else 1
    offset = 0.5 if fancy_sprite else 0

    if all_probes_preds is not None:
        try:
            cycle_probe_idx = [info.probe_type for info in all_probe_infos].index("cycle")
            no_cycle = not np.any(all_probes_preds[cycle_probe_idx])
            if no_cycle:
                title_prefix = " | No Cycle"
                filename = filename.replace(".mp4", "_no_cycle.mp4")
        except ValueError:
            pass
    total_subplots = len(all_probes_preds)
    if overlapped or len(all_probes_preds) <= 1:
        fig, axs = plt.subplots(1, 1, figsize=(4, 4))
        axs = [axs]
    else:
        total_subplots += 0 if all_probe_infos else 1
        rows, cols = np.ceil(total_subplots / 4).astype(int), min(4, total_subplots)
        fig, axs = plt.subplots(rows, cols, figsize=(2 * cols + 1, 2 * rows + 1))
        plt.subplots_adjust(left=0.05, top=0.9, right=0.95, bottom=0.05, hspace=0.2, wspace=0.2)  # manually fine-tuned
        axs = axs.flatten()

    max_fig_dim = max(fig.get_figwidth(), fig.get_figheight())
    heatmap_color_range = None
    if not all_probe_infos and len(all_probes_preds) != 0:  # sae
        heatmap_color_range = (all_probes_preds.min(), all_probes_preds.max())
        norm = plt.Normalize(vmin=heatmap_color_range[0], vmax=heatmap_color_range[1])
        fig.colorbar(cm.ScalarMappable(cmap="viridis", norm=norm), ax=axs)

    all_obs = np.transpose(all_obs, (0, 2, 3, 1))
    total_internal_steps = repeats_per_step * show_internal_steps_until
    total_frames = total_internal_steps + max_len - show_internal_steps_until + 1

    def reset_frame(ax):
        ax.clear()
        if remove_ticks:
            ax.axis("off")

    dataset_name_map = {
        "agent_in_a_cycle": "Cycle",
        "boxes_future_direction_map": "Box Directions",
        "agents_future_direction_map": "Agent Directions",
        "next_box": "Next Box",
        "next_target": "Next Target",
        "agents_future_position_map": "Agent Positions",
        "boxes_future_position_map": "Box Positions",
    }

    def update_frame(i, title_prefix=title_prefix):
        global last_player_pos
        if i == total_frames - 1:
            if all_gt_labels:
                all_metrics = {}
                for pidx, probe_preds in enumerate(all_probes_preds):
                    probe_preds_external = probe_preds[:, repeats_per_step - 1] if show_internal_steps_until else probe_preds
                    probe_metrics = get_metrics(probe_preds_external, all_gt_labels[pidx], classification=True)  # type: ignore
                    prefix = dataset_name_map[all_probe_infos[pidx].dataset_name]
                    probe_metrics = {f"{k}": v for k, v in probe_metrics.items()}
                    all_metrics.update(probe_metrics)
                    ax = axs[pidx]
                    reset_frame(ax)
                    ax.text(0.1, 0.1, prefix + "\n\n" + "\n".join([f"{k}: {100*v:.1f}" for k, v in probe_metrics.items()]))

                # plt.text(0.1, 0.1, "\n".join([f"{k}: {v:.2f}" for k, v in all_metrics.items()]))
            else:
                print("No GT labels provided.")
                plt.text(0.1, 0.1, "No GT labels provided.")
            return
        if i < total_internal_steps:
            obs_idx = i // repeats_per_step
            probe_idx = (obs_idx, i % repeats_per_step)
        else:
            obs_idx = show_internal_steps_until + i - total_internal_steps
            # probe_idx = repeats_per_step * (obs_idx + 1) - 1 if show_internal_steps_until else obs_idx
            probe_idx = (obs_idx, repeats_per_step - 1) if show_internal_steps_until else obs_idx
        obs = all_obs[obs_idx]
        img = obs
        if fancy_sprite:
            img = fancy_obs(obs)
        if len(all_probes_preds) == 0:
            reset_frame(axs[0])
            axs[0].imshow(img)
        for pidx, probe_preds in enumerate(all_probes_preds):
            ax = axs[pidx]
            if not all_probe_infos and len(all_probes_preds) != 0:  # sae
                if pidx == 0:
                    reset_frame(ax)
                    ax.imshow(img)
                    ax.set_title("Observation")
                ax = axs[pidx + 1]
                ax.clear()
            elif (not overlapped) or (pidx == 0):
                reset_frame(ax)
                ax.imshow(img)

            if (not overlapped) and len(all_probes_preds) > 1:
                if all_probe_infos:
                    title = dataset_name_map[all_probe_infos[pidx].dataset_name]
                    if truncate_title > 0:
                        title = title[:truncate_title] + ("..." if len(title) > truncate_title else "")
                    ax.set_title(title)
                elif len(all_probes_preds) != 0:  # sae
                    ax.set_title(f"Feature {sae_feature_offset + pidx}")
            probe_out = probe_preds[probe_idx]
            try:
                gt_label = all_gt_labels[pidx][obs_idx]
            except IndexError:
                gt_label = None

            if not all_probe_infos:
                plt_obs_with_position_probe(
                    probe_out,
                    gt_label,
                    ax,
                    heatmap_color_range=heatmap_color_range,
                    scale=scale,
                    offset=offset,
                )  # sae
            elif "cycle" == all_probe_infos[pidx].probe_type:
                title_prefix, last_player_pos = plt_obs_with_cycle_probe(
                    obs,
                    probe_preds[(probe_idx[0] - 1, probe_idx[1]) if show_internal_steps_until else probe_idx - 1],  # type: ignore
                    probe_preds[probe_idx],
                    gt_label,
                    last_player_pos,
                    show_dot=(obs_idx == 0) or (show_internal_steps_until > 0 and probe_idx[1] < repeats_per_step - 1),
                    ax=ax,
                    scale=scale,
                    offset=offset,
                )
            elif "position" == all_probe_infos[pidx].probe_type:
                plt_obs_with_position_probe(probe_out, gt_label, ax, scale=scale, offset=offset)
            elif "direction" in all_probe_infos[pidx].probe_type:
                plt_obs_with_direction_probe(
                    probe_out, gt_label, ax, all_probe_infos[pidx].color_scheme, scale=scale, offset=offset
                )
            else:
                raise ValueError(f"Unknown probe type: {all_probe_infos[pidx].probe_type}")

            # Draw box and target labels. Aids colorblind people
            if box_labels is not None:
                plt_obs_with_box_labels(box_labels[i], ax)
            if target_labels is not None:
                plt_obs_with_target_labels(target_labels, ax)

        internal_step_suffix = ": Internal Step " + str(i % repeats_per_step) if i < total_internal_steps else ""
        plt.suptitle(f"Step {obs_idx}{internal_step_suffix}" + title_prefix, y=0.99)
        return (fig,)

    anim = animation.FuncAnimation(
        fig,
        update_frame,  # type: ignore
        save_count=total_frames,
        repeat=False,
    )
    dpi = np.ceil(720 / max_fig_dim).astype(int)
    dpi = dpi if dpi % 2 == 0 else dpi + 1
    assert anim is not None
    full_path = os.path.join(base_dir, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    t0 = time.time()
    anim.save(full_path, fps=2, writer="ffmpeg")
    print(f"Saved video to {full_path} in {time.time() - t0:.2f} seconds.")
    return full_path


def save_video_sae(
    filename,
    all_obs,
    all_probes_preds=[],
    show_internal_steps_until=0,
    sae_feature_indices: int | list[int] = 0,
    base_dir="videos",
    box_labels=None,
    target_labels=None,
):
    """Save the video of the level given by all_obs. Video will be saved in the folder videos_{probe_type}.

    Args:
        filename (str): Name of the video file (with extension).
        all_obs (np.ndarray): observations of the level of shape (num_steps, 3, 10, 10).
        all_probes_preds (Optional[list[np.ndarray]]): list of predictions from multiple probes.
            The np arrays can be of the shape (num_steps,) or (num_steps, 10, 10) depending on the `probe_type`.
            Default is None.
        show_internal_steps_until (int): Number of internal steps to show. Default is 0.
        box_labels (np.ndarray): labels of the boxes in the level of shape (num_steps, 10, 10).
        target_labels (np.ndarray): labels of the targets of shape (10, 10).
    """
    plt.rcParams.update({"font.size": 18})
    max_len = len(all_obs)
    repeats_per_step = all_probes_preds[0].shape[1] if show_internal_steps_until else 1
    global last_player_pos
    last_player_pos = None
    os.makedirs(base_dir, exist_ok=True)
    title_prefix = ""

    total_subplots = len(all_probes_preds) + 1
    rows, cols = np.ceil(total_subplots / 4).astype(int), min(4, total_subplots)
    figsize = (2 * cols + 1, 2 * rows + 1)
    max_fig_dim = max(figsize)
    dpi = np.ceil(720 / max_fig_dim).astype(int)
    dpi = dpi if dpi % 2 == 0 else dpi + 1
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    plt.subplots_adjust(left=0.05, top=0.9, right=1.05, bottom=0.05, hspace=0.5, wspace=0.5)  # manually fine-tuned
    try:
        axs = axs.flatten()
    except AttributeError:
        axs = [axs]

    heatmap_color_range = (all_probes_preds.min(), all_probes_preds.max())
    norm = plt.Normalize(vmin=heatmap_color_range[0], vmax=heatmap_color_range[1])
    fig.colorbar(cm.ScalarMappable(cmap="viridis", norm=norm), ax=axs)
    all_obs = np.transpose(all_obs, (0, 2, 3, 1))
    imshow_outs = [axs[0].imshow(all_obs[0])]
    imshow_outs += [
        plt_obs_with_position_probe(all_probes_preds[i, 0, 0], None, ax, heatmap_color_range=heatmap_color_range)
        for i, ax in enumerate(axs[1:])
    ]

    total_internal_steps = repeats_per_step * show_internal_steps_until
    total_frames = total_internal_steps + max_len - show_internal_steps_until

    def ft_idx(idx):
        return sae_feature_indices + idx - 1 if isinstance(sae_feature_indices, int) else sae_feature_indices[idx - 1]

    [ax.set_title("Observation" if i == 0 else f"F{ft_idx(i)}") for i, ax in enumerate(axs)]
    imshow_outs.append(plt.suptitle("", fontsize=18, y=0.99))

    def update_frame(i, title_prefix=title_prefix):
        global last_player_pos
        if i < total_internal_steps:
            obs_idx = i // repeats_per_step
            probe_idx = (obs_idx, i % repeats_per_step)
        else:
            obs_idx = show_internal_steps_until + i - total_internal_steps
            # probe_idx = repeats_per_step * (obs_idx + 1) - 1 if show_internal_steps_until else obs_idx
            probe_idx = (obs_idx, repeats_per_step - 1) if show_internal_steps_until else obs_idx
        obs = all_obs[obs_idx]
        imshow_outs[0].set_data(obs)
        for pidx, probe_preds in enumerate(all_probes_preds):
            probe_out = probe_preds[probe_idx]
            plt_obs_with_position_probe(probe_out, None, imshow_outs[pidx + 1], heatmap_color_range=heatmap_color_range)  # sae
        internal_step_suffix = ": Internal Step " + str(i % repeats_per_step) if i < total_internal_steps else ""
        imshow_outs[-1].set_text(f"Step {obs_idx}{internal_step_suffix}" + title_prefix)

        return imshow_outs

    anim = animation.FuncAnimation(
        fig,
        update_frame,  # type: ignore
        save_count=total_frames,
        repeat=False,
    )

    assert anim is not None
    full_path = os.path.join(base_dir, filename)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    t0 = time.time()
    anim.save(full_path, fps=2, writer="ffmpeg")
    print(f"Saved video to {full_path} in {time.time() - t0:.2f} seconds.")
    return full_path


def plotly_feature_vis(feature_acts, obs, feature_labels, show=False):
    """Feature activations visualized with observations along with time slider.

    Args:
        feature_acts (np.ndarray): Activations of top features. Shape: (time, num_features, height, width).
        obs (np.ndarray): Observations. Shape: (time, channels, height, width).
        feature_labels (list[str]): Labels for the features. Shape: (num_features,).
    """
    cmap = plt.get_cmap("viridis")
    normed = (feature_acts - feature_acts.min()) / (feature_acts.max() - feature_acts.min())

    repeated_obs = np.transpose(obs, (0, 2, 3, 1))[:, None, :, :, :]
    to_plot = np.concatenate([repeated_obs[: len(normed)], cmap(normed)[..., :3] * 255], axis=1)
    labels = ["Observation"] + feature_labels
    fig = px.imshow(
        to_plot[:, :],
        facet_col=1,
        animation_frame=0,
        facet_col_wrap=8,
        binary_string=True,
    )
    fig.for_each_annotation(lambda a: a.update(text=labels[int(a.text.split("=")[-1])]))
    if show:
        fig.show()
    return fig
