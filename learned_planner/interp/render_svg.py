from typing import Sequence, Tuple

import cairosvg
import imageio
import numpy as np
import torch as th
from jinja2 import Environment, FileSystemLoader

from learned_planner import LP_DIR

file_loader = FileSystemLoader(LP_DIR / "plot")
env = Environment(loader=file_loader)
level_template = env.get_template("level.svg")

WALL = np.array([0, 0, 0])
TARGET = np.array([254, 126, 125])
BOX_ON_TARGET = np.array([254, 95, 56])
BOX = np.array([142, 121, 56])
PLAYER = np.array([160, 212, 56])
PLAYER_ON_TARGET = np.array([219, 212, 56])
FLOOR = np.array([243, 248, 238])


def render_level_svg(
    wall_positions: Sequence[Tuple[int, int]] | np.ndarray,
    box_positions: Sequence[Tuple[int, int]] | np.ndarray,
    target_positions: Sequence[Tuple[int, int]] | np.ndarray,
    player_position: Tuple[int, int] | np.ndarray,
) -> str:
    """Renders the level template with sprites and outputs the resulting SVG markup."""
    all_positions = [*wall_positions, *box_positions, *target_positions]
    if isinstance(wall_positions, np.ndarray):
        wall_positions = wall_positions.tolist()
    output = level_template.render(
        wall_positions=wall_positions,
        box_positions=box_positions,
        target_positions=target_positions,
        player_position=player_position,
        total_height=max(y for (_, y) in all_positions) + 1,
        total_width=max(x for (x, _) in all_positions) + 1,
    )
    return output


def level_to_svg(level: str, *, level_sz: int = 10) -> str:
    assert len(level) == level_sz**2
    wall_pos = []
    box_pos = []
    target_pos = []
    player_pos = (-1, -1)
    for y in range(level_sz):
        for x in range(level_sz):
            thing_at_pos = level[y * level_sz + x]
            if thing_at_pos == "#":
                wall_pos.append((x, y))
            elif thing_at_pos == ".":
                target_pos.append((x, y))
            elif thing_at_pos == "$":
                box_pos.append((x, y))
            elif thing_at_pos == "@":
                player_pos = (x, y)

    return render_level_svg(wall_pos, box_pos, target_pos, player_pos)


def tiny_world_rgb_to_svg(rgb, return_info=False):
    assert len(rgb.shape) == 3 and rgb.shape[2] == 3  # and rgb.shape[0] == rgb.shape[1]
    if isinstance(rgb, th.Tensor):
        rgb = rgb.cpu().numpy()
    # H,W transpose needed for argwhere
    rgb = np.transpose(rgb, (1, 0, 2))
    box_on_target_cond = np.all(rgb == BOX_ON_TARGET, axis=-1)
    player_on_target_cond = np.all(rgb == PLAYER_ON_TARGET, axis=-1)
    wall_pos = np.argwhere(np.all(rgb == WALL, axis=-1))
    box_pos = np.argwhere(np.any([np.all(rgb == BOX, axis=-1), box_on_target_cond], axis=0))
    target_pos = np.argwhere(np.any([np.all(rgb == TARGET, axis=-1), box_on_target_cond, player_on_target_cond], axis=0))
    player_pos = np.argwhere(np.any([np.all(rgb == PLAYER, axis=-1), player_on_target_cond], axis=0))
    player_pos = player_pos[0]
    svg = render_level_svg(wall_pos, box_pos, target_pos, player_pos)
    if return_info:
        return svg, (wall_pos, box_pos, target_pos, player_pos)
    return svg


def tiny_world_rgb_to_txt(rgb):
    txt = ""
    for y in range(rgb.shape[0]):
        for x in range(rgb.shape[1]):
            if np.all(rgb[y, x] == WALL):
                txt += "#"
            elif np.all(rgb[y, x] == TARGET):
                txt += "."
            elif np.all(rgb[y, x] == BOX):
                txt += "$"
            elif np.all(rgb[y, x] == PLAYER):
                txt += "@"
            elif np.all(rgb[y, x] == BOX_ON_TARGET):
                txt += "*"
            elif np.all(rgb[y, x] == PLAYER_ON_TARGET):
                txt += "+"
            else:
                txt += " "
        txt += "\n"
    return txt


def episode_obs_to_svgs(episode_obs, max_len):
    all_svg_and_infos = [tiny_world_rgb_to_svg(obs, return_info=True) for obs in episode_obs]
    all_svgs = [svg for svg, _ in all_svg_and_infos]
    last_info = all_svg_and_infos[-1][1]
    assert isinstance(last_info, tuple) and len(last_info) == 4
    if len(episode_obs) < max_len:
        w, b, t, old_p = last_info
        box_not_on_target: set[tuple[int, int]] = set(b) - set(t)
        assert len(box_not_on_target) == 1
        p = box_not_on_target.pop()
        assert abs(p[0] - old_p[0]) + abs(p[1] - old_p[1]) == 1
        b = t  # all boxes are on targets
        last_svg = render_level_svg(w, b, t, p)
        all_svgs.append(last_svg)
    return all_svgs


def svg_to_plt(svg_content, scale=1):
    bytes = cairosvg.svg2png(bytestring=svg_content.encode(), scale=scale)
    img = imageio.v3.imread(bytes)  # type: ignore
    return img


def fancy_obs(rgb_obs):
    return svg_to_plt(tiny_world_rgb_to_svg(rgb_obs))


if __name__ == "__main__":
    output = render_level_svg(
        wall_positions=[(0, 0), (0, 1), (0, 2), (0, 3), (1, 2)],
        box_positions=[(5, 5), (5, 6), (4, 1)],
        target_positions=[(4, 2), (5, 5), (4, 3)],
        player_position=(4, 2),
    )
    print(output)
