from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from jinja2 import Environment, FileSystemLoader

file_loader = FileSystemLoader(Path(__file__).parent)
env = Environment(loader=file_loader)
level_template = env.get_template("level.svg")


def render_level_svg(
    wall_positions: Sequence[Tuple[int, int]],
    box_positions: Sequence[Tuple[int, int]],
    target_positions: Sequence[Tuple[int, int]],
    player_position: Tuple[int, int],
) -> str:
    """Renders the level template with sprites and outputs the resulting SVG markup."""
    output = level_template.render(
        wall_positions=wall_positions,
        box_positions=box_positions,
        target_positions=target_positions,
        player_position=player_position,
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
    assert len(rgb.shape) == 3 and rgb.shape[2] == 3 and rgb.shape[0] == rgb.shape[1]
    rgb = np.array(rgb)

    wall = [0, 0, 0]
    target = [254, 126, 125]
    box_on_target = [254, 95, 56]
    box = [142, 121, 56]
    player = [160, 212, 56]
    player_on_target = [219, 212, 56]

    wall_pos = []
    box_pos = []
    target_pos = []
    player_pos = (-1, -1)

    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            pixel = rgb[i, j]
            if np.all(pixel == wall):
                wall_pos.append((j, i))
            elif np.all(pixel == target):
                target_pos.append((j, i))
            elif np.all(pixel == box_on_target):
                box_pos.append((j, i))
                target_pos.append((j, i))
            elif np.all(pixel == box):
                box_pos.append((j, i))
            elif np.all(pixel == player):
                player_pos = (j, i)
            elif np.all(pixel == player_on_target):
                player_pos = (j, i)
                target_pos.append((j, i))

    svg = render_level_svg(wall_pos, box_pos, target_pos, player_pos)
    if return_info:
        return svg, (wall_pos, box_pos, target_pos, player_pos)
    return svg


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


if __name__ == "__main__":
    output = render_level_svg(
        wall_positions=[(0, 0), (0, 1), (0, 2), (0, 3), (1, 2)],
        box_positions=[(5, 5), (5, 6), (4, 1)],
        target_positions=[(4, 2), (5, 5), (4, 3)],
        player_position=(4, 2),
    )
    print(output)
