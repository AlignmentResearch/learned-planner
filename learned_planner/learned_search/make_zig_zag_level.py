# %%
import shutil

import numpy as np
from cleanba.environments import SokobanConfig
from matplotlib import pyplot as plt

from learned_planner import LP_DIR
from learned_planner.interp.render_svg import tiny_world_rgb_to_txt

single_box = True
agent_close = True
file_path = LP_DIR / f"alternative-levels/levels/boxoban-levels-master/unfiltered/train/0{24+single_box}.txt"

with open(file_path, "w") as file:
    file.write("---\n")
    sizes = [10, 15, 16, 20, 25, 30, 35, 40]
    for size_idx in range(len(sizes)):
        size = sizes[size_idx]
        walls = [(i, 0) for i in range(size)]
        walls += [(i, size - 1) for i in range(size)]
        walls += [(0, i) for i in range(1, size - 1)]
        walls += [(size - 1, i) for i in range(1, size - 1)]

        if size <= 10:
            gap = 3
            vert_gap_ratio = 0.3
        else:
            gap = int(0.2 * size)
            vert_gap_ratio = 0.2
        for i, vert_wall in enumerate(range(gap, size - 1, gap)):
            if i % 2 == 0:
                walls += [(y, vert_wall) for y in range(1, int((1 - vert_gap_ratio) * size))]
                if size > 20:
                    walls += [(y, vert_wall + 1) for y in range(1, int((1 - vert_gap_ratio) * size))]
            else:
                walls += [(y, vert_wall) for y in range(int(vert_gap_ratio * size), size - 1)]
                if size > 20:
                    walls += [(y, vert_wall + 1) for y in range(int(vert_gap_ratio * size), size - 1)]

        if single_box:
            boxes = [(gap, gap - 1)]
            targets = [(gap, size - 2)]
        else:
            boxes = [(gap, x) for x in range(2, gap)]
            targets = [(gap, x) for x in range(size - gap + 1, size - 1)]
        assert len(boxes) == len(targets) and len(boxes) > 0, f"{len(boxes)=}, {len(targets)=}"
        cfg = SokobanConfig(
            dim_room=(size, size),
            tinyworld_obs=True,
            asynchronous=False,
            max_episode_steps=1000,
            min_episode_steps=1000,
            num_envs=1,
        )
        env = cfg.make()
        player = (gap - 1, gap - 2) if agent_close else (1, 1)
        obs, _ = env.reset(options=dict(walls=walls, boxes=boxes, player=player, targets=targets))
        obs = np.transpose(obs.squeeze(), (1, 2, 0))
        plt.imshow(obs)

        txt = tiny_world_rgb_to_txt(obs)
        file.write(f"\n; {size_idx}\n\n")
        file.write(txt)
print("Saved to", file_path)
shutil.copy(file_path, LP_DIR / f"alternative-levels/levels/zig_zag_levels{'_single_box' if single_box else ''}.txt")


# %%
