import glob
from pathlib import Path

import matplotlib.animation as animation
import pandas as pd
from gym_sokoban.envs import boxoban_env
from matplotlib import pyplot as plt

action_map = [0, 3, 1, 2]


def map_action(action_sequence: str):
    try:
        return [action_map[int(c)] for c in action_sequence.strip()]
    except ValueError:
        return None


file_idx, level_idx = 127, 0
levels_per_file = 1000


def main(file_idx, level_idx, show_plot=False):
    cache_path = Path(__file__).parent.parent / "training/.sokoban_cache/"
    path = cache_path / "boxoban-levels-master/unfiltered/train/"
    e = boxoban_env.BoxobanEnv(cache_path=cache_path, reset=False)
    e.reset(options={"level_file_idx": file_idx, "level_idx": level_idx})

    astar_logs_path = path / "logs"
    astar_logs = sorted(glob.glob(str(astar_logs_path / "log_*.csv")))
    log_file = astar_logs[file_idx]
    df = pd.read_csv(log_file, index_col=0)
    df = df.rename(columns=lambda x: x.strip())
    df["Actions"] = df["Actions"].apply(map_action)
    level = df.iloc[level_idx]

    if show_plot:
        fig = plt.figure()  # Create a figure for the animation

        def update_frame(i):
            if i > len(level["Actions"]):
                return
            if i != 0:
                act = level["Actions"][i - 1]
                e.step(act)
            img = e.render()
            plt.imshow(img)
            plt.title(f"Step {i}")  # Add index to title

        anim = animation.FuncAnimation(
            fig,
            update_frame,  # type: ignore
            frames=len(level["Actions"]) + 1,
            interval=500,
            repeat=False,
        )
        assert anim is not None
        plt.show()
    else:
        for i, act in enumerate(level["Actions"]):
            e.step(act)
    return e._check_if_all_boxes_on_target()


if __name__ == "__main__":
    print(main(file_idx, level_idx, show_plot=True))
