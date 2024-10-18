# %%
import collections
import imp
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from learned_planner.interp import render_svg
from learned_planner.interp.render_svg import fancy_obs
from learned_planner.interp.utils import get_solved_obs

imp.reload(render_svg)

# %%
trace_dir = Path("/training/bigger_levels")

level_names = [
    s.removesuffix(".txt")
    for s in """
david-holland-1.txt
david-holland-2.txt
deluxe.txt
dimitri-and-yorick.txt
howards-1st-set.txt
howards-2nd-set.txt
howards-3rd-set.txt
howards-4th-set.txt
intro.txt
mas-sasquatch.txt
microban.txt
microcosmos.txt
nabokosmos.txt
sasquatch-iii.txt
sasquatch-iv.txt
sasquatch.txt
simple-sokoban.txt
sokoban-jr-1.txt
sokoban-jr-2.txt
sokoban.txt
sokogen-990602.txt
still-more-levels.txt
xsokoban.txt
yoshio-automatic.txt
""".split()
]

df = []
for fname in os.listdir(trace_dir):
    _steps_, thinking_steps, _file_, file_idx = fname.split(".")[0].split("_")
    metrics = pd.read_pickle(trace_dir / fname)

    df.append(
        {
            "think": int(thinking_steps),
            "file_idx": int(file_idx),
            "levels": level_names[int(file_idx)],
            **{k.lstrip("0123456789_"): v for k, v in metrics.items()},
        }
    )
df = pd.DataFrame(df)

print(df)

# %%

plt.figure(figsize=(10, 6))
for file_idx, group in sorted(df.sort_values("think").groupby("file_idx"), key=lambda x: -x[1]["episode_successes"].iloc[-1]):
    plt.plot(group["think"], group["episode_successes"], label=group["levels"].unique(), marker="+")

plt.xlabel("Thinking steps")
plt.ylabel("Episode Successes")
plt.legend()

# %%

new_table = []
for file_idx, group in sorted(df.sort_values("think").groupby("file_idx"), key=lambda x: -x[1]["episode_successes"].iloc[0]):
    new_table.append(
        {
            "Name": next(iter(group["levels"].unique())),
            "N. levels": len(list(group["all_episode_info"])[0]["episode_obs"]),
            "success at 0": "{:.01f}\\%".format(list(group["episode_successes"])[0] * 100),
            "max success": "{:.01f}\\%".format(np.max(list(group["episode_successes"])) * 100),
            "max success steps": list(group["think"])[np.argmax(list(group["episode_successes"]))],
        }
    )

all_levels_table = pd.DataFrame(new_table)
# print(pd.DataFrame(new_table).to_latex())
# %%

new_table = []
for file_idx, group in sorted(df.sort_values("think").groupby("file_idx"), key=lambda x: -x[1]["episode_successes"].iloc[0]):
    big_levels = [
        i for i, obs in enumerate(list(group["all_episode_info"])[0]["episode_obs"]) if obs.shape[3] > 10 and obs.shape[2] > 10
    ]
    if not big_levels:
        new_table.append(
            {
                "Name": next(iter(group["levels"].unique())),
                "N. levels >10": 0,
                "success at 0 >10": "---",
                "max success >10": "---",
                "max success steps >10": "---",
            }
        )
        continue

    success_per_steps = [
        np.mean(all_episode_info["episode_successes"][big_levels], axis=0) for all_episode_info in group["all_episode_info"]
    ]
    print(success_per_steps)

    new_table.append(
        {
            "Name": next(iter(group["levels"].unique())),
            "N. levels >10": len(big_levels),
            "success at 0 >10": "{:.01f}\\%".format(float(success_per_steps[0]) * 100),
            "max success >10": "{:.01f}\\%".format(float(np.max(success_per_steps)) * 100),
            "max success steps >10": list(group["think"])[np.argmax(success_per_steps)],
        }
    )

big_levels_table = pd.DataFrame(new_table)
# print(pd.DataFrame(new_table).to_latex())

# %%
combined_table = pd.merge(all_levels_table, big_levels_table, on="Name", how="inner")


def slugify(text):
    return text.lower().replace(" ", "-").replace(".", "").replace("'", "")


# List of level names
pretty_level_names = """
Intro
Sokoban
Sokoban Jr. 1
Sokoban Jr. 2
Deluxe
Sokogen 990602
Xsokoban
David Holland 1
David Holland 2
Howard's 1st set
Howard's 2nd set
Howard's 3rd set
Howard's 4th set
Sasquatch
Mas Sasquatch
Sasquatch III
Sasquatch IV
Still more levels
Nabokosmos
Microcosmos
Microban
Simple sokoban
Dimitri and Yorick
Yoshio Automatic
"""

reverse_slug = {slugify(n): n for n in pretty_level_names.strip().split("\n")}
combined_table["Name"] = combined_table["Name"].map(reverse_slug.__getitem__)


print(pd.DataFrame(combined_table).to_latex(index=False))


# %%


buckets = collections.defaultdict(list)
bucket_names = collections.defaultdict(list)


def fn(all_episode_info):
    assert len(all_episode_info["episode_successes"]) == len(all_episode_info["episode_obs"])
    for i, (episode_success, episode_obs) in enumerate(
        zip(all_episode_info["episode_successes"], all_episode_info["episode_obs"])
    ):
        if episode_obs.shape[-2] > 10:
            if episode_obs.shape[-1] > 10:
                buckets["both >10"].append(episode_success)
                bucket_names["both >10"].append(
                    (
                        all_episode_info["episode_obs"][i],
                        all_episode_info["level_infos"]["level_file_idx"][i],
                        all_episode_info["level_infos"]["level_idx"][i],
                    )
                )
            else:
                buckets["one >10"].append(episode_success)
                bucket_names["one >10"].append(
                    (
                        all_episode_info["episode_obs"][i],
                        all_episode_info["level_infos"]["level_file_idx"][i],
                        all_episode_info["level_infos"]["level_idx"][i],
                    )
                )
        else:
            if episode_obs.shape[-1] > 10:
                buckets["one >10"].append(episode_success)
                bucket_names["one >10"].append(
                    (
                        all_episode_info["episode_obs"][i],
                        all_episode_info["level_infos"]["level_file_idx"][i],
                        all_episode_info["level_infos"]["level_idx"][i],
                    )
                )
            else:
                buckets["≤10"].append(episode_success)
                bucket_names["≤10"].append(
                    (
                        all_episode_info["episode_obs"][i],
                        all_episode_info["level_infos"]["level_file_idx"][i],
                        all_episode_info["level_infos"]["level_idx"][i],
                    )
                )


# %%
df[df["think"] == 128]["all_episode_info"].map(fn)
{k: np.sum(v) for k, v in buckets.items()}, {k: len(v) for k, v in buckets.items()}


# %%
#
fancy_sprite = True
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = None


for THINK_STEPS in [0, 2, 4, 8, 16, 32, 64, 128]:
    buckets.clear()
    bucket_names.clear()
    df[df["think"] == THINK_STEPS]["all_episode_info"].map(fn)

    def frame_to_bgr(o):
        o = np.transpose(o, (1, 2, 0))
        if fancy_sprite:
            o = fancy_obs(o)
        else:
            o = np.kron(o, np.ones((10, 10, 1), dtype=np.uint8))
        o = o.astype(np.uint8)
        return cv2.cvtColor(o, cv2.COLOR_RGB2BGR)

    # for obs, level_file_idx, level_idx, success in zip(*zip(*bucket_names["≤10"]), buckets["≤10"]):
    for obs, level_file_idx, level_idx, success in zip(*zip(*bucket_names["both >10"]), buckets["both >10"]):
        if not success:
            continue
        fpath = Path(
            f"big-level-videos/{THINK_STEPS}_steps/{'success' if success else 'failure'}/{level_names[level_file_idx]}-{level_idx}.mp4"
        )
        fpath.parent.mkdir(exist_ok=True, parents=True)

        zero_fpath = Path(
            f"big-level-svg/0_steps/{'success' if success else 'failure'}/{level_names[level_file_idx]}-{level_idx}.svg"
        )
        zero_fpath.parent.mkdir(exist_ok=True, parents=True)
        # if not zero_fpath.exists():
        #     fpath_svg = Path(
        #         f"big-level-svg/{THINK_STEPS}_steps/{'success' if success else 'failure'}/{level_names[level_file_idx]}-{level_idx}.svg"
        #     )
        #     fpath_svg.parent.mkdir(exist_ok=True, parents=True)

        #     with fpath_svg.open("w") as f:
        #         f.write(plot.render_svg.tiny_world_rgb_to_svg(np.transpose(obs[0], (1, 2, 0))))
        if (
            (THINK_STEPS, level_names[level_file_idx], level_idx)
            in [
                # (8, "sokogen-990602", 66),
                # (0, "howards-2nd-set", 14),
                # (0, "xsokoban", 31),
                # (32, "microban", 144),
                (128, "xsokoban", 29),
                # (32, "microban", 104),
                # (0, "microban", 104),
                # (16, "sokoban-jr", 2),
                # (16, "sokoban-jr", 45),
            ]
            # or (not zero_fpath.exists())
            # or (level_names[level_file_idx], level_idx) in [("mas-sasquatch", 14), ("microban", 143)]
        ):
            # if level_names[level_file_idx] != "xsokoban":
            #     continue
            if success:
                name = f"{level_names[level_file_idx]}-{level_idx}"
                print(name)
            writer = None
            for o in obs:
                img_bgr = frame_to_bgr(o)
                if writer is None:
                    writer = cv2.VideoWriter(str(fpath), fourcc, 20.0, (img_bgr.shape[1], img_bgr.shape[0]))
                writer.write(img_bgr)
            if success:
                img_bgr = frame_to_bgr(get_solved_obs(obs[-1]))
                writer.write(img_bgr)
            writer.release()
            writer = None
