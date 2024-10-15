import argparse
import pickle
import subprocess

import torch as th
from sklearn.multioutput import MultiOutputClassifier

from learned_planner import LP_DIR, MODEL_PATH_IN_REPO, ON_CLUSTER
from learned_planner.interp.collect_dataset import DatasetStore
from learned_planner.interp.plot import save_video
from learned_planner.interp.train_probes import TrainOn
from learned_planner.interp.utils import get_boxoban_cfg, load_jax_model_to_torch, play_level
from learned_planner.policies import download_policy_from_huggingface

MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--difficulty", type=str, default="unfiltered")
parser.add_argument("--split", type=str, default="valid")
parser.add_argument("--lfi", type=int, default=0)
parser.add_argument("--thinking_steps", type=int, default=0)
parser.add_argument("--show_internal_steps_until", type=int, default=0)
parser.add_argument("--num_videos_to_save", type=int, default=30)
parser.add_argument("--videos_base_dir_name", type=str, default="all_probes_separate_with_internal_steps")
parser.add_argument("--fancy_sprite", action="store_true")

args = parser.parse_args()
difficulty = args.difficulty
split = args.split
if split.lower() == "none" or split.lower() == "null":
    split = None
lfi = args.lfi
thinking_steps = args.thinking_steps
show_internal_steps_until = args.show_internal_steps_until
num_videos_to_save = args.num_videos_to_save
videos_base_dir_name = args.videos_base_dir_name

boxo_cfg = get_boxoban_cfg(difficulty=difficulty, split=split)
boxo_env = boxo_cfg.make()
cfg_th, policy_th = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)

if ON_CLUSTER:
    wandb_ids_and_infos = [
        ("dirnsbf3", TrainOn(layer=-1, dataset_name="agents_future_direction_map")),
        ("vb6474rg", TrainOn(layer=-1, dataset_name="boxes_future_direction_map")),
        ("42qs0bh1", TrainOn(layer=-1, dataset_name="next_target")),
        ("6e1w1bb6", TrainOn(layer=-1, dataset_name="next_box")),
    ]
    probe_files, probe_infos = [], []
    for wandb_id, probe_info in wandb_ids_and_infos:
        command = f"/training/findprobe.sh {wandb_id}"
        file_name = subprocess.run(command, shell=True, capture_output=True, text=True).stdout
        file_name = file_name.strip()
        probe_files.append(file_name)
        probe_infos.append(probe_info)
else:
    probe_name_infos = [
        ("agents_future_direction_map_l-all.pkl", TrainOn(layer=-1, dataset_name="agents_future_direction_map")),
        ("boxes_future_direction_map_l-all.pkl", TrainOn(layer=-1, dataset_name="boxes_future_direction_map")),
        # ("boxes_future_direction_map_multioutput_l_all.pkl", TrainOn(layer=-1, dataset_name="boxes_future_direction_map")),
        # ("next_target_l-all.pkl", TrainOn(layer=-1, dataset_name="next_target")),
        # ("next_box_l-all.pkl", TrainOn(layer=-1, dataset_name="next_box")),
    ]
    probe_files = [LP_DIR / "probes/best" / file for file, _ in probe_name_infos]
    probe_infos = [info for _, info in probe_name_infos]

probes = []
for file_name in probe_files:
    with open(file_name, "rb") as f:
        probes.append(pickle.load(f))

for li in range(num_videos_to_save):
    out = play_level(
        boxo_env,
        policy_th=policy_th,
        reset_opts={"level_file_idx": lfi, "level_idx": li},
        probes=probes,
        probe_train_ons=probe_infos,
        thinking_steps=thinking_steps,
        internal_steps=(show_internal_steps_until > 0),
    )
    all_obs = out.obs.squeeze(1)
    ds = DatasetStore(None, all_obs[thinking_steps:], out.rewards, out.solved, out.acts, th.zeros(len(all_obs)), {})
    gts = []
    for pidx, probe_info in enumerate(probe_infos):
        kwargs = {}
        if "direction" in probe_info.dataset_name:
            kwargs["multioutput"] = isinstance(probes[pidx], MultiOutputClassifier)
        gts.append(getattr(ds, probe_info.dataset_name)(**kwargs).numpy())

    box_labels = ds.boxes_label_map().numpy()
    target_labels = ds.target_labels_map().numpy()

    name = LP_DIR / "plot/interp/videos/" f"{videos_base_dir_name}/{difficulty}_{lfi}_{li}.mp4"

    if thinking_steps > 0:
        name = name.with_name(name.stem + f"_ts{thinking_steps}" + name.suffix)
    save_video(
        name,
        all_obs,
        out.probe_outs,
        gts,
        all_probe_infos=probe_infos,
        overlapped=False,
        show_internal_steps_until=show_internal_steps_until,
        # box_labels=box_labels,
        # target_labels=target_labels,
        fancy_sprite=args.fancy_sprite,
    )
