import argparse
import pathlib

import torch as th
from sae_lens import SAE
from safetensors import safe_open

from learned_planner import LP_DIR, MODEL_PATH_IN_REPO
from learned_planner.interp.collect_dataset import DatasetStore
from learned_planner.interp.plot import save_video_sae
from learned_planner.interp.utils import get_boxoban_cfg, load_jax_model_to_torch, play_level
from learned_planner.policies import download_policy_from_huggingface

MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--difficulty", type=str, default="unfiltered")
parser.add_argument("--split", type=str, default="valid")
parser.add_argument("--lfi", type=int, default=0)
parser.add_argument("--thinking_steps", type=int, default=0)
parser.add_argument("--show_internal_steps_until", type=int, default=5)
parser.add_argument("--num_levels_to_save", type=int, default=2)
parser.add_argument("--videos_base_dir_name", type=str, default="sae_with_internal_steps")
parser.add_argument("--sae_path", type=str, default="")
parser.add_argument("--topkfeatures", type=int, default=512, help="comma separated feature indices")
parser.add_argument("--levels_file_path", type=str, default="")
FEATURES_PER_VIDEO = 15

args = parser.parse_args()
difficulty = args.difficulty
split = args.split
if split.lower() == "none" or split.lower() == "null" or not split:
    split = None
lfi = args.lfi
thinking_steps = args.thinking_steps
show_internal_steps_until = args.show_internal_steps_until
num_levels_to_save = args.num_levels_to_save
videos_base_dir_name = args.videos_base_dir_name
sae_path = args.sae_path
sae_name = sae_path.rstrip("/").split("/")[-1]
topkfeatures = args.topkfeatures

boxo_cfg = get_boxoban_cfg(difficulty=difficulty, split=split)
boxo_env = boxo_cfg.make()
cfg_th, policy_th = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)

sae = SAE.load_from_pretrained(sae_path)
sae.eval()
with safe_open(sae_path + "/sparsity.safetensors", "pt", device="cpu") as f:
    log_sparsity = f.get_tensor("sparsity")

top_activating_features = th.argsort(log_sparsity, descending=True).tolist()

if args.levels_file_path:
    with open(args.levels_file_path, "r") as f:
        levels = f.readlines()
    levels = [map(int, lev.strip().split(",")) for lev in levels][:num_levels_to_save]
else:
    levels = [(lfi, li) for li in range(num_levels_to_save)]

for lfi, li in levels:
    print(f"Processing level {lfi}, {li}")
    out = play_level(
        boxo_env,
        policy_th=policy_th,
        reset_opts={"level_file_idx": lfi, "level_idx": li},
        thinking_steps=thinking_steps,
        internal_steps=(show_internal_steps_until > 0),
        sae=sae,
    )
    all_obs = out.obs.squeeze(1)
    ds = DatasetStore(None, all_obs[thinking_steps:], out.rewards, out.solved, out.acts, th.zeros(len(all_obs)), {})

    thinking_steps_dir = f"ts{thinking_steps}/" if thinking_steps > 0 else ""
    name = (
        LP_DIR / "plot/interp/videos/"
        f"{videos_base_dir_name}/{sae_name}/{thinking_steps_dir}{difficulty}_{lfi}_{li}/ftseg{{ft_seg}}.mp4"
    )

    sae_acts = out.sae_outs.detach()
    if len(sae_acts.shape) == 4:
        sae_acts = sae_acts.permute(3, 0, 1, 2)
    else:
        sae_acts = sae_acts.permute(4, 0, 1, 2, 3)
    for feature_start_idx in range(0, topkfeatures, FEATURES_PER_VIDEO):
        curr_name = pathlib.Path(str(name).replace("{ft_seg}", f"{feature_start_idx}"))
        feature_idx = top_activating_features[feature_start_idx : feature_start_idx + FEATURES_PER_VIDEO]
        save_video_sae(
            curr_name,
            all_obs,
            sae_acts[feature_idx],
            show_internal_steps_until=show_internal_steps_until,
            sae_feature_indices=feature_idx,
        )
