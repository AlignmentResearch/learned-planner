"""Save observation (tinyworld or fancy high-res) as svg."""

# %%
import numpy as np
import plotly.express as px
import plotly.io as pio
import torch as th
from cleanba.environments import BoxobanConfig

from learned_planner import BOXOBAN_CACHE, LP_DIR
from learned_planner.interp.act_patch_utils import get_obs
from learned_planner.interp.render_svg import tiny_world_rgb_to_svg
from learned_planner.interp.utils import load_jax_model_to_torch
from learned_planner.notebooks.emacs_plotly_render import set_plotly_renderer
from learned_planner.policies import download_policy_from_huggingface

set_plotly_renderer("emacs")
pio.renderers.default = "notebook"
th.set_printoptions(sci_mode=False, precision=2)


# %%
# MODEL_PATH_IN_REPO = "drc11/eue6pax7/cp_2002944000"  # DRC(1, 1) 2B checkpoint
MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000"  # DRC(1, 1) 2B checkpoint
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)

boxo_cfg = BoxobanConfig(
    cache_path=BOXOBAN_CACHE,
    num_envs=1,
    max_episode_steps=120,
    min_episode_steps=120,
    asynchronous=False,
    tinyworld_obs=True,
    split=None,
    difficulty="hard",
    # split="train",
    # difficulty="medium",
)

model_cfg, model = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)
reps = model_cfg.features_extractor.repeats_per_step

envs = boxo_cfg.make()
# %%

# lfi = 8
# li = 4
svg = True
# lfi, li = 233, 466
lfi, li = 0, 23
obs = get_obs(lfi, li, envs)
obs = np.transpose(obs, (1, 2, 0))
fig = px.imshow(obs)
# remove ticks
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()

if svg:
    svg, info = tiny_world_rgb_to_svg(obs, return_info=True)
    with open(LP_DIR / "iclr_logs" / f"level-{lfi}-{li}.svg", "w") as f:
        f.write(svg)
# fig.write_image(LP_DIR / "iclr_logs" / f"level-{lfi}-{li}.svg", height=200, width=200)


# %%

lfi, li = 0, 23
obs = get_obs(lfi, li, envs)
obs = np.transpose(obs, (1, 2, 0))
fig = px.imshow(obs)
# remove ticks
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_image(LP_DIR / "iclr_logs" / f"level-{lfi}-{li}-tinyobs.svg", height=200, width=200)

# %%
