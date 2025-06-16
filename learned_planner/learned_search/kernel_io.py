"""LPE, TPE kernel input output visualization."""

# %%
import argparse
from pathlib import Path

import plotly.express as px
import torch as th

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, default="")
args = parser.parse_args()

output_path = args.output_path

def plot(a, name, remove_colorscale=True):
    fig = px.imshow(a, color_continuous_scale="Viridis", zmax=1, zmin=-1)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False).update_yaxes(
        showticklabels=False, showgrid=False, zeroline=False
    )
    if remove_colorscale:
        fig.update_layout(coloraxis_showscale=False)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    fig.write_image(Path(output_path) / name)


def convolve(a, b, prefix="", remove_colorscale=True):
    o = th.nn.functional.conv2d(th.as_tensor(b[None, None, :, :]), th.as_tensor(a[None, None, :, :]), padding="same")[0, 0]
    plot(a, name=prefix + "_kernel.svg", remove_colorscale=True)
    plot(b, name=prefix + "_input.svg", remove_colorscale=True)
    plot(o, name=prefix + "_output.svg", remove_colorscale=True)


b = th.zeros((3, 5))
b[1, 1:4] = 1

tk = th.tensor([[0, 0, 0], [0, -1, 1], [0.0, 1.0, -1.0]])
tk = tk[[1, 2, 0]][:, [1, 2, 0]]

convolve(tk, b, prefix="turn_offset")

lk = th.tensor([[0, 0, 0], [0, 0, 1.0], [0, 0, 0]])
convolve(lk, b, prefix="linear")

# %%
