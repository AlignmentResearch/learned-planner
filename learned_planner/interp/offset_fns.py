from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch as th


def identity(feature: np.ndarray | th.Tensor, last_dim_grid: bool = False):
    return feature


def left_shift(feature: np.ndarray | th.Tensor, last_dim_grid: bool = False):
    if isinstance(feature, th.Tensor):
        return th.roll(feature, -1, dims=-2 + int(last_dim_grid))
    else:
        return np.roll(feature, -1, axis=-2 + int(last_dim_grid))


def right_shift(feature: np.ndarray | th.Tensor, last_dim_grid: bool = False):
    if isinstance(feature, th.Tensor):
        return th.roll(feature, 1, dims=-2 + int(last_dim_grid))
    else:
        return np.roll(feature, 1, axis=-2 + int(last_dim_grid))


def up_shift(feature: np.ndarray | th.Tensor, last_dim_grid: bool = False):
    if isinstance(feature, th.Tensor):
        return th.roll(feature, -1, dims=-3 + int(last_dim_grid))
    else:
        return np.roll(feature, -1, axis=-3 + int(last_dim_grid))


def down_shift(feature: np.ndarray | th.Tensor, last_dim_grid: bool = False):
    if isinstance(feature, th.Tensor):
        return th.roll(feature, 1, dims=-3 + int(last_dim_grid))
    else:
        return np.roll(feature, 1, axis=-3 + int(last_dim_grid))


def up_left_shift(feature: np.ndarray | th.Tensor, last_dim_grid: bool = False):
    return up_shift(left_shift(feature, last_dim_grid), last_dim_grid)


def up_right_shift(feature: np.ndarray | th.Tensor, last_dim_grid: bool = False):
    return up_shift(right_shift(feature, last_dim_grid), last_dim_grid)


def down_left_shift(feature: np.ndarray | th.Tensor, last_dim_grid: bool = False):
    return down_shift(left_shift(feature, last_dim_grid), last_dim_grid)


def down_right_shift(feature: np.ndarray | th.Tensor, last_dim_grid: bool = False):
    return down_shift(right_shift(feature, last_dim_grid), last_dim_grid)


def up_up_left_shift(feature: np.ndarray | th.Tensor, last_dim_grid: bool = False):
    return up_shift(up_shift(left_shift(feature, last_dim_grid), last_dim_grid), last_dim_grid)


def down_down_right_shift(feature: np.ndarray | th.Tensor, last_dim_grid: bool = False):
    return down_shift(down_shift(right_shift(feature, last_dim_grid), last_dim_grid), last_dim_grid)


OFFSET_FNS = [
    identity,
    up_shift,
    down_shift,
    left_shift,
    right_shift,
    up_left_shift,
    up_right_shift,
    down_left_shift,
    down_right_shift,
    up_up_left_shift,
]
INV_OFFSET_FNS = [
    identity,
    down_shift,
    up_shift,
    right_shift,
    left_shift,
    down_right_shift,
    down_left_shift,
    up_right_shift,
    up_left_shift,
    down_down_right_shift,
]

OFFSET_FNS_DICT = {fn.__name__: fn for fn in OFFSET_FNS}
INV_OFFSET_FNS_DICT = {fn.__name__: INV_OFFSET_FNS[i] for i, fn in enumerate(OFFSET_FNS)}


def apply_offset(feature: np.ndarray | th.Tensor, offset_fn_name: str):
    if "," in offset_fn_name:
        offset_fn_name, rest = offset_fn_name.split(",", 1)
        return OFFSET_FNS_DICT[offset_fn_name](apply_offset(feature, rest))
    return OFFSET_FNS_DICT[offset_fn_name](feature)


def apply_inv_offset(feature: np.ndarray | th.Tensor, offset_fn_name: str):
    if "," in offset_fn_name:
        offset_fn_name, rest = offset_fn_name.split(",", 1)
        return apply_inv_offset(INV_OFFSET_FNS_DICT[offset_fn_name](feature), rest)
    return INV_OFFSET_FNS_DICT[offset_fn_name](feature)


def apply_offset_lc(feature: np.ndarray | th.Tensor, layer: int, channel: int, last_dim_grid: bool = False):
    return CHANNEL_OFFSET_FNS[layer][channel](feature, last_dim_grid)


def apply_inv_offset_lc(feature: np.ndarray | th.Tensor, layer: int, channel: int, last_dim_grid: bool = False):
    return INV_CHANNEL_OFFSET_FNS[layer][channel](feature, last_dim_grid)


CHANNEL_OFFSET_FNS_STR = """
0 0 up_left_shift
0 1 identity
0 2 left_shift
0 3 identity
0 4 up_left_shift
0 5 left_shift
0 6 identity
0 7 up_shift
0 8 left_shift
0 9 identity
0 10 up_left_shift
0 11 up_shift
0 12 left_shift
0 13 up_shift
0 14 identity
0 15 identity
0 16 up_left_shift
0 17 up_shift
0 18 up_shift
0 19 up_left_shift
0 20 up_shift
0 21 up_shift
0 22 identity
0 23 up_left_shift
0 24 up_left_shift
0 25 up_shift
0 26 up_shift
0 27 identity
0 28 identity
0 29 identity
0 30 up_shift
0 31 up_left_shift
1 0 identity
1 1 up_left_shift
1 2 identity
1 3 up_shift
1 4 up_left_shift
1 5 up_up_left_shift
1 6 up_left_shift
1 7 up_shift
1 8 identity
1 9 identity
1 10 up_shift
1 11 left_shift
1 12 left_shift
1 13 up_shift
1 14 left_shift
1 15 identity
1 16 identity
1 17 identity
1 18 identity
1 19 up_shift
1 20 left_shift
1 21 up_shift
1 22 identity
1 23 up_shift
1 24 left_shift
1 25 up_left_shift
1 26 left_shift
1 27 up_left_shift
1 28 identity
1 29 up_shift
1 30 identity
1 31 left_shift
2 0 up_shift
2 1 up_left_shift
2 2 identity
2 3 identity
2 4 left_shift
2 5 up_shift
2 6 up_left_shift
2 7 identity
2 8 up_shift
2 9 up_shift
2 10 up_shift
2 11 left_shift
2 12 left_shift
2 13 up_shift
2 14 up_left_shift
2 15 up_left_shift
2 16 up_left_shift
2 17 up_shift
2 18 up_left_shift
2 19 up_left_shift
2 20 left_shift
2 21 identity
2 22 up_shift
2 23 left_shift
2 24 up_shift
2 25 up_left_shift
2 26 identity
2 27 identity
2 28 up_shift
2 29 left_shift
2 30 up_left_shift
2 31 left_shift
"""


def offset_yx(y: int, x: int, channels: Sequence, layer: int, inverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
    offset_fn_list = INV_CHANNEL_OFFSET_FNS[layer] if inverse else CHANNEL_OFFSET_FNS[layer]
    offset_y, offset_x = [], []
    for i, channel in enumerate(channels):
        mat = np.zeros((5, 5))
        mat[2, 2] = 1
        mat = offset_fn_list[channel](mat, last_dim_grid=True)
        new_y, new_x = np.where(mat)
        offset_y.append(new_y[0] - 2)
        offset_x.append(new_x[0] - 2)
    offset_y, offset_x = np.array(offset_y), np.array(offset_x)
    offset_y += y
    offset_x += x
    return offset_y, offset_x


def load_channel_offset_fns() -> list[list[Callable]]:
    channel_offset_fns = [[] for _ in range(3)]
    for line in CHANNEL_OFFSET_FNS_STR.strip().split("\n"):
        layer, ch, name = line.split()
        layer, ch = int(layer), int(ch)
        assert len(channel_offset_fns[layer]) == ch
        channel_offset_fns[layer].append(OFFSET_FNS_DICT[name])
    return channel_offset_fns


CHANNEL_OFFSET_FNS = load_channel_offset_fns()
INV_CHANNEL_OFFSET_FNS = [[INV_OFFSET_FNS_DICT[fn.__name__] for fn in layer] for layer in CHANNEL_OFFSET_FNS]

OFFSET_VALUES_LAYER_WISE_LIST = [offset_yx(0, 0, range(32), layer) for layer in range(3)]
OFFSET_VALUES_LAYER_WISE = np.array(OFFSET_VALUES_LAYER_WISE_LIST)
OFFSET_VALUES_LAYER_WISE = OFFSET_VALUES_LAYER_WISE.transpose(0, 2, 1)


def test_offset_yx():
    y, x = 5, 7
    channels = [0, 1, 2, 3, 4]
    layer = 0
    offset_y, offset_x = offset_yx(y, x, channels, layer)
    print(offset_y, offset_x)
    assert np.all(offset_y == np.array([6, 5, 5, 5, 4]))
    assert np.all(offset_x == np.array([7, 7, 6, 7, 6]))


def latex_table():
    l, c, d = OFFSET_VALUES_LAYER_WISE.shape
    layer_cols = [[(y, x) for y, x in OFFSET_VALUES_LAYER_WISE[layer]] for layer in range(l)]

    df = pd.DataFrame({f"Layer {layer}": layer_cols[layer] for layer in range(l)})
    df.index = [f"Channel {i}" for i in range(c)]
    latex_str = df.to_latex(
        index=True,
        escape=True,
        column_format="l" + "r" * l,
        caption="Offset along (row, column) in the grid for each layer and channel",
        label="tab:offsets",
    )
    # add \centering
    latex_str = latex_str.replace("\\caption", "\\centering\n\\caption")
    print(latex_str)


if __name__ == "__main__":
    test_offset_yx()
    latex_table()
