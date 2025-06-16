"""Functions for getting or visualizing weights or inputs to the network."""

from typing import Sequence

import numpy as np
import torch as th

ALL_INP_TYPES, ALL_OUT_TYPES = ["e", "lh", "ch"], ["i", "j", "f", "o"]
INP_TYPE_TO_HOOK = {"e": "hook_layer_input", "lh": "hook_prev_layer_hidden", "ch": "hook_pool_project"}


def inp_type_to_hook_fn(inp_types):
    ret = [v for k, v in INP_TYPE_TO_HOOK.items() if k in inp_types]
    assert len(ret) > 0, f"Invalid inp_types: {inp_types}"
    return ret


def get_conv_weights(layer, out, inp, model, out_type="o", inp_types="lh", ih=True):
    assert out_type in ALL_OUT_TYPES
    if ih:
        inp_type_present = [inp_type in inp_types for inp_type in ALL_INP_TYPES]
        assert any(inp_type_present), f"Invalid inp_types: {inp_types}"
        if isinstance(inp, int):
            assert sum(inp_type_present) == 1, f"Invalid inp_types: {inp_types} for inp: {inp}"
            inp += 32 * inp_type_present.index(True)
        elif isinstance(inp, list) or isinstance(inp, tuple):
            assert sum(inp_type_present) == 1, f"Invalid inp_types: {inp_types} for inp: {inp}"
            inp = [32 * inp_type_present.index(True) + i for inp_type, i in zip(inp_types, inp)]
        else:
            start_inp_type = inp_type_present.index(True)
            end_inp_type = len(ALL_INP_TYPES) - inp_type_present[::-1].index(True)
            assert all(inp_type_present[start_inp_type:end_inp_type]), f"Not contiguous inp_types: {inp_types} with inp: {inp}"
            inp = slice(32 * start_inp_type, 32 * end_inp_type)
    else:
        if not isinstance(inp, int):
            inp = slice(None)
    if isinstance(out, int):
        out += 32 * ALL_OUT_TYPES.index(out_type)
    else:
        out = slice(32 * ALL_OUT_TYPES.index(out_type), 32 * (ALL_OUT_TYPES.index(out_type) + 1))
    try:
        comp = model.features_extractor.cell_list[layer]
    except AttributeError:
        comp = model
    comp = comp.conv_ih if ih else comp.conv_hh
    return comp.weight.data[out, inp]


def top_weights(layer, out, model, out_type="o", inp_type="lh", ih=True):
    top_channels = get_conv_weights(layer, out, None, model, out_type, inp_type, ih).abs().max(dim=1).values.max(dim=1).values
    return top_channels.argsort(descending=True)


def top_weights_out(layer, inp, model, out_type="o", inp_type="lh"):
    next_layer = (layer + 1) % 3
    inp = 32 * ALL_INP_TYPES.index(inp_type) + inp
    out_idx = ALL_OUT_TYPES.index(out_type)
    top_channels = (
        model.features_extractor.cell_list[next_layer]
        .conv_ih.weight.data[32 * out_idx : 32 * (out_idx + 1), inp]
        .abs()
        .max(dim=1)
        .values.max(dim=1)
        .values
    )
    return top_channels.argsort(descending=True)


# attribution based on max conv output
def visualize_top_conv_inputs(
    layer,
    channel,
    model,
    cache,
    num_channels=14,
    out_type="o",
    inp_types="lh",
    ih=True,
    top_channel_sum=True,
):
    conv_weights = get_conv_weights(layer, channel, None, model, out_type, inp_types, ih)
    if ih:
        hook_types = inp_type_to_hook_fn(inp_types)
        print(hook_types)
        inputs = np.concatenate(
            [cache[f"features_extractor.cell_list.{layer}.{hook_type}"] for hook_type in hook_types], axis=1
        )
    else:
        inputs = cache[f"features_extractor.cell_list.{layer}.hook_input_h"]
    # conv_kernels = conv_weights[top_channels][:, None]
    assert inputs.shape[1] % 32 == 0, f"Invalid inputs shape: {inputs.shape}"
    conv_kernels = conv_weights[:, None]
    inputs = th.tensor(inputs)
    conv_output = th.nn.functional.conv2d(inputs, conv_kernels, padding=1, groups=len(conv_kernels))
    values = conv_output.abs().max(dim=0).values.max(dim=1).values.max(dim=1).values
    top_channels = values.argsort(descending=True)
    top_channels = top_channels[:num_channels]
    sum_conv_output = (conv_output[:, top_channels] if top_channel_sum else conv_output).sum(dim=1, keepdim=True)
    conv_output = th.cat([conv_output[:, top_channels], sum_conv_output], dim=1)
    values = values[top_channels].tolist() + [-1]
    top_channels = top_channels.tolist() + [-1]
    return conv_output, top_channels, values


def get_top_conv_inputs(layer, model, cache, out_type="o", inp_type="lh", ih=True):
    conv_weights = get_conv_weights(layer, None, None, model, out_type, inp_type, ih)
    if ih:
        hook_type = INP_TYPE_TO_HOOK[inp_type]
        inputs = cache[f"features_extractor.cell_list.{layer}.{hook_type}"]
    else:
        inputs = cache[f"features_extractor.cell_list.{layer}.hook_input_h"]
    conv_kernels = conv_weights.reshape(conv_weights.shape[0] * conv_weights.shape[1], 1, 3, 3)
    inputs = th.tensor(inputs).repeat(1, len(conv_weights), 1, 1)
    conv_output = th.nn.functional.conv2d(inputs, conv_kernels, padding=1, groups=len(conv_kernels))
    conv_output = conv_output.reshape(
        conv_output.shape[0], conv_weights.shape[0], conv_weights.shape[1], *conv_output.shape[-2:]
    )
    values = conv_output.abs().max(dim=0).values.max(dim=2).values.max(dim=2).values
    top_channels = values.argsort(dim=1, descending=True)
    conv_output = th.take_along_dim(conv_output, top_channels[None, ..., None, None], dim=2)
    return conv_output


def apply_conv(x, kernel, center_mean=False, sum_channels=True):
    if len(x.shape) == 3:
        x = x[:, None]
        kernel = kernel[None]
    assert len(x.shape) == 4 and len(kernel.shape) == 3, (x.shape, kernel.shape)
    groups = 1 if sum_channels else x.shape[1]
    kernel = kernel[None] if sum_channels else kernel[:, None]
    ret = th.nn.functional.conv2d(x, kernel, padding="same", groups=groups)
    ret = ret[:, 0] if sum_channels else ret
    if center_mean:
        ret -= ret.mean(dim=(-1, -2), keepdim=True)
    return ret


def find_ijfo_contribution(
    h_cur: th.Tensor,
    channels: Sequence,
    layer_idx: int,
    c_out: int,
    model,
    ih: bool = True,
    inp_type: str = "lh",
):
    assert len(h_cur.shape) == 4  # (b, c, h, w)
    inp_acts = h_cur[:, channels]
    ijfo_acts, total_ijfo_acts = [], []
    for ijfo_str in ["i", "j", "f", "o"]:
        ifjo = apply_conv(
            inp_acts, get_conv_weights(layer_idx, c_out, None, model, ijfo_str, ih=ih)[channels], sum_channels=False
        )
        total_ijfo = apply_conv(
            h_cur,
            get_conv_weights(layer_idx, c_out, None, model, ijfo_str, inp_type, ih=ih),
            sum_channels=True,
        )
        ijfo_acts.append(ifjo)
        total_ijfo_acts.append(total_ijfo)
    ijfo_acts = th.stack(ijfo_acts, -1)
    total_ijfo_acts = th.stack(total_ijfo_acts, -1)
    return ijfo_acts, total_ijfo_acts
