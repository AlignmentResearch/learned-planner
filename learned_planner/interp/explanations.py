# %%
import argparse
import concurrent.futures
import dataclasses
from copy import deepcopy
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import torch as th
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from learned_planner import LP_DIR, MODEL_PATH_IN_REPO, ON_CLUSTER
from learned_planner.interp.channel_group import layer_groups
from learned_planner.interp.collect_dataset import DatasetStore, HelperFeatures  # noqa
from learned_planner.interp.offset_fns import CHANNEL_OFFSET_FNS, INV_OFFSET_FNS_DICT, OFFSET_FNS, OFFSET_FNS_DICT
from learned_planner.interp.plot import plotly_feature_vis
from learned_planner.interp.utils import get_boxoban_cfg, load_jax_model_to_torch, play_level
from learned_planner.policies import download_policy_from_huggingface

parser = argparse.ArgumentParser()
parser.add_argument("--cache_path", type=str, default=None, help="Path to the cache directory")
parser.add_argument("--num_envs", type=int, default=1000, help="Number of environments to run")
parser.add_argument("--internal_steps", action="store_true", help="Use internal steps for the model")
parser.add_argument("--play_level", action="store_true", help="Run the model to collect data")
parser.add_argument("--search_offset", action="store_true", help="Search for the best offset function for each channel")
parser.add_argument(
    "--evaluate", action="store_true", help="Evaluate the model using saved coefs. Otherwise, train and evaluate."
)
parser.add_argument("--skip_first_n", type=int, default=0, help="Skip the first n steps in each episode")
parser.add_argument("--keep_walls", action="store_true", help="Keep walls in the observation")
try:
    args = parser.parse_args()
    cache_path = args.cache_path
    num_envs = args.num_envs
    internal_steps = args.internal_steps
    load_cache = not args.play_level
    search_offset = args.search_offset
    evaluate = args.evaluate
    skip_first_n = args.skip_first_n
    keep_walls = args.keep_walls
except SystemExit:
    args = parser.parse_args([])
    cache_path = None
    num_envs = 2
    internal_steps = False
    load_cache = False
    search_offset = False
    evaluate = True
    skip_first_n = 0
    keep_walls = False

MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)
boxo_cfg = get_boxoban_cfg(num_envs=num_envs, episode_steps=120, split=None, difficulty="hard")
cfg_th, policy_th = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)
boxo_env = boxo_cfg.make()


def run_model():
    out = play_level(
        boxo_env,
        reset_opts={"level_file_idx": 0, "level_idx": 3},
        policy_th=policy_th,
        max_steps=120,
        internal_steps=internal_steps,
        names_filter=[
            f"features_extractor.cell_list.{layer}.hook_h.0.{int_step}"
            for layer in range(cfg_th.features_extractor.n_recurrent)  # type: ignore
            for int_step in range(cfg_th.features_extractor.repeats_per_step)  # type: ignore
        ],
    )
    obs = out.obs.squeeze(1)
    cache = out.cache
    if internal_steps:
        cache_h = [cache[f"features_extractor.cell_list.{layer}.hook_h"] for layer in range(3)]
        obs = obs.repeat_interleave(3, 0)
    else:
        cache_h = [cache[f"features_extractor.cell_list.{layer}.hook_h"][2::3, :, :] for layer in range(3)]
        obs = obs.numpy()

    features = [HelperFeatures.from_play_output(out, i, internal_steps=internal_steps) for i in range(num_envs)]
    return obs, out.lengths.numpy(), cache_h, features


if cache_path:
    cache_path = Path(cache_path)
elif ON_CLUSTER:
    cache_path = Path(f"/training/activations_dataset/{'valid' if evaluate else 'train'}_medium/0_think_step")
else:
    cache_path = LP_DIR / "drc33_cache/0_think_step"


def load_data():
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:

        def map_fn(i):
            data = pd.read_pickle(cache_path / f"idx_{i}.pkl")
            features = HelperFeatures.from_ds(data)
            cache_h = [data.model_cache[f"features_extractor.cell_list.{layer}.hook_h"] for layer in range(3)]
            cache_h = [cache if internal_steps else cache[2::3] for cache in cache_h]
            return data.obs, cache_h, features

        loaded_data = list(tqdm(executor.map(map_fn, range(num_envs)), total=num_envs))
        obs = th.nn.utils.rnn.pad_sequence([d[0] for d in loaded_data])
        obs = obs.repeat_interleave(3, 0) if internal_steps else obs
        obs = obs.numpy()
        lengths = th.tensor([d[0].shape[0] for d in loaded_data]).numpy()
        layer_wise_h = [
            th.nn.utils.rnn.pad_sequence([th.tensor(d[1][layer]) for d in loaded_data]).numpy() for layer in range(3)
        ]
        features = [d[2] for d in loaded_data]
    return obs, lengths, layer_wise_h, features


obs, lengths, cache_h, features = load_data() if load_cache else run_model()

# %%


def reconstruction_error(orig_act, recons_act, normalize=False):
    error = np.sqrt((orig_act - recons_act) ** 2).mean()
    if normalize:
        error /= np.sqrt((orig_act - orig_act.mean()) ** 2).mean()
    return error


def correlation(orig_act, recons_act):
    return np.corrcoef(orig_act.flatten(), recons_act.flatten())[0, 1]


def visualize(orig_act, recons_act, batch_idx=0):
    stacked = np.stack([orig_act, recons_act], axis=1)
    return plotly_feature_vis(stacked, obs[:, batch_idx], ["Original", "Reconstructed"])


# %%

layer, channel, batch_idx = 1, 19, 0
fn_name = f"l{layer}h{channel}_fn"
orig = cache_h[layer][:, batch_idx, channel]

# %%


@dataclasses.dataclass
class ChannelCoefs:
    name: str
    shift_fn: str
    model: LinearRegression
    resid: float
    corr: float
    only_base_features: bool = False


def learn_coefs(orig_acts, feature_tensor: np.ndarray, bias=False, alpha=1e-4):
    """Learns coefficients for each feature function using linear regression."""
    X, y = feature_tensor, orig_acts

    assert len(X.shape) == len(y.shape) + 1, f"{X.shape=}, {y.shape=}"
    if len(X.shape) > 2:
        X = X.reshape(-1, X.shape[-1])
        y = y.reshape(-1)

    # model = LinearRegression(fit_intercept=bias)
    model = sklearn.linear_model.Lasso(alpha=alpha, fit_intercept=bias)
    model.fit(X, y)
    y_pred = model.predict(X)
    resid = np.mean((y - y_pred) ** 2) / np.mean((y - y.mean()) ** 2)
    corr = correlation(y, y_pred)

    return model, resid, corr


# %%


def remove_walls_fn(obs, acts):
    # assert len(acts.shape) == 3
    assert len(obs.shape) == 3

    non_wall_sqs = ~((obs == 0).all(axis=0))
    return acts[..., non_wall_sqs]


def evaluation_fn(cache_h, features, coefs_for_channels, remove_walls=True):
    """Evaluate the learned coefficients on the cache and features."""
    assert len(features) == num_envs, f"Expected {num_envs} features, got {len(features)}"
    only_base_features = coefs_for_channels[0][0].only_base_features
    if remove_walls:
        features_tensor = [
            np.transpose(ft.to_nparray(bias=False, only_base_features=only_base_features), (0, 3, 1, 2)) for ft in features
        ]
        features_tensor = np.concatenate(
            [
                einops.rearrange(remove_walls_fn(obs[0, i], ft[skip_first_n:]), "s c w -> (s w) c")
                for i, ft in enumerate(features_tensor)
            ]
        )
    else:
        features_tensor = [ft.to_nparray(bias=False, only_base_features=only_base_features) for ft in features]
        features_tensor = np.concatenate([ft.reshape(-1, ft.shape[-1]) for ft in features_tensor])

    for layer in range(0, 3):
        for channel in range(32):
            channel_shift_fn = coefs_for_channels[layer][channel].shift_fn
            orig = [
                INV_OFFSET_FNS_DICT[channel_shift_fn](
                    cache_h[layer][skip_first_n : lengths[i], i, channel], last_dim_grid=True
                )
                for i in range(num_envs)
            ]
            orig = np.concatenate(
                [
                    remove_walls_fn(obs[0, i], orig[i]).reshape(-1) if remove_walls else orig[i].reshape(-1)
                    for i in range(num_envs)
                ]
            )
            channel_info = coefs_for_channels[layer][channel]
            model = channel_info.model
            y_pred = np.einsum("nc, c -> n", features_tensor, model.coef_) + model.intercept_
            resid = np.mean((orig - y_pred) ** 2) / np.mean((orig - y_pred.mean()) ** 2)  # normalized residual
            corr = correlation(orig, y_pred)
            channel_info.resid = resid
            channel_info.corr = corr


# %%

if evaluate:
    base_path = Path("/training/circuit_logs/") if ON_CLUSTER else LP_DIR / "circuit_logs/"
    coefs_for_channels = pd.read_pickle(base_path / "coefs_for_channels.pkl")
    base_coefs_for_channels = pd.read_pickle(base_path / "base_coefs_for_channels.pkl")

    evaluation_fn(cache_h, features, coefs_for_channels, remove_walls=not keep_walls)
    evaluation_fn(cache_h, features, base_coefs_for_channels, remove_walls=not keep_walls)

else:
    coefs_for_channels = []
    base_coefs_for_channels = []
    for layer in range(0, 3):
        coefs_for_channels.append([])
        base_coefs_for_channels.append([])
        for channel in range(32):
            orig = np.concatenate([cache_h[layer][: lengths[i], i, channel].reshape(-1) for i in range(num_envs)])

            for only_base_features in [True, False]:
                models = []
                for shift_fn in OFFSET_FNS if search_offset else [CHANNEL_OFFSET_FNS[layer][channel]]:
                    features_tensor = [ft.to_nparray(bias=False, only_base_features=only_base_features) for ft in features]
                    features_tensor = np.concatenate([shift_fn(ft).reshape(-1, ft.shape[-1]) for ft in features_tensor])
                    model, resid, corr = learn_coefs(orig, features_tensor, bias=True)
                    models.append((shift_fn, model, resid, corr))
                shift_fn, model, resid, corr = max(models, key=lambda x: x[3])
                name = f"L{layer}H{channel}"
                channel_coef = ChannelCoefs(name, shift_fn.__name__, model, resid, corr, only_base_features)
                if only_base_features:
                    base_coefs_for_channels[-1].append(channel_coef)
                else:
                    coefs_for_channels[-1].append(channel_coef)

    min_corr_channel = sorted(
        [x for xl in coefs_for_channels for x in xl],
        key=lambda x: x.corr,
    )

    if search_offset:
        with open("feature_offsets.txt", "w") as f:
            for layer in range(0, 3):
                for channel in range(32):
                    print(layer, channel, coefs_for_channels[layer][channel].shift_fn)
                    f.write(f"{layer} {channel} {coefs_for_channels[layer][channel].shift_fn}\n")
    base_path = "/training/circuit_logs/"
    pd.to_pickle(coefs_for_channels, base_path + "coefs_for_channels.pkl")
    pd.to_pickle(base_coefs_for_channels, base_path + "base_coefs_for_channels.pkl")

    evaluation_fn(cache_h, features, coefs_for_channels, remove_walls=not keep_walls)
    evaluation_fn(cache_h, features, base_coefs_for_channels, remove_walls=not keep_walls)

print("Mean correlation:", np.mean([x.corr for xl in coefs_for_channels for x in xl]))

# %% Correlation by group

box_corr, box_resid = [], []
agent_corr, agent_resid = [], []
group_corr = {}
base_box_corr, base_box_resid = [], []
base_agent_corr, base_agent_resid = [], []
base_group_corr = {}


for group_name, group in layer_groups.items():
    resids, corrs = [], []
    base_resids, base_corrs = [], []
    for c_dict in group:
        layer, channel = c_dict["layer"], c_dict["idx"]
        coefs = coefs_for_channels[layer][channel]
        base_coefs = base_coefs_for_channels[layer][channel]
        if group_name.startswith("B"):
            box_resid.append(coefs.resid)
            box_corr.append(coefs.corr)
            base_box_resid.append(base_coefs.resid)
            base_box_corr.append(base_coefs.corr)
        elif group_name.startswith("A"):
            agent_resid.append(coefs.resid)
            agent_corr.append(coefs.corr)
            base_agent_resid.append(base_coefs.resid)
            base_agent_corr.append(base_coefs.corr)
        resids.append(coefs.resid)
        corrs.append(coefs.corr)
        base_resids.append(base_coefs.resid)
        base_corrs.append(base_coefs.corr)
    group_corr[group_name] = np.mean(corrs)
    base_group_corr[group_name] = np.mean(base_corrs)


group_corr["B"] = np.mean(box_corr)
group_corr["A"] = np.mean(agent_corr)
base_group_corr["B"] = np.mean(base_box_corr)
base_group_corr["A"] = np.mean(base_agent_corr)
print(f"Box Residual: {100*np.mean(box_resid):.4f}, Correlation: {100*np.mean(box_corr):.4f}")
print(f"Agent Residual: {100*np.mean(agent_resid):.4f}, Correlation: {100*np.mean(agent_corr):.4f}")
print(f"Base Box Residual: {100*np.mean(base_box_resid):.4f}, Correlation: {100*np.mean(base_box_corr):.4f}")
print(f"Base Agent Residual: {100*np.mean(base_agent_resid):.4f}, Correlation: {100*np.mean(base_agent_corr):.4f}")

# %% Latex table using df


def latex_table_of_corr(grouped=False):
    """Generates a latex table of the correlation values for each layer and channel."""
    if grouped:
        clean_group_corr = {
            k.replace("B", "Box").replace("A", "Agent").replace("T", "Target"): v for k, v in group_corr.items()
        }
        clean_base_group_corr = {
            k.replace("B", "Box").replace("A", "Agent").replace("T", "Target"): v for k, v in base_group_corr.items()
        }
        df = pd.DataFrame({"Correlation": clean_group_corr, "Base correlation": clean_base_group_corr})
        df.index.name = "Group"
        df = df * 100  # Convert to percentage
        # df replace "Misc plan" with "Combined plan"
        df = df.rename(index={"Misc plan": "Combined plan"})
        df = df.rename(index={"Box": "Box (all dir)", "Agent": "Agent (all dir)"})
        latex_str = df.to_latex(
            index=True,
            escape=True,
            column_format="lrr",  # Adjusted for the new column
            caption="Correlation of linear regression model's predictions with the original activations averaged over channels for each group. Includes correlation using only base features for comparison.",
            label="tab:correlation_grouped",
            float_format="%.2f",
        )
    else:
        l, c = len(coefs_for_channels), len(coefs_for_channels[0])
        layer_cols = [[100 * coefs_for_channels[layer][channel].corr for channel in range(c)] for layer in range(l)]

        df = pd.DataFrame({f"Layer {layer}": layer_cols[layer] for layer in range(l)})
        df.index = [f"Channel {i}" for i in range(c)]
        latex_str = df.to_latex(
            index=True,
            escape=True,
            column_format="l" + "r" * l,
            caption="Correlation of linear regression model's predictions with the original activations for each channel.",
            label="tab:correlation",
            float_format="%.2f",
        )
    # add \centering
    latex_str = latex_str.replace("\\caption", "\\centering\n\\caption")
    print(latex_str)


latex_table_of_corr(grouped=False)

latex_table_of_corr(grouped=True)

# %%


layer_wise_offset_fns = [[OFFSET_FNS_DICT[coefs_for_channels[i][c].shift_fn] for c in range(32)] for i in range(3)]


zero_padded_cache_h = [np.concatenate([np.zeros_like(cache[0, None]), cache[:-1]]) for cache in cache_h]


def learn_coefs_with_h(layer, channel, cache_h, feature_tensor: np.ndarray, layer_wise_shift_fns, bias=False, alpha=1e-4):
    assert len(feature_tensor.shape) == 4
    feature_tensor = layer_wise_shift_fns[layer][channel](feature_tensor)
    feature_tensor = feature_tensor.reshape(-1, feature_tensor.shape[-1])

    gt_acts = np.concatenate(
        [layer_wise_shift_fns[layer][channel](cache_h[layer][: lengths[i], i, channel]).reshape(-1) for i in range(num_envs)]
    )  # samples
    prev_layer = (layer + 2) % 3
    prev_tick_acts = [
        np.concatenate(
            [
                layer_wise_shift_fns[layer][channel](zero_padded_cache_h[layer][: lengths[i], i, channel]).reshape(-1)
                for i in range(num_envs)
            ]
        )
        for channel in range(32)
    ]

    prev_layer_acts = [
        np.concatenate(
            [
                layer_wise_shift_fns[prev_layer][channel](cache_h[prev_layer][: lengths[i], i, channel]).reshape(-1)
                for i in range(num_envs)
            ]
        )
        for channel in range(32)
    ]

    input_h_features = np.stack([prev_layer_acts, prev_tick_acts], axis=-1)
    input_features = np.concatenate([feature_tensor, input_h_features], axis=-1)

    # model = LinearRegression(fit_intercept=bias)
    model = sklearn.linear_model.Lasso(alpha=alpha, fit_intercept=bias)
    model.fit(input_features, gt_acts)
    y_pred = model.predict(input_features)
    resid = np.mean((gt_acts - y_pred) ** 2) / np.mean((gt_acts - gt_acts.mean()) ** 2)
    corr = correlation(gt_acts, y_pred)

    return model, resid, corr


def train_with_h():
    coefs_with_h_for_channels = []
    for layer in range(0, 3):
        coefs_with_h_for_channels.append([])
        for channel in range(32):
            features_tensor = [ft.to_nparray(bias=False) for ft in features]
            features_tensor = np.stack([ft for ft in features_tensor])
            model, resid, corr = learn_coefs_with_h(
                layer, channel, cache_h, features_tensor, layer_wise_shift_fns=layer_wise_offset_fns
            )
            name = f"L{layer}H{channel}"
            print(f"L{layer}H{channel}:", resid, corr)
            # coefs_for_channels[-1].append((name, shift_fn, model, resid, corr))
            coefs_with_h_for_channels[-1].append(
                ChannelCoefs(name, coefs_for_channels[layer][channel].shift_fn, model, resid, corr)
            )
    print("Mean correlation:", np.mean([x.corr for xl in coefs_with_h_for_channels for x in xl]))


# %% left channel
sort_by_agent_left = sorted(
    [x for xl in coefs_for_channels for x in xl],
    key=lambda x: -abs(x.model.coef_[-7]),
)

# %%

coefs_without_dir_ft = []
for layer in range(0, 3):
    coefs_without_dir_ft.append([])
    for channel in range(32):
        # coefs_for_channels[layer][channel] = ChannelCoefs(*coefs_for_channels[layer][channel])
        coefs = deepcopy(coefs_for_channels[layer][channel])
        coefs.model.coef_[-12:] = 0
        orig = np.concatenate([cache_h[layer][: lengths[i], i, channel].reshape(-1) for i in range(num_envs)])
        ft_tensor = np.concatenate(
            [
                OFFSET_FNS_DICT[coefs.shift_fn](ft.to_nparray(bias=False)).reshape(-1, coefs.model.coef_.shape[0])
                for ft in features
            ]
        )
        pred = np.einsum("nc, c -> n", ft_tensor, coefs.model.coef_) + coefs.model.intercept_
        coefs.corr = correlation(orig, pred)
        coefs.resid = np.mean((orig - pred) ** 2) / np.mean((orig - pred.mean()) ** 2)
        coefs_without_dir_ft[-1].append(coefs)


# %%
# layer, channel, batch_idx, without_dir_ft = 0, 6, 0, False
# orig = cache_h[layer][:, batch_idx, channel]
# features_tensor = features[batch_idx].to_nparray(bias=False)
# # zeros = np.zeros((1, 3, 3, 1))
# # zeros[0, 1, 1, 0] = 1
# # print("shift", coefs_for_channels[layer][channel].shift_fn(zeros)[0, :, :, 0])
# channel_coefs = (coefs_without_dir_ft if without_dir_ft else coefs_for_channels)[layer][channel]
# model = channel_coefs.model
# # features_tensor = coefs.model(features_tensor)
# # model, resid, corr = learn_coefs(orig, features_tensor, bias=True)
# resid, corr = channel_coefs.resid, channel_coefs.corr
# print(resid, corr)
# recons = np.einsum("nhwc, c -> nhw", features_tensor, model.coef_) + model.intercept_
# visualize(orig, recons, batch_idx)

# # %%
# weight_fc0 = policy_th.mlp_extractor.policy_net.fc0.weight.data
# weight_value = policy_th.value_net.weight.data
# weight_fc0 = weight_fc0.reshape(weight_fc0.shape[0], 32, 10, 10)
# top_neurons_inp = weight_fc0[:, 25, 8, 9].abs().argsort()
# top_neurons_value = weight_value[0].abs().argsort()
# # %%

# inp = th.zeros(1, 32, 10, 10, dtype=th.float32)
# inp[0, 25, 2, 3] = -1

# inp = inp.reshape(1, -1)
# out = policy_th.mlp_extractor.policy_net(inp)
# out_value = policy_th.value_net(out)
# print(out_value)

# inp = th.zeros(1, 32, 10, 10, dtype=th.float32)
# inp[0, 25, 2, 3] = 1

# inp = inp.reshape(1, -1)
# out = policy_th.mlp_extractor.policy_net(inp)
# out_value = policy_th.value_net(out)
# print(out_value)

# # %%
# down_channels = [(0, 2), (0, 14), (0, 28), (1, 17), (1, 18), (2, 3), (2, 4)]
# for layer, channel in down_channels:
#     print("Layer", layer, "Channel", channel, "Corr", coefs_for_channels[layer][channel][4])

# # %%
