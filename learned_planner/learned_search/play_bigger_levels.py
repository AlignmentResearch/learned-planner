import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th

from learned_planner import LP_DIR, ON_CLUSTER
from learned_planner.environments import BoxobanConfig
from learned_planner.interp.channel_group import get_group_channels, get_group_connections
from learned_planner.interp.utils import load_jax_model_to_torch
from learned_planner.policies import download_policy_from_huggingface

# %%
parser = argparse.ArgumentParser(description="Parse file index and steps to think.")

# Add the arguments
# Example command: python play_bigger_levels.py --file_idx 25 --steps_to_think 0 --weight_factor 1.2
parser.add_argument("--file_idx", type=int, required=True, help="The index of the file to process.")
parser.add_argument("--steps_to_think", type=int, required=True, help="The number of steps to think.")
parser.add_argument("--max_steps", type=int, required=False, default=1000)
parser.add_argument("--n_episodes", type=int, required=False, default=1000)
parser.add_argument("--start_level_idx", type=int, required=False, default=0)
parser.add_argument("--output_base_path", type=str, default="bigger_levels/", help="Path to save plots and cache.")
parser.add_argument("--store_cache", action="store_true", help="Whether to store the cache.")
parser.add_argument(
    "--weight_factor", type=float, default=1.0, help="factor to multiply core weights with (in conv_ih and conv_hh)."
)
args = parser.parse_args()


# %%
boxo_cfg = BoxobanConfig(
    n_envs=1,
    n_envs_to_render=1,
    min_episode_steps=args.max_steps,
    max_episode_steps=args.max_steps,
    tinyworld_obs=True,
    cache_path=LP_DIR / "alternative-levels/levels",
    seed=1234,
    difficulty="unfiltered",
    split="train",
)
env = boxo_cfg.make()

# %%

MODEL_PATH_IN_REPO = "drc33/bkynosqi/cp_2002944000"  # DRC(3, 3) 2B checkpoint
MODEL_PATH = download_policy_from_huggingface(MODEL_PATH_IN_REPO)
model_cfg, model = load_jax_model_to_torch(MODEL_PATH, boxo_cfg)


# %%

group_channels = get_group_channels("box_agent")
group_connections = get_group_connections(group_channels)
factor = args.weight_factor

# for g1 in group_connections:
#     for g2 in g1:
#         for inplc, outlc in g2:
#             inpl, inc = inplc
#             outl, outc = outlc
#             outc_ijfo = [idx * 32 + outc for idx in range(4)]
#             if inpl == outl:
#                 model.features_extractor.cell_list[outl].conv_hh.weight.data[outc_ijfo, inc] *= factor
#             else:
#                 model.features_extractor.cell_list[outl].conv_ih.weight.data[outc_ijfo, inc + 32] *= factor


for layer in range(3):
    model.features_extractor.cell_list[layer].conv_ih.weight.data[:, 32:64] *= factor
    model.features_extractor.cell_list[layer].conv_hh.weight.data *= factor


# %%
def obs_to_torch(obs):
    out = th.as_tensor(obs).unsqueeze(0).permute((0, 3, 1, 2))
    return out


obs, info = env.reset(options=dict(level_file_idx=3, level_idx=1))


# %%

# probe_fnames = [
#     "/training/TrainProbeConfig/09-probe-action/wandb/run-20240919_005641-4astgbed/local-files/probe_l-2_x-all_y-all_c-all_ds-boxes_future_direction_map.pkl",
#     "/training/TrainProbeConfig/09-probe-action/wandb/run-20240919_230748-6tndl9lb/local-files/probe_l-2_x-all_y-all_c-all_ds-boxes_future_direction_map.pkl",
#     "/training/TrainProbeConfig/09-probe-action/wandb/run-20240919_230752-dhejplrt/local-files/probe_l-2_x-all_y-all_c-all_ds-boxes_future_direction_map.pkl",
#     "/training/TrainProbeConfig/09-probe-action/wandb/run-20240919_230757-vaou9qcc/local-files/probe_l-2_x-all_y-all_c-all_ds-boxes_future_direction_map.pkl",
# ]

# combined_probe = []
# combined_intercepts = []
# for fname in probe_fnames:
#     probe = pd.read_pickle(fname)
#     combined_probe.append(th.as_tensor(probe.coef_).squeeze(0))
#     combined_intercepts.append(th.as_tensor(probe.intercept_).squeeze(0))

# combined_probe = th.stack(combined_probe, dim=0)
# combined_intercepts = th.stack(combined_intercepts, dim=0)
# th.save((combined_probe, combined_intercepts), "learned_planner/notebooks/action_l2_probe.pt")

combined_probe, combined_intercepts = th.load(LP_DIR / "learned_planner/notebooks/action_l2_probe.pt", weights_only=True)

# %%


@dataclasses.dataclass
class EvalConfig:
    env: BoxobanConfig
    n_episodes: int
    level_file_idx: int = 3  # dimitri & yorick
    steps_to_think: list[int] = dataclasses.field(default_factory=lambda: [0])
    temperature: float = 0.0

    safeguard_max_episode_steps: int = 30000

    def run(self, get_action_fn, initialize_carry_fn) -> dict[str, float]:
        # assert isinstance(self.env, EnvpoolBoxobanConfig)
        max_steps = min(self.safeguard_max_episode_steps, self.env.max_episode_steps)
        episode_starts_no = th.zeros(1, dtype=th.bool)

        metrics = {}
        try:
            env = self.env.make()
            for steps_to_think in self.steps_to_think:
                all_episode_returns = []
                all_episode_lengths = []
                all_episode_successes = []
                all_obs = []
                all_acts = []
                all_rewards = []
                all_level_infos = []
                all_cache = []
                # envs = dataclasses.replace(self.env, seed=env_seed).make()
                for episode_i in range(args.start_level_idx, args.start_level_idx + self.n_episodes):
                    try:
                        obs, level_infos = env.reset(options=dict(level_idx=episode_i, level_file_idx=self.level_file_idx))
                    except AssertionError as e:
                        if "not in range" in str(e):
                            print(f"Level {episode_i} not in range, skipping.")
                            break
                        raise e

                    obs = obs_to_torch(obs)
                    carry = initialize_carry_fn(obs)

                    for think_step in range(steps_to_think):
                        _, carry = get_action_fn(obs, carry, episode_starts_no)

                    eps_done = np.zeros(1, dtype=np.bool_)
                    episode_success = np.zeros(1, dtype=np.bool_)
                    episode_returns = np.zeros(1, dtype=np.float64)
                    episode_lengths = np.zeros(1, dtype=np.int64)
                    episode_obs = np.zeros((max_steps + 1, *obs.shape), dtype=np.int64)
                    episode_acts = np.zeros((max_steps, 1), dtype=np.int64)
                    episode_rewards = np.zeros((max_steps, 1), dtype=np.float64)
                    # 3 for the 3 layers and 64 for the number of neurons in h and c
                    cache = np.zeros((max_steps, 3, 64, obs.shape[-2], obs.shape[-1]), dtype=np.float32)

                    episode_obs[0] = obs
                    i = 0
                    while not np.all(eps_done):
                        if i >= self.safeguard_max_episode_steps:
                            break

                        action, carry = get_action_fn(obs, carry, episode_starts_no)

                        cpu_action = action.item()
                        obs, rewards, terminated, truncated, infos = env.step(cpu_action)
                        obs = obs_to_torch(obs)
                        for layer in range(3):
                            cache[i][layer] = th.cat(carry[layer], dim=2).squeeze().squeeze().numpy()

                        episode_returns += rewards  # type: ignore
                        episode_lengths += 1
                        episode_success |= terminated  # If episode terminates it's a success

                        episode_obs[i + 1, ...] = obs
                        episode_acts[i] = action
                        episode_rewards[i] = rewards

                        # Set as done the episodes which are done
                        eps_done |= truncated | terminated
                        i += 1

                    all_episode_returns.append(episode_returns)
                    all_episode_lengths.append(episode_lengths)
                    all_episode_successes.append(episode_success)

                    all_obs += [episode_obs[: episode_lengths[i], i] for i in range(1)]
                    all_acts += [episode_acts[: episode_lengths[i], i] for i in range(1)]
                    all_rewards += [episode_rewards[: episode_lengths[i], i] for i in range(1)]

                    if args.store_cache:
                        all_cache += [cache[: episode_lengths[i]] for i in range(1)]

                    all_level_infos.append(level_infos)
                    print(f"{steps_to_think=}, {episode_i=}, success:{episode_success.item()}")

                all_episode_returns = np.stack(all_episode_returns)
                all_episode_lengths = np.stack(all_episode_lengths)
                all_episode_successes = np.stack(all_episode_successes)
                if isinstance(self.env, BoxobanConfig):
                    all_level_infos = {
                        k: np.stack([d[k] for d in all_level_infos])
                        for k in all_level_infos[0].keys()
                        if not k.startswith("_")
                    }
                else:
                    all_level_infos = {
                        k: np.stack([d[k] for d in all_level_infos]) for k in all_level_infos[0].keys() if "level" in k
                    }
                    total = set(zip(all_level_infos["level_file_idx"], all_level_infos["level_idx"]))
                    print(f"Total levels: {len(total)}")

                metrics.update(
                    {
                        f"{steps_to_think:02d}_episode_returns": float(np.mean(all_episode_returns)),
                        f"{steps_to_think:02d}_episode_lengths": float(np.mean(all_episode_lengths)),
                        f"{steps_to_think:02d}_episode_successes": float(np.mean(all_episode_successes)),
                        f"{steps_to_think:02d}_num_episodes": len(all_episode_returns),
                        f"{steps_to_think:02d}_all_episode_info": dict(
                            episode_returns=all_episode_returns,
                            episode_lengths=all_episode_lengths,
                            episode_successes=all_episode_successes,
                            episode_obs=all_obs,
                            episode_acts=all_acts,
                            episode_rewards=all_rewards,
                            level_infos=all_level_infos,
                            all_cache=all_cache,
                        ),
                    }
                )
                print(f"Success rate for {steps_to_think} steps: {np.mean(all_episode_successes)}")
        finally:
            env.close()  # type: ignore
        return metrics


# %%

prediction_record = []
# %%


def initialize_carry(obs):
    carry = [tuple(th.zeros([1, 1, 32, obs.shape[2], obs.shape[3]]) for _ in range(2)) for _ in range(3)]
    return carry


cache = {}


def _save_fn(input, hook):
    cache[hook.name] = input
    return None


weight, bias = th.load(LP_DIR / "learned_planner/notebooks/aggregation.pt")


def get_action_fn(obs, carry, episode_starts):
    # cache.clear()
    _, new_carry = model._recurrent_extract_features(obs, carry, episode_starts)

    # mlp_action, _value, _log_prob, new_carry = model(obs, carry, episode_starts, deterministic=True)
    activations = th.cat(
        # [cache["features_extractor.cell_list.2.hook_h.0.2"], cache["features_extractor.cell_list.2.hook_c.0.2"]],
        new_carry[2],
        dim=2,
    ).squeeze(0)
    # Action prediction using channels
    # action_prediction = activations[:, [29, 8, 27, 3]]
    # Action prediction using probes
    action_prediction = th.einsum("nchw,oc->nohw", activations, combined_probe) + combined_intercepts[None, :, None, None]

    # norm_action = action_prediction - action_prediction.min(dim=2, keepdim=True).values.min(dim=3, keepdim=True).values
    # norm_action = norm_action / norm_action.max(dim=2, keepdim=True).values.max(dim=3, keepdim=True).values

    # Prediction record:  0.9080541696364932
    # Prediction record:  0.796
    # Prediction record:  0.7035741064733817
    num_action1 = action_prediction.mean((2, 3))

    # Prediction record:  0.9736279401282965
    # Prediction record:  0.6863284178955261
    # Prediction record:  0.70875
    num_action2 = action_prediction.max(dim=2, keepdim=False).values.max(dim=2, keepdim=False).values

    # Prediction record:  0.940128296507484
    # Prediction record:  0.787363304981774
    # Prediction record:  0.77775
    num_action3 = (action_prediction > 0).float().mean((2, 3))
    # actions = num_action.argmax(1)
    # assert mlp_action.shape == actions.shape
    num_action = num_action1 * weight[0] + num_action2 * weight[1] + num_action3 * weight[2] + bias

    action = num_action.argmax(1)
    # prediction_record.append(action == mlp_action)
    return action, new_carry


# with th.no_grad():
#     with model.input_dependent_hooks_context(
#         obs,
#         fwd_hooks=[
#             ("features_extractor.cell_list.2.hook_h.0", _save_fn),
#             ("features_extractor.cell_list.2.hook_c.0", _save_fn),
#         ],
#         bwd_hooks=None,
#     ):
#         with model.hooks(
#             fwd_hooks=[
#                 ("features_extractor.cell_list.2.hook_h.0.2", _save_fn),
#                 ("features_extractor.cell_list.2.hook_c.0.2", _save_fn),
#             ]
#         ):
#             # for boxo_cfg in [
#             #     dataclasses.replace(boxo_cfg, difficulty="unfiltered", split="train"),
#             #     dataclasses.replace(boxo_cfg, difficulty="medium", split="train"),
#             #     dataclasses.replace(boxo_cfg, difficulty="hard", split=None),
#             # ]:
#             metrics = EvalConfig(
#                 boxo_cfg, n_episodes=1000, level_file_idx=args.file_idx, steps_to_think=[args.steps_to_think]
#             ).run(get_action_fn, initialize_carry)
#             pd.to_pickle(metrics, f"/training/bigger_levels/steps_{args.steps_to_think}_file_{args.file_idx:03d}.pkl.gz")


with th.no_grad():
    # with model.input_dependent_hooks_context(
    #     obs,
    #     fwd_hooks=[
    #         ("features_extractor.cell_list.2.hook_h.0", _save_fn),
    #         ("features_extractor.cell_list.2.hook_c.0", _save_fn),
    #     ],
    #     bwd_hooks=None,
    # ):
    #     with model.hooks(
    #         fwd_hooks=[
    #             ("features_extractor.cell_list.2.hook_h.0.2", _save_fn),
    #             ("features_extractor.cell_list.2.hook_c.0.2", _save_fn),
    #         ]
    #     ):
    # for boxo_cfg in [
    #     dataclasses.replace(boxo_cfg, difficulty="unfiltered", split="train"),
    #     dataclasses.replace(boxo_cfg, difficulty="medium", split="train"),
    #     dataclasses.replace(boxo_cfg, difficulty="hard", split=None),
    # ]:
    metrics = EvalConfig(
        boxo_cfg, n_episodes=args.n_episodes, level_file_idx=args.file_idx, steps_to_think=[args.steps_to_think]
    ).run(get_action_fn, initialize_carry)
    if ON_CLUSTER:
        args.output_base_path = Path("/training/") / args.output_base_path
    args.output_base_path = Path(args.output_base_path)
    args.output_base_path.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(
        metrics, args.output_base_path / f"steps_{args.steps_to_think}_weightfactor_{factor}_file_{args.file_idx:03d}.pkl.gz"
    )

# print("Prediction record: ", np.mean(prediction_record))

# 0.8999 on 10 random levels
# 0.9 unfil, 0.9 medium, 0.4 hard
# th.save(prediction_record, "/home/dev/persistent-storage/prediction.pt")
# %%
# prediction_record[0]

# exes, y = zip(*prediction_record)
# x1, x2, x3 = zip(*exes)

# x1 = th.cat(x1, dim=0)
# x2 = th.cat(x2, dim=0)
# x3 = th.cat(x3, dim=0)
# y = th.cat(y, dim=0)

# x1.shape, x2.shape, x3.shape, y.shape

# %%
# weight = th.ones(3, requires_grad=True)
# bias = th.zeros(4, requires_grad=True)

# optim = th.optim.Adam([weight, bias], lr=0.001)

# criterion = th.nn.CrossEntropyLoss()

# for i in range(1, 10001):
#     optim.param_groups[0]["lr"] = 0.001 * ((10000 - i) / 10000)
#     preds = x1 * weight[0] + x2 * weight[1] + x3 * weight[2] + bias
#     loss = criterion(preds, y)
#     if i % 10 == 0:
#         print(f"{i=}, loss={loss.item():.4f}")
#     loss.backward()
#     optim.step()

# th.save((weight.detach(), bias.detach()), "/home/dev/persistent-storage/aggregation.pt")

# # % 83% accuracy

# %%
