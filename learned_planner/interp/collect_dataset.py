import argparse
import dataclasses
import pathlib
import pickle
import re
from typing import Dict, List, Literal, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th
from gym_sokoban.envs.sokoban_env import CHANGE_COORDINATES

# from optree import tree_map
from stable_baselines3.common import type_aliases
from stable_baselines3.common.pytree_dataclass import tree_index, tree_map
from stable_baselines3.common.type_aliases import TorchGymObs
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecTransposeImage,
)
from stable_baselines3.common.vec_env.util import obs_as_tensor
from tqdm import tqdm

from learned_planner.convlstm import ConvLSTMOptions
from learned_planner.environments import BoxobanConfig, EnvpoolSokobanVecEnvConfig
from learned_planner.interp.utils import load_jax_model_to_torch


@dataclasses.dataclass
class DatasetStore:
    store_path: pathlib.Path
    obs: th.Tensor
    rewards: th.Tensor
    solved: bool
    pred_actions: th.Tensor
    pred_values: th.Tensor
    model_cache: Dict[str, np.ndarray]
    file_idx: Optional[int] = None
    level_idx: Optional[int] = None

    def __post_init__(self):
        full_length = self.pred_actions.shape[0]
        assert self.obs.shape[0] == self.rewards.shape[0], f"{self.obs.shape[0]} != {self.rewards.shape[0]}"
        assert full_length == self.pred_values.shape[0], f"{full_length} != {self.pred_values.shape[0]}"
        for k, v in self.model_cache.items():
            if "hook_pre_model" in k:
                assert self.obs.shape[0] == v.shape[0], f"{self.obs.shape[0]} != {v.shape[0]} for {k}"
            else:
                assert full_length == len(v), f"{full_length} != {v.shape}[0] for {k}"

        assert (self.file_idx is None) == (self.level_idx is None), "file_idx and level_idx must be provided together"

    def save(self):
        self.cpu()
        with open(self.store_path, "wb") as f:
            pickle.dump(self, f)

    def cpu(self):
        self.obs = self.obs.cpu()
        self.rewards = self.rewards.cpu()
        self.pred_actions = self.pred_actions.cpu()
        self.pred_values = self.pred_values.cpu()

    @property
    def n_steps_to_think(self):
        return int(self.store_path.parent.name.split("_")[0])

    @property
    def repeats_per_step(self):
        repeats = len(self.pred_actions) / (len(self.obs) + self.n_steps_to_think)
        assert repeats.is_integer(), f"Repeats per step is not an integer: {repeats}"
        return int(repeats)

    @staticmethod
    def load(store_path: str):
        with open(store_path, "rb") as f:
            return pickle.load(f)

    def actual_steps(
        self,
        n_steps_to_think: Optional[int] = None,
        repeats_per_step: Optional[int] = None,
        include_initial_thinking: bool = False,
    ):
        n_steps_to_think = self.n_steps_to_think if n_steps_to_think is None else n_steps_to_think
        repeats_per_step = self.repeats_per_step if repeats_per_step is None else repeats_per_step
        skips = 0 if include_initial_thinking else n_steps_to_think
        start = (skips + 1) * repeats_per_step - 1
        return th.arange(start, self.pred_actions.shape[0], repeats_per_step)

    def get_actions(self, only_env_steps: bool = True, include_initial_thinking: bool = False):
        if only_env_steps:
            return self.pred_actions[self.actual_steps(include_initial_thinking=include_initial_thinking)].squeeze(-1)
        else:
            self.pred_actions.squeeze(-1)

    def get_values(self, only_env_steps: bool = True, include_initial_thinking: bool = False):
        if only_env_steps:
            return self.pred_values[self.actual_steps(include_initial_thinking=include_initial_thinking)].squeeze(-1)
        else:
            self.pred_values.squeeze(-1)

    def get_cache(self, key: str, only_env_steps: bool = False, include_initial_thinking: bool = False):
        if only_env_steps:
            return self.model_cache[key][self.actual_steps(include_initial_thinking=include_initial_thinking)]
        else:
            return self.model_cache[key]

    def get_true_values(self, gamma: float = 1.0):
        # use gamma and self.rewards to multiply gamma^0, gamma^1, gamma^2, ... to rewards using torch function
        gammas = th.pow(gamma, th.arange(len(self.rewards), device=self.rewards.device))
        discounted_rewards = gammas * self.rewards
        cumsum = th.cumsum(discounted_rewards, dim=0)
        values = discounted_rewards - cumsum + cumsum[-1]
        values = values / gammas
        return values

    def get_success_repeated(self):
        return th.tensor([self.solved] * len(self.obs), dtype=th.int)

    def get_boxing_indices(self):
        next_target_time = th.isclose(self.rewards, th.tensor(0.9)) | th.isclose(self.rewards, th.tensor(-1.1))
        next_target_time[-1] = th.isclose(self.rewards[-1], th.tensor(10.9))
        return th.where(next_target_time)[0].cpu().numpy()

    def get_next_target_positions(self):
        next_target_timesteps = self.get_boxing_indices()
        target_positions = []
        repeats = []
        last_time = -1
        for t in next_target_timesteps:
            agent_pos = self.get_agent_position_per_step(self.obs[t]).cpu()
            action = int(self.pred_actions[self.to_hidden_idx(t)].item())
            # twice away from agent when putting box on target
            multiplier = 1 if th.isclose(self.rewards[t], th.tensor(-1.1)) else 2
            target_pos = agent_pos + multiplier * th.tensor(CHANGE_COORDINATES[action])
            target_positions.append(target_pos)
            repeats.append(t - last_time)
            last_time = t
        if len(target_positions) == 0:
            return th.zeros((0, 2), dtype=th.int64)
        next_target_positions = th.repeat_interleave(th.stack(target_positions), th.tensor(repeats), dim=0)
        return next_target_positions

    def get_agent_positions(self):
        agent_positions = []
        for obs in self.obs:
            agent_positions.append(self.get_agent_position_per_step(obs).cpu())
        return th.stack(agent_positions)

    def get_agent_position_per_step(self, obs):
        assert obs.shape == (3, 10, 10)
        agent_pos = th.where((obs[0] == 160) & (obs[1] == 212) & (obs[2] == 56))
        if len(agent_pos[0]) == 0:
            agent_pos = th.where((obs[0] == 219) & (obs[1] == 212) & (obs[2] == 56))
        return th.stack(agent_pos).squeeze()

    def to_hidden_idx(self, idx):
        return (idx + self.n_steps_to_think + 1) * self.repeats_per_step - 1


def create_eval_env(
    split: Literal["train", "valid", "test", None] = "valid",
    difficulty: Literal["unfiltered", "medium", "hard"] = "medium",
    max_episode_steps=80,
    n_envs=1,
    device=th.device("cpu"),
    BOXOBAN_CACHE: pathlib.Path = pathlib.Path("/training/.sokoban_cache/"),
    envpool: bool = False,
):
    if not envpool:
        cfg = BoxobanConfig(
            max_episode_steps=max_episode_steps,
            n_envs=n_envs,
            n_envs_to_render=0,
            min_episode_steps=max_episode_steps,
            tinyworld_obs=True,
            cache_path=BOXOBAN_CACHE,
            split=split,
            difficulty=difficulty,
            seed=42,
        )
        return cfg, VecTransposeImage(DummyVecEnv([cfg.make] * cfg.n_envs))
    else:
        cfg = EnvpoolSokobanVecEnvConfig(
            max_episode_steps=max_episode_steps,
            n_envs=n_envs,
            n_envs_to_render=0,
            min_episode_steps=max_episode_steps,
            load_sequentially=False,
            cache_path=BOXOBAN_CACHE,
            split=split,
            difficulty=difficulty,
            seed=42,
        )
        return cfg, cfg.make(device=device)


def think_for_n_steps(
    policy,
    n_steps: int,
    obs_tensor: TorchGymObs,
    lstm_states,
    episode_starts: th.Tensor,
    n_envs: int,
    repeats_per_step: int,
):
    if lstm_states is None:
        out = policy.recurrent_initial_state(episode_starts.size(0), device=policy.device)
        lstm_states = out

    if not episode_starts.any() or n_steps == 0:
        return lstm_states, None
    # ignore because TorchGymObs and TensorTree do not match
    obs_for_start_envs: TorchGymObs = tree_index(obs_tensor, (episode_starts,))  # type: ignore[type-var]
    lstm_states_for_start_envs = tree_index(lstm_states, (slice(None), episode_starts))
    num_reset_envs = int(episode_starts.sum().item())
    reset_all = th.ones(num_reset_envs, device=policy.device, dtype=th.bool)
    do_not_reset = ~reset_all
    all_actions = th.zeros((num_reset_envs, n_steps * repeats_per_step, 1), dtype=th.int64, device=policy.device)
    all_values = th.zeros((num_reset_envs, n_steps * repeats_per_step, 1), dtype=th.float32, device=policy.device)
    all_cache = np.zeros((num_reset_envs, n_steps), dtype=object)
    for step_i in range(n_steps):
        (actions, values, log_probs, lstm_states_for_start_envs), cache = policy.run_with_cache(
            obs_for_start_envs,
            lstm_states_for_start_envs,
            reset_all if step_i == 0 else do_not_reset,
            return_repeats=True,
        )
        # remove hook_pre_model as it doesn't while thinking for N steps on the same observation
        cache.pop("features_extractor.hook_pre_model", None)
        all_actions[:, step_i * repeats_per_step : (step_i + 1) * repeats_per_step] = actions.transpose(0, 1)
        all_values[:, step_i * repeats_per_step : (step_i + 1) * repeats_per_step] = values.transpose(0, 1)
        all_cache[:, step_i] = split_cache(cache, num_reset_envs)

        assert (
            th.take_along_dim(th.log_softmax(cache["hook_action_net"], dim=-1), indices=actions, dim=-1)
            .squeeze(-1)
            .allclose(log_probs)
        )  # taken action's log prob from cache should match returned log probs

    def _set_thinking(x, y) -> th.Tensor:
        x = x.clone()  # Don't overwrite previous tensor
        x[:, episode_starts] = y
        return x

    lstm_states = tree_map(_set_thinking, lstm_states, lstm_states_for_start_envs)
    return lstm_states, (all_actions, all_values, all_cache)


def split_cache(cache, num_envs):
    new_cache = [{} for _ in range(num_envs)]
    for k, v in cache.items():
        if "features_extractor." in k and "hook_pre_model" not in k:
            assert v.shape[0] == num_envs, f"{v.shape}[0] != {num_envs} for {k}"
            for i in range(num_envs):
                new_cache[i][k] = v[i].cpu().numpy()
        else:
            assert v.shape[1] == num_envs, f"{v.shape}[1] != {num_envs} for {k}"
            for i in range(num_envs):
                new_cache[i][k] = v[:, i, ...].cpu().numpy()
    return new_cache


def join_cache_across_steps(cache_list):
    """Finds the cache items, whose HookPoint names are of the form
    `prefix.{Pos}.{Repeat}`. Then, it stacks all of the tensors from those parts
    of the cache, so that the 0th dimension is now of size (pos * repeat).

    Input: a list of caches, which optionally contain keys of the form
      `prefix.{Pos}.{Repeat}`
    Returns: a single cache, whose arrays are [pos*repeat, ...]-sized, with the
      cache entries in order.
    """
    new_cache = [{} for _ in range(len(cache_list))]
    for i, cache in enumerate(cache_list):
        for k, v in cache.items():
            match = re.match(r"(.*)\.(\d+)\.(\d+)$", k)
            if match is not None:
                prefix, pos, rep = match.groups()
                if prefix not in new_cache[i]:
                    new_cache[i][prefix] = [(int(pos), int(rep), v[None, ...])]
                else:
                    new_cache[i][prefix].append((int(pos), int(rep), v[None, ...]))
            else:
                new_cache[i][k] = [(0, 0, v)]
    final_cache = {}
    for k in new_cache[0].keys():
        for lx in new_cache:
            assert sorted(lx[k], key=lambda e: (e[0], e[1])) == lx[k]
        final_cache[k] = np.concatenate([x for lx in new_cache for _, _, x in sorted(lx[k], key=lambda e: (e[0], e[1]))])
    return final_cache


def evaluate_policy_and_collect_dataset(
    model,
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    n_steps_to_think: int = 0,
    max_episode_steps: int = 80,
    repeats_per_step: int = 1,
    solve_reward: float = 10.9,
) -> Union[Tuple[float, float, int], Tuple[List[float], List[int], int]]:
    """
    Runs policy for n_eval_episodes. For the ith episode, saves the output of all
    the policy's hooks to `idx_{i}.pkl`.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method and `run_with_cache` method for logging cache.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :param n_steps_to_think: how many steps should the model think before taking the first action of an episode?
    :param max_episode_steps: maximum number of steps in an episode.
    :param repeats_per_step: the number of forward passes to take for each step. N in DRC(D, N).
    :param solve_reward: the reward value at the end of a level that indicates the episode is solved.
    :return: Mean reward per episode, std of reward per episode, number of episodes solved.
        Returns ([float], [int], int) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    n_envs = env.num_envs
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []

    device = model.device
    observations = env.reset()
    observations = obs_as_tensor(observations, device)
    assert isinstance(observations, th.Tensor)

    # Hardcode episode counts and the reward accumulators to use CPU. They're used for bookkeeping and don't involve
    # much computation.

    episode_counts = th.zeros(n_envs, dtype=th.int64, device=device)

    states: tuple[th.Tensor, ...] | None = model.recurrent_initial_state(n_envs, device=device)
    current_rewards = th.zeros(n_envs, dtype=th.float32, device=device)
    episode_starts = th.ones((env.num_envs,), dtype=th.bool, device=device)

    steps_including_repeats = (n_steps_to_think + max_episode_steps) * repeats_per_step

    all_obs = th.zeros((n_envs, max_episode_steps, *observations.shape[1:]), dtype=observations.dtype)
    all_rewards = th.zeros((n_envs, max_episode_steps), dtype=th.float32)
    all_pred_actions = th.zeros((n_envs, steps_including_repeats, 1), dtype=th.int64)
    all_pred_values = th.zeros((n_envs, steps_including_repeats, 1), dtype=th.float32)
    all_model_cache = np.zeros((n_envs, n_steps_to_think + max_episode_steps), dtype=object)
    all_level_infos = -np.ones((n_envs, 2), dtype=int)
    if "level_file_idx" in env.reset_infos[0]:
        all_level_infos[:] = [(info["level_file_idx"], info["level_idx"]) for info in env.reset_infos]

    idx_in_eps = th.zeros((n_envs,), dtype=th.int64)
    idx_in_eps_with_repeats = th.zeros((n_envs,), dtype=th.int64)
    env_idx = th.arange(n_envs)
    num_finished_episodes = 0

    episodes_solved = 0
    save_dir = pathlib.Path(args.output_path) / f"{n_steps_to_think}_think_step"
    save_dir.mkdir(exist_ok=True, parents=True)

    with th.no_grad(), tqdm(total=n_eval_episodes) as pbar:
        while num_finished_episodes < n_eval_episodes:
            if n_steps_to_think > 0:
                states, actions_values_cache = think_for_n_steps(
                    model,
                    n_steps_to_think,
                    observations,
                    states,
                    episode_starts,
                    n_envs,
                    repeats_per_step,
                )
                if actions_values_cache is not None:
                    thinking_actions, thinking_values, thinking_cache = actions_values_cache
                    all_pred_actions[episode_starts, : n_steps_to_think * repeats_per_step] = thinking_actions.cpu()
                    all_pred_values[episode_starts, : n_steps_to_think * repeats_per_step] = thinking_values.cpu()
                    episode_starts_cpu = episode_starts.cpu()
                    all_model_cache[episode_starts_cpu, :n_steps_to_think] = thinking_cache
                    idx_in_eps_with_repeats[episode_starts_cpu] += n_steps_to_think * repeats_per_step
                episode_starts = th.zeros_like(episode_starts)

            (acts, values, log_probs, states), cache = model.run_with_cache(
                observations,  # type: ignore[arg-type]
                state=states,
                episode_starts=episode_starts,
                deterministic=deterministic,
                return_repeats=True,
            )
            states = tree_map(th.clone, states, namespace=type_aliases.SB3_TREE_NAMESPACE, none_is_leaf=False)  # type: ignore

            new_observations, rewards, dones, infos = env.step(acts[-1, :, 0])  # indexing due to return_repeats=True
            new_observations = obs_as_tensor(new_observations, device)
            assert isinstance(new_observations, th.Tensor)
            rewards, dones = rewards.to(device), dones.to(device)

            current_rewards += rewards

            all_obs[env_idx, idx_in_eps] = observations.cpu()
            all_rewards[env_idx, idx_in_eps] = rewards.cpu()

            indices = (idx_in_eps_with_repeats.unsqueeze(-1) + th.arange(repeats_per_step)).view(-1)
            env_idx_repeated = env_idx.repeat_interleave(repeats_per_step)
            all_pred_actions[env_idx_repeated, indices] = acts.cpu().transpose(0, 1).reshape(-1, 1)
            all_pred_values[env_idx_repeated, indices] = values.cpu().transpose(0, 1).reshape(-1, 1)
            all_model_cache[env_idx, n_steps_to_think + idx_in_eps] = split_cache(cache, n_envs)

            idx_in_eps += 1
            idx_in_eps_with_repeats += repeats_per_step
            episode_starts = dones
            for i in th.where(dones)[0]:
                reward = rewards[i].item()
                info = infos[i]
                episode_solved = np.isclose(reward, solve_reward, atol=1e-4).item()

                episode_rewards.append(current_rewards[i].item())
                episode_lengths.append(int(idx_in_eps[i].item()))
                DatasetStore(
                    store_path=save_dir / f"idx_{num_finished_episodes}.pkl",
                    obs=all_obs[i][: idx_in_eps[i]],
                    rewards=all_rewards[i][: idx_in_eps[i]],
                    solved=episode_solved,
                    pred_actions=all_pred_actions[i][: idx_in_eps_with_repeats[i]],
                    pred_values=all_pred_values[i][: idx_in_eps_with_repeats[i]],
                    model_cache=join_cache_across_steps(all_model_cache[i][: n_steps_to_think + idx_in_eps[i]]),
                    file_idx=all_level_infos[i][0],
                    level_idx=all_level_infos[i][1],
                ).save()
                if "level_idx" in env.reset_infos[i]:
                    all_level_infos[i] = (info["level_file_idx"], info["level_idx"])
                episode_counts[i] += 1
                num_finished_episodes += 1
                episodes_solved += episode_solved
                current_rewards[i] = 0
                idx_in_eps[i] = 0
                idx_in_eps_with_repeats[i] = 0

            observations = new_observations
            pbar.update(num_finished_episodes - pbar.n)
            solve_perc = 100 * episodes_solved / num_finished_episodes if num_finished_episodes > 0 else 0
            pbar.set_postfix_str(f"solved={solve_perc:.1f}")

    mean_reward = np.mean(episode_rewards).item()
    std_reward = np.std(episode_rewards).item()
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episodes_solved
    return mean_reward, std_reward, episodes_solved


def collect_dataset(model_path, args):
    if TEST:
        max_episode_steps = 10
        n_eval_episodes = 1
        n_steps_to_think = [2]
        n_envs = 2
        device = th.device("cpu")
    else:
        max_episode_steps = 80
        n_eval_episodes = args.n_eval_episodes
        n_steps_to_think = [8, 0]
        n_envs = 64
        device = th.device(args.device)
    print("Device:", device)

    for i, steps_to_think in enumerate(n_steps_to_think):
        env_cfg, eval_env = create_eval_env(
            split=args.split,
            difficulty=args.difficulty,
            max_episode_steps=max_episode_steps,
            n_envs=n_envs,
            device=device,
            BOXOBAN_CACHE=pathlib.Path(args.boxoban_cache),
            envpool=args.envpool,
        )
        cfg, policy = load_jax_model_to_torch(model_path, eval_env)
        assert isinstance(cfg.features_extractor, ConvLSTMOptions)
        policy = policy.to(device)
        solve_reward = env_cfg.reward_finished + env_cfg.reward_box + env_cfg.reward_step
        mean, std, solved = evaluate_policy_and_collect_dataset(
            policy,
            eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            n_steps_to_think=steps_to_think,
            repeats_per_step=cfg.features_extractor.repeats_per_step,
            solve_reward=solve_reward,
            max_episode_steps=max_episode_steps,
        )
        print(f"Steps to think: {steps_to_think}, mean return: {mean}, std return: {std}")
        print("Fraction of solved episodes: ", solved / n_eval_episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str)
    parser.add_argument("-t", "--test", action="store_true", help="Enable test mode")
    parser.add_argument("-e", "--envpool", action="store_true", help="Use EnvpoolSokobanVecEnv")
    parser.add_argument("-c", "--boxoban_cache", type=str, default="/training/.sokoban_cache/")
    parser.add_argument("-d", "--device", type=str, default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("-s", "--split", type=str, default="valid")
    parser.add_argument("-l", "--difficulty", type=str, default="medium")
    parser.add_argument("-n", "--n_eval_episodes", type=int, default=5000)
    parser.add_argument("-o", "--output_path", type=str, default=".")
    args = parser.parse_args()

    TEST = args.test
    model_path = pathlib.Path(args.model_path)

    collect_dataset(model_path, args)
