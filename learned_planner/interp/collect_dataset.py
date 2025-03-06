import argparse
import concurrent.futures as cf
import dataclasses
import pathlib
import pickle
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th
from gym_sokoban.envs.sokoban_env import CHANGE_COORDINATES
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
from learned_planner.interp import alternative_plans
from learned_planner.interp.render_svg import BOX, BOX_ON_TARGET, FLOOR, PLAYER, PLAYER_ON_TARGET, TARGET
from learned_planner.interp.utils import join_cache_across_steps, load_jax_model_to_torch
from learned_planner.policies import download_policy_from_huggingface

SOLVE_REWARD = 10.9
BOX_IN_REWARD = 0.9  # Reward for pushing the box onto a target
BOX_OUT_PENALTY = -1.1  # Penalty for removing box from a target
NUM_BOXES = 4  # Number of boxes and targets in the Sokoban environment
EMPTY_SQUARE = -1  # Value for a grid cell that does not contain a box or target


@dataclasses.dataclass
class DatasetStore:
    store_path: Optional[pathlib.Path]
    obs: th.Tensor  # Observations. Dimension[steps or steps*layers, 3 RGB, 10, 10]
    rewards: Optional[th.Tensor] = None
    solved: bool = False
    pred_actions: Optional[th.Tensor] = None
    pred_values: Optional[th.Tensor] = None
    model_cache: Dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    file_idx: Optional[int] = None
    level_idx: Optional[int] = None

    def __post_init__(self):
        if self.rewards is not None:
            if self.obs.shape[0] != self.rewards.shape[0]:
                warnings.warn(f"Obs and rewards shape not matching: {self.obs.shape[0]} != {self.rewards.shape[0]}")
        if self.pred_values is not None and self.pred_actions is not None:
            assert self.pred_actions.shape[0] == self.pred_values.shape[0]
        for k, v in self.model_cache.items():
            if "hook_pre_model" in k:
                assert self.obs.shape[0] == v.shape[0], f"{self.obs.shape[0]} != {v.shape[0]} for {k}"
            elif self.pred_actions is not None:
                assert self.pred_actions.shape[0] == len(v), f"{self.pred_actions.shape[0]} != {v.shape}[0] for {k}"

        assert (self.file_idx is None) == (self.level_idx is None), "file_idx and level_idx must be provided together"

    def save(self):
        self.cpu()
        assert self.store_path is not None
        with open(self.store_path, "wb") as f:
            pickle.dump(self, f)

    def cpu(self):
        self.obs = self.obs.cpu()
        self.rewards = self.rewards.cpu() if self.rewards is not None else None
        self.pred_actions = self.pred_actions.cpu() if self.pred_actions is not None else None
        self.pred_values = self.pred_values.cpu() if self.pred_values is not None else None

    @property
    def n_steps_to_think(self):
        if self.store_path is None:
            warnings.warn("store_path is None, using default n_steps_to_think=0")
            return 0
        try:
            return int(self.store_path.parent.name.split("_")[0])
        except AttributeError:
            warnings.warn(f"Could not find n_steps_to_think in the store path: {self.store_path}")
            return 0

    @property
    def repeats_per_step(self):
        if self.pred_actions is None:
            warnings.warn("pred_actions is None, using default repeats_per_step=3")
            return 3
        repeats = len(self.pred_actions) / (len(self.obs) + self.n_steps_to_think)
        assert repeats.is_integer(), f"Repeats per step is not an integer: {repeats}"
        return int(repeats)

    @staticmethod
    def load(store_path: str):
        with open(store_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_from_play_output(play_output, batch_idx: int = 0):
        length = play_output.lengths[batch_idx]
        return DatasetStore(
            store_path=None,
            obs=play_output.obs[:length, batch_idx],
            rewards=play_output.rewards[:length, batch_idx],
            solved=play_output.solved[batch_idx],
            pred_actions=play_output.acts[:length, batch_idx],
            pred_values=None,
            model_cache={k: v[:length, batch_idx].numpy() for k, v in play_output.cache.items()},
        )

    def actual_steps(
        self,
        n_steps_to_think: Optional[int] = None,
        repeats_per_step: Optional[int] = None,
        include_initial_thinking: bool = False,
    ):
        assert self.pred_actions is not None, "pred_actions is None"
        n_steps_to_think = self.n_steps_to_think if n_steps_to_think is None else n_steps_to_think
        repeats_per_step = self.repeats_per_step if repeats_per_step is None else repeats_per_step
        skips = 0 if include_initial_thinking else n_steps_to_think
        start = (skips + 1) * repeats_per_step - 1
        return th.arange(start, self.pred_actions.shape[0], repeats_per_step)

    def get_actions(self, only_env_steps: bool = True, include_initial_thinking: bool = False) -> th.Tensor:
        assert self.pred_actions is not None, "pred_actions is None"
        if only_env_steps:
            return self.pred_actions[self.actual_steps(include_initial_thinking=include_initial_thinking)].squeeze(-1)
        elif include_initial_thinking:
            return self.pred_actions.squeeze(-1)
        else:
            return self.pred_actions[self.n_steps_to_think * self.repeats_per_step :].squeeze(-1)

    def get_values(self, only_env_steps: bool = True, include_initial_thinking: bool = False) -> th.Tensor:
        assert self.pred_values is not None, "pred_values is None"
        if only_env_steps:
            return self.pred_values[self.actual_steps(include_initial_thinking=include_initial_thinking)].squeeze(-1)
        elif include_initial_thinking:
            return self.pred_values.squeeze(-1)
        else:
            return self.pred_values[self.n_steps_to_think * self.repeats_per_step :].squeeze(-1)

    def get_cache(self, key: str, only_env_steps: bool = False, include_initial_thinking: bool = False):
        if only_env_steps:
            return self.model_cache[key][self.actual_steps(include_initial_thinking=include_initial_thinking)]
        elif include_initial_thinking:
            return self.model_cache[key]
        else:
            return self.model_cache[key][self.n_steps_to_think * self.repeats_per_step :]

    def get_true_values(self, gamma: float = 1.0) -> th.Tensor:
        assert self.rewards is not None, "rewards is None"
        # use gamma and self.rewards to multiply gamma^0, gamma^1, gamma^2, ... to rewards using torch function
        gammas = th.pow(gamma, th.arange(len(self.rewards), device=self.rewards.device))
        discounted_rewards = gammas * self.rewards
        cumsum = th.cumsum(discounted_rewards, dim=0)
        values = discounted_rewards - cumsum + cumsum[-1]
        values = values / gammas
        return values

    def get_success_repeated(self) -> th.Tensor:
        return th.tensor([self.solved] * len(self.obs), dtype=th.int)

    def get_boxing_indices(self) -> np.ndarray:
        assert self.rewards is not None, "rewards is None"
        self.rewards = self.rewards.to(th.float64)
        next_target_time = th.isclose(self.rewards, th.tensor(BOX_IN_REWARD, dtype=th.float64))
        next_target_time |= th.isclose(self.rewards, th.tensor(BOX_OUT_PENALTY, dtype=th.float64))
        next_target_time[-1] = th.isclose(self.rewards[-1], th.tensor(SOLVE_REWARD, dtype=th.float64))
        return th.where(next_target_time)[0].cpu().numpy()

    def is_wall(self, i: int, j: int) -> bool:
        if i < 0 or i >= self.obs[0].shape[1] or j < 0 or j >= self.obs[0].shape[2]:
            return True
        # If, in step 0, all RGB values are 0, this is a wall.
        return self.obs[0, :, i, j].eq(0).all().item()  # type: ignore

    def is_box(self, i: int, j: int) -> bool:
        # If, in step 0, all RGB values are 0, this is a wall.
        return self.obs[0, :, i, j].eq(th.tensor(BOX)).all().item()

    def is_next_to_a_wall(self, i: int, j: int, box_is_wall: bool = False) -> bool:
        is_wall = any(self.is_wall(x, y) for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)])
        if box_is_wall:
            is_wall = is_wall or any(self.is_box(x, y) for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)])
        return is_wall

    def get_wall_directions(self, i: int, j: int, box_is_wall: bool = False) -> np.ndarray:
        walls = np.array([self.is_wall(x, y) for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]])
        if box_is_wall:
            boxes = np.array([self.is_box(x, y) for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]])
            walls |= boxes
        return walls

    @staticmethod
    def map_idx_to_grid(step_wise_idxs: th.Tensor) -> th.Tensor:
        assert step_wise_idxs.ndim == 2 and step_wise_idxs.shape[1] == 2
        step_wise_grids = th.zeros((step_wise_idxs.shape[0], 10, 10), dtype=th.int64)
        step_wise_grids[th.arange(step_wise_idxs.shape[0]), step_wise_idxs[:, 0], step_wise_idxs[:, 1]] = 1
        return step_wise_grids

    def actions_for_probe(self, action_idx: int, grid_wise: bool = True) -> th.Tensor:
        gt = self.get_actions(only_env_steps=True, include_initial_thinking=False) == action_idx
        if grid_wise:
            return gt[:, None, None].repeat(1, 10, 10)
        return gt

    def next_target(self, map_to_grid: bool = True) -> th.Tensor:
        assert self.rewards is not None and self.pred_actions is not None, "rewards or pred_actions is None"
        next_target_timesteps = self.get_boxing_indices()
        target_positions = []
        repeats = []
        last_time = -1
        for t in next_target_timesteps:
            agent_pos = self.get_agent_position_per_step(self.obs[t]).cpu()
            action = int(self.pred_actions[self.to_hidden_idx(t)].item())
            # twice away from agent when putting box on target
            multiplier = 1 if th.isclose(self.rewards[t], th.tensor(BOX_OUT_PENALTY, dtype=th.float64)) else 2
            target_pos = agent_pos + multiplier * th.tensor(CHANGE_COORDINATES[action])
            target_positions.append(target_pos)
            repeats.append(t - last_time)
            last_time = t
        if len(target_positions) == 0:
            next_target_positions = th.zeros((0, 2), dtype=th.int64)
        else:
            next_target_positions = th.repeat_interleave(th.stack(target_positions), th.tensor(repeats), dim=0)
        if map_to_grid:
            next_target_positions = self.map_idx_to_grid(next_target_positions)
        return next_target_positions

    def next_box(self) -> th.Tensor:
        if not self.solved:
            return th.zeros((0, 10, 10), dtype=th.int64)
        target_positions = self.get_target_positions()
        all_steps_box_positions = self.get_box_positions()

        last_moving_box = self.different_positions(all_steps_box_positions[-1], target_positions)[0].pop()
        reversed_next_box_positions = [last_moving_box]
        for idx in range(len(all_steps_box_positions) - 2, -1, -1):
            moved_box = self.different_positions(all_steps_box_positions[idx], all_steps_box_positions[idx + 1])[0]
            if len(moved_box) == 1:
                last_moving_box = moved_box.pop()
            else:
                assert len(moved_box) == 0, f"Expected 0, got {len(moved_box)}"
            reversed_next_box_positions.append(last_moving_box)
        next_box_positions = th.tensor(reversed_next_box_positions[::-1])
        next_box_positions = self.map_idx_to_grid(next_box_positions)
        return next_box_positions

    def get_agent_positions(self, return_map: bool = False) -> th.Tensor:
        agent_positions = []
        for obs in self.obs:
            agent_positions.append(self.get_agent_position_per_step(obs, return_map=return_map).cpu())
        return th.stack(agent_positions)

    def get_agent_position_per_step(self, obs, return_map: bool = False) -> th.Tensor:
        assert obs.shape == (3, 10, 10)
        agent_pos = ((obs[0] == 160) | (obs[0] == 219)) & (obs[1] == 212) & (obs[2] == 56)
        if return_map:
            return agent_pos
        agent_pos = th.where(agent_pos)
        return th.stack(agent_pos).squeeze()

    def agents_future_position_map(self, include_current_position: bool = False) -> th.Tensor:
        agent_positions = self.get_agent_positions()
        future_positions_map = th.zeros((len(agent_positions), 10, 10), dtype=th.int64)
        for i, pos in enumerate(agent_positions):
            upto_idx = i + 1 if include_current_position else i
            future_positions_map[:upto_idx, pos[0], pos[1]] = 1
        return future_positions_map

    def agents_future_direction_map(
        self,
        move_out: bool = True,
        include_current_position: bool = True,
        multioutput: bool = True,
        variable_boxes: bool = False,
        return_timestep_map: bool = False,
        future_n_steps: Optional[int] = None,
    ) -> th.Tensor:
        agent_positions = self.get_agent_positions()
        init_val, shape, dtype = -1, [len(agent_positions), 10, 10], th.int32
        if multioutput:
            shape.append(4)
            init_val, dtype = False, th.bool
        future_direction_map = init_val * th.ones(shape, dtype=dtype)
        future_direction_timestep_map = -1 * th.ones((len(agent_positions), 10, 10), dtype=th.int32)
        inv_change_coordinates = {tuple(v): k for k, v in CHANGE_COORDINATES.items()}
        correct_indices = range(len(agent_positions) - 1) if move_out else range(1, len(agent_positions))
        start_idx_for_pos = th.zeros((10, 10), dtype=th.int64)
        upto_idx = 0
        for i in correct_indices:
            pos = agent_positions[i]
            upto_idx = i + 1 if include_current_position else i
            if move_out:
                change = agent_positions[i + 1] - pos
            else:
                change = pos - agent_positions[i - 1]
            if th.all(change == 0):
                continue

            if multioutput:
                from_idx = 0 if future_n_steps is None else max(0, upto_idx - future_n_steps)
                future_direction_map[from_idx:upto_idx, *pos, inv_change_coordinates[*change.tolist()]] = True
            else:
                from_idx = start_idx_for_pos[*pos].item()
                from_idx = from_idx if future_n_steps is None else max(0, from_idx, upto_idx - future_n_steps)
                future_direction_map[from_idx:upto_idx, *pos] = inv_change_coordinates[*change.tolist()]
                start_idx_for_pos[*pos] = upto_idx
            future_direction_timestep_map[from_idx:upto_idx, *pos] = i
        if self.solved:
            last_agent_pos = agent_positions[-1]
            target_positions = self.get_target_positions(variables_boxes=variable_boxes)
            last_box_positions = self.get_box_positions(idx=-1, variable_boxes=variable_boxes)
            found = False
            for dir_idx, direction in enumerate(CHANGE_COORDINATES.values()):
                direction = th.tensor(direction)
                final_agent_pos = last_agent_pos + direction
                final_target_pos = final_agent_pos + direction
                found_target = (final_target_pos == target_positions).all(dim=1).any()
                found_box = (final_agent_pos == last_box_positions).all(dim=1).any()
                if found_target and found_box:
                    found = True
                    break
            assert found, "Could not find target position for agent on the final step"
            pos = last_agent_pos if move_out else final_target_pos  # type: ignore
            from_idx = start_idx_for_pos[*pos].item()
            upto_idx += 1
            if multioutput:
                from_idx = 0 if future_n_steps is None else max(0, upto_idx - future_n_steps)
                future_direction_map[from_idx:, *pos, dir_idx] = True  # type: ignore
            else:
                from_idx = from_idx if future_n_steps is None else max(0, from_idx, upto_idx - future_n_steps)
                future_direction_map[from_idx:, *pos] = dir_idx  # type: ignore
            future_direction_timestep_map[from_idx:, *pos] = len(agent_positions) - 1
        if return_timestep_map:
            return future_direction_map, future_direction_timestep_map
        return future_direction_map

    def get_box_positions(
        self,
        idx: Optional[int] = None,
        variable_boxes: bool = False,
        return_map: bool = False,
        only_solved: bool = False,
        only_unsolved: bool = False,
    ) -> th.Tensor:
        if idx is not None:
            return self.get_box_position_per_step(self.obs[idx], variable_boxes, return_map, only_solved, only_unsolved).cpu()  # type: ignore
        box_positions = []
        for obs in self.obs:
            bpps = self.get_box_position_per_step(obs, variable_boxes, return_map, only_solved, only_unsolved).cpu()  # type: ignore
            box_positions.append(bpps)
        return th.stack(box_positions)

    @staticmethod
    def get_target_positions_from_obs(
        obs,
        variables_boxes: bool = False,
        return_map: bool = False,
        only_solved: bool = False,
        only_unsolved: bool = False,
    ) -> th.Tensor:
        assert obs.shape[-3] == 3, f"Expected (..., 3, 10, 10), got {obs.shape}"
        player_on_target = (obs == PLAYER_ON_TARGET[:, None, None]).all(dim=-3)
        unsolved_target_positions = (obs == TARGET[:, None, None]).all(dim=-3) | player_on_target
        solved_target_positions = (obs == BOX_ON_TARGET[:, None, None]).all(dim=-3)
        ret = (
            solved_target_positions
            if only_solved
            else unsolved_target_positions
            if only_unsolved
            else unsolved_target_positions | solved_target_positions
        )
        if return_map:
            return ret
        if len(obs.shape) > 3:
            raise NotImplementedError("Not implemented for batched observations")
        target_positions = th.where(ret)

        positions_list = [tuple(pos.tolist()) for pos in th.stack(target_positions).T]
        if not variables_boxes:
            assert len(positions_list) == NUM_BOXES, f"Expected {NUM_BOXES} targets, got {len(positions_list)}"
        unique_positions = set(positions_list)
        if not variables_boxes:
            assert (
                len(unique_positions) == NUM_BOXES
            ), f"Expected {NUM_BOXES} unique target positions, but found {len(unique_positions)}"
        return th.stack(target_positions).T  # type: ignore

    @staticmethod
    def get_floor_positions_from_obs(obs, return_map: bool = False) -> th.Tensor:
        assert obs.shape[-3] == 3, f"Expected (..., 3, 10, 10), got {obs.shape}"
        floor_positions = (obs == FLOOR[:, None, None]).all(dim=-3)
        if return_map:
            return floor_positions
        if len(obs.shape) > 3:
            raise NotImplementedError("Not implemented for batched observations")
        floor_positions = th.where(floor_positions)
        return th.stack(floor_positions).T  # type: ignore

    @staticmethod
    def get_box_position_per_step(
        obs: th.Tensor | np.ndarray,
        variable_boxes=False,
        return_map=False,
        only_solved=False,
        only_unsolved=False,
    ) -> th.Tensor | np.ndarray:
        # assert obs.shape == (3, 10, 10), f"Expected (3, 10, 10), got {obs.shape}"
        th_or_np = th if isinstance(obs, th.Tensor) else np
        if only_solved:
            box_pos_map = (obs == BOX_ON_TARGET[:, None, None]).all(0)
        elif only_unsolved:
            box_pos_map = (obs == BOX[:, None, None]).all(0)
        else:
            solved = (obs == BOX_ON_TARGET[:, None, None]).all(0)
            unsolved = (obs == BOX[:, None, None]).all(0)
            box_pos_map = solved | unsolved
        if return_map:
            return box_pos_map
        box_pos = th_or_np.where(box_pos_map)
        if not (only_solved or only_unsolved or variable_boxes):
            assert len(box_pos[0]) == NUM_BOXES, f"Expected {NUM_BOXES} boxes, got {len(box_pos[0])}"
        return th_or_np.stack(box_pos).T  # type: ignore

    def get_target_positions(self, obs=None, variables_boxes: bool = False, return_map: bool = False) -> th.Tensor:
        if obs is None:
            obs = self.obs[0]
        return self.get_target_positions_from_obs(obs, variables_boxes, return_map)

    def get_floor_positions(self, obs=None, return_map: bool = False) -> th.Tensor:
        if obs is None:
            obs = self.obs[0]
        return self.get_floor_positions_from_obs(obs, return_map)

    def target_labels_map(self, obs=None, incremental_labels=True) -> th.Tensor:
        """Return a map of each target positions. Map values are used to title the targets as T0 to T3."""
        target_positions = self.get_target_positions(obs)

        label_map = EMPTY_SQUARE * th.ones((10, 10), dtype=th.int64)
        # Set the 4 squares where the targets are to 0, 1, 2, 3
        labels = th.arange(NUM_BOXES) if incremental_labels else th.ones(NUM_BOXES, dtype=th.int64)
        label_map[target_positions[:, 0], target_positions[:, 1]] = labels

        unique_locations = th.nonzero(label_map != EMPTY_SQUARE, as_tuple=False)
        assert (
            unique_locations.shape[0] == NUM_BOXES
        ), f"Expected {NUM_BOXES} unique target locations, but found {unique_locations.shape[0]}"

        return label_map

    def boxes_future_position_map(self, include_current_position: bool = False) -> th.Tensor:
        box_positions = self.get_box_positions()
        future_positions_map = th.zeros((len(box_positions), 10, 10), dtype=th.int64)
        for i, pos in enumerate(box_positions):
            upto_idx = i + 1 if include_current_position else i
            future_positions_map[:upto_idx, pos[:, 0], pos[:, 1]] = 1
        return future_positions_map

    @staticmethod
    def different_positions(pos1: th.Tensor, pos2: th.Tensor, successive_positions: bool = False) -> Tuple[set, set]:
        """Return the positions that are unique to each of the position sets"""
        set1 = set(tuple(x.tolist()) for x in pos1)
        set2 = set(tuple(x.tolist()) for x in pos2)
        change1 = set1 - set2
        change2 = set2 - set1

        if successive_positions:
            """ If params are from successive steps, then at most one box position changes position by at most one square """
            assert len(change1) == len(change2), f"{len(change1)} != {len(change2)} in different_positions"
            assert len(change1) <= 1, f"Expected 0/1 change, got {len(change1)} changes in different_positions"

        return change1, change2

    def boxes_future_direction_map(
        self,
        move_out: bool = True,
        include_current_position: bool = True,
        multioutput: bool = True,
        variable_boxes: bool = False,
        return_timestep_map: bool = False,
        future_n_steps: Optional[int] = None,
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        box_positions = self.get_box_positions(variable_boxes=variable_boxes)
        init_val, shape, dtype = -1, [len(box_positions), 10, 10], th.int32
        if multioutput:
            shape.append(4)
            init_val, dtype = False, th.bool
        future_direction_map = init_val * th.ones(shape, dtype=dtype)
        future_direction_timestep_map = -1 * th.ones((len(box_positions), 10, 10), dtype=th.int32)
        inv_change_coordinates = {tuple(v): k for k, v in CHANGE_COORDINATES.items()}
        correct_indices = range(len(box_positions) - 1) if move_out else range(1, len(box_positions))
        start_idx_for_pos = th.zeros((10, 10), dtype=th.int64)
        for i in correct_indices:
            pos = box_positions[i]
            upto_idx = i + 1 if include_current_position else i
            if move_out:
                change_pos, change_pos_rev = self.different_positions(pos, box_positions[i + 1], True)
            else:
                change_pos, change_pos_rev = self.different_positions(pos, box_positions[i - 1], True)
            if len(change_pos) == 0:
                continue
            else:
                pos = change_pos.pop()
                change = th.tensor(pos) - th.tensor(change_pos_rev.pop())
                change = -change if move_out else change

            if multioutput:
                from_idx = 0 if future_n_steps is None else max(0, upto_idx - future_n_steps)
                future_direction_map[from_idx:upto_idx, *pos, inv_change_coordinates[*change.tolist()]] = True
            else:
                from_idx = start_idx_for_pos[*pos].item()
                from_idx = from_idx if future_n_steps is None else max(0, from_idx, upto_idx - future_n_steps)
                future_direction_map[from_idx:upto_idx, *pos] = inv_change_coordinates[*change.tolist()]
                start_idx_for_pos[*pos] = upto_idx
            future_direction_timestep_map[from_idx:upto_idx, *pos] = i

        if self.solved:
            final_target, final_box = self.different_positions(
                self.get_target_positions(variables_boxes=variable_boxes),
                box_positions[-1],
            )
            assert len(final_target) == len(final_box) == 1, f"{len(final_target)} != {len(final_box)} != 1"
            final_target = th.tensor(list(final_target)[0])
            final_box = th.tensor(list(final_box)[0])
            change = final_target - final_box
            pos = final_box if move_out else final_target
            from_idx = start_idx_for_pos[*pos].item()
            upto_idx = len(box_positions)
            if multioutput:
                from_idx = 0 if future_n_steps is None else max(0, upto_idx - future_n_steps)
                future_direction_map[from_idx:, *pos, inv_change_coordinates[*change.tolist()]] = True
            else:
                from_idx = from_idx if future_n_steps is None else max(0, from_idx, upto_idx - future_n_steps)
                future_direction_map[from_idx:, *pos] = inv_change_coordinates[*change.tolist()]
            future_direction_timestep_map[from_idx:, *pos] = len(box_positions) - 1

        if return_timestep_map:
            return future_direction_map, future_direction_timestep_map
        return future_direction_map

    def boxes_label_map(self) -> th.Tensor:
        """Return a map of each box position over steps. Map values are used to title the boxes as B0 to B3."""
        """The map values are 0 to NUM_BOXES-1 and are maintained as the boxes move. A map value of -1 is ignored."""
        box_positions = self.get_box_positions()

        label_map = EMPTY_SQUARE * th.ones((len(box_positions), 10, 10), dtype=th.int64)

        # Randomly assign box labels 0, 1, 2, 3 in step 0
        pos = box_positions[0]
        label_map[0, pos[:, 0], pos[:, 1]] = th.arange(NUM_BOXES)

        for i in range(len(box_positions) - 1):
            pos = box_positions[i]
            assert len(pos) == NUM_BOXES, f"{len(pos)} != {NUM_BOXES} in boxes_label_map"

            label_map[i + 1] = label_map[i].clone()

            # Will a box change position in this step?
            curr_box_pos, next_box_pos = self.different_positions(pos, box_positions[i + 1], True)
            if len(curr_box_pos) == 1:
                curr_pos = curr_box_pos.pop()
                next_pos = next_box_pos.pop()

                box_title = label_map[i, curr_pos[0], curr_pos[1]]
                assert box_title != EMPTY_SQUARE, f"Expected box title in step {i-1} location {curr_pos[0]}, {curr_pos[1]}"

                # Move the box title from the previous position to the new position
                label_map[i + 1, curr_pos[0], curr_pos[1]] = EMPTY_SQUARE
                label_map[i + 1, next_pos[0], next_pos[1]] = box_title

        return label_map

    def agent_in_a_cycle(self) -> th.Tensor:
        try:
            last_box_time_step = self.get_boxing_indices()[-1]
        except IndexError:
            warnings.warn(f"No box was put on target in the level: {str(self.store_path)}")
            last_box_time_step = len(self.obs)
        all_obs = self.obs[:last_box_time_step]
        all_obs = all_obs.reshape(all_obs.shape[0], 1, *all_obs.shape[1:]).numpy()
        obs_repeat = np.all(all_obs == all_obs.transpose(1, 0, 2, 3, 4), axis=(2, 3, 4))
        np.fill_diagonal(obs_repeat, False)
        obs_repeat = [np.where(obs_repeat[j])[0] for j in range(len(obs_repeat))]
        is_a_cycle = th.zeros(len(self.obs), dtype=th.bool)

        i = 0
        while i < len(obs_repeat):
            if obs_repeat[i].size > 0 and obs_repeat[i][-1] - i > 0:
                is_a_cycle[i : obs_repeat[i][-1] + 1] = True  # include the last step as part of the cycle
                i = obs_repeat[i][-1]
            i += 1
        return is_a_cycle

    def to_hidden_idx(self, idx: int) -> int:
        return (idx + self.n_steps_to_think + 1) * self.repeats_per_step - 1

    def alternative_boxes_future_direction_map(self):
        # these tensors are booleans with dimensions (directions, N, box, target, y, x)
        directions_bw_false = alternative_plans.compute_directions(self, boxes_are_walls=False)
        directions_bw_true = alternative_plans.compute_directions(self, boxes_are_walls=False)

        box_target = (2, 3)
        plan = directions_bw_false.any(box_target) | directions_bw_true.any(box_target)

        # make it (N, y, x, direction)
        return plan.moveaxis(0, -1)


@dataclasses.dataclass
class HelperFeatures:
    ds: Optional[DatasetStore]
    agent: np.ndarray  # (seq_len, y, x)
    floor: np.ndarray
    unsolved_boxes: np.ndarray
    solved_boxes: np.ndarray
    unsolved_targets: np.ndarray
    box_directions: Optional[np.ndarray] = None  # action along last dim
    agent_directions: Optional[np.ndarray] = None  # action along last dim
    actions: Optional[np.ndarray] = None  # action along last dim

    def to_nparray(self, bias=False) -> np.ndarray:
        base = [np.ones_like(self.agent)] if bias else []
        base = np.stack(
            base + [self.agent, self.floor, self.unsolved_boxes, self.solved_boxes, self.unsolved_targets], axis=-1
        )
        if self.box_directions is not None:
            base = np.concatenate([base, self.box_directions, self.agent_directions, self.actions], axis=-1)  # type: ignore
        return base

    def to_tensor(self, bias=False) -> th.Tensor:
        return th.tensor(self.to_nparray(bias))

    def myopic(self, feature_name="box_directions", future_n_steps=1):
        assert self.ds is not None
        if feature_name == "box_directions":
            ret = self.ds.boxes_future_direction_map(multioutput=True, future_n_steps=future_n_steps, variable_boxes=True)
            assert isinstance(ret, th.Tensor)
            return ret.float().numpy()
        elif feature_name == "agent_directions":
            return (
                self.ds.agents_future_direction_map(multioutput=True, future_n_steps=future_n_steps, variable_boxes=True)
                .float()
                .numpy()
            )
        else:
            raise ValueError("Invalid feature_name")

    @staticmethod
    def from_ds(ds: DatasetStore, use_future_features: bool = True) -> "HelperFeatures":
        agent_pos = ds.get_agent_positions(return_map=True)
        floor_pos = ds.get_floor_positions(return_map=True)
        unsolved_box_pos = ds.get_box_positions(return_map=True, only_unsolved=True)
        solved_box_pos = ds.get_box_positions(return_map=True, only_solved=True)

        unsolved_target_pos = ds.get_target_positions(return_map=True)
        seq_len = agent_pos.shape[0]
        features = HelperFeatures(
            ds,
            agent_pos.float().numpy(),
            floor_pos[None].repeat(seq_len, 1, 1).float().numpy(),
            unsolved_box_pos.float().numpy(),
            solved_box_pos.float().numpy(),
            unsolved_target_pos[None].repeat(seq_len, 1, 1).float().numpy(),
        )

        if use_future_features:
            box_directions = ds.boxes_future_direction_map(multioutput=True, variable_boxes=True)
            assert isinstance(box_directions, th.Tensor)
            agent_directions = ds.agents_future_direction_map(multioutput=True, variable_boxes=True)

            action_pred = ds.get_actions(only_env_steps=True)
            action_pred = th.nn.functional.one_hot(action_pred, num_classes=4)
            action_pred = action_pred[:, None, None, :].repeat(1, 10, 10, 1).float().numpy()

            features.box_directions = box_directions.float().numpy()
            features.agent_directions = agent_directions.float().numpy()
            features.actions = action_pred

        return features

    @staticmethod
    def from_play_output(play_output, batch_idx: int = 0, use_future_features: bool = True) -> "HelperFeatures":
        ds = DatasetStore.load_from_play_output(play_output, batch_idx)
        return HelperFeatures.from_ds(ds, use_future_features)

    @staticmethod
    def feature_from_obs(obs) -> th.Tensor:
        agent = th.all(obs == PLAYER[:, None, None], dim=-3)
        agent_on_target = th.all(obs == (PLAYER_ON_TARGET)[:, None, None], dim=-3)
        agent = agent | agent_on_target

        floor = th.all(obs == FLOOR[:, None, None], dim=-3)

        unsolved_boxes = th.all(obs == BOX[:, None, None], dim=-3)
        solved_boxes = th.all(obs == BOX_ON_TARGET[:, None, None], dim=-3)

        unsolved_targets = th.all(obs == TARGET[:, None, None], dim=-3)
        unsolved_targets = unsolved_targets | agent_on_target

        return th.stack([agent, floor, unsolved_boxes, solved_boxes, unsolved_targets], dim=-1).float()


@dataclasses.dataclass
class ChannelCoefs:
    name: str
    shift_fn: str
    model: Any
    resid: float
    corr: float


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
    names_filter: Optional[List[str]] = None,
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
            names_filter=names_filter,
            feature_extractor_kwargs={"return_repeats": True},
        )
        actions = actions.unsqueeze(-1)
        # remove hook_pre_model as it doesn't while thinking for N steps on the same observation
        cache.pop("features_extractor.hook_pre_model", None)
        all_actions[:, step_i * repeats_per_step : (step_i + 1) * repeats_per_step] = actions.transpose(0, 1)
        all_values[:, step_i * repeats_per_step : (step_i + 1) * repeats_per_step] = values.transpose(0, 1)
        all_cache[:, step_i] = split_cache(cache, num_reset_envs)

        if "hook_action_net" in cache:
            assert (
                th.take_along_dim(th.log_softmax(cache["hook_action_net"], dim=-1), indices=actions, dim=-1)
                .squeeze(-1)
                .allclose(log_probs, atol=1e-4)
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
    solve_reward: float = SOLVE_REWARD,
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

    names_filter = None
    if args.cache_keys:
        names_filter = []
        for name in args.cache_keys.split(","):
            name = name.strip()
            if "cell_list.*" in name:
                for layer in range(len(model.features_extractor.cell_list)):
                    names_filter += [
                        (name.replace("*", str(layer)) + f".{0}.{int_pos}") for int_pos in range(repeats_per_step)
                    ]
            else:
                names_filter.append(name)
        print("Filtering cache keys:", names_filter)

    episodes_solved = 0
    save_dir = pathlib.Path(args.output_path) / f"{n_steps_to_think}_think_step"
    save_dir.mkdir(exist_ok=True, parents=True)

    with th.no_grad(), tqdm(total=n_eval_episodes) as pbar, cf.ThreadPoolExecutor(max_workers=32) as executor:
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
                    names_filter=names_filter,
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
                names_filter=names_filter,
                feature_extractor_kwargs={"return_repeats": True},
            )
            states = tree_map(th.clone, states, namespace=type_aliases.SB3_TREE_NAMESPACE, none_is_leaf=False)  # type: ignore

            new_observations, rewards, dones, infos = env.step(acts[-1, :])  # indexing due to return_repeats=True
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
                ds = DatasetStore(
                    store_path=save_dir / f"idx_{num_finished_episodes}.pkl",
                    obs=all_obs[i][: idx_in_eps[i]],
                    rewards=all_rewards[i][: idx_in_eps[i]],
                    solved=episode_solved,
                    pred_actions=all_pred_actions[i][: idx_in_eps_with_repeats[i]],
                    pred_values=all_pred_values[i][: idx_in_eps_with_repeats[i]],
                    model_cache=join_cache_across_steps(all_model_cache[i][: n_steps_to_think + idx_in_eps[i]]),
                    file_idx=all_level_infos[i][0],
                    level_idx=all_level_infos[i][1],
                )
                executor.submit(ds.save)

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
        n_steps_to_think = [int(x) for x in args.n_steps_to_think.split(",")] if args.n_steps_to_think else [0]
        n_envs = 128
        device = th.device(args.device)
    print("Device:", device)

    split = None if args.split == "none" or not args.split else args.split
    for i, steps_to_think in enumerate(n_steps_to_think):
        env_cfg, eval_env = create_eval_env(
            split=split,
            difficulty=args.difficulty,
            max_episode_steps=max_episode_steps,
            n_envs=n_envs,
            device=device,
            BOXOBAN_CACHE=pathlib.Path(args.boxoban_cache),
            envpool=args.envpool,
        )
        cfg, policy = load_jax_model_to_torch(model_path, env_cfg)
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
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="drc33/bkynosqi/cp_2002944000",
        help="local path or relative huggingface path",
    )
    parser.add_argument("-t", "--test", action="store_true", help="Enable test mode")
    parser.add_argument("-e", "--envpool", action="store_true", help="Use EnvpoolSokobanVecEnv")
    parser.add_argument("-c", "--boxoban_cache", type=str, default="/training/.sokoban_cache/")
    parser.add_argument("-d", "--device", type=str, default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("-s", "--split", type=str, default="valid")
    parser.add_argument("-l", "--difficulty", type=str, default="medium")
    parser.add_argument("-n", "--n_eval_episodes", type=int, default=5000)
    parser.add_argument("-o", "--output_path", type=str, default=".")
    parser.add_argument(
        "-k",
        "--cache_keys",
        type=str,
        default=None,
        help="Comma separated full keys to cache (shouldn't contain position or repeat). Replace layer idx with * to cache all layers.",
    )
    parser.add_argument("--n_steps_to_think", type=str, default="0", help="Comma separated list of steps to think")
    args = parser.parse_args()

    TEST = args.test
    model_path = download_policy_from_huggingface(args.model_path)

    collect_dataset(model_path, args)
