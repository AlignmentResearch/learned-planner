import dataclasses
import re
from functools import partial
from glob import glob
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch as th
import wandb
from stable_baselines3.common.pytree_dataclass import tree_index, tree_map
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.util import obs_as_tensor
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from learned_planner import __main__ as lp_main
from learned_planner.environments import BoxobanConfig, EnvConfig
from learned_planner.interp.collect_dataset import DatasetStore
from learned_planner.interp.utils import load_jax_model_to_torch
from learned_planner.policies import ConvLSTMOptions
from learned_planner.train import ABCCommandConfig


def set_seed(seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


class ActivationsDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        labels_type: str = "pred_value",
        keys: list[str] = [".*hook_h$", ".*hook_c$"],
        num_data_points: Optional[int] = None,
        fetch_all_boxing_data_points: bool = False,
        gamma_value: float = 0.99,
    ):
        """Create a dataset of model activations to train probes on.

        Args:
            dataset_path (Path): Path to the dataset containing cache files of `DataStore` class.
            labels_type (str, optional): Type of labels to use for training. Can be one of
                ["pred_value", "true_value", "reward", "next_box_pos", "next_target_pos"].
                Defaults to "pred_value".
            keys (list, optional): List of regex patterns to match the keys in the cache.
                Defaults to [".*hook_h$", ".*hook_c$"].
            num_data_points (int, optional): Number of data points to keep in the dataset. Not strictly
                followed when `fetch_all_boxing_data_points = True`. Defaults to None which keeps all datapoints.
            fetch_all_boxing_data_points (bool, optional): If True, all datapoints where box is pushed or removed from a target
                are kept in the dataset. Defaults to False.

        Raises:
            ValueError: If `labels_type` is not one of ["pred_value", "true_value", "reward", "next_box_pos", "next_target_pos"].
        """
        super().__init__()
        self.dataset_path = dataset_path
        if labels_type not in [
            "pred_value",
            "true_value",
            "reward",
            "next_box_pos",
            "next_target_pos",
            "success",
        ]:
            raise ValueError(f"Unknown labels type {labels_type}")
        self.labels_type = labels_type
        self.level_files = glob(str(dataset_path / "*.pkl"))
        np.random.shuffle(self.level_files)

        self.n_layers = 3
        self.n_features = 32
        self.obs_size = 10 * 10
        self.keys = self.get_full_keys(keys)
        self.num_data_points = num_data_points
        self.fetch_all_boxing_data_points = fetch_all_boxing_data_points
        self.gamma_value = gamma_value

        self.load_data(num_data_points)

    def get_full_keys(self, keys):
        ds_cache = DatasetStore.load(self.level_files[0])
        full_keys = [fk for fk in ds_cache.model_cache.keys() if any(re.match(k, fk) for k in keys)]
        assert len(full_keys) == len(keys) * self.n_layers
        return full_keys

    def load_file(self, file_name, num_data_points: Optional[int] = None):
        ds_cache = DatasetStore.load(file_name)
        all_cache_values = [ds_cache.get_cache(k, only_env_steps=True, include_initial_thinking=True) for k in self.keys]
        self.classification = False
        if self.labels_type == "pred_value":
            gt_output = ds_cache.get_values(only_env_steps=True, include_initial_thinking=True)
        elif self.labels_type == "true_value":
            gt_output = ds_cache.get_true_values(gamma=self.gamma_value)
        elif self.labels_type == "reward":
            gt_output = ds_cache.rewards
        elif self.labels_type == "next_box_pos":
            gt_output = ds_cache.get_next_box_positions()
        elif self.labels_type == "next_target_pos":
            gt_output = ds_cache.get_next_target_positions()
        elif self.labels_type == "success":
            gt_output = ds_cache.get_success_repeated()
            self.classification = True
        else:
            raise ValueError(f"Unknown labels type {self.labels_type}")

        data = []
        gt_output_data = []
        num_data_points = num_data_points or len(gt_output)
        indices = np.random.permutation(len(gt_output))[:num_data_points]
        if self.fetch_all_boxing_data_points:
            boxing_indices = ds_cache.get_boxing_indices()
            boxing_indices = np.setdiff1d(boxing_indices, indices)
            indices = np.concatenate([indices, boxing_indices])

        for v_idx in indices:
            cache_data = th.zeros((self.num_features,))
            for comp_idx in range(len(all_cache_values)):
                v = all_cache_values[comp_idx][v_idx]
                cache_data[comp_idx * v.size : (comp_idx + 1) * v.size] = th.tensor(v.flatten())
            data.append(cache_data)
            gt_output_data.append(gt_output[v_idx].item() if gt_output[v_idx].size == 1 else gt_output[v_idx])
        boxing_indices_bool = np.zeros(len(indices), dtype=bool)
        if self.fetch_all_boxing_data_points:
            boxing_indices_bool[-len(boxing_indices) :] = True  # type: ignore
        return data, gt_output_data, boxing_indices_bool

    def load_data(self, num_data_points: Optional[int] = None):
        if num_data_points is not None:
            num_data_points_per_file = int(np.ceil(num_data_points / len(self.level_files)))
            if len(self.level_files) > num_data_points:
                num_files = num_data_points
                self.level_files = np.random.choice(self.level_files, num_files, replace=False)
        else:
            num_data_points_per_file = None

        results = list(map(partial(self.load_file, num_data_points=num_data_points_per_file), self.level_files))
        self.data = [item for result in results for item in result[0]]
        self.gt_output = [item for result in results for item in result[1]]
        self.boxing_indices = [item for result in results for item in result[2]]
        assert (
            len(self.data) == len(self.gt_output) == len(self.boxing_indices)
        ), f"{len(self.data)=}, {len(self.gt_output)=}, {len(self.boxing_indices)=}"

        if not (num_data_points is None or self.fetch_all_boxing_data_points):
            self.data = self.data[:num_data_points]
            self.gt_output = self.gt_output[:num_data_points]
            self.boxing_indices = self.boxing_indices[:num_data_points]

        self.data = th.stack(self.data)
        self.gt_output = (
            th.tensor(self.gt_output) if not isinstance(self.gt_output[0], th.Tensor) else th.stack(self.gt_output)
        )
        self.boxing_indices = th.tensor(self.boxing_indices)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.gt_output[idx]

    def get_labels(self):
        return self.gt_output

    def get_data(self):
        return self.data

    def avg_eps_size(self):
        return len(self.data) / len(self.level_files)

    @property
    def num_features(self):
        return len(self.keys) * self.n_features * self.obs_size


class LinearProbeModel(th.nn.Module):
    def __init__(self, input_dim, output_dim, classification=False):
        super().__init__()
        self.model = th.nn.Sequential(th.nn.Linear(input_dim, output_dim))
        self.classification = classification
        if classification:
            if output_dim == 1:
                self.model.add_module("sigmoid", th.nn.Sigmoid())

    def forward(self, x):
        out = self.model(x)
        return out

    def predict(self, next_obs):
        """Predict the action index given a batch of all the next observations.

        Args:
            next_obs (th.Tensor): Batch of all the next observations. (batch_size, num_features)

        Returns:
            int: Predicted action index.
        """
        assert len(next_obs.shape) == 2, f"{next_obs.shape=}"
        values = self(next_obs)

        return values.argmax().item()

    @property
    def device(self):
        return next(self.parameters()).device


@dataclasses.dataclass
class TrainProbeConfig(ABCCommandConfig):
    """Arguments to train or evaluate probes (and their corresponding policies)."""

    dataset_path: Path = Path("asdf")
    learning_rate: float = 1e-4
    epochs: int = 10
    batch_size: int = 512
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    weight_decay_norm: int = 1
    dataset_size: Optional[int] = None
    wandb_name: Optional[str] = None
    policy_path: Optional[Path] = None
    eval_epoch_interval: int = 10
    eval_only: bool = False
    eval_cache_path: Path = Path("/training/.sokoban_cache")
    eval_split: Literal["train", "valid", "test"] = "valid"
    eval_difficulty: Literal["unfiltered", "medium", "hard"] = "unfiltered"
    eval_episodes: int = 20
    eval_type: str = "probe"
    eval_model: Optional[str] = None

    def __post_init__(self):
        assert self.dataset_path, "Dataset path must be provided"
        if self.eval_type not in ["probe", "policy_action", "policy_value"]:
            raise ValueError(f"Unknown eval type {self.eval_type}")

    def run(self, run_dir: Path):  # type: ignore
        return main(self)


def train(args: TrainProbeConfig, eval_cfg: EnvConfig, eval_env):
    device = args.th_device
    if args.policy_path:
        policy_cfg, policy = load_jax_model_to_torch(Path(args.policy_path), eval_env)
        policy.to(device)
        policy.eval()
    else:
        policy = None
    dataset_path = Path(args.dataset_path)
    if dataset_path.is_dir():
        acts_ds = ActivationsDataset(dataset_path, num_data_points=args.dataset_size)
    else:
        acts_ds = th.load(dataset_path)
    input_dim, output_dim = acts_ds.num_features, 1
    train_ds, test_ds = random_split(acts_ds, [0.8, 0.2])
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    classification = acts_ds.classification

    model = LinearProbeModel(input_dim, output_dim, classification=classification)
    model.to(device)

    if classification:
        if output_dim == 1:
            criterion = th.nn.BCELoss()
        else:
            criterion = th.nn.CrossEntropyLoss()
    else:
        criterion = th.nn.MSELoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss, avg_test_loss = th.tensor(0.0, requires_grad=True).to(device), 0.0
    for epoch in tqdm(range(args.epochs)):
        model.train()
        for i, (inputs, labels) in enumerate(train_dl):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            reg = th.tensor(0.0, requires_grad=True).to(device)
            for name, param in model.named_parameters():
                if "weight" in name:
                    reg = reg + th.norm(param, args.weight_decay_norm)
            loss_w_reg = loss + args.weight_decay * reg
            loss_w_reg.backward()

            th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            wandb.log({"train/loss": loss.item(), "train/reg": reg.item()})

        if epoch % args.eval_epoch_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            with th.no_grad():
                total_test_loss = 0
                for inputs, labels in test_dl:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    test_loss = criterion(outputs, labels.unsqueeze(1))
                    total_test_loss += test_loss.item()

                avg_test_loss = total_test_loss / len(test_dl)
                active_features = (model.model[0].weight.abs() < 1e-4).sum().item() * 100 / (input_dim * output_dim)
                log_dict = {"test/loss": avg_test_loss, "test/active_features": active_features}
                if policy and acts_ds.labels_type in ["pred_value", "true_value"]:
                    success_rate, max_value_match = evaluate(
                        model,
                        policy,
                        acts_ds.keys,
                        policy_cfg.features_extractor.repeats_per_step,  # type: ignore
                        eval_cfg,
                        eval_env,
                        args.eval_episodes,
                        args.eval_type,
                    )
                    log_dict["test/success_rate"] = success_rate
                    log_dict["test/max_value_match"] = max_value_match
                wandb.log(log_dict)
    wandb.finish()
    return model, {"train/loss": loss.item(), "test/loss": avg_test_loss}


def evaluate(
    probe_model,
    policy,
    cache_keys,
    repeats_per_step,
    eval_cfg,
    eval_env,
    eval_episodes=20,
    eval_type="probe",
):
    final_reward, reward = eval_cfg.reward_finished + eval_cfg.reward_box + eval_cfg.reward_step, 0.0
    num_successes = 0
    probe_matches_with_value_head = 0
    num_steps = 0
    for i in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        state = policy.recurrent_initial_state(n_envs=1, device=policy.device)
        ep_len = 0
        while not done:
            obs = obs_as_tensor(obs, probe_model.device)
            if eval_type == "policy_action":
                pred_acts, _, _, state = policy(obs, state, th.tensor([ep_len == 0], device=policy.device), deterministic=True)
                action = pred_acts.item()
            else:
                if ep_len == 0:
                    _, _, _, state = policy(obs, state, th.tensor([True], device=policy.device), deterministic=True)
                next_acts, next_obs = eval_env.envs[0].next_observations()
                num_valid_actions = len(next_acts)
                next_obs = th.tensor(np.array(next_obs), dtype=th.int32, device=policy.device).permute(0, 3, 1, 2)
                pol_state = tree_map(partial(th.repeat_interleave, repeats=num_valid_actions, dim=1), state)
                eps_start = th.tensor([False] * num_valid_actions, device=policy.device)
                (pol_acts, pol_values, _, pol_state), cache = policy.run_with_cache(
                    next_obs,
                    pol_state,
                    eps_start,
                    deterministic=True,
                )
                if eval_type == "probe":
                    input_features = th.cat(
                        [cache[k + f".0.{repeats_per_step-1}"].reshape(num_valid_actions, -1) for k in cache_keys], dim=1
                    )
                    assert input_features.shape[0] == next_obs.shape[0]
                    action_idx_in_next_acts = probe_model.predict(input_features)
                    probe_matches_with_value_head += pol_values.argmax().item() == action_idx_in_next_acts
                elif eval_type == "policy_value":
                    action_idx_in_next_acts = pol_values.argmax().item()
                else:
                    raise ValueError(f"Unknown eval type {eval_type}")
                action = next_acts[action_idx_in_next_acts]
                state = tree_index(pol_state, (slice(None), action_idx_in_next_acts))
                state = tree_map(partial(th.unsqueeze, dim=1), state)
            obs, reward, done, _ = eval_env.step(th.tensor([action]))
            ep_len += 1
        if np.isclose(reward, final_reward):
            num_successes += 1
        num_steps += ep_len
    return num_successes / eval_episodes, probe_matches_with_value_head / num_steps


def main(args: TrainProbeConfig):
    set_seed(args.seed)

    eval_cfg = BoxobanConfig(
        max_episode_steps=80,
        min_episode_steps=80,
        tinyworld_obs=True,
        cache_path=args.eval_cache_path,
        split=args.eval_split,
        difficulty=args.eval_difficulty,
    )
    vec_env = VecTransposeImage(DummyVecEnv([eval_cfg.make]))

    if args.eval_only:
        dataset_path = Path(args.dataset_path)
        if dataset_path.is_dir():
            acts_ds = ActivationsDataset(dataset_path, num_data_points=args.dataset_size)
        else:
            acts_ds = th.load(dataset_path)

        if not args.eval_model:
            model = LinearProbeModel(acts_ds.num_features, 1)
        else:
            model = th.load(args.eval_model)
        assert args.policy_path, "Policy path must be provided for evaluation"
        policy_cfg, policy = load_jax_model_to_torch(args.policy_path, vec_env)
        assert isinstance(policy_cfg.features_extractor, ConvLSTMOptions)
        policy.to(model.device)
        policy.eval()
        success_rate, max_value_match = evaluate(
            model,
            policy,
            acts_ds.keys,
            policy_cfg.features_extractor.repeats_per_step,
            eval_type=args.eval_type,
            eval_cfg=eval_cfg,
            eval_env=vec_env,
        )
        print(f"Success rate: {success_rate}, Max value match: {max_value_match}")
        return None, {"test/success_rate": success_rate, "test/max_value_match": max_value_match}
    else:
        return train(args, eval_cfg, vec_env)


if __name__ == "__main__":
    lp_main.main()
