import concurrent.futures
import dataclasses
import pickle
import re
from functools import partial
from glob import glob
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np
import torch as th
import wandb
from sklearn import linear_model as sk_linear_model
from sklearn.multioutput import MultiOutputClassifier
from stable_baselines3.common.pytree_dataclass import tree_index, tree_map
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env.util import obs_as_tensor
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from learned_planner import __main__ as lp_main
from learned_planner.environments import BoxobanConfig, EnvConfig
from learned_planner.interp.collect_dataset import DatasetStore
from learned_planner.interp.utils import get_metrics, load_jax_model_to_torch, process_cache_for_sae
from learned_planner.policies import ConvLSTMOptions, download_policy_from_huggingface
from learned_planner.train import ABCCommandConfig


def set_seed(seed):
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


DATASET_TO_PROBE_TYPE = {
    "boxes_future_direction_map": "direction",
    "boxes_future_position_map": "position",
    "agents_future_direction_map": "direction",
    "agents_future_position_map": "position",
    "next_box": "position",
    "next_target": "position",
    "agent_in_a_cycle": "cycle",
    "alternative_boxes_future_direction_map": "direction",
}
GRID_WISE_DATASETS = [
    "agents_future_position_map",
    "agents_future_direction_map",
    "boxes_future_position_map",
    "boxes_future_direction_map",
    "next_box",
    "next_target",
    "alternative_boxes_future_direction_map",
    "actions_for_probe",
]


def is_grid_wise_dataset(dataset_name: str):
    return any(dataset_name.startswith(d) for d in GRID_WISE_DATASETS)


class ActivationsDataset(Dataset):
    def __init__(
        self,
        dataset_path: str | Path | list[Path] | list[str],
        labels_type: str = "pred_value",
        keys: list[str] = [".*hook_h$", ".*hook_c$"],
        num_data_points: Optional[int] = None,
        fetch_all_boxing_data_points: bool = False,
        gamma_value: float = 0.99,
        balance_classes: bool = False,
        skip_first_n: int = 0,
        skip_walls: bool = False,
        multioutput: bool = False,
        only_env_steps: bool = True,
        load_data: bool = True,
        seed: int = 42,
        train: bool = True,
    ):
        """Create a dataset of model activations to train probes on.

        Args:
            dataset_path (str | Path): Path to the dataset containing cache files of `DatasetStore` class.
            labels_type (str, optional): Type of labels to use for training. Can be one of
                ["pred_value", "true_value", "reward", "next_box_pos", "next_target_pos"].
                Defaults to "pred_value".
            keys (list, optional): List of regex patterns to match the keys in the cache.
                Defaults to [".*hook_h$", ".*hook_c$"].
            num_data_points (int, optional): Number of data points to keep in the dataset. Not strictly
                followed when `fetch_all_boxing_data_points = True`. Defaults to None which keeps all datapoints.
            fetch_all_boxing_data_points (bool, optional): If True, all datapoints where box is pushed or removed from a target
                are kept in the dataset. Defaults to False.
            gamma_value (float, optional): Gamma value to use for calculating true values. Defaults to 0.99.
            balance_classes (bool, optional): If True, balance the classes in the dataset. Defaults to False.
                Not used for grid-wise data.
            skip_first_n (int, optional): Number of initial timesteps to skip from each episode. Defaults to 0.
            skip_walls (bool, optional): For grid-wise datasets, if True, skip the datapoints where the pixel is a wall.
                Defaults to False.
            multioutput (bool, optional): If True, the labels are multioutput for classification datasets. Defaults to False.
        """
        super().__init__()
        self.LEVELS_PER_BUFFER = 1000
        if isinstance(dataset_path, (list, tuple)):
            self.dataset_path = [Path(p) if isinstance(p, str) else p for p in dataset_path]
        else:
            self.dataset_path = [dataset_path if isinstance(dataset_path, Path) else Path(dataset_path)]

        self.labels_type = labels_type
        self.level_files = sum((glob(str(p / "*.pkl")) for p in self.dataset_path), [])
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.level_files)
        self.train = train
        train_files = np.ceil(0.8 * len(self.level_files)).astype(int)
        if self.train:
            self.level_files = self.level_files[:train_files]
        else:
            self.level_files = self.level_files[train_files:]
            if num_data_points is not None:
                num_data_points = num_data_points // 4
        self.n_layers = 3
        self.n_features = 32
        self.obs_size = 10 * 10
        self.keys = self.get_full_keys(keys)
        self.num_data_points = num_data_points
        self.fetch_all_boxing_data_points = fetch_all_boxing_data_points
        self.gamma_value = gamma_value
        self.balance_classes = balance_classes
        self.skip_first_n = skip_first_n
        self.skip_walls = skip_walls
        self.multioutput = multioutput
        self.only_env_steps = only_env_steps

        if load_data:
            self.load_data(num_data_points)

    def get_full_keys(self, keys):
        ds_cache = DatasetStore.load(self.level_files[0])
        full_keys = [fk for fk in ds_cache.model_cache.keys() if any(re.match(k, fk) for k in keys)]
        return full_keys

    def total_est_buffers(self):
        return len(self.level_files) // self.LEVELS_PER_BUFFER + 1

    def load_model_cache(self, file_name, include_initial_thinking: bool = False):
        ds_cache = DatasetStore.load(file_name)
        all_cache_values = [
            ds_cache.get_cache(
                k,
                only_env_steps=self.only_env_steps,
                include_initial_thinking=include_initial_thinking,
            )
            for k in self.keys
        ]
        return ds_cache, all_cache_values

    def load_file(self, file_name, num_data_points: Optional[int] = None):
        include_initial_thinking = False
        ds_cache, all_cache_values = self.load_model_cache(file_name, include_initial_thinking)

        self.classification = True
        self.grid_wise = is_grid_wise_dataset(self.labels_type)
        if self.labels_type == "pred_value":
            gt_output = ds_cache.get_values(
                only_env_steps=self.only_env_steps, include_initial_thinking=include_initial_thinking
            )
            self.classification = False
        elif self.labels_type == "true_value":
            gt_output = ds_cache.get_true_values(gamma=self.gamma_value)
            self.classification = False
        elif self.labels_type == "reward":
            gt_output = ds_cache.rewards
            self.classification = False
        elif self.labels_type == "success":
            gt_output = ds_cache.get_success_repeated()
        elif self.labels_type == "agents_future_position_map":
            gt_output = ds_cache.agents_future_position_map()
        elif self.labels_type == "agents_future_direction_map":
            gt_output = ds_cache.agents_future_direction_map(multioutput=self.multioutput)
        elif self.labels_type == "boxes_future_position_map":
            gt_output = ds_cache.boxes_future_position_map()
        elif self.labels_type == "boxes_future_direction_map":
            gt_output = ds_cache.boxes_future_direction_map(multioutput=self.multioutput)
        elif self.labels_type == "next_box":
            gt_output = ds_cache.next_box()
            all_cache_values = [v[: len(gt_output)] for v in all_cache_values]
        elif self.labels_type == "next_target":
            gt_output = ds_cache.next_target()
            all_cache_values = [v[: len(gt_output)] for v in all_cache_values]
        elif self.labels_type.startswith("actions_for_probe"):
            args = self.labels_type[len("actions_for_probe") + 1 :].split("_")
            action = int(args[0])
            if len(args) > 1:
                self.grid_wise = bool(args[1])
            gt_output = ds_cache.actions_for_probe(action, grid_wise=self.grid_wise)
        elif self.labels_type == "agent_in_a_cycle":
            gt_output = ds_cache.agent_in_a_cycle()
        elif self.labels_type == "alternative_boxes_direction_map":
            gt_output = ds_cache.alternative_boxes_future_direction_map()
            self.multioutput = True
        else:
            raise ValueError(f"Unknown labels type {self.labels_type}")

        if self.grid_wise and self.balance_classes:
            raise ValueError("Cannot balance classes for grid-wise data")

        assert len(gt_output) == len(all_cache_values[0]), f"{len(gt_output)=}, {len(all_cache_values[0])=}"

        data = []
        gt_output_data = []
        num_data_points = num_data_points or len(gt_output)
        if len(gt_output) == 0:
            return data, gt_output_data, np.array([])
        indices = self.sample_random_indices(gt_output, num_data_points, balance_classes=self.balance_classes)
        if self.fetch_all_boxing_data_points:
            boxing_indices = ds_cache.get_boxing_indices()
            boxing_indices = np.setdiff1d(boxing_indices, indices)
            indices = np.concatenate([indices, boxing_indices])

        for v_idx in indices:
            if self.grid_wise:
                for cell_idx in range(self.obs_size):
                    i, j = cell_idx % 10, cell_idx // 10
                    if self.skip_walls and ds_cache.is_wall(i, j):
                        continue
                    cache_data = {}
                    for comp_idx in range(len(all_cache_values)):
                        v = all_cache_values[comp_idx][v_idx][:, i, j]
                        cache_data[self.keys[comp_idx]] = v
                    data.append(cache_data)
                    gt_output_data.append(
                        gt_output[v_idx, i, j].item() if gt_output[v_idx, i, j].size == 1 else gt_output[v_idx, i, j]
                    )
            else:
                cache_data = {}
                for comp_idx in range(len(all_cache_values)):
                    v = all_cache_values[comp_idx][v_idx]
                    cache_data[self.keys[comp_idx]] = v
                data.append(cache_data)
                gt_output_data.append(gt_output[v_idx].item() if gt_output[v_idx].size == 1 else gt_output[v_idx])
        boxing_indices_bool = np.zeros(len(indices), dtype=bool)
        if self.fetch_all_boxing_data_points:
            boxing_indices_bool[-len(boxing_indices) :] = True  # type: ignore
        return data, gt_output_data, boxing_indices_bool

    def sample_random_indices(self, gt_output, num_data_points: int, balance_classes: bool = False):
        if self.classification and balance_classes and not self.grid_wise:
            pos_indices = np.where(gt_output == 1)[0]
            neg_indices = np.where(gt_output[self.skip_first_n :] == 0)[0] + self.skip_first_n
            num_pos = int(num_data_points / 2)
            num_neg = num_data_points - num_pos
            if num_neg < len(neg_indices):
                neg_indices = self.rng.choice(neg_indices, num_neg, replace=False)
            indices = np.concatenate([pos_indices, neg_indices])
            self.rng.shuffle(indices)
        else:
            indices = np.arange(self.skip_first_n, len(gt_output))
            indices = self.rng.choice(indices, min(num_data_points, len(indices)), replace=False)
        return indices

    def load_data(self, num_data_points: Optional[int] = None):
        if num_data_points is not None:
            num_data_points_per_file = int(np.ceil(num_data_points / len(self.level_files)))
            if len(self.level_files) > num_data_points:
                num_files = num_data_points
                self.level_files = self.level_files[:num_files]
        else:
            num_data_points_per_file = None

        results = []
        map_fn = partial(self.load_file, num_data_points=num_data_points_per_file)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(map_fn, self.level_files), total=len(self.level_files)))
        self.data = [item for result in results for item in result[0]]
        self.gt_output = [item for result in results for item in result[1]]
        self.boxing_indices = [item for result in results for item in result[2]]
        assert len(self.data) == len(self.gt_output), f"{len(self.data)=}, {len(self.gt_output)=}"

        if not (num_data_points is None or self.fetch_all_boxing_data_points):
            self.data = self.data[:num_data_points]
            self.gt_output = self.gt_output[:num_data_points]
            self.boxing_indices = self.boxing_indices[:num_data_points]

        self.gt_output = (
            np.array(self.gt_output) if not isinstance(self.gt_output[0], np.ndarray) else np.stack(self.gt_output)
        )
        self.boxing_indices = np.array(self.boxing_indices)

    @staticmethod
    def process_cache_for_sae(cache_tensor, grid_wise: bool = False):
        return process_cache_for_sae(cache_tensor, grid_wise=grid_wise)

    def load_only_activations(self, buffer_idx, grid_wise: bool = False):
        if buffer_idx <= 0:
            if self.num_data_points is not None and len(self.level_files) > self.num_data_points:
                num_files = self.num_data_points
                self.level_files = self.level_files[:num_files]

        map_fn = partial(self.load_model_cache, include_initial_thinking=False)
        start, end = 0, len(self.level_files)
        if buffer_idx >= 0:
            start = buffer_idx * self.LEVELS_PER_BUFFER
            end = min((buffer_idx + 1) * self.LEVELS_PER_BUFFER, len(self.level_files))
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            cache_values = list(tqdm(executor.map(lambda x: map_fn(x)[1], self.level_files[start:end]), total=end - start))

        cache = {
            k: np.concatenate([self.process_cache_for_sae(cv[i], grid_wise=grid_wise) for cv in cache_values])  # type: ignore
            for i, k in enumerate(self.keys)
        }
        if self.num_data_points is not None and len(cache[self.keys[0]]) > self.num_data_points:
            indices = self.rng.choice(len(cache[self.keys[0]]), self.num_data_points, replace=False)
        else:
            indices = self.rng.permutation(len(cache[self.keys[0]]))
        cache = {k: v[indices] for k, v in cache.items()}
        return cache

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
        """Predict the probe output given a batch of inputs.

        Args:
            next_obs (th.Tensor): Batch of inputs. (batch_size, num_features)

        Returns:
            int: Predicted action index.
        """
        assert len(next_obs.shape) == 2, f"{next_obs.shape=}"
        values = self(next_obs)

        return values

    @property
    def device(self):
        return next(self.parameters()).device


@dataclasses.dataclass
class TrainOn:
    layer: int | Sequence[int] = -1  # -1 means all layers
    grid_wise: bool = True
    mean_pool_grid: bool = False
    channel: int = -1  # -1 means all channels
    dataset_name: str = "boxes_future_direction_map"
    hooks: list[str] = dataclasses.field(default_factory=lambda: ["hook_h", "hook_c"])

    def __post_init__(self):
        if isinstance(self.layer, int):
            assert self.layer >= -1, "Layer must be -1 or greater"
        assert self.channel >= -1, "Channel must be -1 or greater"

    def __repr__(self):
        return f"l-{self.rename(self.layer)}_c-{self.rename(self.channel)}_ds-{self.dataset_name}_mpg-{self.mean_pool_grid}"

    @property
    def action_idx(self):
        if self.dataset_name.startswith("actions_for_probe"):
            return int(self.dataset_name.split("_")[3])

    @property
    def probe_type(self):
        return DATASET_TO_PROBE_TYPE[self.dataset_name]

    @property
    def color_scheme(self):
        return ["red", "green", "blue"]

    @staticmethod
    def rename(o):
        if o == slice(None) or o == -1:
            return "all"
        return str(o)


@dataclasses.dataclass
class TrainProbeConfig(ABCCommandConfig):
    """Arguments to train or evaluate probes (and their corresponding policies)."""

    dataset_path: Path = Path("asdf")
    learning_rate: float = 1e-4
    epochs: int = 10
    batch_size: int = 512
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    weight_decay_type: str = "l1"
    dataset_size: Optional[int] = None
    wandb_name: Optional[str] = None
    policy_path: Optional[str] = None
    eval_epoch_interval: int = 10
    eval_only: bool = False
    eval_cache_path: Path = Path("/training/.sokoban_cache")
    eval_split: Literal["train", "valid", "test"] = "valid"
    eval_difficulty: Literal["unfiltered", "medium", "hard"] = "unfiltered"
    eval_episodes: int = 20
    eval_type: str = "probe"
    probe_path: Optional[str] = None  # None if you only want to evaluate the policy
    use_sklearn: bool = True
    sklearn_class_weight: Optional[str | dict] = None
    sklearn_solver: str = "liblinear"  # liblinear, saga support l1 penalty
    sklearn_n_jobs: int = 1
    sklearn_l1_ratio: float = 1.0
    train_on: TrainOn = dataclasses.field(default_factory=TrainOn)

    def __post_init__(self):
        if self.eval_type not in ["probe", "policy_action", "policy_value"]:
            raise ValueError(f"Unknown eval type {self.eval_type}")

    def run(self, run_dir: Path):  # type: ignore
        return main(self, run_dir)


def train_setup(args: TrainProbeConfig, eval_cfg: EnvConfig):
    device = args.th_device
    if args.policy_path:
        policy_path = download_policy_from_huggingface(args.policy_path)
        policy_cfg, policy = load_jax_model_to_torch(policy_path, eval_cfg)
        policy.to(device)
        policy.eval()
    else:
        policy_cfg, policy = None, None
    dataset_path = args.dataset_path
    if dataset_path.is_dir():
        acts_ds = ActivationsDataset(dataset_path, num_data_points=args.dataset_size)
        train_ds, test_ds = random_split(acts_ds, [0.8, 0.2])
    else:
        try:
            train_ds = th.load(str(dataset_path).replace(".pt", "_train.pt"))
            test_ds = th.load(str(dataset_path).replace(".pt", "_test.pt"))
            acts_ds = train_ds
            print("Loaded separate train and test datasets")
        except FileNotFoundError:
            acts_ds = th.load(dataset_path)
            train_ds, test_ds = random_split(acts_ds, [0.8, 0.2])
            print("Loaded single dataset and split into train and test")
        if "multioutput" not in acts_ds.__dict__:
            acts_ds.multioutput = False
    classification = acts_ds.classification
    input_dim, output_dim = acts_ds.num_features, 1

    return policy, policy_cfg, acts_ds, classification, input_dim, output_dim, train_ds, test_ds


def fit_probe(
    classification: bool,
    args: TrainProbeConfig,
    train_X,
    train_y,
    test_X,
    test_y,
    run_dir: Path,
    multioutput: bool,
):
    if classification:
        probe = sk_linear_model.LogisticRegression(
            penalty=None if args.weight_decay_type == "none" else args.weight_decay_type,  # type: ignore
            C=1 / args.weight_decay,
            class_weight=args.sklearn_class_weight,
            solver=args.sklearn_solver,
            n_jobs=args.sklearn_n_jobs,
            l1_ratio=args.sklearn_l1_ratio,
        )
        if multioutput:
            probe = MultiOutputClassifier(probe)
    else:
        if args.weight_decay_type == "l1":
            probe = sk_linear_model.Lasso(alpha=args.weight_decay)
        elif args.weight_decay_type == "l2":
            probe = sk_linear_model.Ridge(alpha=args.weight_decay)
        elif args.weight_decay_type == "none":
            probe = sk_linear_model.LinearRegression()
        else:
            raise ValueError(f"Unknown weight decay type {args.weight_decay_type}")

    if args.eval_only and args.probe_path is not None and Path(args.probe_path).exists():
        with open(args.probe_path, "rb") as f:
            probe = pickle.load(f)
    else:
        probe.fit(train_X, train_y)

    test_preds = probe.predict(test_X)
    train_preds = probe.predict(train_X)

    probe_subkey = str(args.train_on)
    if classification:
        test_metrics = get_metrics(test_preds, test_y, classification, f"test/{probe_subkey}")
        train_metrics = get_metrics(train_preds, train_y, classification, f"train/{probe_subkey}")
        metrics = {**test_metrics, **train_metrics}
    else:
        test_metrics = get_metrics(test_preds, test_y, classification, f"test/{probe_subkey}")
        train_metrics = get_metrics(train_preds, train_y, classification, f"train/{probe_subkey}")
        metrics = {**test_metrics, **train_metrics}
        metrics[f"test/{probe_subkey}/accuracy"] = probe.score(test_X, test_y)
        metrics[f"train/{probe_subkey}/accuracy"] = probe.score(train_X, train_y)
        # if "value" in acts_ds.labels_type:
        #     success_rate, max_value_match = evaluate(
        #         probe,
        #         policy,
        #         acts_ds.keys,
        #         policy_cfg.features_extractor.repeats_per_step,  # type: ignore
        #         eval_cfg,
        #         eval_env,
        #         args.eval_episodes,
        #         args.eval_type,
        #     )
        #     metrics[f"test/{probe_subkey}/success_rate"] = success_rate
        #     metrics[f"test/{probe_subkey}/max_value_match"] = max_value_match
    try:
        coefs = probe.coef_  # type: ignore
    except AttributeError:
        assert isinstance(probe, MultiOutputClassifier)
        coefs = np.stack([c.coef_ for c in probe.estimators_])  # type: ignore
    nonzero_weights = (coefs != 0).sum()
    nonzero_weights_ratio = nonzero_weights * 100 / coefs.size

    metrics[f"{probe_subkey}/nonzero_weights"] = nonzero_weights
    metrics[f"{probe_subkey}/nonzero_weights_ratio"] = nonzero_weights_ratio
    with open(run_dir / f"probe_{probe_subkey}.pkl", "wb") as f:
        pickle.dump(probe, f)
    print(f"Saved probe weights to probe_{probe_subkey}.pkl")
    wandb.log(metrics)  # type: ignore
    print(metrics)
    return probe, metrics


def train_with_sklearn(args: TrainProbeConfig, eval_cfg: EnvConfig, eval_env, run_dir: Path):
    policy, policy_cfg, acts_ds, classification, _, _, train_ds, test_ds = train_setup(args, eval_cfg)
    if classification:
        pos_ratio = 100 * (acts_ds.get_labels() == 1).sum() / len(acts_ds)  # type: ignore
        print(f"Positive ratio: {pos_ratio:.2f}%")
    layer = args.train_on.layer
    channel = args.train_on.channel
    channel = slice(None) if channel == -1 else channel

    train_X, train_y = list(zip(*train_ds))
    train_y = np.stack(train_y)
    if layer == -1:
        layer_keys = [k for k in acts_ds.keys if any(k.endswith(h) for h in args.train_on.hooks)]
    elif isinstance(layer, list) or isinstance(layer, tuple):
        layer_keys = [k for k in acts_ds.keys if any(k.endswith(h) for h in args.train_on.hooks)]
        layer_keys = [k for k in layer_keys if any(f"cell_list.{ly}" in k for ly in layer)]
    else:
        layer_keys = [k for k in acts_ds.keys if f"cell_list.{layer}" in k and any(k.endswith(h) for h in args.train_on.hooks)]
    print(f"Layer keys: {layer_keys}")
    train_X = np.stack(
        [
            np.concatenate(
                [
                    example[k][channel].mean(axis=(-1, -2)) if args.train_on.mean_pool_grid else example[k][channel]
                    for k in layer_keys
                ]
            )
            for example in train_X
        ]
    )
    if len(train_X.shape) == 4:  # everything is grid-wise now
        train_y = train_y.repeat((train_X.shape[-1] * train_X.shape[-2],))
        train_X = np.transpose(train_X, (2, 3, 0, 1)).reshape(-1, train_X.shape[1])

    if acts_ds.multioutput:
        assert len(train_X.shape) == 2 and len(train_y.shape) == 2 and train_y.shape == (len(train_X), 4)
    else:
        assert (
            len(train_X.shape) == 2 and len(train_y.shape) == 1 and train_X.shape[0] == len(train_y)
        ), f"{train_X.shape=}, {train_y.shape=}"

    test_X, test_y = list(zip(*test_ds))
    test_y = np.stack(test_y)
    test_X = np.stack(
        [
            np.concatenate(
                [
                    example[k][channel].mean(axis=(-1, -2)) if args.train_on.mean_pool_grid else example[k][channel]
                    for k in layer_keys
                ]
            )
            for example in test_X
        ]
    )
    if len(test_X.shape) == 4:
        test_y = test_y.repeat((test_X.shape[-1] * test_X.shape[-2],))
        test_X = np.transpose(test_X, (2, 3, 0, 1)).reshape(-1, test_X.shape[1])
    if acts_ds.multioutput:
        assert len(test_X.shape) == 2 and len(test_y.shape) == 2 and test_y.shape == (len(test_X), 4)
    else:
        assert (
            len(test_X.shape) == 2 and len(test_y.shape) == 1 and test_X.shape[0] == len(test_y)
        ), f"{test_X.shape=}, {test_y.shape=}"

    probe, metrics = fit_probe(classification, args, train_X, train_y, test_X, test_y, run_dir, acts_ds.multioutput)
    return probe, metrics


def train(args: TrainProbeConfig, eval_cfg: EnvConfig, eval_env, run_dir: Path):
    device = args.th_device
    policy, policy_cfg, acts_ds, classification, input_dim, output_dim, train_ds, test_ds = train_setup(args, eval_cfg)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

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
                weight_decay_norm = int(args.weight_decay_type[1])
                if "weight" in name:
                    reg = reg + th.norm(param, weight_decay_norm)
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
    is_sklearn=True,
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
            if not is_sklearn:
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
                    action_idx_in_next_acts = probe_model.predict(input_features).argmax().item()
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


def main(args: TrainProbeConfig, run_dir: Path):
    set_seed(args.seed)

    eval_cfg = BoxobanConfig(
        n_envs=1,
        n_envs_to_render=0,
        max_episode_steps=80,
        min_episode_steps=80,
        tinyworld_obs=True,
        cache_path=args.eval_cache_path,
        split=args.eval_split,
        difficulty=args.eval_difficulty,
    )
    vec_env = VecTransposeImage(DummyVecEnv([eval_cfg.make]))

    if args.eval_only and not args.use_sklearn:
        dataset_path = args.dataset_path
        if dataset_path.is_dir():
            acts_ds = ActivationsDataset(dataset_path, num_data_points=args.dataset_size)
        else:
            acts_ds = th.load(dataset_path)

        if not args.probe_path:
            model = LinearProbeModel(acts_ds.num_features, 1)
        else:
            model = th.load(args.probe_path)
        assert args.policy_path, "Policy path must be provided for evaluation"
        policy_path = download_policy_from_huggingface(args.policy_path)
        policy_cfg, policy = load_jax_model_to_torch(policy_path, eval_cfg)
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
        if args.use_sklearn:
            return train_with_sklearn(args, eval_cfg, vec_env, run_dir)
        else:
            return train(args, eval_cfg, vec_env, run_dir)


if __name__ == "__main__":
    lp_main.main()
