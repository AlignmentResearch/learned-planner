from pathlib import Path
from typing import Literal, Optional

from learned_planner.configs.command_config import WandbCommandConfig
from learned_planner.configs.misc import DEFAULT_TRAINING, random_seed
from learned_planner.interp.train_probes import TrainProbeConfig

DeviceLiteral = Literal["cuda", "cpu"]


def train_probe(
    device: DeviceLiteral = "cuda",
    training_mount: Path = DEFAULT_TRAINING,
    dataset_path: Path = DEFAULT_TRAINING / "activations_dataset/valid_medium/8ts_reward_500.pt",
    seed: int = random_seed(),
):
    return WandbCommandConfig(
        base_save_prefix=training_mount,
        cmd=TrainProbeConfig(
            dataset_path=dataset_path,
            policy_path="drc33/bkynosqi/cp_2002944000",
            learning_rate=1e-4,
            epochs=5000,
            eval_epoch_interval=500,
            batch_size=512,
            weight_decay=1e-1,
            weight_decay_type="l1",
            eval_type="probe",
            device=device,
            seed=seed,
        ),
    )


def eval_probe(
    training_mount: Path = DEFAULT_TRAINING,
    eval_type: str = "probe",
    policy_path: str = "drc33/bkynosqi/cp_2002944000",
    probe_path: Optional[str] = None,
    seed: int = random_seed(),
):
    return WandbCommandConfig(
        base_save_prefix=training_mount,
        cmd=TrainProbeConfig(
            policy_path=policy_path,
            eval_only=True,
            eval_type=eval_type,
            probe_path=probe_path,
            seed=seed,
        ),
    )


# fmt: off
def train_local(): return train_probe("cpu", Path("."))
def train_cluster(): return train_probe("cpu", DEFAULT_TRAINING)
