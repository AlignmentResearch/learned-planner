import dataclasses
from pathlib import Path
from typing import Literal, Optional

from sae_lens import DRCSAERunnerConfig, SAETrainingRunner

from learned_planner import __main__ as lp_main
from learned_planner.interp.collect_dataset import DatasetStore  # noqa: F401  # pyright: ignore
from learned_planner.interp.utils import load_policy
from learned_planner.train import ABCCommandConfig


@dataclasses.dataclass
class TrainSAEConfig(ABCCommandConfig):
    hook_name: str = "blocks.0.hook_mlp_out"
    hook_layer: int = 0
    cached_activations_path: str | list[str] = "/training/activations_dataset/hard/0_think_step/"
    architecture: Literal["standard", "gated"] = "standard"
    lr: float = 3e-4
    d_in: int = 512
    expansion_factor: int = 4
    activation_fn: str = "relu"  # relu, tanh-relu, topk
    topk: int = 8
    normalize_sae_decoder: bool = True
    init_encoder_as_decoder_transpose: bool = False
    scale_sparsity_penalty_by_decoder_norm: bool = False
    apply_b_dec_to_input: bool = True
    b_dec_init_method: str = "geometric_median"

    training_tokens: int = 2_000_000
    train_batch_size_tokens: int = 4096
    normalize_activations: str = (
        "none"  # none, expected_average_only_in (Anthropic April Update), constant_norm_rescale (Anthropic Feb Update)
    )
    mse_loss_normalization: Optional[str] = None  # None, dense_batch
    l1_coefficient: float = 1e-3
    l1_warm_up_steps: int = 0
    lp_norm: float = 1
    lr: float = 3e-4
    lr_scheduler_name: str = "constant"  # constant, cosineannealing, cosineannealingwarmrestarts
    lr_warm_up_steps: int = 0
    lr_decay_steps: int = 0

    ## Adam
    adam_beta1: float = 0
    adam_beta2: float = 0.999

    use_ghost_grads: bool = False  # want to change this to true on some timeline.
    feature_sampling_window: int = 2000
    dead_feature_window: int = 1000  # unless this window is larger feature sampling,

    wandb_log_frequency: int = 10
    eval_every_n_wandb_logs: int = 500  # logs every 1000 steps.
    checkpoint_path: str = "checkpoints"
    n_checkpoints: int = 0

    ## DRC
    grid_wise: bool = True
    epochs: int = 1  # Number of epochs to train for
    num_envs: int = 1  # Number of environments to use during evaluation
    envpool: bool = True  # Whether to use the Envpool environment

    def run(self, run_dir: Path):  # type: ignore
        self.checkpoint_path = str(run_dir / "checkpoint")
        self_args = dataclasses.asdict(self)
        topk = self_args.pop("topk")
        if self_args["activation_fn"] == "topk":
            self_args["activation_fn_kwargs"] = {"k": topk}

        default_args = dict(
            model_name="drc33",  # not used
            hook_head_index=None,
            wandb_project="lp_sae",
        )
        cfg = DRCSAERunnerConfig(**self_args, **default_args)  # type: ignore
        return main(cfg, run_dir)


def main(cfg: DRCSAERunnerConfig, run_dir: Path):
    _, policy_th = load_policy("drc33/bkynosqi/cp_2002944000/")
    sparse_autoencoder = SAETrainingRunner(cfg, override_model=policy_th).run() # type: ignore
    return sparse_autoencoder


if __name__ == "__main__":
    lp_main.main()
