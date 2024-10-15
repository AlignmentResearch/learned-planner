from pathlib import Path
from typing import Literal

from learned_planner.configs.command_config import WandbCommandConfig
from learned_planner.configs.misc import DEFAULT_TRAINING, random_seed
from learned_planner.interp.train_sae import TrainSAEConfig

DeviceLiteral = Literal["cuda", "cpu"]


def train_sae(
    device: DeviceLiteral = "cuda",
    training_mount: Path = DEFAULT_TRAINING,
    seed: int = random_seed(),
):
    hook_layer = 0
    return WandbCommandConfig(
        base_save_prefix=training_mount,
        cmd=TrainSAEConfig(
            architecture="standard",
            hook_name=f"features_extractor.cell_list.{hook_layer}.hook_h",
            d_in=32,
            expansion_factor=32,
            activation_fn="topk",
            topk=8,
            l1_coefficient=0,
            lp_norm=1.0,
            scale_sparsity_penalty_by_decoder_norm=False,
            # Learning Rate
            lr_scheduler_name="constant",  # we set this independently of warmup and decay steps.
            l1_warm_up_steps=10_000,
            lr_warm_up_steps=0,
            lr_decay_steps=40_000,
            ## No ghost grad term.
            apply_b_dec_to_input=True,
            b_dec_init_method="geometric_median",
            normalize_sae_decoder=True,
            init_encoder_as_decoder_transpose=True,
            # Optimizer
            lr=4e-5,
            ## adam optimizer has no weight decay by default so worry about this.
            adam_beta1=0.9,
            adam_beta2=0.999,
            # Buffer details won't matter in we cache / shuffle our activations ahead of time.
            training_tokens=int(3e7),
            train_batch_size_tokens=4096,
            normalize_activations="none",
            cached_activations_path=["/training/activations_dataset/hard/0_think_step/"],
            eval_every_n_wandb_logs=500,
            wandb_log_frequency=100,
            n_checkpoints=5,
            ###
            grid_wise=True,
            num_envs=64,
            envpool=True,
            device=device,
            seed=seed,
        ),
    )


# fmt: off
def train_local(): return train_sae("cpu", Path("."))
def train_cluster(): return train_sae("cuda", DEFAULT_TRAINING)
