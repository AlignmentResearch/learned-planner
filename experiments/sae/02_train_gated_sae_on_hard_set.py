from pathlib import Path

from farconf import update_fns_to_cli
from stable_baselines3.common.type_aliases import check_cast

from learned_planner.configs.command_config import WandbCommandConfig
from learned_planner.configs.train_sae import train_local
from learned_planner.interp.train_sae import TrainSAEConfig
from learned_planner.launcher import FlamingoRun, group_from_fname, launch_jobs

clis: list[list[str]] = []
for lr in [5e-5, 1e-6]:
    for k in [4, 8]:
        for expansion_factor in [16, 32]:
            for normalize_activations in ["none", "expected_average_only_in", "constant_norm_rescale", "layer_norm"]:

                def update(config: WandbCommandConfig) -> WandbCommandConfig:
                    config.cmd = check_cast(TrainSAEConfig, config.cmd)
                    config.cmd.lr = lr
                    config.cmd.device = "cpu"
                    config.cmd.activation_fn = "topk"
                    config.cmd.topk = k
                    config.cmd.expansion_factor = expansion_factor
                    config.cmd.normalize_activations = normalize_activations
                    config.cmd.l1_coefficient = 1e-20
                    config.cmd.n_checkpoints = 5
                    return config

                cli, _ = update_fns_to_cli(train_local, update)
                clis.append(cli)


runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 1
for update_fns_i in range(0, len(clis), RUNS_PER_MACHINE):
    this_run_clis = [
        ["python", "-m", "learned_planner", *clis[update_fns_i + j]]
        for j in range(min(RUNS_PER_MACHINE, len(clis) - update_fns_i))
    ]
    runs.append(
        FlamingoRun(
            this_run_clis,
            CONTAINER_TAG="b12433f-main",
            CPU=1,
            MEMORY="20G",
            GPU=0,
            PRIORITY="normal-batch",
            parallel=False,
        )
    )


GROUP: str = group_from_fname(__file__)

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=GROUP,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner.yaml",
        project="lp_sae",
    )
