from pathlib import Path

from farconf import update_fns_to_cli
from stable_baselines3.common.type_aliases import check_cast

from learned_planner.configs.command_config import WandbCommandConfig
from learned_planner.configs.train_probe import train_local
from learned_planner.interp.train_probes import TrainProbeConfig
from learned_planner.launcher import FlamingoRun, group_from_fname, launch_jobs

dataset_path = Path(
    "/training/activations_dataset/hard/0ts_alternative_boxes_direction_map_20000_skip5_gridwise_multioutput.pt"
)

clis: list[list[str]] = []
for layer in range(-1, 3):
    for weight_decay in [5000, 1000, 500, 100, 1]:

        def update(config: WandbCommandConfig) -> WandbCommandConfig:
            config.cmd = check_cast(TrainProbeConfig, config.cmd)
            config.cmd.train_on.layer = layer
            config.cmd.sklearn_solver = "saga"
            config.cmd.weight_decay = weight_decay
            config.cmd.sklearn_n_jobs = 4
            config.cmd.dataset_path = dataset_path
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
            CPU=4,
            MEMORY="40G",
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
        project="learned-planners",
    )
