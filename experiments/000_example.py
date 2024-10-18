from pathlib import Path

from farconf import update_fns_to_cli

from learned_planner.launcher import FlamingoRun, group_from_fname, launch_jobs
from learned_planner.train import BaseCommandConfig, TrainConfig

clis: list[list[str]] = []
for name in ["Alice"]:

    def update(config: BaseCommandConfig) -> BaseCommandConfig:
        ...
        return config

    cli, _ = update_fns_to_cli(TrainConfig, update)
    clis.append(cli)


runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 2
for update_fns_i in range(0, len(clis), RUNS_PER_MACHINE):
    this_run_clis = [
        ["python", "-m", "learned_planner", *clis[update_fns_i + j]]
        for j in range(min(RUNS_PER_MACHINE, len(clis) - update_fns_i))
    ]
    runs.append(
        FlamingoRun(
            this_run_clis,
            CONTAINER_TAG="f2198c5-main",
            CPU=2,
            MEMORY="2G",
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
        project="lp-interp",
    )
