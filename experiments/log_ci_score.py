from pathlib import Path

from learned_planner.launcher import FlamingoRun, group_from_fname, launch_jobs

difficulty = "medium"
split = "valid"
agent = True
dataset_name = "agents_future_direction_map" if agent else "boxes_future_direction_map"
probe_wandb_id = "dirnsbf3" if agent else "vb6474rg"
runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 1
for lfi in [0]:
    for li in range(0, 1000):
        this_run_clis = [
            [
                "python",
                "/workspace/plot/interp/ci_score.py",
                "--probe_wandb_id",
                probe_wandb_id,
                "--dataset_name",
                dataset_name,
                "--level",
                str(lfi),
                str(li),
                "--difficulty",
                difficulty,
                "--split",
                split,
                "--hook_steps",
                "-1",
                "--output_base_path",
                "/training/iclr_logs/ci_score/" + "agents_direction_probe/" if agent else "",
                "--logits",
                "10,15,20,25,30",
            ]
        ]
        runs.append(
            FlamingoRun(
                this_run_clis,
                CPU=1,
                MEMORY="3.1G" if agent else "2.1G",
                GPU=0,
                PRIORITY="cpu-normal-batch",
                parallel=False,
            )
        )


GROUP: str = group_from_fname(__file__)

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=GROUP,
        project="learned-planners",
        # wandb_mode="disabled",
    )
