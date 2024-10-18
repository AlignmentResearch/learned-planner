from pathlib import Path

from learned_planner.launcher import FlamingoRun, group_from_fname, launch_jobs

difficulty = "medium"
split = "valid"
box = True
dataset_name = "next_box" if box else "next_target"
probe_wandb_id = "6e1w1bb6" if box else "42qs0bh1"
runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = 1
for lfi in [0]:
    for li in range(0, 1000):
        this_run_clis = [
            [
                "python",
                "/workspace/plot/interp/ci_score_box_target_probe.py",
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
                "/training/iclr_logs/ci_score/" + "box/" if box else "target/",
                "--logits",
                "10,15,20,25,30,35,40",
                "--probe_wandb_id",
                probe_wandb_id,
                "--dataset_name",
                dataset_name,
            ]
        ]
        runs.append(
            FlamingoRun(
                this_run_clis,
                CPU=1,
                MEMORY="2.1G",
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
