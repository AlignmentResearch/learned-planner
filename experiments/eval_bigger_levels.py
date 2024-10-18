from pathlib import Path

from learned_planner.launcher import FlamingoRun, group_from_fname, launch_jobs

clis: list[list[str]] = []
for steps_to_think in [0, 2, 4, 8, 16, 32, 64, 128]:
    for file_idx in range(24):
        cli = [
            "python",
            "learned_planner/notebooks/play_bigger_levels.py",
            f"--file_idx={file_idx}",
            f"--steps_to_think={steps_to_think}",
        ]
        clis.append(cli)


runs: list[FlamingoRun] = []
for cli in clis:
    runs.append(
        FlamingoRun(
            [cli],
            CPU=2,
            MEMORY="5G",
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
        job_template_path=Path(__file__).parent.parent / "k8s/bigger-levels.yaml",
        # project="learned-planners",
        wandb_mode="disabled",
    )
