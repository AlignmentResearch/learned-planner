import dataclasses
import functools
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import wandb
from farconf import parse_cli
from git.repo import Repo
from names_generator import generate_name, random_names
from typing_extensions import Self


@functools.lru_cache()
def git_latest_commit() -> str:
    repo = Repo(".")
    commit_hash = str(repo.head.object.hexsha)
    return commit_hash


def group_from_fname(fname: str, suffix: str = "") -> str:
    base_group = Path(fname).name.replace(".py", "").replace("_", "-")
    if suffix:
        return f"{base_group}-{suffix}"
    return base_group


@dataclasses.dataclass
class FlamingoRun:
    commands: list[list[str]]
    CONTAINER_TAG: str = ""
    COMMIT_HASH: str = dataclasses.field(default_factory=git_latest_commit)
    CPU: int | str = 4
    MEMORY: str = "20G"
    SHM_SIZE: str = "10G"
    GPU: int = 1
    TRAINING_MOUNT: Path = Path("/training")
    PRIORITY: str = "normal-batch"
    parallel: bool = True

    def format_args(self) -> dict[str, str | int]:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "cfg"}

    @classmethod
    def field_defaults(cls) -> dict[str, Any]:
        return {f.name: f.default for f in dataclasses.fields(cls)}

    @classmethod
    def from_cfg(cls: type[Self], cfg: type, **kwargs) -> Self:
        to_parse_cli = cfg.to_cli()
        parsed_cfg = parse_cli(to_parse_cli, type(cfg))
        check_parsing(cfg, parsed_cfg)
        assert parsed_cfg == cfg, f"The CLI {to_parse_cli} won't be correctly parsed"
        return cls([to_parse_cli], TRAINING_MOUNT=cfg.base_save_prefix, **kwargs)


def check_parsing(cfg1, cfg2, prefix=""):
    if not hasattr(cfg1, "__dict__"):
        if cfg1 != cfg2:
            print(f"Attribute is not being parsed correctly for {prefix}: {cfg1} != {cfg2}")
        return
    for k, v in cfg1.__dict__.items():
        if getattr(cfg1, k) != getattr(cfg2, k):
            check_parsing(getattr(cfg1, k), getattr(cfg2, k), prefix=f"{prefix}.{k}")


def create_jobs(
    start_number: int,
    runs: Sequence[FlamingoRun],
    group: str,
    project: str = "learned-planners",
    wandb_mode: str = "online",
    job_template_path: Optional[Path] = None,
    run_locally: bool = True,
) -> tuple[Sequence[str], str]:
    launch_id = generate_name(style="hyphen")

    if job_template_path is None:
        job_template_path = Path(__file__).parent.parent / "k8s" / "runner.yaml"
    with job_template_path.open() as f:
        job_template = f.read()

    jobs = []
    for i, run in enumerate(runs):
        split_command: list[str] = []

        job_name = "lp"
        wandb_job_name: Optional[str] = None
        for run_cli in run.commands:
            name1, name2 = random_names()
            job_number = start_number + i

            wandb_job_name = f"{name1}-{name2}-{job_number}"
            job_name += f"-{name1[:3]}{name2[:3]}{job_number}"

            split_command.extend(
                [
                    f"WANDB_JOB_NAME={shlex.quote(wandb_job_name)}",
                    f"WANDB_RUN_GROUP={shlex.quote(group)}",
                    f"WANDB_PROJECT={shlex.quote(project)}",
                    f"WANDB_MODE={shlex.quote(wandb_mode)}",
                    *map(shlex.quote, run_cli),
                ]
                + [("&" if run.parallel else " || true ;") if len(run.commands) > 1 else ""],
            )
        if run.parallel and len(run.commands) > 1:
            split_command.append("wait")
        assert wandb_job_name is not None

        job = job_template.format(
            WANDB_RUN_GROUP=group,
            WANDB_JOB_NAME=wandb_job_name,
            NAME=job_name,
            LAUNCH_ID=launch_id,
            WANDB_ENTITY="",
            WANDB_PROJECT=project,
            WANDB_MODE=wandb_mode,
            COMMAND=" ".join(split_command),
            OMP_NUM_THREADS=json.dumps(str(run.CPU if isinstance(run.CPU, int) else 1)),
            **run.format_args(),
        )
        if run_locally:
            jobs.append(" ".join(split_command))
        else:
            jobs.append(job)

    return jobs, launch_id


def launch_jobs(
    runs: Sequence[FlamingoRun],
    group: str,
    project: str = "learned-planners",
    wandb_mode: str = "online",
    job_template_path: Optional[Path] = None,
    run_locally: bool = True,
) -> tuple[str, str]:
    repo = Repo(".")
    repo.remote("origin").push(repo.active_branch.name)  # Push to an upstream branch with the same name
    start_number = 1 + len(wandb.Api().runs(f"{project}"))
    jobs, launch_id = create_jobs(
        start_number,
        runs,
        group=group,
        project=project,
        wandb_mode=wandb_mode,
        job_template_path=job_template_path,
        run_locally=run_locally,
    )
    if run_locally:
        for cmd in jobs:
            print("Running:")
            print(cmd)
            print()
            subprocess.run(cmd, shell=True)
        return "\n\n".join(jobs), launch_id
    else:
        yamls_for_all_jobs = "\n\n---\n\n".join(jobs)

        if not any(s in sys.argv for s in ["--dryrun", "--dry-run", "-d"]):
            subprocess.run(["kubectl", "create", "-f", "-"], check=True, input=yamls_for_all_jobs.encode())
            print(f"Jobs launched. To delete them run:\nkubectl delete jobs -l launch-id={launch_id}")
        return yamls_for_all_jobs, launch_id