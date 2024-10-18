import subprocess

from learned_planner import LP_DIR


def test_measure_plan_quality_script(tmp_path):
    script_path = LP_DIR / "plot/interp/probes/measure_plan_quality.py"
    assert script_path.exists(), f"Script {script_path} does not exist"

    cmd = [
        "python",
        str(script_path),
        "--num_envs",
        "2",
        "--num_levels",
        "2",
        "--output_base_path",
        str(tmp_path),
    ]
    result = subprocess.run(cmd)

    assert result.returncode == 0, f"Command '{' '.join(cmd)}' failed with exit code {result}"


def test_measure_plan_recall_script(tmp_path):
    script_path = LP_DIR / "plot/interp/probes/measure_plan_recall.py"
    assert script_path.exists(), f"Script {script_path} does not exist"

    cmd = [
        "python",
        str(script_path),
        "--num_envs",
        "2",
        "--num_levels",
        "2",
        "--output_base_path",
        str(tmp_path),
    ]
    result = subprocess.run(cmd)

    assert result.returncode == 0, f"Command '{' '.join(cmd)}' with exit code {result}"


def test_ci_score_direction_probe_script(tmp_path):
    script_path = LP_DIR / "plot/interp/probes/ci_score_direction_probe.py"
    assert script_path.exists(), f"Script {script_path} does not exist"

    cmd = [
        "python",
        str(script_path),
        "--num_levels",
        "1",
        "--logits",
        "15",
        "--num_workers",
        "1",
        "--output_base_path",
        str(tmp_path),
    ]
    result = subprocess.run(cmd)

    assert result.returncode == 0, f"Command '{' '.join(cmd)}' with exit code {result}"


def test_ci_score_box_target_probe_script(tmp_path):
    script_path = LP_DIR / "plot/interp/probes/ci_score_box_target_probe.py"
    assert script_path.exists(), f"Script {script_path} does not exist"

    cmd = [
        "python",
        str(script_path),
        "--num_levels",
        "1",
        "--logits",
        "15",
        "--num_workers",
        "1",
        "--output_base_path",
        str(tmp_path),
    ]
    result = subprocess.run(cmd)

    assert result.returncode == 0, f"Command '{' '.join(cmd)}' with exit code {result}"


def test_evaluate_probe_script(tmp_path):
    script_path = LP_DIR / "plot/interp/probes/evaluate_probe.py"
    dataset_path = LP_DIR / "tests/probes_dataset/"
    assert script_path.exists(), f"Script {script_path} does not exist"

    cmd = [
        "python",
        str(script_path),
        "--dataset_path",
        str(dataset_path),
        "--num_workers",
        "1",
        "--output_base_path",
        str(tmp_path),
    ]
    result = subprocess.run(cmd)

    assert result.returncode == 0, f"Command '{' '.join(cmd)}' with exit code {result}"
