from glob import glob
from pathlib import Path

from learned_planner.interp.collect_dataset import DatasetStore

# dataset_path = Path("/training/activations_dataset/8_think_step")
dataset_path = Path("/training/activations_dataset/valid_medium/0_think_step")
level_files = glob(str(dataset_path / "*.pkl"))

total_mse = 0
total_eps = 100
for level_file in level_files[:total_eps]:
    ds_cache = DatasetStore.load(level_file)
    pred_values = ds_cache.get_values()
    true_values = ds_cache.get_true_values(0.97)
    assert pred_values.shape == true_values.shape, f"{pred_values.shape} != {true_values.shape}"

    mse = ((pred_values - true_values) ** 2).mean().item()
    last_20_mse = ((pred_values[-20:] - true_values[-20:]) ** 2).mean().item()
    print(mse, last_20_mse, ds_cache.solved)
    print("Predicted Values: [" + ", ".join([f"{x:.2f}" for x in pred_values.tolist()]) + "]")
    print("True Values: [" + ", ".join([f"{x:.2f}" for x in true_values.tolist()]) + "]")
    total_mse += mse

print(total_mse / total_eps)
