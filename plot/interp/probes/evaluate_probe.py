import argparse
import glob
import multiprocessing
import os
import pathlib
import warnings

import numpy as np
from scipy.stats import bootstrap
from sklearn.multioutput import MultiOutputClassifier

from learned_planner.interp.collect_dataset import DatasetStore
from learned_planner.interp.train_probes import TrainOn
from learned_planner.interp.utils import get_metrics, load_probe, predict

on_cluster = os.path.exists("/training")
LP_DIR = pathlib.Path(__file__).parent.parent.parent


def process_file(file):
    with warnings.catch_warnings(action="ignore"):
        ds = DatasetStore.load(file)
    if "direction" in probe_info.dataset_name:
        gts = getattr(ds, probe_info.dataset_name)(multioutput=multioutput)
    else:
        gts = getattr(ds, probe_info.dataset_name)()
    if probe_info.dataset_name in ["next_box"] and not ds.solved:
        return [], []

    cache = {k: ds.get_cache(key=k, only_env_steps=True) for k in keys}
    probe_preds = predict(cache, probe, probe_info, 0, is_concatenated_cache=True)
    if probe_info.dataset_name not in ["next_box", "next_target"]:
        assert len(probe_preds) == len(gts)
    else:
        probe_preds = probe_preds[: len(gts)]
    return probe_preds, gts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/training/activations_dataset/valid_medium/0_think_step/")
    parser.add_argument("--probe_path", type=str, default="")
    parser.add_argument("--probe_wandb_id", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="boxes_future_direction_map")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--hooks", type=str, default="hook_h,hook_c")
    parser.add_argument("--num_levels", type=int, default=1000)
    parser.add_argument("--all", action="store_true", help="Show all metrics.")

    args = parser.parse_args()

    probe, grid_wise = load_probe(args.probe_path, args.probe_wandb_id)

    multioutput = isinstance(probe, MultiOutputClassifier)

    probe_info = TrainOn(layer=args.layer, grid_wise=grid_wise, dataset_name=args.dataset_name, hooks=args.hooks.split(","))

    dataset_path = pathlib.Path(args.dataset_path)
    layers = [args.layer] if args.layer >= 0 else range(3)

    all_probe_preds = []
    all_gts = []
    keys = [f"features_extractor.cell_list.{layer}.{hook}" for layer in layers for hook in probe_info.hooks]
    files = glob.glob(str(dataset_path / "*.pkl"))[: args.num_levels]

    save_dir = pathlib.Path("/training/iclr_logs/") if on_cluster else LP_DIR / "plot/interp/"
    save_dir = save_dir / "evaluate_probe" / args.dataset_name / args.probe_wandb_id
    save_dir.mkdir(parents=True, exist_ok=True)

    if (save_dir / "probe_preds.npy").exists():
        probe_preds = np.load(save_dir / "probe_preds.npy")
        gts = np.load(save_dir / "gts.npy")
    else:
        with multiprocessing.Pool(8) as pool:
            results = list(pool.imap(process_file, files))
        probe_preds = np.concatenate([r[0] for r in results if len(r[0]) > 0])
        gts = np.concatenate([r[1] for r in results if len(r[0]) > 0])
        np.save(save_dir / "probe_preds.npy", probe_preds)
        np.save(save_dir / "gts.npy", gts)

    metrics = get_metrics(probe_preds, gts, classification=True)
    print("Dataset Name:", args.dataset_name)
    coefs = np.stack([c.coef_ for c in probe.estimators_]) if multioutput else probe.coef_
    metrics["nonzero_weights"] = np.count_nonzero(coefs)
    if args.all:
        print(metrics)
    else:

        def f1_statistic(preds, gts):
            f1 = get_metrics(preds, gts, classification=True)["f1"]
            return f1

        rng = np.random.default_rng(seed=42)
        if multioutput:
            raise NotImplementedError
        probe_preds = probe_preds.reshape(-1)
        gts = gts.reshape(-1)
        conf_int = bootstrap(
            (probe_preds, gts),
            statistic=f1_statistic,
            random_state=rng,
            batch=100,
            n_resamples=1000,
            vectorized=False,
            paired=True,
            method="basic",
        ).confidence_interval
        print("F1 Score:", 100 * metrics["f1"])
        print("Confidence interval:", (100 * conf_int[0], 100 * conf_int[1]))
        print("Nonzero weights:", metrics["nonzero_weights"])
