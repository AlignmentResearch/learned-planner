import argparse
import os
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def orthogonal(d1, d2):
    assert d1 >= 0 and d1 < 4
    assert d2 >= 0 and d2 < 4
    return (d1 > 1) ^ (d2 > 1)


def get_confidence_interval(data, rng):
    return bootstrap((data,), statistic=np.mean, method="basic", n_resamples=1000, random_state=rng).confidence_interval


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_base_path", type=str, default="/training/iclr_logs/ci_score/valid_medium/", help="Path to save plots and cache."
)
parser.add_argument("--remove_next_to_wall", action="store_true", help="Remove next_to_wall=True rows.")
parser.add_argument("--ortho_direction", action="store_true", help="Keep only orthogonal directions.")
parser.add_argument("--all", action="store_true", help="Show all logit values & mean of best logit per perturbation.")
args = parser.parse_args()

on_cluster = os.path.exists("/training")
LP_DIR = pathlib.Path(__file__).parent.parent.parent
if on_cluster:
    output_base_path = pathlib.Path(args.output_base_path)
else:
    output_base_path = LP_DIR / args.output_base_path.lstrip("/")

if output_base_path.is_file():
    df = pd.read_csv(output_base_path)
else:
    df = pd.concat([pd.read_csv(f) for f in output_base_path.glob("*.csv")])

logit_group = df.groupby(["logit"])[["ci_success"]].mean().reset_index()
logit_group = logit_group.rename(columns={"ci_success": "Mean CI score for logit"})

best_overall_logit_idx = logit_group["Mean CI score for logit"].idxmax()
best_overall_logit = logit_group.loc[best_overall_logit_idx, "logit"]

best_logit_rows = df[df["logit"] == best_overall_logit]["ci_success"].to_numpy()

if "direction_idx" in df.columns:
    df["next_to_wall"] = df["next_to_wall"].astype(bool)
    if args.remove_next_to_wall:
        df = df[~df["next_to_wall"]]

    if args.ortho_direction:
        df = df[df.apply(lambda x: orthogonal(x["direction_idx"], x["ci_direction"]), axis=1)]

    best_perturbation = (
        df.groupby(["lfi", "li", "i", "j", "next_to_wall", "direction_idx", "logit"])["ci_success"].max().reset_index()
    )

    if args.all:
        best_perturbation_logit_group = best_perturbation.groupby(["logit"])[["ci_success"]].mean().reset_index()
        best_perturbation_logit_group = best_perturbation_logit_group.rename(
            columns={"ci_success": "Mean CI score for logit given best perturbation"}
        )
        logit_group = logit_group.merge(best_perturbation_logit_group, on="logit")

        best_perturbation_and_logit = (
            df.groupby(["lfi", "li", "i", "j", "next_to_wall", "direction_idx"])["ci_success"].max().reset_index()
        )
        ci_given_best_pert_logit = best_perturbation_and_logit["ci_success"].mean()

        best_logit_per_perturbation = (
            df.groupby(["lfi", "li", "i", "j", "next_to_wall", "direction_idx", "ci_direction"])["ci_success"]
            .max()
            .reset_index()
        )
        ci_given_best_logit = best_logit_per_perturbation["ci_success"].mean()

        best_logit_row = {
            "logit": "best_logit",
            "Mean CI score for logit": ci_given_best_logit,
            "Mean CI score for logit given best perturbation": ci_given_best_pert_logit,
        }
        logit_group = logit_group._append(best_logit_row, ignore_index=True)

        logit_group = logit_group.sort_values("Mean CI score for logit given best perturbation")
        print(logit_group.to_string(index=False))
        print()
        print("Total:", len(best_logit_per_perturbation))
        print("Total given best perturbation:", len(best_perturbation_and_logit))
    else:
        print("Best logit:", best_overall_logit)
        rng = np.random.default_rng(seed=42)
        conf_int = get_confidence_interval(best_logit_rows, rng)
        print(f"Mean CI score for best logit: {100 * best_logit_rows.mean()}")
        print("Confidence interval:", (100 * conf_int[0], 100 * conf_int[1]))

        best_perturbation_best_logit = best_perturbation[best_perturbation["logit"] == best_overall_logit][
            "ci_success"
        ].to_numpy()
        conf_int = get_confidence_interval(best_perturbation_best_logit, rng)
        print(f"Mean CI score for best logit given best perturbation: {100 * best_perturbation_best_logit.mean()}")
        print("Confidence interval:", (100 * conf_int[0], 100 * conf_int[1]))
elif "target" in df.columns or "box" in df.columns:
    col_name = "target" if "target" in df.columns else "box"
    print("Best logit:", best_overall_logit)
    rng = np.random.default_rng(seed=42)
    conf_int = get_confidence_interval(best_logit_rows, rng)
    print(f"Mean CI score for best logit: {100 * best_logit_rows.mean()}")
    print("Confidence interval:", (100 * conf_int[0], 100 * conf_int[1]))

    best_ci_per_level = df.groupby(["lfi", "li", "logit"])["ci_success"].max().reset_index()
    best_ci_per_level_best_logit = best_ci_per_level[best_ci_per_level["logit"] == best_overall_logit]["ci_success"].to_numpy()
    conf_int = get_confidence_interval(best_ci_per_level_best_logit, rng)
    print(f"Mean CI score for best logit given best {col_name}: {100 * best_ci_per_level_best_logit.mean()}")
    print("Confidence interval:", (100 * conf_int[0], 100 * conf_int[1]))
else:
    raise ValueError(f"Unknown columns in dataframe: {df.columns}")
