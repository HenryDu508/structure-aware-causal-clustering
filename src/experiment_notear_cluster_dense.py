#!/usr/bin/env python3
import argparse
import json
import os
import random

import numpy as np

from data_generation import generate_clustered_data
from NOTEAR import run_notear_experiment_cluster


def _to_py(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_py(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_py(item) for item in value)
    return value


def save_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_to_py(value), file, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run only the oracle-cluster NOTEARS baseline with a dense alpha grid."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--setting_name", type=str, required=True)
    parser.add_argument("--total_samples", type=int, default=50)
    parser.add_argument("--cluster_proportions", type=float, nargs="+", default=[0.6, 0.4])
    parser.add_argument("--n_vars", type=int, default=15)
    parser.add_argument("--m", type=int, nargs="+", default=[300])
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--s0_list", type=int, nargs="+", default=[20, 20])
    parser.add_argument(
        "--alpha_grid",
        type=float,
        nargs="+",
        default=[
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
            3e-2,
            1e-1,
            3e-1,
            1.0,
        ],
    )
    parser.add_argument("--n_folds", type=int, default=3)
    return parser.parse_args()


def resolve_m(args):
    if args.m == [0] or args.m == 0:
        m_choices = np.arange(50, 310, 10)
        return np.random.choice(m_choices, size=args.total_samples, replace=True).tolist()
    return args.m[0] if isinstance(args.m, list) and len(args.m) == 1 else args.m


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    m_value = resolve_m(args)
    X_list, W_list_gt, W_centers_gt, clusters_gt, label_truth = generate_clustered_data(
        total_samples=args.total_samples,
        cluster_proportions=args.cluster_proportions,
        n_vars=args.n_vars,
        m=m_value,
        W_pos_range=(0.2, 0.5),
        W_neg_range=(-0.5, -0.2),
        mean=0.0,
        std=args.std,
        s0_list=args.s0_list,
        graph_type="UR",
        seed=args.seed,
        permute=True,
    )

    results = run_notear_experiment_cluster(
        X_list,
        W_list_gt,
        W_centers_gt,
        clusters_gt,
        label_truth,
        args.seed,
        args.std,
        args.alpha_grid,
        n_folds=args.n_folds,
        cv_shuffle=True,
        cv_random_state=args.seed,
    )

    payload = {
        "seed": args.seed,
        "data_config": {
            "total_samples": args.total_samples,
            "cluster_proportions": args.cluster_proportions,
            "n_vars": args.n_vars,
            "m": m_value,
            "std": args.std,
            "s0_list": args.s0_list,
        },
        "diagnostic": "oracle_cluster_notears_dense_alpha_grid",
        "notear_cluster_dense": results,
    }

    out_dir = os.path.join(args.out_dir, args.setting_name, f"seed_{args.seed}")
    save_json(os.path.join(out_dir, "summary_notear_cluster_dense_CV.json"), payload)
    print(f"[seed {args.seed}] Dense-grid oracle-cluster NOTEARS complete. Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
