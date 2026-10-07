#!/usr/bin/env python3
"""
DAG-DC-ADMM in the three-cluster scenario (proportions 0.4/0.4/0.2, five edges per cluster).

Run one seed of one setting: data generation, CV over (lambda1, lambda2, tau), final fit, evaluation.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dagdc.three_cluster.data_generation import generate_clustered_data
from dagdc.three_cluster.cross_validation import tune_hyperparameters, compute_total_reconstruction_error, average_skelton_accuracy, clustering_overall_metrics
from dagdc.three_cluster.dc_admm import optimize_dc_admm

DEFAULT_GRID = {
    "lambda1": [0.0001, 0.001, 0.01, 0.1],
    "lambda2": [0.01, 0.001, 0.0001, 0.00001],
    "tau":     [0.05, 0.1, 0.4, 0.7],
    "rho1":    [0.1],
    "rho2":    [0.05],
}

# (number of subjects, measurements per subject, noise std, CV grid) used in the paper
SETTINGS = {
    "N50_std05":  dict(total_samples=50,  m=300, std=0.5, grid=DEFAULT_GRID),
    "N50_std1":   dict(total_samples=50,  m=300, std=1,   grid=DEFAULT_GRID),
    "N50_std2":   dict(total_samples=50,  m=300, std=2,   grid=DEFAULT_GRID),
    "N200_std05": dict(total_samples=200, m=50,  std=0.5, grid=DEFAULT_GRID),
    "N200_std1":  dict(total_samples=200, m=50,  std=1,   grid=DEFAULT_GRID),
    "N200_std2":  dict(total_samples=200, m=50,  std=2,
                       grid={"lambda1": [0.0001, 0.01, 0.1], "lambda2": [0.1, 0.01, 0.001, 0.0001],
                             "tau": [0.1, 0.4, 0.7], "rho1": [0.1], "rho2": [0.05]}),
}


def _to_py(x):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_py(v) for v in x)
    return x
    
def save_artifacts_json_simple(
    out_dir, tag,
    Wc_est,  # (K,d,d) or list of arrays; may be None
    clusters_est,  # list of index sets
    W_list_final,        # (n,d,d)
    label_est,           # (n,)
    W_cluster_est_list,  # (n,d,d): cluster W assigned to each subject
    params=None,
    round_decimals=3  # None disables rounding
):
    os.makedirs(out_dir, exist_ok=True)

    def _tolist(arr):
        a = np.asarray(arr)
        if (round_decimals is not None) and np.issubdtype(a.dtype, np.floating):
            a = np.round(a, round_decimals)
        return a.tolist()

    payload = {
        "Wc_est":               None if Wc_est is None else _tolist(Wc_est),
        "clusters_est":         [sorted(map(int, list(c))) for c in clusters_est],
        "W_list_final":         _tolist(W_list_final),
        "label_est":            np.asarray(label_est, dtype=int).tolist(),
        "W_cluster_est_list":   _tolist(W_cluster_est_list),
        "_meta": {
            "n": int(np.asarray(W_list_final).shape[0]),
            "d": int(np.asarray(W_list_final).shape[1]),
            "K": int(len(clusters_est)),
            "round_decimals": round_decimals,
            "version": "artifacts-json-v1"
        },
        "params": ({k: float(v) for k, v in (params or {}).items()})
    }

    json_path = os.path.join(out_dir, f"{tag}_artifacts.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return json_path
    

def evaluate_best_for_all(
    out_dir,
    cv_table,
    X_list,
    W_list_gt,
    W_centers_gt,
    clusters_gt,
    label_truth,
    generate_seed_fit=False
):
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]

    for metric in ["mle_mean", "recon_mean", "cov_mean"]:
        best_idx = cv_table[metric].idxmin()
        row      = cv_table.loc[best_idx]
        params   = {k: float(row[k]) for k in ["lambda1","lambda2","tau","rho1","rho2"]}

        if generate_seed_fit:
            np.random.seed(int(1e6 * np.random.rand()))

        Wc_est, clusters_est, W_list_final, label_est, W_cluster_est_list = optimize_dc_admm(
            X_list,
            **params,
            max_dc_iter=10,
            max_admm_iter=15,
            plot_graph=False,
            thres_value=0.01    
        )

        skeleton_acc_base = average_skelton_accuracy(W_list_gt, W_cluster_est_list, threshold=0.01)
        recon_gt   = compute_total_reconstruction_error(W_centers_gt, clusters_gt,   X_list)
        recon_est  = compute_total_reconstruction_error(Wc_est,        clusters_est, X_list)
        clust_acc  = clustering_overall_metrics(label_truth,           label_est)

  
        sweep_records = []
        for t in thresholds:
            acc_t = average_skelton_accuracy(W_list_gt, W_cluster_est_list, threshold=t)
            sweep_records.append({"threshold": float(t), "skeleton_accuracy": _to_py(acc_t)})

        df_sweep = pd.DataFrame(sweep_records)
        csv_path = os.path.join(out_dir, f"skeleton_thresholds_{metric}.csv")
        df_sweep.to_csv(csv_path, index=False)

        results[metric] = _to_py({
            "best_params":              params,
            "train_thres_value":        0.01,                
            "skeleton_accuracy@0.01":   skeleton_acc_base,
            "skeleton_threshold_sweep": sweep_records,
            "recon_error_gt":           recon_gt,
            "recon_error_est":          recon_est,
            "clustering_accuracy":      clust_acc,
            "sweep_csv":                csv_path
        })

        artifacts_json = save_artifacts_json_simple(
            out_dir=out_dir, tag=f"{metric}_full",
            Wc_est=Wc_est,
            clusters_est=clusters_est,
            W_list_final=W_list_final,
            label_est=label_est,
            W_cluster_est_list=W_cluster_est_list,
            params=params,
            round_decimals=3
        )

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as fp:
        json.dump(results, fp, indent=2)

    return results


def main():
    p = argparse.ArgumentParser(description="Run the three-cluster experiment for one seed")
    p.add_argument("--setting", choices=list(SETTINGS), required=True)
    p.add_argument("--seed", type=int, required=True, help="Random seed")
    p.add_argument("--out_dir", type=str, default="results", help="Base output directory (default: results)")
    args = p.parse_args()
    seed, cfg = args.seed, SETTINGS[args.setting]

    random.seed(seed)
    np.random.seed(seed)

    X_list, W_list_gt, W_centers_gt, clusters_gt, label_truth = generate_clustered_data(
        total_samples=cfg["total_samples"],
        cluster_proportions=[0.4, 0.4, 0.2],
        n_vars=5,
        m=cfg["m"],
        W_pos_range=(0.2, 0.5),
        W_neg_range=(-0.5, -0.2),
        mean=0.0,
        std=cfg["std"],
        s0_list=[5, 5, 5],
        graph_type="UR",
        seed=seed,
        permute=True
    )

    top5_mle, top5_recon, top5_cov, cv_table = tune_hyperparameters(
        X_list,
        cfg["grid"],
        n_folds=3,
        max_dc_iter=10,
        max_admm_iter=15,
        plot_graph=False
    )

    out_dir = os.path.join(args.out_dir, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    cv_table.to_csv(os.path.join(out_dir, "cv_results.csv"), index=False)
    for name, data in [('top5_mle', top5_mle), ('top5_recon', top5_recon), ('top5_cov', top5_cov)]:
        with open(os.path.join(out_dir, f"{name}.json"), 'w') as f:
            json.dump(data, f, indent=2)

    evaluate_best_for_all(out_dir, cv_table, X_list, W_list_gt, W_centers_gt, clusters_gt, label_truth)
    print(f"[{args.setting} seed {seed}] Experiment complete. Summary written to {out_dir}/summary.json")

if __name__ == "__main__":
    main()