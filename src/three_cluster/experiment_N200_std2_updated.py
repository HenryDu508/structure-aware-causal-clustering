#!/usr/bin/env python3
"""
experiment.py

Run one full experiment (data generation → hyperparameter tuning → final fit → evaluation)
for a given random seed.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import random
import numpy as np
import os
import json
import pandas as pd

from data_generation import generate_clustered_data
from cross_validation_updated import tune_hyperparameters, compute_total_reconstruction_error, average_skelton_accuracy, clustering_overall_metrics
from cluster_algo_updated import optimize_dc_admm


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
    Wc_est,              # (K,d,d) 或 list[np.ndarray]；若没有可传 None
    clusters_est,        # list[Iterable[int]]（也可能是 set）
    W_list_final,        # (n,d,d)
    label_est,           # (n,)
    W_cluster_est_list,  # (n,d,d) —— 你目前的定义：簇W已广播到每个样本
    params=None,
    round_decimals=3     # 为控制体积，可设 None 取消四舍五入
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
    cv_table,        # DataFrame with ['lambda1','lambda2','tau','rho1','rho2','mle_mean','recon_mean','cov_mean']
    X_list,
    W_list_gt,
    W_centers_gt,
    clusters_gt,
    label_truth,
    generate_seed_fit=False
):
    """
    For each metric in ['mle_mean','recon_mean','cov_mean']:
      1) pick the row with minimal metric in cv_table
      2) extract params and fit optimize_dc_admm on full data
      3) compute structure accuracy, reconstruction errors, clustering accuracy
      4) sweep average_skelton_accuracy over thresholds 0.01..0.05, save CSV
    Saves a JSON summary to out_dir/summary.json and returns the results dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    # 预定义要扫的阈值
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05,0.06,0.07,0.08,0.09,0.1]

    for metric in ["mle_mean", "recon_mean", "cov_mean"]:
        # 1) select best hyperparameters
        best_idx = cv_table[metric].idxmin()
        row      = cv_table.loc[best_idx]
        params   = {k: float(row[k]) for k in ["lambda1","lambda2","tau","rho1","rho2"]}

        # 2) optional reseed for reproducibility
        if generate_seed_fit:
            np.random.seed(int(1e6 * np.random.rand()))

        # 3) fit on full data
        Wc_est, clusters_est, W_list_final, label_est, W_cluster_est_list = \
            optimize_dc_admm(
                X_list,
                **params,
                max_dc_iter=10,
                max_admm_iter=15,
                plot_graph=False,
                thres_value=0.01
            )

        # 4) baseline evaluation（和原来一致）
        skeleton_acc_base = average_skelton_accuracy(W_list_gt, W_cluster_est_list, threshold=0.01)
        recon_gt   = compute_total_reconstruction_error(W_centers_gt, clusters_gt,   X_list)
        recon_est  = compute_total_reconstruction_error(Wc_est,        clusters_est, X_list)
        clust_acc  = clustering_overall_metrics(label_truth,           label_est)

        # 5) 扫阈值并保存到 CSV
        sweep_records = []
        for t in thresholds:
            acc_t = average_skelton_accuracy(W_list_gt, W_cluster_est_list, threshold=t)
            sweep_records.append({"threshold": t, "skeleton_accuracy": acc_t})

        df_sweep = pd.DataFrame(sweep_records)
        csv_path = os.path.join(out_dir, f"skeleton_thresholds_{metric}.csv")
        df_sweep.to_csv(csv_path, index=False)

        # 6) bundle into results（把 sweep 结果也放进 summary.json）
        results[metric] = _to_py({
            "best_params":         params,
            "skeleton_accuracy@0.01": skeleton_acc_base,
            "skeleton_threshold_sweep": sweep_records,  # list of {threshold, skeleton_accuracy}
            "recon_error_gt":      recon_gt,
            "recon_error_est":     recon_est,
            "clustering_accuracy": clust_acc,
            "sweep_csv":           csv_path
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
        
    # 7) write JSON summary
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as fp:
        json.dump(results, fp, indent=2)

    return results



def main():
    # 1) Parse command‑line arguments
    p = argparse.ArgumentParser(description="Run experiment for one seed")
    p.add_argument("--seed", type=int, required=True, help="Random seed")
    p.add_argument("--out_dir", type=str, default="results", help="Base output directory (default: results)")
    args = p.parse_args()
    seed = args.seed
    base_out_dir = args.out_dir

    # 2) Seed RNGs
    random.seed(seed)
    np.random.seed(seed)

    X_list, W_list_gt, W_centers_gt, clusters_gt, label_truth = generate_clustered_data(
        total_samples=200,
        cluster_proportions=[0.4, 0.4,0.2],
        n_vars=5,
        m=50,
        W_pos_range=(0.2, 0.5),
        W_neg_range=(-0.5, -0.2),
        mean=0.0,
        std=2,
        s0_list=[5, 5, 5],           # Pass per-cluster edge counts
        graph_type="UR",              # UR ensures edges are in upper triangle
        seed=seed,
        permute=True                 # Disable permutation to maintain upper-triangular form
    )




    # 4) Hyperparameter grid Aug 20th
    param_grid = {
            'lambda1': [0.0001, 0.01, 0.1],     
            'lambda2': [0.1, 0.01, 0.001, 0.0001],
            'tau':     [0.1, 0.4, 0.7],      
            'rho1':    [0.1],                          # 1 value
            'rho2':    [0.05]                          # 1 value
    }
    # param_grid = {
    #         'lambda1': [0.0001, 0.001, 0.1],     
    #         'lambda2': [0.01, 0.0001, 0.000001,0.00000001],
    #         'tau':     [0.05, 0.3, 0.7],          
    #         'rho1':    [0.1],                          # 1 value
    #         'rho2':    [0.05]                          # 1 value
    # }

    # 5) Cross-validate
    top5_mle, top5_recon, top5_cov, cv_table = tune_hyperparameters(
        X_list,
        param_grid,
        n_folds=3,
        max_dc_iter=10,
        max_admm_iter=15,
        plot_graph=False
    )

    # 6) Save CV results and top5
    out_dir = os.path.join(base_out_dir, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    cv_table.to_csv(os.path.join(out_dir, "cv_results.csv"), index=False)
    for name, data in [('top5_mle', top5_mle), ('top5_recon', top5_recon), ('top5_cov', top5_cov)]:
        with open(os.path.join(out_dir, f"{name}.json"), 'w') as f:
            json.dump(data, f, indent=2)

    # 7) Evaluate best-for-each-metric and save summary
    summary = evaluate_best_for_all(
        out_dir,
        cv_table,
        X_list,
        W_list_gt,
        W_centers_gt,
        clusters_gt,
        label_truth
    )

    print(f"[seed {seed}] Experiment complete. Summary written to {out_dir}/summary.json")

if __name__ == "__main__":
    main()