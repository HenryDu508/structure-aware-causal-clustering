#!/usr/bin/env python3
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse, random, json
import numpy as np, pandas as pd

from data_generation import generate_clustered_data
from cross_validation_updated import (
    tune_hyperparameters, compute_total_reconstruction_error,
    average_skeleton_accuracy, clustering_overall_metrics
)
from algorithm_updated import optimize_dc_admm, complete_linkage_clusters
from NOTEAR import (
    run_notear_experiment_pooled,
    run_notear_experiment_individual,
    run_notear_experiment_cluster,
)


def _to_py(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.generic,)): return x.item()
    if isinstance(x, dict): return {k: _to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(_to_py(v) for v in x)
    return x

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_py(obj), f, ensure_ascii=False, indent=2)

def run_dag_dc_admm_experiment(X_list, W_list_gt, W_centers_gt, clusters_gt, label_truth, seed,
                               lambda1_grid=(0.0001, 0.001, 0.01, 0.1),
                               lambda2_grid=(0.01, 0.001, 0.0001, 0.00001),
                               tau_grid=(0.05, 0.1, 0.4, 0.7)):
    param_grid = {
        "lambda1": list(lambda1_grid),
        "lambda2": list(lambda2_grid),
        "tau": list(tau_grid),
        "rho1": [0.1],
        "rho2": [0.05],
    }

    top5_recon, cv_table = tune_hyperparameters(X_list, param_grid,
        n_folds=3, max_dc_iter=10, max_admm_iter=15, plot_graph=False, cv_shuffle=True, cv_random_state=seed)

    row = cv_table.loc[cv_table["recon_mean"].idxmin()]
    params = {k: float(row[k]) for k in ["lambda1", "lambda2", "tau", "rho1", "rho2"]}

    Wc_est, clusters_est, W_list_final, label_est, W_cluster_est_list = optimize_dc_admm(
        X_list, **params, max_dc_iter=10, max_admm_iter=15, plot_graph=False, thres_value=0.01
    )

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    sweep_records = [
        {"threshold": float(t), "skeleton_accuracy": _to_py(average_skeleton_accuracy(W_list_gt, W_cluster_est_list, threshold=t))}
        for t in thresholds
    ]

    return {
        "best_params": params,
        "selected_metric": "recon_mean",
        "train_thres_value": 0.01,
        "top5_recon": top5_recon,
        "cv_table": _to_py(cv_table.to_dict("records")),
        "skeleton_accuracy@0.01": average_skeleton_accuracy(W_list_gt, W_cluster_est_list, threshold=0.01),
        "skeleton_threshold_sweep": sweep_records,
        "recon_error_gt": compute_total_reconstruction_error(W_centers_gt, clusters_gt, X_list),
        "recon_error_est": compute_total_reconstruction_error(Wc_est, clusters_est, X_list),
        "clustering_accuracy": clustering_overall_metrics(label_truth, label_est),
        "Wc_est": _to_py(Wc_est),
        "clusters_est": _to_py([sorted(map(int, list(c))) for c in clusters_est]),
        "W_list_final": _to_py(W_list_final),
        "label_est": _to_py(label_est),
        "W_cluster_est_list": _to_py(W_cluster_est_list),
    }

def run_two_step_experiment(individual_results, label_truth, tau):
    """Two-step baseline: individual NOTEARS DAGs, then complete-linkage clustering cut at tau."""
    W = np.asarray(individual_results["W_notear_list"], dtype=float)
    dist = np.linalg.norm(W[:, None] - W[None, :], ord="fro", axis=(2, 3))
    np.fill_diagonal(dist, 0.0)
    cluster_sets, labels = complete_linkage_clusters(dist, tau)
    labels = np.asarray(labels, dtype=int)
    clusters = [sorted(int(i) for i in c) for c in cluster_sets]
    consensus = np.stack([W[c].mean(axis=0) for c in clusters], axis=0)
    return {
        "method": "individual_notears_then_complete_linkage",
        "tau": tau,
        "selected_notears_lambda1": float(individual_results["best_alpha"]),
        "hierarchical_clustering": {
            "selected_k": len(clusters),
            "label_est": labels,
            "clusters_est_zero_based_subject_indices": clusters,
            "pairwise_distance_matrix": dist,
            "cluster_consensus_matrices": consensus,
            "cluster_consensus_matrix_by_subject": np.stack([consensus[l - 1] for l in labels], axis=0),
            "clustering_accuracy": clustering_overall_metrics(label_truth, labels),
        },
    }

def main():
    p = argparse.ArgumentParser(description="Run experiment for one seed")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--setting_name", type=str, required=True)

    p.add_argument("--total_samples", type=int, default=50)
    p.add_argument("--cluster_proportions", type=float, nargs="+", default=[0.6, 0.4])
    p.add_argument("--n_vars", type=int, default=5)
    
    # Accept m as a list or a single int. Default to 300 for backward compatibility.
    p.add_argument("--m", type=int, nargs="+", default=[300])
    
    p.add_argument("--std", type=float, default=0.5)
    p.add_argument("--s0_list", type=int, nargs="+", default=[5, 5])
    # Hyperparameter grids for DAG-DC-ADMM. Defaults give a 64-combination grid.
    # Main two-cluster tables: --lambda2_grid 0.01 0.001 0.0001 0.00001 0.1 (80 combinations).
    # Higher-dimensional setting (d=15): --tau_grid 0.15 0.3 1.2 2.1.
    p.add_argument("--lambda1_grid", type=float, nargs="+", default=[0.0001, 0.001, 0.01, 0.1])
    p.add_argument("--lambda2_grid", type=float, nargs="+", default=[0.01, 0.001, 0.0001, 0.00001])
    p.add_argument("--tau_grid", type=float, nargs="+", default=[0.05, 0.1, 0.4, 0.7])
    p.add_argument("--baseline_lambda1_grid", type=float, nargs="+", default=[0.0001, 0.001, 0.01, 0.1],
                   help="L1 penalty grid for the NOTEARS baselines.")
    p.add_argument("--two_step_tau", type=float, default=0.7,
                   help="Dendrogram cut height for the two-step baseline.")
    p.add_argument("--methods", nargs="+", default=["all"],
                   choices=["all", "dag_dc_admm", "pooled", "individual", "oracle", "two_step"],
                   help="Methods to run (default: all).")
    args = p.parse_args()

    # Set reproducibility seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    # --- HANDLE FLEXIBLE M LOGIC ---
    # If m is set to [0], trigger the "Random Length Mode"
    # This generates a reproducible list of lengths for each subject
    if args.m == [0] or args.m == 0:
        # Range: 50 to 300, Step: 10
        m_choices = np.arange(50, 310, 10) 
        # Generate a unique length for each of the total_samples
        m_val = np.random.choice(m_choices, size=args.total_samples, replace=True).tolist()
        print(f"[Seed {seed}] Random Mode Triggered: Generated unique lengths for {args.total_samples} samples.")
    else:
        # Standard mode: Extract the single integer if passed as a list, otherwise keep as is
        m_val = args.m[0] if isinstance(args.m, list) and len(args.m) == 1 else args.m
        print(f"[Seed {seed}] Standard Mode: Using fixed length m={m_val}")

    CV_SHUFFLE = True
    CV_RANDOM_STATE = seed

    setting_out_dir = os.path.join(args.out_dir, args.setting_name)
    out_dir = os.path.join(setting_out_dir, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    # Generate data using the m_val (either a single int or the generated list)
    X_list, W_list_gt, W_centers_gt, clusters_gt, label_truth = generate_clustered_data(
        total_samples=args.total_samples, 
        cluster_proportions=args.cluster_proportions,
        n_vars=args.n_vars, 
        m=m_val, 
        W_pos_range=(0.2, 0.5), 
        W_neg_range=(-0.5, -0.2),
        mean=0.0, 
        std=args.std, 
        s0_list=args.s0_list, 
        graph_type="UR", 
        seed=seed, 
        permute=True
    )

    methods = {"dag_dc_admm", "pooled", "individual", "oracle", "two_step"} if "all" in args.methods else set(args.methods)
    alpha_grid = args.baseline_lambda1_grid
    cv = dict(n_folds=3, cv_shuffle=CV_SHUFFLE, cv_random_state=CV_RANDOM_STATE)
    all_results = {
        "seed": seed,
        "data_config": {
            "total_samples": args.total_samples,
            "cluster_proportions": args.cluster_proportions,
            "n_vars": args.n_vars,
            "m": m_val,
            "std": args.std,
            "s0_list": args.s0_list,
            "graph_type": "UR",
            "weight_positive_range": [0.2, 0.5],
            "weight_negative_range": [-0.5, -0.2],
            "mean": 0.0,
            "permute": True,
        },
    }

    # DAG-DC-ADMM
    if "dag_dc_admm" in methods:
        dc_results = run_dag_dc_admm_experiment(X_list, W_list_gt, W_centers_gt, clusters_gt, label_truth, seed,
                                                lambda1_grid=args.lambda1_grid, lambda2_grid=args.lambda2_grid,
                                                tau_grid=args.tau_grid)
        save_json(os.path.join(out_dir, "dag_dc_admm_results.json"), dc_results)
        all_results["dc_admm_file"] = "dag_dc_admm_results.json"

    # NOTEARS baselines: Population, Individual, Oracle
    if "pooled" in methods:
        all_results["notear_pooled"] = run_notear_experiment_pooled(
            X_list, W_list_gt, W_centers_gt, clusters_gt, seed, args.std, alpha_grid, **cv)
        save_json(os.path.join(out_dir, "summary_notear_pooled_CV.json"), all_results["notear_pooled"])
    if "individual" in methods or "two_step" in methods:
        individual_results = run_notear_experiment_individual(
            X_list, W_list_gt, W_centers_gt, clusters_gt, seed, args.std, alpha_grid, **cv)
        if "individual" in methods:
            all_results["notear_individual"] = individual_results
            save_json(os.path.join(out_dir, "summary_notear_individual_CV.json"), individual_results)
    if "oracle" in methods:
        all_results["notear_cluster"] = run_notear_experiment_cluster(
            X_list, W_list_gt, W_centers_gt, clusters_gt, label_truth, seed, args.std, alpha_grid, **cv)
        save_json(os.path.join(out_dir, "summary_notear_cluster_CV.json"), all_results["notear_cluster"])

    # Two-step baseline: Individual NOTEARS + complete-linkage clustering
    if "two_step" in methods:
        two_step = run_two_step_experiment(individual_results, label_truth, args.two_step_tau)
        save_json(os.path.join(out_dir, "summary_notear_individual_hierarchical.json"),
                  {**two_step, "seed": seed, "data_config": all_results["data_config"],
                   "notears_individual": individual_results})

    save_json(os.path.join(out_dir, "summary_all_methods.json"), all_results)

    print(f"[seed {seed}] Experiment complete. Outputs written to {out_dir}")

if __name__ == "__main__":
    main()