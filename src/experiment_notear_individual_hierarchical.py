#!/usr/bin/env python3
"""Two-step baseline: individual NOTEARS followed by complete-linkage clustering."""

import argparse
import json
import os
import random
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import scipy

from algorithm_updated import complete_linkage_clusters
from cross_validation_updated import clustering_overall_metrics
from data_generation import generate_clustered_data
from NOTEAR import run_notear_experiment_individual


def _to_py(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_py(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_py(item) for item in value]
    return value


def save_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_to_py(value), file, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate one NOTEARS DAG per subject and apply complete-linkage "
            "hierarchical clustering to their pairwise Frobenius distances."
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--setting_name", type=str, required=True)
    parser.add_argument("--total_samples", type=int, default=50)
    parser.add_argument(
        "--cluster_proportions", type=float, nargs="+", default=[0.6, 0.4]
    )
    parser.add_argument("--n_vars", type=int, default=5)
    parser.add_argument("--m", type=int, nargs="+", default=[300])
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--s0_list", type=int, nargs="+", default=[5, 5])
    parser.add_argument(
        "--alpha_grid",
        "--lambda1_grid",
        dest="alpha_grid",
        type=float,
        nargs="+",
        default=[1e-4, 1e-3, 1e-2, 1e-1],
        help=(
            "Candidate NOTEARS L1 penalties (lambda1; called alpha internally). "
            "The value minimizing mean validation reconstruction error is used."
        ),
    )
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument(
        "--tau",
        type=float,
        default=0.7,
        help=(
            "Distance at which to cut the complete-linkage dendrogram. To make "
            "the comparison exact, pass the tau selected for the proposed method."
        ),
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="summary_notear_individual_hierarchical.json",
    )
    return parser.parse_args()


def resolve_m(args):
    if args.m == [0] or args.m == 0:
        m_choices = np.arange(50, 310, 10)
        return np.random.choice(
            m_choices, size=args.total_samples, replace=True
        ).tolist()
    return args.m[0] if len(args.m) == 1 else args.m


def validate_args(args):
    if args.total_samples < 2:
        raise ValueError("Hierarchical clustering requires at least two subjects.")
    if args.n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if args.tau < 0:
        raise ValueError("tau must be nonnegative.")
    if not args.alpha_grid or any(alpha < 0 for alpha in args.alpha_grid):
        raise ValueError("alpha_grid must contain nonnegative values.")
    if len(args.cluster_proportions) != len(args.s0_list):
        raise ValueError("cluster_proportions and s0_list must have equal length.")
    if not np.isclose(sum(args.cluster_proportions), 1.0):
        raise ValueError("cluster_proportions must sum to one.")


def pairwise_frobenius_distances(weight_matrices):
    """Return distances d_ij = ||W_i - W_j||_F."""
    weight_matrices = np.asarray(weight_matrices, dtype=float)
    if weight_matrices.ndim != 3:
        raise ValueError("weight_matrices must have shape (N, q, q).")
    n_subjects, n_rows, n_cols = weight_matrices.shape
    if n_rows != n_cols:
        raise ValueError("Every NOTEARS weight matrix must be square.")
    if not np.isfinite(weight_matrices).all():
        raise ValueError("NOTEARS weight matrices contain non-finite values.")

    differences = weight_matrices[:, None, :, :] - weight_matrices[None, :, :, :]
    distance_matrix = np.linalg.norm(differences, ord="fro", axis=(2, 3))
    np.fill_diagonal(distance_matrix, 0.0)
    if distance_matrix.shape != (n_subjects, n_subjects):
        raise RuntimeError("Unexpected pairwise distance matrix shape.")
    return distance_matrix


def hierarchical_cluster_weight_matrices(weight_matrices, tau):
    """Apply the same complete-linkage cutoff rule as the proposed method."""
    weight_matrices = np.asarray(weight_matrices, dtype=float)
    distance_matrix = pairwise_frobenius_distances(weight_matrices)
    cluster_sets, labels_one_based = complete_linkage_clusters(distance_matrix, tau)
    labels_one_based = np.asarray(labels_one_based, dtype=int)
    clusters = [sorted(int(index) for index in cluster) for cluster in cluster_sets]
    consensus_matrices = np.stack(
        [weight_matrices[cluster].mean(axis=0) for cluster in clusters], axis=0
    )
    consensus_by_subject = np.stack(
        [consensus_matrices[label - 1] for label in labels_one_based], axis=0
    )
    return (
        labels_one_based,
        clusters,
        distance_matrix,
        consensus_matrices,
        consensus_by_subject,
    )


def main():
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    total_start = time.perf_counter()

    m_value = resolve_m(args)
    X_list, W_list_gt, W_centers_gt, clusters_gt, label_truth = (
        generate_clustered_data(
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
    )

    notears_start = time.perf_counter()
    notears_result = run_notear_experiment_individual(
        X_list,
        W_list_gt,
        W_centers_gt,
        clusters_gt,
        args.seed,
        args.std,
        args.alpha_grid,
        n_folds=args.n_folds,
        cv_shuffle=True,
        cv_random_state=args.seed,
    )
    notears_seconds = time.perf_counter() - notears_start

    weight_matrices = np.asarray(notears_result["W_notear_list"], dtype=float)
    clustering_start = time.perf_counter()
    (
        labels_one_based,
        clusters_est,
        distance_matrix,
        consensus_matrices,
        consensus_by_subject,
    ) = hierarchical_cluster_weight_matrices(weight_matrices, args.tau)
    clustering_seconds = time.perf_counter() - clustering_start

    selected_k = len(clusters_est)
    clustering_result = {
        "method": "complete_linkage_hierarchical_clustering",
        "distance_metric": "frobenius_between_weighted_adjacency_matrices",
        "linkage": "complete",
        "cut_criterion": "distance",
        "tau": args.tau,
        "selected_k": selected_k,
        "label_est": labels_one_based,
        "clusters_est_zero_based_subject_indices": clusters_est,
        "pairwise_distance_matrix": distance_matrix,
        "cluster_consensus_matrices": consensus_matrices,
        "cluster_consensus_matrix_by_subject": consensus_by_subject,
        "clustering_accuracy": clustering_overall_metrics(
            label_truth, labels_one_based
        ),
    }

    total_seconds = time.perf_counter() - total_start
    payload = {
        "seed": args.seed,
        "method": "individual_notears_then_complete_linkage_hierarchical",
        "data_config": {
            "total_samples": args.total_samples,
            "cluster_proportions": args.cluster_proportions,
            "n_vars": args.n_vars,
            "m": m_value,
            "std": args.std,
            "s0_list": args.s0_list,
            "graph_type": "UR",
            "weight_positive_range": [0.2, 0.5],
            "weight_negative_range": [-0.5, -0.2],
            "mean": 0.0,
            "permute": True,
        },
        "selected_notears_lambda1": float(notears_result["best_alpha"]),
        "notears_individual": notears_result,
        "hierarchical_clustering": clustering_result,
        "runtime_seconds": {
            "notears_individual_including_cv": notears_seconds,
            "hierarchical_clustering": clustering_seconds,
            "total_including_data_generation": total_seconds,
        },
        "software_versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "evaluation_notes": {
            "notears_lambda1_internal_name": "alpha",
            "notears_lambda1_selected_by": "mean validation reconstruction error",
            "ground_truth_labels_used_only_for_final_evaluation": True,
            "cluster_count_determined_by_tau_cutoff": True,
            "hierarchical_rule_matches_proposed_method": True,
            "pairwise_distances_use_individual_notears_estimates": True,
            "consensus_matrices_are_cluster_means_not_refitted_dags": True,
        },
    }

    out_dir = os.path.join(args.out_dir, args.setting_name, f"seed_{args.seed}")
    output_path = os.path.join(out_dir, args.output_filename)
    save_json(output_path, payload)
    print(
        f"[seed {args.seed}] Individual NOTEARS + complete linkage complete: "
        f"tau={args.tau:g}, selected K={selected_k}, output={output_path}"
    )


if __name__ == "__main__":
    main()
