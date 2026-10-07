#!/usr/bin/env python3
"""
A small end-to-end example of DAG-DC-ADMM on simulated data.

Main simulation setting of the paper (n = 50, m = 300, sigma = 1, two clusters with
their own 5-node DAGs), one seed, and a reduced 2 x 2 x 2 CV grid. The script
1. generates the data,
2. selects (lambda1, lambda2, tau) by 3-fold CV on a small grid,
3. refits on all data with the selected values,
4. prints the estimated clusters and the clustering and graph-recovery metrics.

Run from the repository root:
    python examples/two_cluster_example.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from dagdc.data_generation import generate_clustered_data
from dagdc.cross_validation import (
    tune_hyperparameters, compute_total_reconstruction_error,
    average_skeleton_accuracy, clustering_overall_metrics,
)
from dagdc.dc_admm import optimize_dc_admm

SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# 1. Data: 50 subjects (60% / 40%), 300 measurements each, 5 variables, 5 edges per cluster DAG
X_list, W_list_gt, W_centers_gt, clusters_gt, label_truth = generate_clustered_data(
    total_samples=50,
    cluster_proportions=[0.6, 0.4],
    n_vars=5,
    m=300,
    W_pos_range=(0.2, 0.5),
    W_neg_range=(-0.5, -0.2),
    mean=0.0,
    std=1.0,
    s0_list=[5, 5],
    graph_type="UR",
    seed=SEED,
    permute=True,
)

# 2. Cross-validation over a small grid (the paper uses a 4 x 5 x 4 grid)
param_grid = {
    "lambda1": [0.01, 0.1],
    "lambda2": [0.001, 0.0001],
    "tau": [0.4, 0.7],
    "rho1": [0.1],
    "rho2": [0.05],
}
_, cv_table = tune_hyperparameters(
    X_list, param_grid, n_folds=3, max_dc_iter=10, max_admm_iter=15,
    plot_graph=False, cv_shuffle=True, cv_random_state=SEED,
)
best = cv_table.loc[cv_table["recon_mean"].idxmin()]
params = {k: float(best[k]) for k in ["lambda1", "lambda2", "tau", "rho1", "rho2"]}
print("Selected:", {k: params[k] for k in ["lambda1", "lambda2", "tau"]},
      f"(CV reconstruction error {best['recon_mean']:.4f})")

# 3. Refit on all data
Wc_est, clusters_est, W_list_est, label_est, W_cluster_est_list = optimize_dc_admm(
    X_list, **params, max_dc_iter=10, max_admm_iter=15, plot_graph=False, thres_value=0.01,
)

# 4. Results
print("True labels:     ", list(label_truth))
print("Estimated labels:", list(label_est))
clu = clustering_overall_metrics(label_truth, label_est)
print(f"ARI {clu['ARI']:.3f}, AMI {clu['AMI']:.3f}, clusters found {clu['n_pred_clusters']}")
acc = average_skeleton_accuracy(W_list_gt, W_cluster_est_list, threshold=0.01)
print("Graph recovery (subject average):", {k: round(v, 3) for k, v in acc.items()})
print(f"Reconstruction error: estimated {compute_total_reconstruction_error(Wc_est, clusters_est, X_list):.4f}, "
      f"true graphs {compute_total_reconstruction_error(W_centers_gt, clusters_gt, X_list):.4f}")
np.set_printoptions(precision=2, suppress=True)
for k, W in enumerate(Wc_est):
    print(f"\nEstimated DAG of cluster {k + 1}:\n{np.asarray(W)}")
