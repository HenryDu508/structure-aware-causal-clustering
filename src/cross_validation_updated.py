import numpy as np
import pandas as pd
import os
import json
from itertools import product
from sklearn.model_selection import KFold
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from algorithm_updated import optimize_dc_admm



def evaluate_params(params, fold, X_train, X_val, max_dc_iter, max_admm_iter, plot_graph):
    lambda1, lambda2, tau, rho1, rho2 = params
    cluster_W, clusters, W_list, label_list, W_cluster_list = optimize_dc_admm(
        X_train, lambda1=lambda1, lambda2=lambda2, tau=tau, rho1=rho1, rho2=rho2,
        max_dc_iter=max_dc_iter, max_admm_iter=max_admm_iter, plot_graph=plot_graph, thres_value=0.01
    )
    return {
        "fold": fold, "lambda1": lambda1, "lambda2": lambda2, "tau": tau, "rho1": rho1, "rho2": rho2,
        "recon_error": compute_total_reconstruction_error(cluster_W, clusters, X_val)
    }


def tune_hyperparameters(
    X_list,
    param_grid,
    n_folds=3,
    max_dc_iter=10,
    max_admm_iter=15,
    plot_graph=False,
    cv_shuffle=True,
    cv_random_state=None,
):
    """Supports different m_i. Returns top5_recon, cv_table."""
    if not X_list:
        raise ValueError("X_list must be non-empty.")
    if plot_graph:
        print("Warning: plot_graph=True will produce plots for each fold/param.")

    folds_per_X = []
    for i, Xi in enumerate(X_list):
        if Xi.shape[0] < n_folds:
            raise ValueError(f"X_list[{i}] has only {Xi.shape[0]} rows, cannot do {n_folds}-fold CV.")
        kf = KFold(
            n_splits=n_folds,
            shuffle=cv_shuffle,
            random_state=cv_random_state if cv_shuffle else None,
        )
        folds_per_X.append(list(kf.split(np.arange(Xi.shape[0]))))

    param_names = list(param_grid.keys())
    param_combos = list(product(*param_grid.values()))
    records = []

    for fold in range(n_folds):
        X_tr = [Xi[folds[fold][0], :] for Xi, folds in zip(X_list, folds_per_X)]
        X_va = [Xi[folds[fold][1], :] for Xi, folds in zip(X_list, folds_per_X)]
        for combo in param_combos:
            records.append(evaluate_params(combo, fold, X_tr, X_va, max_dc_iter, max_admm_iter, plot_graph))

    df = pd.DataFrame(records)
    cv = df.groupby(param_names)[["recon_error"]].mean().reset_index().rename(columns={"recon_error": "recon_mean"})
    top5_recon = cv.nsmallest(5, "recon_mean")[param_names + ["recon_mean"]].rename(
        columns={"recon_mean": "mean_error"}
    ).to_dict("records")
    return top5_recon, cv






def compute_reconstruction_error(W_est: np.ndarray, X_i: np.ndarray) -> float:
    """
    Compute the reconstruction error (MSE per observation) for a single sample.

    Parameters:
        W_est (d×d array): Representative weight matrix for the cluster.
        X_i   (m_i×d array): Test observations for sample i.

    Returns: loss_i (float): Mean squared reconstruction error for that sample.
    """
    m_i, d = X_i.shape
    if m_i == 0:
        raise ValueError("X_i must have at least one observation.")
    if W_est.shape != (d, d):
        raise ValueError(f"W_est shape {W_est.shape} does not match d={d}.")
    residuals = X_i - X_i @ W_est
    # MSE per observation
    return float(np.sum(residuals**2) / m_i)


def compute_total_reconstruction_error(W_cluster, cluster_label, X_test):
    """
    Compute the average reconstruction error over all test samples.
    Iterates over each cluster k and each sample index i in cluster_label[k],
    calls compute_reconstruction_error_single(W_cluster[k], X_test[i]),
    and returns the mean over all valid samples.

    Parameters:
        W_cluster      : list of d×d representative weight matrices, one per cluster.
        cluster_label  : list of sets of sample indices (into X_test) for each cluster.
        X_test         : list of test data arrays, each of shape (m_i, d).

    Returns:
        total_reconstruction_error (float): Mean of all single-sample reconstruction errors.
    """
    N = len(X_test)

    errors = []
    for k, cluster in enumerate(cluster_label):
        for i in cluster:
            if not (0 <= i < N):
                raise IndexError(f"Invalid sample index {i} in cluster labels.")
            try:
                e_i = compute_reconstruction_error(W_cluster[k], X_test[i])
            except ValueError:
                continue
            errors.append(e_i)

    return float(np.mean(errors))


def compute_penalized_error(W, X_i):
    """
    Compute the penalized maximum likelihood objective for a single sample.

    Objective:
        (m_i * d / 2) * log(trace((I - W)^T (I - W) Σ̂)) + (log(m_i) / 2) * ||W||_0

    Parameters
    W : (d, d) ndarray: Candidate coefficient matrix.
    X_i : (m_i, d) ndarray test data matrix for one sample.

    Returns: The penalized MLE objective for this sample.
    """
    m_i, d = X_i.shape
    if m_i < 2:
        raise ValueError(f"Need at least 2 observations, got {m_i}")

    # Center data & compute sample covariance (unbiased)
    Xc = X_i - np.mean(X_i, axis=0, keepdims=True)
    Sigma_hat = (Xc.T @ Xc) / (m_i - 1)

    # Trace term
    I = np.eye(d)
    T = np.trace((I - W).T @ (I - W) @ Sigma_hat)
    if T <= 0:
        raise ValueError("Trace term must be positive to take log.")

    # Sparsity penalty
    lambda_val = np.log(m_i) / 2
    sparsity = lambda_val * np.count_nonzero(W)
    return float((m_i * d / 2) * np.log(T) + sparsity)

def compute_average_penalized_mle_error(W_cluster, cluster_label, X_test):
    """
    Compute the average penalized MLE objective across test samples.

    Parameters
    ----------
    W_cluster : list of (d, d) ndarrays
        Representative coefficient matrices, one per cluster.
    cluster_label : list of lists of int
        Indices of X_test assigned to each cluster.
    X_test : list of (m_i, d) ndarrays
        Test data matrices.

    Returns: Mean penalized MLE objective over all valid test samples.
    """
    errors = []
    N = len(X_test)
    for k, indices in enumerate(cluster_label):
        Wk = W_cluster[k]
        d = Wk.shape[0]
        if Wk.shape != (d, d):
            raise ValueError(f"W_cluster[{k}] must be square, got {Wk.shape}")

        for i in indices:
            if not (0 <= i < N):
                raise IndexError(f"Invalid index {i} for X_test of length {N}")

            X_i = X_test[i]
            if X_i.shape[1] != d:
                raise ValueError(f"X_test[{i}] has wrong dimension: {X_i.shape[1]} vs {d}")

            try:
                ei = compute_penalized_error(Wk, X_i)
            except ValueError as e:
                # Skip samples that cannot be evaluated
                continue
            errors.append(ei)

    return float(np.mean(errors))


def count_accuracy_with_skeleton(W_true, W_est, threshold=0.01):
    """
    Evaluate both:
    (1) Skeleton accuracy (undirected): whether any edge exists between i and j.
    (2) DAG accuracy (directed): whether edge direction is correctly predicted.

    Args:
        W_true (ndarray): Ground truth weight matrix (d x d)
        W_est (ndarray): Estimated weight matrix (d x d)
        threshold (float): Threshold for binarizing the weight matrix

    Returns:
        dict: skeleton and DAG accuracy metrics
    """
    # Binarize: 1 if edge weight > threshold, else 0
    B_true = (np.abs(W_true).round(3) > threshold).astype(int)
    B_est  = (np.abs(W_est ).round(3) > threshold).astype(int)
    d = B_true.shape[0]

    # ===== Part 1: Skeleton (undirected) accuracy =====
    skel_true = ((B_true + B_true.T) > 0).astype(int)
    skel_est  = ((B_est  + B_est.T ) > 0).astype(int)

    triu_mask = np.triu(np.ones((d, d), dtype=bool), k=1)
    true_flat = skel_true[triu_mask]
    est_flat  = skel_est[triu_mask]

    tp_skel = int(np.sum((true_flat == 1) & (est_flat == 1)))
    fp_skel = int(np.sum((true_flat == 0) & (est_flat == 1)))
    fn_skel = int(np.sum((true_flat == 1) & (est_flat == 0)))
    tn_skel = int(np.sum((true_flat == 0) & (est_flat == 0)))

    skeleton_tpr = tp_skel / (tp_skel + fn_skel) if (tp_skel + fn_skel) > 0 else 0.0
    skeleton_fdr = fp_skel / (tp_skel + fp_skel) if (tp_skel + fp_skel) > 0 else 0.0
    skeleton_tnr = tn_skel / (tn_skel + fp_skel) if (tn_skel + fp_skel) > 0 else 0.0
    skeleton_shd = fp_skel + fn_skel

    # ===== Part 2: DAG (directed) accuracy =====
    mask = ~np.eye(d, dtype=bool)
    true_flat_dir = B_true[mask]
    est_flat_dir  = B_est[mask]

    tp = int(np.sum((est_flat_dir == 1) & (true_flat_dir == 1)))
    fp = int(np.sum((est_flat_dir == 1) & (true_flat_dir == 0)))
    fn = int(np.sum((est_flat_dir == 0) & (true_flat_dir == 1)))
    tn = int(np.sum((est_flat_dir == 0) & (true_flat_dir == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    dag_fdr = 1 - precision
    dag_tpr = recall
    dag_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    dag_tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    dag_shd = fp + fn
    dag_nnz = tp + fp

    return {
        # Skeleton metrics
        'skeleton_tpr': skeleton_tpr,
        'skeleton_fdr': skeleton_fdr,
        'skeleton_tnr': skeleton_tnr,
        'skeleton_shd': skeleton_shd,

        # Directed (DAG) metrics
        'dag_fdr': dag_fdr,
        'dag_tpr': dag_tpr,
        'dag_fpr': dag_fpr,
        'dag_tnr': dag_tnr,
        'dag_shd': dag_shd,
        'dag_nnz': dag_nnz
    }

def average_skeleton_accuracy(W_true_list, W_est_list, threshold=0.01):
    """
    Compute average (macro‑avg) of count_accuracy over lists of weight matrices.

    Args:
        W_true_list (list of np.ndarray): ground‑truth weight matrices.
        W_est_list  (list of np.ndarray): estimated weight matrices.
        threshold (float): binarization cutoff passed to count_accuracy.

    Returns:
        dict: average of each metric across all pairs:
            'fdr', 'tpr', 'fpr', 'shd', 'nnz'
    """
    metrics = [
        count_accuracy_with_skeleton(Wt, We, threshold)
        for Wt, We in zip(W_true_list, W_est_list)
    ]

    avg = { key: float(np.mean([m[key] for m in metrics]))
            for key in metrics[0].keys() }
    return avg


def clustering_overall_metrics(y_true, y_pred, average_method='arithmetic'):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "AMI": adjusted_mutual_info_score(y_true, y_pred, average_method=average_method),
        "Homogeneity": homogeneity_score(y_true, y_pred),
        "Completeness": completeness_score(y_true, y_pred),
        "V_measure": v_measure_score(y_true, y_pred),
        "n_true_clusters": int(np.unique(y_true).size),
        "n_pred_clusters": int(np.unique(y_pred).size),
    }


