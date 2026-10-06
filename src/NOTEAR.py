"""
NOTEAR.py

This module provides functions to learn DAG structures using the NO TEARS approach:
- notears_linear_raw: solves the constrained optimization for linear SEMs.
- from_numpy_lasso2: wrapper to learn structure with L1 penalty.
"""

import numpy as np
from collections import defaultdict
from scipy.optimize import minimize
import scipy.optimize as sopt
import scipy.linalg as slin
from scipy.special import expit as sigmoid
import warnings
from sklearn.model_selection import KFold
from cross_validation_updated import (
    compute_reconstruction_error,
    compute_total_reconstruction_error,
    average_skeleton_accuracy,
)

def _to_py(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.generic,)): return x.item()
    if isinstance(x, dict): return {k: _to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return type(x)(_to_py(v) for v in x)
    return x
    
def notears_linear_raw(X, lambda1, loss_type, max_iter=100, h_tol=1e-8,
                       rho_max=1e+16, w_threshold=0.3):
    """Solve min_W L(W; X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian."""
    def _loss(W):
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = -1.0 / X.shape[0] * X.T @ R
        elif loss_type == 'logistic':
            loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        elif loss_type == 'poisson':
            S = np.exp(M)
            loss = 1.0 / X.shape[0] * (S - X * M).sum()
            G_loss = 1.0 / X.shape[0] * X.T @ (S - X)
        else:
            raise ValueError('unknown loss type')
        return loss, G_loss

    def _h(W):
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h_val, G_h = _h(W)
        obj = loss + 0.5 * rho * h_val**2 + alpha * h_val + lambda1 * np.abs(w).sum()
        G_smooth = G_loss + (rho * h_val + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est = np.zeros(2 * d * d)
    rho, alpha, h_val = 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2)
            for i in range(d) for j in range(d)]

    for _ in range(max_iter):
        while rho < rho_max:
            sol = minimize(_func, w_est, method='SLSQP', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h_val:
                rho *= 10
            else:
                break
        w_est, h_val = w_new, h_new
        alpha += rho * h_val
        if h_val <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est)
    # Threshold small weights
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def from_numpy_lasso2(
    X,
    beta,
    max_iter=100,
    h_tol=1e-8,
    w_threshold=0.0,
    tabu_edges=None,
    tabu_parent_nodes=None,
    tabu_child_nodes=None,
):
    """Learn a StructureModel using NOTEARS with L1 regularization."""
    n, d = X.shape
    bnds = []
    for i in range(d):
        for j in range(d):
            if i == j or (tabu_edges and (i, j) in tabu_edges)                or (tabu_parent_nodes and i in tabu_parent_nodes)                or (tabu_child_nodes and j in tabu_child_nodes):
                bnds.append((0, 0))
            else:
                bnds.append((0, None))
    bnds = bnds * 2

    return _learn_structure_lasso(X, beta, bnds, max_iter, h_tol, w_threshold)


def _learn_structure_lasso(X, beta, bnds, max_iter=100, h_tol=1e-8, w_threshold=0.0):
    n, d = X.shape

    def _h(W):
        return np.trace(slin.expm(W * W)) - d

    def _adj(w_vec):
        return (w_vec[:d**2] - w_vec[d**2:]).reshape([d, d])

    def _func(w_vec):
        W = _adj(w_vec)
        loss = 0.5 / n * np.square(np.linalg.norm(X.dot(np.eye(d) - W), 'fro'))
        h_val = _h(W)
        return loss + 0.5 * rho * h_val**2 + alpha * h_val + beta * w_vec.sum()

    def _grad(w_vec):
        W = _adj(w_vec)
        loss_grad = -1.0 / n * X.T.dot(X).dot(np.eye(d) - W)
        expm_h = slin.expm(W * W)
        h_val = np.trace(expm_h) - d
        grad_h = expm_h.T * W * 2

        obj_grad = loss_grad + (rho * h_val + alpha) * grad_h
        l1_grad = beta * np.ones(d**2)

        grad = np.zeros(2 * d**2)
        grad[:d**2] = obj_grad.flatten() + l1_grad
        grad[d**2:] = -obj_grad.flatten() + l1_grad
        return grad

    w_est = np.zeros(2 * d**2)
    rho, alpha, h_val = 1.0, 0.0, np.inf

    for n_iter in range(max_iter):
        while rho < 1e20:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=_grad, bounds=bnds)
            w_new = sol.x
            h_new = _h(_adj(w_new))
            if h_new > 0.25 * h_val:
                rho *= 10
                continue
            w_est, h_val = w_new, h_new
            break

        alpha += rho * h_val
        if h_val <= h_tol:
            break
        if n_iter == max_iter - 1 and h_val > h_tol:
            warnings.warn("Failed to converge. Consider increasing max_iter.")

    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

def compute_notear_reconstruction_error(W_list: np.ndarray, X_list: list[np.ndarray]) -> float:
    """
    Compute average reconstruction error using per-sample NO-TEARS initialized weights.

    Parameters:
        W_list : (n × d × d) ndarray, one W matrix per sample
        X_list : list of (m_i × d) arrays, one for each sample

    Returns:
        float : mean reconstruction error across all samples
    """
    if len(W_list) != len(X_list):
        raise ValueError(f"Length mismatch: W_list has {len(W_list)} elements, X_list has {len(X_list)}")

    errors = []
    for W_i, X_i in zip(W_list, X_list):
        if X_i.shape[0] == 0:
            continue
        try:
            err = compute_reconstruction_error(W_i, X_i)
            errors.append(err)
        except ValueError:
            continue

    return float(np.mean(errors))

def generate_notear_pooled(X_list, alpha=0.05, round_decimals=3):
    """
    Population mode: Stack all samples to estimate a single global W.
    Returns an array of shape (n, d, d) where every sample shares the same W.
    """
    n = len(X_list)
    if n == 0:
        raise ValueError("X_list must contain at least one sample")

    # Vertical stack all data to estimate a single W
    X_vert = np.vstack(X_list)
    
    # Estimate W from the stacked data
    W_single = from_numpy_lasso2(X_vert, alpha)

    # Round entries
    if round_decimals is not None:
        W_single = np.round(W_single, round_decimals)

    # Use np.tile to replicate the single W into shape (n, d, d)
    # This is more memory-efficient than a list comprehension
    return np.tile(W_single[np.newaxis, :, :], (n, 1, 1))


def generate_notear_per_sample(X_list, alpha=0.05, round_decimals=3):
    """
    Sample mode: Apply Lasso independently to each sample in X_list.
    Returns an array of shape (n, d, d).
    """
    W_list = []
    for i, X_i in enumerate(X_list):
        if X_i.ndim != 2 or X_i.shape[0] == 0:
            raise ValueError(f"Sample {i} must be non-empty 2D array, got shape {X_i.shape}")
        
        # Estimate W for the individual sample
        W_i = from_numpy_lasso2(X_i, alpha)
        
        if round_decimals is not None:
            W_i = np.round(W_i, round_decimals)

        W_list.append(W_i)

    return np.stack(W_list, axis=0)


def generate_notear_per_cluster(X_list, cluster_label_list, alpha=0.05, round_decimals=3):
    """
    Cluster mode: Fit one W per unique cluster label using stacked data of that cluster.
    Returns an array of shape (n, d, d) mapping each sample to its cluster's W.
    """
    n = len(X_list)
    if len(cluster_label_list) != n:
        raise ValueError("Length of cluster_label_list must match X_list")

    # Group sample indices by cluster ID
    cluster_to_indices = defaultdict(list)
    for i, label in enumerate(cluster_label_list):
        cluster_to_indices[label].append(i)

    # Compute W for each unique cluster once to avoid redundant calculations
    cluster_W_map = {}
    for label, indices in cluster_to_indices.items():
        X_stack = np.vstack([X_list[i] for i in indices])
        
        W_c = from_numpy_lasso2(X_stack, alpha)
        
        if round_decimals is not None:
            W_c = np.round(W_c, round_decimals)
            
        cluster_W_map[label] = W_c

    # Map each sample back to its respective cluster's estimated W
    W_final = np.stack([cluster_W_map[label] for label in cluster_label_list], axis=0)
    return W_final

def run_notear_experiment_pooled(X_list, W_list_gt, W_centers_gt, clusters_gt, seed, variance, alpha_grid, n_folds=3, cv_shuffle=True, cv_random_state=None):
    folds_per_X = []
    for i, Xi in enumerate(X_list):
        if Xi.shape[0] < n_folds:
            raise ValueError(f"X_list[{i}] has only {Xi.shape[0]} rows, cannot do {n_folds}-fold CV.")
        kf = KFold( n_splits=n_folds, shuffle=cv_shuffle, random_state=cv_random_state if cv_shuffle else None,
)
        folds_per_X.append(list(kf.split(np.arange(Xi.shape[0]))))

    fold_errors_by_alpha = {a: [] for a in alpha_grid}
    for fold_id in range(n_folds):
        X_train = [Xi[folds[fold_id][0], :] for Xi, folds in zip(X_list, folds_per_X)]
        X_val = [Xi[folds[fold_id][1], :] for Xi, folds in zip(X_list, folds_per_X)]
        for a in alpha_grid:
            W_list_cv = generate_notear_pooled(X_train, alpha=a, round_decimals=3)
            fold_errors_by_alpha[a].append(float(compute_notear_reconstruction_error(W_list_cv, X_val)))

    mean_err_by_alpha = {a: float(np.mean(v)) for a, v in fold_errors_by_alpha.items()}
    best_alpha = min(mean_err_by_alpha, key=mean_err_by_alpha.get)
    W_notear_list = generate_notear_pooled(X_list, alpha=best_alpha, round_decimals=3)

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    skeleton_acc_by_thresh = {str(t): _to_py(average_skeleton_accuracy(W_list_gt, W_notear_list, threshold=float(t))) for t in thresholds}

    return {
    "seed": seed,
    "mode": "pooled",
    "std": float(variance),
    "alpha_grid": [float(a) for a in alpha_grid],
    "cv_fold_errors": {str(a): [float(x) for x in fold_errors_by_alpha[a]] for a in alpha_grid},
    "cv_mean_error_by_alpha": {str(a): float(v) for a, v in mean_err_by_alpha.items()},
    "best_alpha": float(best_alpha),
    "eval_thresholds": [float(t) for t in thresholds],
    "skeleton_accuracy_by_thresh": skeleton_acc_by_thresh,
    "recon_error_gt": _to_py(compute_total_reconstruction_error(W_centers_gt, clusters_gt, X_list)),
    "recon_error_est": _to_py(compute_notear_reconstruction_error(W_notear_list, X_list)),
    "W_notear_list": _to_py(W_notear_list),
}

def run_notear_experiment_individual(X_list, W_list_gt, W_centers_gt, clusters_gt, seed, variance, alpha_grid, n_folds=3, cv_shuffle=True, cv_random_state=None):
    folds_per_X = []
    for i, Xi in enumerate(X_list):
        if Xi.shape[0] < n_folds:
            raise ValueError(f"X_list[{i}] has only {Xi.shape[0]} rows, cannot do {n_folds}-fold CV.")
        kf = KFold( n_splits=n_folds, shuffle=cv_shuffle, random_state=cv_random_state if cv_shuffle else None,
)
        folds_per_X.append(list(kf.split(np.arange(Xi.shape[0]))))

    fold_errors_by_alpha = {a: [] for a in alpha_grid}
    for fold_id in range(n_folds):
        X_train = [Xi[folds[fold_id][0], :] for Xi, folds in zip(X_list, folds_per_X)]
        X_val = [Xi[folds[fold_id][1], :] for Xi, folds in zip(X_list, folds_per_X)]
        for a in alpha_grid:
            W_list_cv = generate_notear_per_sample(X_train, alpha=a, round_decimals=3)
            fold_errors_by_alpha[a].append(float(compute_notear_reconstruction_error(W_list_cv, X_val)))

    mean_err_by_alpha = {a: float(np.mean(v)) for a, v in fold_errors_by_alpha.items()}
    best_alpha = min(mean_err_by_alpha, key=mean_err_by_alpha.get)
    W_notear_list = generate_notear_per_sample(X_list, alpha=best_alpha, round_decimals=3)

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    skeleton_acc_by_thresh = {str(t): _to_py(average_skeleton_accuracy(W_list_gt, W_notear_list, threshold=float(t))) for t in thresholds}

    return {
    "seed": seed,
    "mode": "individual",
    "std": float(variance),
    "alpha_grid": [float(a) for a in alpha_grid],
    "cv_fold_errors": {str(a): [float(x) for x in fold_errors_by_alpha[a]] for a in alpha_grid},
    "cv_mean_error_by_alpha": {str(a): float(v) for a, v in mean_err_by_alpha.items()},
    "best_alpha": float(best_alpha),
    "eval_thresholds": [float(t) for t in thresholds],
    "skeleton_accuracy_by_thresh": skeleton_acc_by_thresh,
    "recon_error_gt": _to_py(compute_total_reconstruction_error(W_centers_gt, clusters_gt, X_list)),
    "recon_error_est": _to_py(compute_notear_reconstruction_error(W_notear_list, X_list)),
    "W_notear_list": _to_py(W_notear_list),
}


def run_notear_experiment_cluster(X_list, W_list_gt, W_centers_gt, clusters_gt, label_truth, seed, variance, alpha_grid, n_folds=3, cv_shuffle=True, cv_random_state=None):
    unique_labels = np.unique(label_truth)
    best_alpha_by_cluster, cv_errors_by_cluster = {}, {}

    for cluster_id in unique_labels:
        idxs = [i for i, lab in enumerate(label_truth) if lab == cluster_id]
        folds_per_sample = []
        for i in idxs:
            Xi = X_list[i]
            if Xi.shape[0] < n_folds:
                raise ValueError(f"Sample {i} in cluster {cluster_id} has only {Xi.shape[0]} rows, cannot do {n_folds}-fold CV.")
            kf = KFold( n_splits=n_folds, shuffle=cv_shuffle, random_state=cv_random_state if cv_shuffle else None,)
            folds_per_sample.append(list(kf.split(np.arange(Xi.shape[0]))))

        cv_log = {float(a): [] for a in alpha_grid}
        for fold_id in range(n_folds):
            X_train = [X_list[i][folds_per_sample[j][fold_id][0], :] for j, i in enumerate(idxs)]
            X_val = [X_list[i][folds_per_sample[j][fold_id][1], :] for j, i in enumerate(idxs)]
            labels_sub = [int(cluster_id)] * len(X_train)
            for a in alpha_grid:
                W_list_cv = generate_notear_per_cluster(X_train, labels_sub, alpha=a, round_decimals=3)
                cv_log[float(a)].append(float(compute_notear_reconstruction_error(W_list_cv, X_val)))

        mean_by_a = {a: float(np.mean(v)) for a, v in cv_log.items()}
        best_alpha = min(mean_by_a, key=mean_by_a.get)
        best_alpha_by_cluster[int(cluster_id)] = float(best_alpha)
        cv_errors_by_cluster[str(int(cluster_id))] = {str(a): [float(x) for x in v] for a, v in cv_log.items()}

    W_center_est = {}
    for cluster_id in unique_labels:
        idxs = [i for i, lab in enumerate(label_truth) if lab == cluster_id]
        X_sub = [X_list[i] for i in idxs]
        labels_sub = [int(cluster_id)] * len(X_sub)
        W_list = generate_notear_per_cluster(X_sub, labels_sub, alpha=best_alpha_by_cluster[int(cluster_id)], round_decimals=3)
        W_center_est[int(cluster_id)] = np.asarray(W_list[0])

    W_notear_list = np.stack([W_center_est[int(lab)] for lab in label_truth], axis=0)
    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    skeleton_acc_by_thresh = {str(t): _to_py(average_skeleton_accuracy(W_list_gt, W_notear_list, threshold=float(t))) for t in thresholds}

    return {
    "seed": seed,
    "mode": "cluster",
    "std": float(variance),
    "alpha_grid": [float(a) for a in alpha_grid],
    "best_alpha_by_cluster": best_alpha_by_cluster,
    "cv_errors_by_cluster": cv_errors_by_cluster,
    "eval_thresholds": [float(t) for t in thresholds],
    "skeleton_accuracy_by_thresh": skeleton_acc_by_thresh,
    "recon_error_gt": _to_py(compute_total_reconstruction_error(W_centers_gt, clusters_gt, X_list)),
    "recon_error_est": _to_py(compute_notear_reconstruction_error(W_notear_list, X_list)),
    "W_center_est": _to_py(W_center_est),
    "W_notear_list": _to_py(W_notear_list),
}