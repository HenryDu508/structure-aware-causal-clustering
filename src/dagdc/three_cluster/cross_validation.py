import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import KFold
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from .dc_admm import optimize_dc_admm



def evaluate_params(params, fold, X_train, X_val,
                    max_dc_iter, max_admm_iter, plot_graph):
    """Evaluate a single parameter combination on a specific fold."""
    lambda1, lambda2, tau, rho1, rho2 = params

    # unpack the five outputs of your updated DC‑ADMM
    cluster_W, clusters, W_list, label_list, W_cluster_list = optimize_dc_admm(
        X_train,
        lambda1=lambda1,
        lambda2=lambda2,
        tau=tau,
        rho1=rho1,
        rho2=rho2,
        max_dc_iter=max_dc_iter,
        max_admm_iter=max_admm_iter,
        plot_graph=plot_graph,
        thres_value=0.01
    )

    # now compute errors using `clusters`
    cov_error   = compute_total_cov_err(cluster_W, clusters,   X_val)
    recon_error = compute_total_reconstruction_error(cluster_W, clusters, X_val)
    mle_error   = compute_average_penalized_mle_error(cluster_W, clusters,   X_val)

    return {
        "fold":        fold,
        "lambda1":     lambda1,
        "lambda2":     lambda2,
        "tau":         tau,
        "rho1":        rho1,
        "rho2":        rho2,
        "mle_error":   mle_error,
        "cov_error":   cov_error,
        "recon_error": recon_error
    }


def tune_hyperparameters(
    X_list,
    param_grid,
    n_folds=3,
    max_dc_iter=10,
    max_admm_iter=15,
    plot_graph=False
):
    """
    Sequential grid‑search over params × folds.
    Returns top5 for each metric and the full CV table.
    """
    # 0) alignment check
    T0 = X_list[0].shape[0]
    if any(X.shape[0] != T0 for X in X_list):
        raise ValueError("All X_i must have same time steps.")
    if plot_graph:
        # plotting per‐fold will clutter; ensure you really want this
        print("Warning: plot_graph=True will produce plots for each fold/param.")

    # 1) prepare splits
    kf = KFold(n_splits=n_folds, shuffle=False)
    splits = list(kf.split(X_list[0]))

    # 2) build all param combinations
    param_names  = list(param_grid.keys())
    param_combos = list(product(*param_grid.values()))

    # 3) run sequentially
    records = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        # slice once per fold
        X_tr = [X[tr_idx] for X in X_list]
        X_va = [X[va_idx] for X in X_list]

        for combo in param_combos:
            res = evaluate_params(
                combo, fold, X_tr, X_va,
                max_dc_iter, max_admm_iter, plot_graph
            )
            records.append(res)

    # 4) aggregate into DataFrame
    df = pd.DataFrame(records)
    # compute means across folds
    cv = (
        df
        .groupby(param_names)[["cov_error","recon_error","mle_error"]]
        .mean()
        .reset_index()
        .rename(columns={
            "cov_error":  "cov_mean",
            "recon_error":"recon_mean",
            "mle_error":  "mle_mean"
        })
    )

    # 5) extract top5 for each metric
    def top5(metric):
        return (
            cv
            .nsmallest(5, metric)
            [param_names + [metric]]
            .rename(columns={metric: "mean_error"})
            .to_dict("records")
        )

    return top5("mle_mean"), top5("recon_mean"), top5("cov_mean"), cv


def compute_covariance_error(W, X_i):
    """
    Compute the Frobenius‐norm covariance error for a single sample.
    Error = ‖cov_model(W) – cov_sample(X_i)‖_F

    Parameters:
      W    (d×d array): representative weight matrix for one cluster.
      X_i  (m_i×d array): observations for sample i.

    Returns:
      e_i  (float): Frobenius‐norm error.
    """
    m_i, d = X_i.shape
    if m_i < 2:
        raise ValueError(f"Need at least 2 observations, got {m_i}")

    cov_sample = np.cov(X_i, rowvar=False, ddof=1)  

    I = np.eye(d)
    A = I - W
    try:
        invA = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        invA = np.linalg.pinv(A)
    cov_model = invA @ invA.T
    return float(np.linalg.norm(cov_model - cov_sample, ord='fro'))


def compute_total_cov_err(W_cluster, cluster_label, X_test):
    """
    Compute the average covariance error over all test samples.

    Iterates over clusters k and sample indices i∈cluster_label[k], calls
    compute_covariance_error(W_cluster[k], X_test[i])

    Parameters:
      W_cluster      : list of d×d repr. weight matrices, one per cluster.
      cluster_label  : list of sets of test‐sample indices for each cluster.
      X_test         : list of test samples X_i, each m_i×d.

    Returns:
      total_error    : mean of all individual errors.
    """
    errors = []
    N = len(X_test)
    for k, cluster in enumerate(cluster_label):
        for i in cluster:
            if not (0 <= i < N):
                raise IndexError(f"Invalid test index {i}")
            try:
                e_i = compute_covariance_error(W_cluster[k], X_test[i])
            except ValueError:
                continue
            errors.append(e_i)

    if not errors:
        raise ValueError("No valid test samples to compute covariance error.")
    return float(np.mean(errors))

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


def compute_total_reconstruction_error(W_cluster,
                                       cluster_label,
                                       X_test):
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

def average_skelton_accuracy(W_true_list, W_est_list, threshold=0.01):
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


# def clustering_overall_metrics(y_true, y_pred):
#     """
#     Align y_pred to y_true (handles extra clusters) and compute:
#       - TP, FP, FN counts
#       - precision, recall, accuracy

#     Args:
#         y_true (array-like of shape [N,]): true labels
#         y_pred (array-like of shape [N,]): predicted labels (may have extra classes)

#     Returns:
#         dict: {
#           'TP': int,
#           'FP': int,
#           'FN': int,
#           'precision': float,
#           'recall': float,
#           'accuracy': float,
#           'y_pred_aligned': np.ndarray
#         }
#     """
#     y_true = np.asarray(y_true)
#     y_pred = np.asarray(y_pred)
#     if y_true.shape != y_pred.shape:
#         raise ValueError("y_true and y_pred must have same length")

#     # 1) Build confusion matrix of size (K_true, K_pred)
#     labels_true = np.unique(y_true)
#     labels_pred = np.unique(y_pred)
#     Kt, Kp = labels_true.size, labels_pred.size

#     cm = np.zeros((Kt, Kp), int)
#     for t, p in zip(y_true, y_pred):
#         # find index positions explicitly
#         i = np.nonzero(labels_true == t)[0][0]
#         j = np.nonzero(labels_pred == p)[0][0]
#         cm[i, j] += 1

#     # 2) Hungarian matching to maximize correct assignments
#     row_ind, col_ind = linear_sum_assignment(-cm)
#     mapping = { labels_pred[j]: labels_true[i]
#                 for i, j in zip(row_ind, col_ind) }
#     # any extra predicted label → dummy mismatch (-1)
#     for p in labels_pred:
#         mapping.setdefault(p, -1)

#     # 3) Remap all predictions
#     y_aligned = np.array([mapping[p] for p in y_pred], dtype=int)

#     # 4) Compute counts and rates
#     N  = len(y_true)
#     TP = int((y_aligned == y_true).sum())
#     FP = N - TP
#     FN = FP   # in multiclass micro setting, FP == FN

#     precision = TP / (TP + FP) if TP + FP else 0.0
#     recall    = TP / (TP + FN) if TP + FN else 0.0
#     accuracy  = TP / N

#     return {
#         'TP': TP,
#         'FP': FP,
#         'FN': FN,
#         'precision': precision,
#         'recall': recall,
#         'accuracy': accuracy,
#         'y_pred_aligned': y_aligned
#     }

def evaluate_and_save(
    out_dir: str,
    W_true_list,
    W_cluster_true,
    cluster_label_true,
    X_list,
    W_cluster_est,
    cluster_label_est,
    W_est_list,
    label_pred,
):
    """
    Runs graph‐recovery, reconstruction, and clustering‐accuracy metrics
    and writes both a JSON and a one‐row CSV to `out_dir`.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Macro‐avg graph‐recovery
    graph_metrics = average_skelton_accuracy(W_true_list, W_est_list, threshold=0.01)

    # 2) Recon errors
    recon_true = compute_total_reconstruction_error(
        W_cluster_true, cluster_label_true, X_list
    )
    recon_est  = compute_total_reconstruction_error(
        W_cluster_est,  cluster_label_est,  X_list
    )

    # 3) Clustering accuracy
    cluster_metrics = clustering_overall_metrics(
        y_true=[i for cl in cluster_label_true for i in cl],
        y_pred=label_pred
    )

    # Combine everything
    results = {}
    results.update({f"graph_{k}": v for k, v in graph_metrics.items()})
    results["recon_error_true"] = recon_true
    results["recon_error_est"]  = recon_est
    results.update(cluster_metrics)

    # Save JSON
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save one‐row CSV
    pd.DataFrame([results]).to_csv(
        os.path.join(out_dir, "results.csv"), index=False
    )

    return results