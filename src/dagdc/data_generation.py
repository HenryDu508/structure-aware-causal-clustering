"""
Synthetic clustered linear SEM data.

This module provides functions for:
- Generating initial W_list via NO-TEARS lasso.
- Generating strict DAG adjacency matrices.
- Generating synthetic data X_i given a DAG.
- Generating clustered datasets with shuffle.
- Utility functions: threshold_array, split_X_list.
"""

import numpy as np
import networkx as nx
from typing import List, Tuple
from numpy.random import default_rng




def threshold_array(arr, threshold=0.01):
    """
    Set all values in `arr` with absolute value <= threshold to zero.
    """
    result = np.round(arr, 3).copy()
    result[np.abs(result) <= threshold] = 0
    return result


def generate_strict_dag_type(n_vars, s0, graph_type, W_pos_range, W_neg_range, seed=None, permute=True):
    """
    Generate a DAG by type ('ER','SF','BP','UR') with weighted edges.
    
    Args:
        n_vars (int): Number of variables (nodes)
        s0 (int): Number of edges
        graph_type (str): One of 'ER', 'SF', 'BP', 'UR'
        W_pos_range (tuple): Range for positive edge weights
        W_neg_range (tuple): Range for negative edge weights
        seed (int): Random seed
        permute (bool): If True, permutes variable order after graph generation
    
    Returns:
        np.ndarray: Weighted adjacency matrix W (n_vars x n_vars)
    """
    rng = default_rng(seed)
    B = np.zeros((n_vars, n_vars), dtype=int)

    if graph_type == 'ER':
        p = min(0.3, 2 * s0 / (n_vars * (n_vars - 1))) if n_vars > 1 else 0
        mask = rng.random((n_vars, n_vars)) < p
        B = np.triu(mask.astype(int), k=1)

    elif graph_type == 'SF':
        for j in range(1, n_vars):
            degs = B.sum(axis=1)[:j] + 1e-8
            probs = degs / degs.sum()
            m = max(1, s0 // n_vars)
            srcs = rng.choice(j, size=min(m, j), p=probs, replace=False)
            B[srcs, j] = 1

    elif graph_type == 'BP':
        split = max(1, n_vars // 5)
        max_bp = split * (n_vars - split)
        s0 = min(s0, max_bp)
        avail = [(i, j) for i in range(split) for j in range(split, n_vars)]
        picks = rng.choice(len(avail), size=s0, replace=False)
        for idx in picks:
            i, j = avail[idx]
            B[i, j] = 1

    elif graph_type == 'UR':
        avail = [(i, j) for i in range(n_vars) for j in range(i + 1, n_vars)]
        picks = rng.choice(len(avail), size=s0, replace=False)
        for idx in picks:
            i, j = avail[idx]
            B[i, j] = 1

    else:
        raise ValueError(f"Unknown graph_type '{graph_type}'")

    # Optional permutation
    if permute:
        perm = rng.permutation(n_vars)
        B = B[perm][:, perm]

    # Assign edge weights
    edge_idx = np.argwhere(B == 1)
    n_edges = len(edge_idx)
    signs = rng.choice([1, -1], size=n_edges)
    pos = rng.uniform(*W_pos_range, size=n_edges)
    neg = rng.uniform(*W_neg_range, size=n_edges)
    weights = np.where(signs > 0, pos, neg)

    W = np.zeros_like(B, dtype=float)
    for (i, j), w in zip(edge_idx, np.round(weights, 2)):
        W[i, j] = np.round(w, 3)

    W[np.abs(W) <= 1e-3] = 0
    return W

def generate_X_i(m, d, W_c, mean=0, std=1, seed=None):
    """
    Simulate one multivariate time series X_i = Z_i @ (I - W_c)^{-1}.
    """
    rng = default_rng(seed)
    I_W = np.eye(d) - W_c
    if np.linalg.det(I_W) == 0:
        raise ValueError("(I - W_c) is singular; invalid DAG.")
    Z = rng.normal(loc=mean, scale=std, size=(m, d))
    X_i = Z @ np.linalg.inv(I_W)
    return X_i


def generate_clustered_data(
    total_samples,
    cluster_proportions,
    n_vars,
    m,
    W_pos_range,
    W_neg_range,
    mean,
    std,
    s0_list,
    graph_type,
    seed=None,
    permute=True
):
    """
    Generate synthetic data for clustering experiments with flexible sample lengths.

    Args:
        total_samples (int): Total number of series to generate.
        cluster_proportions (list): Proportions for each cluster (sum to 1).
        n_vars (int): Number of variables (nodes) in each DAG.
        m (int or list): Number of time steps. If int, all samples share the same length.
                         If list, must have length equal to total_samples.
        s0_list (list): Number of edges for each cluster's center DAG.
        ... (other args)

    Returns:
        X_list, W_list, W_centers, cluster_assignment, sample_labels
    """
    rng = default_rng(seed)
    K = len(cluster_proportions)

    # --- Handle flexible m (int or list) ---
    if isinstance(m, (int, np.integer)):
        # Assign same length to all samples
        m_list = [int(m)] * total_samples
    elif isinstance(m, (list, np.ndarray)):
        # Validate list length
        if len(m) != total_samples:
            raise ValueError(f"Length of m list ({len(m)}) must match total_samples ({total_samples})")
        m_list = [int(val) for val in m]
    else:
        raise TypeError("m must be an int or a list/array of ints")

    assert len(s0_list) == K, "Length of s0_list must match number of clusters."

    # Determine number of samples per cluster
    counts = [int(total_samples * p) for p in cluster_proportions]
    rem = total_samples - sum(counts)
    for i in range(rem):
        counts[i % K] += 1

    # Generate Center DAGs for each cluster
    W_centers = []
    for k in range(K):
        center_seed = int(rng.integers(1e9))
        Wc = generate_strict_dag_type(
            n_vars, s0_list[k], graph_type, W_pos_range, W_neg_range, seed=center_seed, permute=permute
        )
        W_centers.append(Wc)

    # Generate individual samples
    samples = []
    m_idx = 0  # Pointer to track m for each sample
    for k, cnt in enumerate(counts):
        for _ in range(cnt):
            sample_seed = int(rng.integers(1e9))
            # Use current m from the processed m_list
            current_m = m_list[m_idx]
            Xi = generate_X_i(current_m, n_vars, W_centers[k], mean, std, seed=sample_seed)
            # Store data, the ground truth DAG, and the true cluster ID
            samples.append((Xi, W_centers[k], k))
            m_idx += 1

    # Shuffle to ensure cluster order isn't sequential in the final list
    rng.shuffle(samples)

    # Unpack shuffled results
    X_list, W_list, cluster_assignment, sample_labels = [], [], [set() for _ in range(K)], []
    for idx, (Xi, Wi, k) in enumerate(samples):
        X_list.append(Xi)
        W_list.append(Wi)
        cluster_assignment[k].add(idx)
        sample_labels.append(k + 1)  # 1-based labeling for metrics

    return X_list, W_list, W_centers, cluster_assignment, sample_labels



def split_X_list(X_list, split_ratio=0.8, shuffle=False, seed=None):
    """
    Split each X in X_list into train/test along the time axis.

    Args:
        X_list:      list of (m_i × d) arrays.
        split_ratio: fraction for training (0 < split_ratio < 1).
        shuffle:     whether to shuffle the time‐steps before splitting.
        seed:        random seed for reproducibility.

    Returns:
        (X_train_list, X_test_list)
    """
    if not (0 < split_ratio < 1):
        raise ValueError("split_ratio must be between 0 and 1")
    rng = default_rng(seed)
    X_train, X_test = [], []

    for X in X_list:
        m = X.shape[0]
        idx = rng.permutation(m) if shuffle else np.arange(m)
        cut = int(m * split_ratio)
        # ensure at least one in each split
        cut = max(1, min(cut, m - 1))
        X_train.append(X[idx[:cut]])
        X_test.append( X[idx[cut:]] )

    return X_train, X_test