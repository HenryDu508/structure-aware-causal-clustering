"""
DAG-DC-ADMM optimizer.

This module provides functions for:
- Loss and gradient computation for reconstruction loss.
- Acyclicity constraint (h-function) and its gradient.
- ADMM-based updates for W matrices and clustering.
- Hierarchical clustering (complete linkage) utilities.
- End-to-end DC-ADMM optimization function.
"""

import numpy as np
from scipy.optimize import minimize
import scipy.linalg as slin
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
import networkx as nx
import time


h_times = []
minimize_times = []
dc_times = []
all_admm_times = []
minimize_successes = []
all_admm_primal_res = []
all_admm_dual_res  = []


def precompute_second_moment(X):
    """
    Precompute S_i = X_i^T X_i / m_i for each individual.
    This is exactly equivalent to the original reconstruction loss,
    without assuming mean-zero data.
    """
    S_list = []
    for X_i in X:
        m_i = X_i.shape[0]
        S_i = (X_i.T @ X_i) / m_i
        S_list.append(S_i)
    return S_list

def loss_func(S_i, W_i):
    """
    Compute the same reconstruction loss and gradient as
    0.5/m_i * ||X_i W_i - X_i||_F^2, but using
    S_i = X_i^T X_i / m_i for speed.
    """
    d = W_i.shape[0]
    I = np.eye(d)
    A = W_i - I

    loss = 0.5 * np.trace(A.T @ S_i @ A)
    grad_loss = S_i @ A
    return loss, grad_loss

def h_func(W):
    """
    Compute the acyclicity constraint value and its gradient.
    Input:
        W (np.ndarray): Weight matrix, shape (d, d).
    Output:
        h (float): Acyclicity constraint value.
        grad_h (np.ndarray): Gradient of the constraint, shape (d, d).
    """
    t0 = time.perf_counter()
    d = W.shape[0]

    W = np.clip(W, -5.0, 5.0)
    W_squared = W * W  # Hadamard product
    E = slin.expm(W_squared)     # Compute matrix exponential


    h = np.trace(E) - d     # Compute h(W_i)
    grad_h = 2 * W * E.T    # Compute gradient
    h = float(np.clip(h, 0.0, 1e3)) 
    grad_h = np.clip(grad_h, -1e3, 1e3)  # Clip gradient to prevent overflow

    h_times.append(time.perf_counter() - t0)
    return h, grad_h

def update_W_i(i, S_i, W_list, W_list_new, theta_prev_admm,
    u, alpha, rho1, rho2, lambda1):
    """
    Update the weight matrix W_i for individual i.

    Input:
        i (int): Index of the individual being updated.
        S_i (np.ndarray): Precomputed second-moment matrix for individual i, shape (d, d).
        W_list (np.ndarray): Previous weight matrices, shape (n, d, d).
        W_list_new (np.ndarray): Updated weight matrices (current iteration), shape (n, d, d).
        theta_prev_admm (np.ndarray): Previous θ estimates from the last ADMM iteration, shape (n, n, d, d).
        u (np.ndarray): Scaled dual variables for clustering constraints, shape (n, n, d, d).
        alpha(np.ndarray): Scaled dual variables for DAG constraints, shape (n,).
        rho1 (float): Augmented Lagrangian parameter for DAG constraints.
        rho2 (float): Augmented Lagrangian parameter for ADMM consensus.
        lambda1 (float): L1 regularization strength.

    Output:
        W_i_new (np.ndarray): Updated weight matrix for individual i, shape (d, d).
    """
    n, d = len(W_list), S_i.shape[0]
    bounds = [(0, 0) if row == col else (0, None) for row in range(d) for col in range(d)] * 2

    def objective(w):
        """
        Compute the objective function and its gradient.

        Input:
            w (np.ndarray): Flattened parameter vector containing both w_pos and w_neg, shape (2 * d * d,).

        Output:
            total_obj (float): Total objective function value.
            total_grad (np.ndarray): Gradient of the objective, shape (2 * d * d,).
        """
        w_pos = w[:d*d].reshape((d,d))
        w_neg = w[d*d:].reshape((d,d))
        W_i = w_pos - w_neg
        
        # Clip W_i to prevent numerical instability
        W_i = np.clip(W_i, -10, 10)  # Adjust the clipping range as needed

        # Compute reconstruction loss
        loss, grad_loss = loss_func(S_i, W_i)

        l1_term = lambda1 * (np.sum(w_pos) + np.sum(w_neg))
        grad_l1_pos = lambda1 * np.ones_like(w_pos)
        grad_l1_neg = lambda1 * np.ones_like(w_neg)

        # Compute DAG constraint terms
        # h_val, grad_h = h_func(W_i)
        if alpha[i] == 0.0 and rho1 == 0.0:
            h_val = 0.0
            grad_h = np.zeros_like(W_i)
        else:
            h_val, grad_h = h_func(W_i)
        
        dag_term = alpha[i] * h_val + 0.5 * rho1 * h_val**2
        grad_dag = (alpha[i] + rho1 * h_val) * grad_h

        # Compute ADMM consensus constraints
        consensus_penalty = 0.0
        grad_consensus_pos = np.zeros_like(w_pos)
        grad_consensus_neg = np.zeros_like(w_neg)

        if rho2 !=0:
            for j in range(len(W_list)):
                if j == i: continue
                Wj = W_list_new[j] if j < i else W_list[j]
                residual = theta_prev_admm[i, j] - (W_i - Wj) + u[i, j]
                    
                consensus_penalty += 0.5 * rho2 * np.sum(residual*residual)
                grad_consensus_pos -= rho2 * residual  # d/d(w_pos) = -rho2 * residual
                grad_consensus_neg += rho2 * residual  # d/d(w_neg) = +rho2 * residual

        # Compute total objective and gradient
        total_obj = loss + l1_term + dag_term + consensus_penalty
        total_grad_pos = grad_loss + grad_l1_pos + grad_dag + grad_consensus_pos
        total_grad_neg = -grad_loss + grad_l1_neg - grad_dag + grad_consensus_neg
        total_grad = np.concatenate([total_grad_pos.flatten(), total_grad_neg.flatten()])
        return total_obj, total_grad

    # Solve with bounded L-BFGS-B
    # w_init = np.concatenate([W_list[i].flatten(), np.zeros(d * d)])  # Initialize w_pos and w_neg
    W0 = W_list[i].copy()
    np.fill_diagonal(W0, 0.0)
    
    w_pos0 = np.maximum(W0, 0.0).reshape(-1)
    w_neg0 = np.maximum(-W0, 0.0).reshape(-1)
    
    w_init = np.concatenate([w_pos0, w_neg0])
    
    t0 = time.perf_counter()
    res = minimize(objective, w_init, method='L-BFGS-B', jac=True, bounds=bounds)
    minimize_times.append(time.perf_counter() - t0)
    minimize_successes.append(res.success)

    # Reconstruct W_i from w_pos and w_neg
    w_pos = res.x[:d * d].reshape([d, d])
    w_neg = res.x[d * d:].reshape([d, d])
    W_i_new = w_pos - w_neg
    return W_i_new


def soft_threshold(diff, gamma):
    """
    Soft-thresholding function for a batch of matrices using Frobenius norm.

    Input:
        diff (np.ndarray): Difference matrices, shape (batch_size, d, d).
        gamma (float): Threshold parameter (λ2 / ρ2).

    Output:
        thresholded_diff (np.ndarray): Soft-thresholded matrices, same shape as diff.
    """
    # Compute Frobenius norm for each matrix in the batch
    norm_diff = np.linalg.norm(diff, ord='fro', axis=(1, 2), keepdims=True) + 1e-12
    scale_factor = np.maximum(1 - gamma / norm_diff, 0)
    return diff * scale_factor

def update_theta(W_list,u,rho2,lambda2,tau,theta_hat_prev):
    """
    Update auxiliary variables θ with truncation logic (vectorized version).

    Input:
        W_list (np.ndarray): Current weight matrices, shape (n, d, d).
        u (np.ndarray): Scaled dual variables, shape (n, n, d, d).
        rho2 (float): Augmented Lagrangian parameter for ADMM consensus.
        lambda2 (float): Clustering penalty strength.
        tau (float): Truncation threshold.
        theta_hat_prev (np.ndarray): Previous θ estimates, shape (n, n, d, d).

    Output:
        theta (np.ndarray): Updated θ values, shape (n, n, d, d).
    """
    n, d, _ = W_list.shape
    theta = np.zeros_like(theta_hat_prev)  # (n, n, d, d)

    # Get all i < j index pairs
    i_indices, j_indices = np.triu_indices(n, k=1)

    # Compute differences for all i < j pairs
    W_diff = W_list[i_indices] - W_list[j_indices] - u[i_indices, j_indices]
    W_diff = np.nan_to_num(W_diff, nan=0.0, posinf=1e3, neginf=-1e3)  # (n_pairs, d, d)
    

    # Compute previous θ's Frobenius norms
    norm_prev = np.linalg.norm(theta_hat_prev[i_indices, j_indices], ord='fro', axis=(1, 2))  # (n_pairs,)
    norm_diff = np.linalg.norm(W_diff,                                   ord='fro', axis=(1, 2))  # (n_pairs,)

    # Case handling with vectorized operations
    gamma = lambda2 / rho2

    if np.max(norm_prev) < 1e-12:
        mask = np.zeros_like(norm_prev, dtype=bool)  # no pair is soft-thresholded
    else:
        mask = (norm_prev < tau)
    
    # mask = norm_prev < tau  # (n_pairs,)

    frac_mask   = np.mean(mask)
    frac_zero = np.mean(norm_diff[mask] <= gamma) if np.any(mask) else 0.0
    med_prev = np.median(norm_prev) if norm_prev.size > 0 else 0.0
    med_diff = np.median(norm_diff) if norm_diff.size > 0 else 0.0
    
    # print(f"tau={tau:.3g} | frac_mask={frac_mask:.2f} | "
    #       f"gamma={gamma:.3g} | med||diff||={med_diff:.3g} | zero_rate_in_mask={frac_zero:.2f}")

    # Initialize theta_ij and apply thresholds
    theta_ij = np.zeros_like(W_diff)  # (n_pairs, d, d)
    
    # Case 1: ||θ_prev||_F >= τ → θ_ij = W_diff (no thresholding)
    theta_ij[~mask] = W_diff[~mask]

    # Case 2: ||θ_prev||_F < τ → apply soft-thresholding
    theta_ij[mask] = soft_threshold(W_diff[mask], gamma)

    # Fill upper and lower triangles
    theta[i_indices, j_indices] = theta_ij
    theta[j_indices, i_indices] = -theta_ij  # Ensure symmetry

    return theta

def optimize_admm(S_list, lambda1, lambda2, tau, theta_hat_prev,
    u_init, alpha_init, theta_prev_admm_init,
    max_iter=100, rho1=0.1, rho2=0.4,
    dual_tol=1e-3, rho_max=1e16, W_init=None
):
    """
    Inner ADMM optimization loop with warm-started duals and theta history.

    Inputs:
        S_list (list of np.ndarray): Precomputed second-moment matrices,
            each S_i = X_i^T X_i / m_i, shape [(d, d), ...].
        lambda1, lambda2, tau: regularization and threshold parameters.
        theta_hat_prev (np.ndarray): Previous DC-level theta, shape (n, n, d, d).
        u_init (np.ndarray): Warm-start dual variables for consensus, shape (n, n, d, d).
        alpha_init (np.ndarray): Warm-start dual vars for DAG, shape (n,).
        theta_prev_admm_init (np.ndarray): Warm-start theta history for ADMM residuals, shape (n, n, d, d).
        max_iter, rho1, rho2, dual_tol, rho_max: ADMM control params.
        W_init (np.ndarray, optional): Warm-start for W_list, shape (n, d, d).

    Returns:
        W_list (np.ndarray): Optimized weight matrices, shape (n, d, d).
        admm_times (list): Per-iteration timings.
        admm_primal_res (list): Primal residuals.
        admm_dual_res (list): Theta residuals.
        u (np.ndarray): Updated dual variables for consensus.
        alpha (np.ndarray): Updated dual variables for DAG.
        theta_prev_admm (np.ndarray): Final theta history from ADMM.
        rho1 (float): Final DAG penalty (carried across DC).
    """
    n = len(S_list)
    d = S_list[0].shape[0]

    # Primal vars
    W_list = W_init.copy() if W_init is not None else np.random.uniform(-0.01, 0.01, (n, d, d))
    W_list_new = np.zeros_like(W_list)

    # Duals & history
    u = u_init.copy()
    alpha = alpha_init.copy()
    theta_prev_admm = theta_prev_admm_init.copy()

    # Metrics
    admm_times, admm_primal_res, admm_dual_res = [], [], []
    h_prev = None
    eta_up = 5.0
    min_admm_iter = 5

    for k in range(max_iter):
        t0 = time.perf_counter()

        # 1. Update W_list
        for i in range(n):
            W_list_new[i] = update_W_i(i, S_list[i], W_list, W_list_new,
                theta_prev_admm, u, alpha, rho1, rho2, lambda1)

        # 2. Update theta
        theta = update_theta(W_list_new, u, rho2, lambda2, tau, theta_hat_prev)

        # 3. Residuals
        theta_old = theta_prev_admm
        r = theta - (W_list_new[:, None] - W_list_new[None, :])
        s = rho2 * (theta - theta_old)

        mask_ij = ~np.eye(n, dtype=bool)
        r_norm = np.mean(np.linalg.norm(r[mask_ij], ord='fro', axis=(1, 2)))
        s_norm = np.mean(np.linalg.norm(s[mask_ij], ord='fro', axis=(1, 2)))

        theta_prev_admm = theta.copy()

        # 4. Dual update
        u += r

        # 5. DAG dual updates
        h_vals = np.array([h_func(Wi)[0] for Wi in W_list_new])
        h_curr = float(np.sum(h_vals))

        if not np.isfinite(h_curr) or h_curr <= 0.0:
            h_curr = 1e-8

        if h_prev is not None:
            if h_curr > 0.25 * h_prev and rho1 < rho_max:
                rho1 = min(rho1 * eta_up, rho_max)

        alpha += rho1 * h_vals

        # Record metrics
        admm_times.append(time.perf_counter() - t0)
        admm_primal_res.append(r_norm)
        admm_dual_res.append(s_norm)

        # Convergence
        tol_r = dual_tol
        tol_s = dual_tol * max(rho2, 1e-12)

        if (k + 1) >= min_admm_iter and (r_norm < tol_r) and (s_norm < tol_s):
            print(f"Converged at ADMM iteration {k+1}")
            break

        W_list = W_list_new.copy()
        h_prev = h_curr

    return W_list, admm_times, admm_primal_res, admm_dual_res, u, alpha, theta_prev_admm, rho1


def compute_Sm(S_list, W_list, theta_hat_prev, lambda1, lambda2, tau):
    """
    Compute the current objective value S^{(m)}(W, θ)
    Input:
        S_list (list of np.ndarray): List of second-moment matrices, each shape (d, d).
        W_list (np.ndarray): List of optimized weight matrices, shape (n, d, d).
        theta_hat_prev (np.ndarray): Previous theta matrix (θ^{(m)}), shape (n, n, d, d).
        lambda1 (float): L1 regularization strength.
        lambda2 (float): Clustering penalty strength.
        tau (float): Truncation threshold.
    Output:
        S (float): Computed objective function value.
    """
    n, d = len(S_list), S_list[0].shape[0]
    S = 0.0

    for i in range(n):
        if W_list[i].shape != (d, d):
            print(f"Warning: W_list[{i}] has shape {W_list[i].shape}, expected ({d}, {d})")

        # reconstruction loss using precomputed S_i
        loss_i, _ = loss_func(S_list[i], W_list[i])
        S += loss_i

        # L1 penalty
        S += lambda1 * np.sum(np.abs(W_list[i]))

        # pairwise clustering penalty
        for j in range(i + 1, n):
            norm_theta_prev = np.linalg.norm(theta_hat_prev[i, j], ord='fro')
            norm_theta_current = np.linalg.norm(W_list[i] - W_list[j], ord='fro')

            if norm_theta_prev < tau:
                S += lambda2 * norm_theta_current
            else:
                S += lambda2 * tau

    return S

def complete_linkage_clusters(dist_mat, tau):
    """
    Perform complete‐link hierarchical clustering on a symmetric distance matrix.

    Parameters
    ----------
    dist_matrix : (n×n) array
        Pairwise distances between n items (must be symmetric, zero diagonal).
    threshold   : float
        Cutoff on the linkage dendrogram: any cluster must have maximum linkage ≤ threshold.

    Returns
    clusters : List[Set[int]]
        A list of sets, each containing the zero‐based indices of items in that cluster.
    labels : List[int]
        A list where labels[i] is the cluster index of sample i (starting from 1).
    """
    # 1) extract upper‐triangular distances as a condensed vector
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method="complete")
    labels = fcluster(Z, t=tau, criterion="distance")
    clusters = [set(np.where(labels==lab)[0]) for lab in np.unique(labels)]
    
    return clusters, labels.tolist()



def plot_clusters(dist_matrix,
                  clusters,
                  threshold,
                  figsize=(5,5),
                  node_size=300):
    """
    Plot a graph where nodes in the same complete‐link cluster are
    connected (if their pairwise distance ≤ threshold) and colored by cluster.

    Parameters
    ----------
    dist_matrix : (n×n) array
        Symmetric distance matrix between n nodes.
    clusters    : List[Set[int]]
        Clusters as returned by `complete_linkage_clusters`.
    threshold   : float
        Only draw edges for node‐pairs within the same cluster whose distance ≤ threshold.
    figsize     : tuple, optional
        Matplotlib figure size.
    node_size   : int, optional
        Size of each node in the plot.
    """
    n = dist_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # add intra‐cluster edges only if distance ≤ threshold
    for cluster in clusters:
        for i in cluster:
            for j in cluster:
                if i < j and dist_matrix[i,j] <= threshold:
                    G.add_edge(i, j)

    # layout and color map
    pos = nx.spring_layout(G, seed=42)
    cmap = plt.get_cmap("tab10", len(clusters))

    plt.figure(figsize=figsize)
    for cid, cluster in enumerate(clusters):
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=list(cluster),
            node_color=[cmap(cid)],
            node_size=node_size,
            label=f"Cluster {cid}"
        )
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5)
    nx.draw_networkx_labels(
        G, pos,
        labels={i: str(i) for i in G.nodes()},
        font_color="white",
        font_weight="bold"
    )

    plt.title("Complete‐Link Clustering Graph")
    plt.legend(scatterpoints=1)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    

def build_warm_start_W(S_list, lambda1):
    """
    Only build W0 by per-individual L1 reconstruction (DAG/consensus OFF).
    """
    n, d = len(S_list), S_list[0].shape[0]
    W0 = np.zeros((n, d, d))
    zeros_theta = np.zeros((n, n, d, d))
    zeros_u = np.zeros_like(zeros_theta)
    zeros_alpha = np.zeros(n)

    for i in range(n):
        W0[i] = update_W_i(
            i, S_list[i], W_list=W0, W_list_new=W0,
            theta_prev_admm=zeros_theta,
            u=zeros_u,
            alpha=zeros_alpha,
            rho1=0.0, rho2=0.0,
            lambda1=lambda1
        )
        np.fill_diagonal(W0[i], 0.0)
    return W0

def optimize_dc_admm(X, lambda1=0.1, lambda2=0.5, tau=0.5, rho1=0.1, rho2=0.4,
    max_dc_iter=10, max_admm_iter=20,
    plot_graph=False, thres_value=0.01, W_init=None):
    """
    Outer DC-ADMM optimization loop with warm-started ADMM duals and theta history.
    Returns:
        cluster_W, clusters, W_list, label_list, W_cluster_list
    Globals updated:
        dc_times, all_admm_times, all_admm_primal_res, all_admm_dual_res
    """
    global dc_times, all_admm_times, all_admm_primal_res, all_admm_dual_res


    if isinstance(X, np.ndarray) and X.ndim == 3:
        X = [X[i] for i in range(X.shape[0])]
    n, d = len(X), X[0].shape[1]

    
    # Precompute second-moment matrices once for speed:
    # S_i = X_i^T X_i / m_i
    S_list = precompute_second_moment(X)

    # Initialize primal and dual variables
    if W_init is None:
        W_list = build_warm_start_W(S_list, lambda1)
    else:
        W_list = W_init.copy()
        for i in range(n):
            np.fill_diagonal(W_list[i], 0.0)

    theta_hat = W_list[:, None, :, :] - W_list[None, :, :, :]   # shape (n,n,d,d)
    u = np.zeros_like(theta_hat)
    alpha = np.zeros(n)
    prev_S = float('inf')

    for dc_iter in range(max_dc_iter):
        t0_dc = time.perf_counter()

        theta_hat_prev = theta_hat.copy()
        theta_prev_admm = theta_hat.copy()

        # Call ADMM with warm-started duals and theta history
        (W_list, admm_times, admm_primal_res, admm_dual_res, u, alpha, theta_prev_admm, rho1) = optimize_admm(
            S_list, lambda1, lambda2, tau, theta_hat_prev, u, alpha, theta_prev_admm,
            max_iter=max_admm_iter, rho1=rho1, rho2=rho2, W_init=W_list)

        # Record metrics
        dc_times.append(time.perf_counter() - t0_dc)
        all_admm_times.append(admm_times)
        all_admm_primal_res.append(admm_primal_res)
        all_admm_dual_res.append(admm_dual_res)

        # Print summary
        print(f"DC iter {dc_iter+1}: {dc_times[-1]:.3f}s | ADMM iters: {len(admm_times)}")

        # Update DC-level theta_hat
        for i in range(n):
            for j in range(i + 1, n):
                theta_hat[i, j] = W_list[i] - W_list[j]
                theta_hat[j, i] = -theta_hat[i, j]

        # Check DC convergence
        current_S = compute_Sm(S_list, W_list, theta_hat_prev, lambda1, lambda2, tau)
        if abs(current_S - prev_S) < 5e-3:
            print(f"Converged at DC iteration {dc_iter+1}")
            break
        prev_S = current_S

    # Final clustering
    theta_norm_matrix = np.linalg.norm(theta_hat, axis=(2, 3))
    clusters, label_list = complete_linkage_clusters(theta_norm_matrix, tau)

    cluster_W = []
    for cluster in clusters:
        mats = np.stack([W_list[i] for i in cluster], axis=0)
        rep = mats.mean(axis=0)
        if thres_value is not None:
            rep[np.abs(rep) < thres_value] = 0
            rep = np.round(rep, 2)
        cluster_W.append(rep)

    W_cluster_list = [cluster_W[label_list[i] - 1] for i in range(n)]

    if plot_graph:
        plot_clusters(theta_norm_matrix, clusters, tau)

    if thres_value is not None:
        W_list[np.abs(W_list) < thres_value] = 0

    return cluster_W, clusters, W_list, label_list, W_cluster_list