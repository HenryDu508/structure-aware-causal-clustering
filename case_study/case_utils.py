"""Preprocessing, DAG-DC-ADMM, NOTEARS baselines and plots for the Sachs et al. (2005) case study.

The DAG-DC-ADMM solver here is the version used for the case study. It differs from
src/algorithm_updated.py only in the final clustering step, which cuts the complete-linkage
dendrogram at the midpoint of its largest height jump.
"""
import os
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
from joblib import Parallel, delayed
from matplotlib.cm import ScalarMappable
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
PERTURBATION_FILES = [
    "1. cd3cd28.csv", "2. cd3cd28icam2.csv", "3. cd3cd28+aktinhib.csv",
    "4. cd3cd28+g0076.csv", "5. cd3cd28+psitect.csv", "6. cd3cd28+u0126.csv",
    "7. cd3cd28+ly.csv", "8. pma.csv", "9. b2camp.csv",
]
RENAME = {"p44/42": "Erk", "praf": "Raf", "pakts473": "Akt", "pjnk": "Jnk", "pmek": "Mek",
          "plcg": "Plcg", "pip2": "PIP2", "pip3": "PIP3"}
FEATURES = ["Jnk", "P38", "Mek", "PKA", "PKC", "PIP2", "Raf", "PIP3", "Akt", "Plcg", "Erk"]


def load_perturbations(data_dir):
    """Read the nine perturbation files into {'stimulus1': DataFrame, ...}."""
    out = {}
    for k, name in enumerate(PERTURBATION_FILES, start=1):
        df = pd.read_csv(os.path.join(data_dir, name), encoding="utf-8-sig")
        out[f"stimulus{k}"] = df.rename(columns=RENAME)
    return out


# ---------------------------------------------------------------------------
# Preclustering within each perturbation
# ---------------------------------------------------------------------------
def eval_kmeans_silhouette(Z, k_range=range(2, 11), n_init=50, random_state=0):
    rows = []
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit_predict(Z)
        rows.append({"k": k, "silhouette": silhouette_score(Z, labels)})
    return pd.DataFrame(rows)


def select_k_by_silhouette(df_sil, rel_tol=0.01):
    """Smallest k whose silhouette is within rel_tol of the maximum."""
    k_vals, sil = df_sil["k"].to_numpy(), df_sil["silhouette"].to_numpy()
    feasible = k_vals[sil >= (1.0 - rel_tol) * sil.max()]
    return int(feasible.min())


def select_preclusters(stimuli, pca_var=0.8, k_range=range(2, 11), n_init=50, random_state=0, rel_tol=0.01):
    """Number of PCA components and of K-means clusters for each perturbation."""
    rows, curves = [], []
    for name, df in stimuli.items():
        X = df.to_numpy(dtype=float)
        Z = PCA(n_components=pca_var, random_state=random_state).fit_transform(StandardScaler().fit_transform(X))
        df_sil = eval_kmeans_silhouette(Z, k_range=k_range, n_init=n_init, random_state=random_state)
        rows.append({"stimulus": name, "n_cells": Z.shape[0], "pca_dims": Z.shape[1],
                     "k_selected": select_k_by_silhouette(df_sil, rel_tol=rel_tol)})
        curves.append(df_sil.assign(stimulus=name))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def assign_preclusters(stimuli, summary, n_init=50, random_state=0):
    """Fit K-means in PCA space with the selected settings and return all cells with labels."""
    cfg = summary.set_index("stimulus")
    parts = []
    for name, df in stimuli.items():
        X = df.select_dtypes(include=[np.number])
        X = X.fillna(X.mean(numeric_only=True))
        Z = PCA(n_components=int(cfg.loc[name, "pca_dims"]), random_state=random_state).fit_transform(
            StandardScaler().fit_transform(X))
        km = KMeans(n_clusters=int(cfg.loc[name, "k_selected"]), n_init=n_init, random_state=random_state)
        parts.append(df.assign(cluster=km.fit_predict(Z).astype(int), stimulus=name))
    return pd.concat(parts, ignore_index=True)


def zscore_nan(X):
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    Z = (X - mu) / sd
    return np.where(np.isnan(Z), 0.0, Z)


def to_scaled_X(df, cols):
    num = df[cols]
    num = num.fillna(num.mean(numeric_only=True))
    return np.ascontiguousarray(StandardScaler().fit_transform(num.values.astype(float)))


# ---------------------------------------------------------------------------
# NOTEARS with L1 penalty (pooled and per-perturbation baselines)
# ---------------------------------------------------------------------------
def notears_lasso(X, beta, max_iter=100, h_tol=1e-8):
    """NOTEARS (Zheng et al., 2018) with an L1 penalty, solved by L-BFGS-B on (W+, W-)."""
    n, d = X.shape
    bnds = [(0, 0) if i == j else (0, None) for i in range(d) for j in range(d)] * 2

    def _h(W):
        return np.trace(slin.expm(W * W)) - d

    def _split(w):
        return w[: d ** 2].reshape([d, d]) - w[d ** 2:].reshape([d, d])

    def _func(w):
        W = _split(w)
        h = _h(W)
        loss = 0.5 / n * np.square(np.linalg.norm(X.dot(np.eye(d) - W), "fro"))
        return loss + 0.5 * rho * h * h + alpha * h + beta * w.sum()

    def _grad(w):
        W = _split(w)
        E = slin.expm(W * W)
        g = -1.0 / n * X.T.dot(X).dot(np.eye(d) - W) + (rho * (np.trace(E) - d) + alpha) * E.T * W * 2
        return np.concatenate([g.flatten() + beta, -g.flatten() + beta])

    w_est, w_new = np.zeros(2 * d * d), np.zeros(2 * d * d)
    rho, alpha, h_val, h_new = 1.0, 0.0, np.inf, np.inf
    for it in range(max_iter):
        while rho < 1e20 and (h_new > 0.25 * h_val or h_new == np.inf):
            w_new = sopt.minimize(_func, w_est, method="L-BFGS-B", jac=_grad, bounds=bnds).x
            h_new = _h(_split(w_new))
            if h_new > 0.25 * h_val:
                rho *= 10
        w_est, h_val = w_new, h_new
        alpha += rho * h_val
        if h_val <= h_tol:
            break
        if it == max_iter - 1:
            warnings.warn("NOTEARS did not converge. Consider increasing max_iter.")
    return _split(w_est)


def reconstruction_error(W, X):
    """Mean squared reconstruction error per observation, ||X - XW||_F^2 / m."""
    return float(np.sum((X - X @ W) ** 2) / X.shape[0])


def tune_notears(X, alpha_grid, n_splits=3, seed=42):
    """K-fold CV over the L1 weight; refit on all rows with the best value."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    errs = {a: [] for a in alpha_grid}
    for tr, va in kf.split(X):
        for a in alpha_grid:
            errs[a].append(reconstruction_error(notears_lasso(X[tr], a), X[va]))
    cv = {a: float(np.mean(v)) for a, v in errs.items()}
    best = min(cv, key=cv.get)
    return best, np.round(notears_lasso(X, best), 3), cv


# ---------------------------------------------------------------------------
# DAG-DC-ADMM
# ---------------------------------------------------------------------------
def loss_func(X_i, W_i):
    m_i = X_i.shape[0]
    residuals = X_i @ W_i - X_i
    return 0.5 * np.sum(residuals ** 2) / m_i, (X_i.T @ residuals) / m_i


def h_func(W):
    d = W.shape[0]
    W = np.clip(W, -5.0, 5.0)
    E = slin.expm(W * W)
    h = float(np.clip(np.trace(E) - d, 0.0, 1e3))
    return h, np.clip(2 * W * E.T, -1e3, 1e3)


def update_W_i(i, X_i, W_list, W_list_new, theta_prev_admm, u, alpha, rho1, rho2, lambda1):
    d = X_i.shape[1]
    bounds = [(0, 0) if r == c else (0, None) for r in range(d) for c in range(d)] * 2

    def objective(w):
        w_pos, w_neg = w[:d * d].reshape((d, d)), w[d * d:].reshape((d, d))
        W_i = np.clip(w_pos - w_neg, -10, 10)
        loss, grad_loss = loss_func(X_i, W_i)
        if alpha[i] == 0.0 and rho1 == 0.0:
            h_val, grad_h = 0.0, np.zeros_like(W_i)
        else:
            h_val, grad_h = h_func(W_i)
        grad_dag = (alpha[i] + rho1 * h_val) * grad_h
        consensus, grad_cons = 0.0, np.zeros_like(W_i)
        if rho2 != 0:
            for j in range(len(W_list)):
                if j == i:
                    continue
                Wj = W_list_new[j] if j < i else W_list[j]
                res = theta_prev_admm[i, j] - (W_i - Wj) + u[i, j]
                consensus += 0.5 * rho2 * np.sum(res * res)
                grad_cons -= rho2 * res
        obj = loss + lambda1 * (w_pos.sum() + w_neg.sum()) + alpha[i] * h_val + 0.5 * rho1 * h_val ** 2 + consensus
        g = grad_loss + grad_dag + grad_cons
        return obj, np.concatenate([(g + lambda1).flatten(), (-g + lambda1).flatten()])

    W0 = W_list[i].copy()
    np.fill_diagonal(W0, 0.0)
    w_init = np.concatenate([np.maximum(W0, 0.0).reshape(-1), np.maximum(-W0, 0.0).reshape(-1)])
    x = sopt.minimize(objective, w_init, method="L-BFGS-B", jac=True, bounds=bounds).x
    return x[:d * d].reshape([d, d]) - x[d * d:].reshape([d, d])


def soft_threshold(diff, gamma):
    norm = np.linalg.norm(diff, ord="fro", axis=(1, 2), keepdims=True) + 1e-12
    return diff * np.maximum(1 - gamma / norm, 0)


def update_theta(W_list, u, rho2, lambda2, tau, theta_hat_prev):
    n = W_list.shape[0]
    theta = np.zeros_like(theta_hat_prev)
    ii, jj = np.triu_indices(n, k=1)
    W_diff = np.nan_to_num(W_list[ii] - W_list[jj] - u[ii, jj], nan=0.0, posinf=1e3, neginf=-1e3)
    norm_prev = np.linalg.norm(theta_hat_prev[ii, jj], ord="fro", axis=(1, 2))
    mask = np.zeros_like(norm_prev, dtype=bool) if np.max(norm_prev) < 1e-12 else (norm_prev < tau)
    theta_ij = np.zeros_like(W_diff)
    theta_ij[~mask] = W_diff[~mask]
    theta_ij[mask] = soft_threshold(W_diff[mask], lambda2 / rho2)
    theta[ii, jj] = theta_ij
    theta[jj, ii] = -theta_ij
    return theta


def optimize_admm(X, lambda1, lambda2, tau, theta_hat_prev, u_init, alpha_init, theta_prev_admm_init,
                  max_iter=100, rho1=0.1, rho2=0.4, dual_tol=1e-3, rho_max=1e16, W_init=None):
    n, d = len(X), X[0].shape[1]
    W_list = W_init.copy()
    W_list_new = np.zeros_like(W_list)
    u, alpha, theta_prev_admm = u_init.copy(), alpha_init.copy(), theta_prev_admm_init.copy()
    h_prev, eta_up, min_admm_iter = None, 5.0, 5
    off = ~np.eye(n, dtype=bool)
    for k in range(max_iter):
        for i in range(n):
            W_list_new[i] = update_W_i(i, X[i], W_list, W_list_new, theta_prev_admm, u, alpha, rho1, rho2, lambda1)
        theta = update_theta(W_list_new, u, rho2, lambda2, tau, theta_hat_prev)
        r = theta - (W_list_new[:, None] - W_list_new[None, :])
        s = rho2 * (theta - theta_prev_admm)
        r_norm = np.mean(np.linalg.norm(r[off], ord="fro", axis=(1, 2)))
        s_norm = np.mean(np.linalg.norm(s[off], ord="fro", axis=(1, 2)))
        theta_prev_admm = theta.copy()
        u += r
        h_vals = np.array([h_func(Wi)[0] for Wi in W_list_new])
        h_curr = float(np.sum(h_vals))
        if not np.isfinite(h_curr) or h_curr <= 0.0:
            h_curr = 1e-8
        if h_prev is not None and h_curr > 0.25 * h_prev and rho1 < rho_max:
            rho1 = min(rho1 * eta_up, rho_max)
        alpha += rho1 * h_vals
        if (k + 1) >= min_admm_iter and r_norm < dual_tol and s_norm < dual_tol * max(rho2, 1e-12):
            break
        W_list = W_list_new.copy()
        h_prev = h_curr
    return W_list, u, alpha, theta_prev_admm, rho1


def compute_Sm(X, W_list, theta_hat_prev, lambda1, lambda2, tau):
    n = len(X)
    S = 0.0
    for i in range(n):
        S += loss_func(X[i], W_list[i])[0] + lambda1 * np.sum(np.abs(W_list[i]))
        for j in range(i + 1, n):
            if np.linalg.norm(theta_hat_prev[i, j], "fro") < tau:
                S += lambda2 * np.linalg.norm(W_list[i] - W_list[j], "fro")
            else:
                S += lambda2 * tau
    return S


def _largest_jump_midpoint(heights):
    """Midpoint of the largest jump in merge heights, avoiding the first and last jump when possible."""
    diffs = np.diff(heights)
    order = np.argsort(diffs)[::-1]
    j = int(order[0])
    if j == 0 or j == len(diffs) - 1:
        for k in order:
            if 0 < k < len(diffs) - 1:
                j = int(k)
                break
    return 0.5 * (heights[j] + heights[j + 1])


def complete_linkage_clusters(dist_mat):
    D = 0.5 * (np.asarray(dist_mat, dtype=float) + np.asarray(dist_mat, dtype=float).T)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="complete")
    if len(Z) < 2:
        return [{i} for i in range(D.shape[0])], list(range(1, D.shape[0] + 1))
    labels = fcluster(Z, t=_largest_jump_midpoint(Z[:, 2]), criterion="distance")
    return [set(np.where(labels == lab)[0]) for lab in np.unique(labels)], labels.tolist()


def build_warm_start_W(X, lambda1):
    """Per-subject L1-penalized least squares, used as the initial W."""
    n, d = len(X), X[0].shape[1]
    W0 = np.zeros((n, d, d))
    zeros_theta = np.zeros((n, n, d, d))
    for i in range(n):
        W0[i] = update_W_i(i, X[i], W0, W0, zeros_theta, zeros_theta, np.zeros(n), 0.0, 0.0, lambda1)
        np.fill_diagonal(W0[i], 0.0)
    return W0


def optimize_dc_admm(X, lambda1=0.1, lambda2=0.5, tau=0.5, rho1=0.1, rho2=0.4,
                     max_dc_iter=10, max_admm_iter=20, thres_value=0.01):
    """DAG-DC-ADMM. Returns cluster DAGs, clusters (sets of subject indices), subject DAGs and labels."""
    n = len(X)
    W_list = build_warm_start_W(X, lambda1)
    theta_hat = W_list[:, None, :, :] - W_list[None, :, :, :]
    u, alpha = np.zeros_like(theta_hat), np.zeros(n)
    prev_S = float("inf")
    for _ in range(max_dc_iter):
        theta_hat_prev = theta_hat.copy()
        W_list, u, alpha, _, rho1 = optimize_admm(
            X, lambda1, lambda2, tau, theta_hat_prev, u, alpha, theta_hat.copy(),
            max_iter=max_admm_iter, rho1=rho1, rho2=rho2, W_init=W_list)
        for i in range(n):
            for j in range(i + 1, n):
                theta_hat[i, j] = W_list[i] - W_list[j]
                theta_hat[j, i] = -theta_hat[i, j]
        S = compute_Sm(X, W_list, theta_hat_prev, lambda1, lambda2, tau)
        if abs(S - prev_S) < 5e-3:
            break
        prev_S = S

    dist = np.linalg.norm(theta_hat, axis=(2, 3))
    clusters, labels = complete_linkage_clusters(dist)
    cluster_W = []
    for cluster in clusters:
        rep = np.stack([W_list[i] for i in cluster]).mean(axis=0)
        if thres_value is not None and thres_value > 0:
            rep[np.abs(rep) < thres_value] = 0.0
        np.fill_diagonal(rep, 0.0)
        cluster_W.append(rep)
    return cluster_W, clusters, W_list, labels


def cluster_reconstruction_error(cluster_W, clusters, X_val):
    errors = [reconstruction_error(cluster_W[k], X_val[i]) for k, c in enumerate(clusters) for i in c]
    return float(np.mean(errors))


def _evaluate(params, fold, X_train, X_val, max_dc_iter, max_admm_iter):
    lambda1, lambda2, tau, rho1, rho2 = params
    cluster_W, clusters, _, _ = optimize_dc_admm(
        X_train, lambda1, lambda2, tau, rho1, rho2, max_dc_iter, max_admm_iter, thres_value=0.01)
    return {"fold": fold, "lambda1": lambda1, "lambda2": lambda2, "tau": tau, "rho1": rho1, "rho2": rho2,
            "recon_error": cluster_reconstruction_error(cluster_W, clusters, X_val)}


def tune_dc_admm(X_list, param_grid, n_folds=3, seed=42, max_dc_iter=10, max_admm_iter=15, n_jobs=-1):
    """Grid search with K-fold CV over the rows of every subpopulation. Returns mean CV error per combination."""
    splits = [list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X)) for X in X_list]
    combos = list(product(*param_grid.values()))
    records = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_evaluate)(
            combo, f,
            [X_list[i][splits[i][f][0]] for i in range(len(X_list))],
            [X_list[i][splits[i][f][1]] for i in range(len(X_list))],
            max_dc_iter, max_admm_iter)
        for f in range(n_folds) for combo in combos)
    return (pd.DataFrame(records).groupby(list(param_grid))["recon_error"].mean()
            .reset_index().rename(columns={"recon_error": "recon_mean"}).sort_values("recon_mean"))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_adjacency(A_list, labels, titles, thr=0.04, linthresh=0.05, vmax_percentile=95,
                   fontsize=10, out_path=None, dpi=300):
    """Signed adjacency matrices side by side with one color bar. Entries with |A| <= thr are blank."""
    nz = np.concatenate([np.abs(A).ravel() for A in A_list])
    nz = nz[nz > thr]
    vmax = float(min(nz.max(), np.percentile(nz, vmax_percentile))) if nz.size else 1.0
    vmax = max(vmax, linthresh)
    norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax) if linthresh else TwoSlopeNorm(0.0, -vmax, vmax)
    cmap, d = "coolwarm", A_list[0].shape[0]
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    fig, axes = plt.subplots(1, len(A_list), figsize=(5 * len(A_list), 5))
    for ax, A, title in zip(np.atleast_1d(axes), A_list, titles):
        A = np.asarray(A, dtype=float)
        mask = np.abs(A) <= thr
        im = ax.imshow(np.ma.array(A, mask=mask), cmap=cmap, norm=norm)
        im.cmap.set_bad(color="white")
        ax.set_xticks(range(d))
        ax.set_yticks(range(d))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
        ax.set_yticklabels(labels, fontsize=12)
        ax.set_title(title, fontsize=16)
        for i in range(d):
            for j in range(d):
                if not mask[i, j]:
                    r, g, b, _ = mapper.to_rgba(A[i, j])
                    color = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.6 else "black"
                    ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", color=color,
                            fontsize=fontsize, fontweight="bold")
    fig.subplots_adjust(right=0.90)
    cbar = fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.015, 0.7]))
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout(rect=[0, 0.03, 0.90, 0.95])
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.show()
