"""
NOTEAR.py

This module provides functions to learn DAG structures using the NO TEARS approach:
- notears_linear_raw: solves the constrained optimization for linear SEMs.
- from_numpy_lasso2: wrapper to learn structure with L1 penalty.
"""

import numpy as np
from scipy.optimize import minimize
import scipy.optimize as sopt
import scipy.linalg as slin
from scipy.special import expit as sigmoid
import warnings


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


def _learn_structure_lasso(X,beta, bnds,max_iter= 100,h_tol= 1e-8,w_threshold= 0.0):
    n, d = X.shape

    def _h(W_vec):
        Wmat = W_vec.reshape([d, d])
        return np.trace(slin.expm(Wmat * Wmat)) - d

    def _func(w_vec):
        w_pos = w_vec[: d**2]
        w_neg = w_vec[d**2:]
        W = (w_pos - w_neg).reshape([d, d])
        loss = 0.5 / n * np.square(np.linalg.norm(X.dot(np.eye(d) - W), 'fro'))
        h_val = _h(np.concatenate([w_pos, w_neg]))
        return loss + 0.5 * rho * h_val**2 + alpha * h_val + beta * w_vec.sum()

    def _grad(w_vec):
        w_pos = w_vec[: d**2]
        w_neg = w_vec[d**2:]
        W = (w_pos - w_neg).reshape([d, d])
        loss_grad = -1.0 / n * X.T.dot(X).dot(np.eye(d) - W)
        expm_h = slin.expm(W * W)
        grad_h = expm_h.T * W * 2
        obj_grad = loss_grad + (rho * (np.trace(expm_h) - d) + alpha) * grad_h
        l1_grad = beta * np.ones(d**2)
        grad = np.zeros(2 * d**2)
        grad[:d**2] = obj_grad.flatten() + l1_grad
        grad[d**2:] = -obj_grad.flatten() + l1_grad
        return grad

    w_est = np.zeros(2 * d**2)
    rho, alpha = 1.0, 0.0
    h_val = np.inf

    for n_iter in range(max_iter):
        while rho < 1e20:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=_grad, bounds=bnds)
            w_new = sol.x
            h_new = _h(w_new)
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

    W_est = (w_est[: d**2].reshape([d, d]) - w_est[d**2:].reshape([d, d]))
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est
