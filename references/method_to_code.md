# Method-to-Code Reference

This file maps each component of DAG-DC-ADMM in the manuscript to the function that implements it.
All paths are relative to the repository root. The main library is `src/dagdc/`.

---

## Model and objective

For subjects `i = 1, ..., n`, each with data `X_i` of size `m_i x d`, DAG-DC-ADMM minimizes

```
sum_i  (1 / 2 m_i) ||X_i - X_i W_i||_F^2  +  lambda1 sum_i ||W_i||_1
       +  lambda2 sum_{i<j} min(||W_i - W_j||_F, tau)
subject to  h(W_i) = 0 for every i,
```

where `h` is the NOTEARS acyclicity function and the last term is the groupwise truncated Lasso
(gTLP) fusion penalty.

| Component | Function | File |
|---|---|---|
| Second-moment matrices `S_i = X_i^T X_i / m_i`, computed once | `precompute_second_moment` | `src/dagdc/dc_admm.py` |
| Least-squares SEM loss and its gradient, written with `S_i` | `loss_func` | `src/dagdc/dc_admm.py` |
| Acyclicity function `h(W)` and its gradient | `h_func` | `src/dagdc/dc_admm.py` |
| Objective value of the DC subproblem `S^(m)(W, theta)` | `compute_Sm` | `src/dagdc/dc_admm.py` |

## Optimization

| Step | What it does | Function | File |
|---|---|---|---|
| Initialization | Per-subject L1 fits with acyclicity and fusion switched off | `build_warm_start_W` | `src/dagdc/dc_admm.py` |
| DC outer loop | Linearizes the truncated penalty at the previous `theta_hat`. Stops when the change in `S^(m)` is below `5e-3` or after `max_dc_iter` iterations | `optimize_dc_admm` | `src/dagdc/dc_admm.py` |
| ADMM inner loop | Alternates the `W`, `theta` and dual updates. Stops at `max_admm_iter`, or after at least 5 iterations once both the primal and dual residuals are below their tolerances (`dual_tol`) | `optimize_admm` | `src/dagdc/dc_admm.py` |
| `W_i` update | L-BFGS-B on the split `W_i = W_i^+ - W_i^-` (nonnegative bounds, zero diagonal), with the augmented-Lagrangian acyclicity term `alpha_i h + (rho1 / 2) h^2` and the ADMM fusion term | `update_W_i` | `src/dagdc/dc_admm.py` |
| `theta_ij` update | Pairs with `||theta_hat_ij||_F < tau` get group soft-thresholding with threshold `lambda2 / rho2`. Other pairs are not penalized | `update_theta`, `soft_threshold` | `src/dagdc/dc_admm.py` |
| Dual updates | Scaled fusion duals `u`. Acyclicity multipliers `alpha`, with `rho1` multiplied by 5 when `h` is not reduced below a quarter of its previous value | inside `optimize_admm` | `src/dagdc/dc_admm.py` |
| Cluster assignment | Complete-linkage clustering of `||theta_hat_ij||_F`, cut at `tau` | `complete_linkage_clusters` | `src/dagdc/dc_admm.py` |
| Cluster DAGs | Mean of the member `W_i`, entries below `thres_value = 0.01` set to zero | end of `optimize_dc_admm` | `src/dagdc/dc_admm.py` |

`optimize_dc_admm` returns `(cluster_W, clusters, W_list, label_list, W_cluster_list)`.

## Model selection and evaluation

| Step | Function | File |
|---|---|---|
| 3-fold CV over `(lambda1, lambda2, tau)` with shuffled folds within each subject. Selects the minimum mean held-out reconstruction error | `tune_hyperparameters`, `evaluate_params` | `src/dagdc/cross_validation.py` |
| Reconstruction error of a clustered fit | `compute_total_reconstruction_error` | `src/dagdc/cross_validation.py` |
| Skeleton and DAG metrics (FDR, TPR, FPR, SHD) after thresholding | `count_accuracy_with_skeleton`, `average_skeleton_accuracy` | `src/dagdc/cross_validation.py` |
| Clustering metrics (ARI, AMI, homogeneity, completeness, V-measure) | `clustering_overall_metrics` | `src/dagdc/cross_validation.py` |

## Data generation

| Step | Function | File |
|---|---|---|
| Random DAG with a given number of edges and weights in `[-0.5, -0.2] U [0.2, 0.5]` | `generate_strict_dag_type` | `src/dagdc/data_generation.py` |
| Samples from a linear SEM with Gaussian noise | `generate_X_i` | `src/dagdc/data_generation.py` |
| Clustered subjects, one DAG per cluster, equal or unequal `m_i` | `generate_clustered_data` | `src/dagdc/data_generation.py` |

## Baselines

| Baseline | Function | File |
|---|---|---|
| Population: one NOTEARS DAG for all subjects | `run_notear_experiment_pooled` | `src/dagdc/notears.py` |
| Individual: one NOTEARS DAG per subject | `run_notear_experiment_individual` | `src/dagdc/notears.py` |
| Oracle: one NOTEARS DAG per true cluster | `run_notear_experiment_cluster` | `src/dagdc/notears.py` |
| Two-step: Individual NOTEARS, then complete-linkage clustering cut at `--two_step_tau` | `run_two_step_experiment` | `experiments/run_experiment.py` |
| Dirichlet process mixture of Gaussian DAGs | not redistributed (see `references/reproducing_experiments.md`) | |

The NOTEARS penalty `lambda1` of each baseline is chosen by 3-fold CV over `--baseline_lambda1_grid`.

## Other code versions

- `src/dagdc/three_cluster/` is the library version used for the DAG-DC-ADMM runs in the
  three-cluster scenario. It also records the penalized-likelihood and covariance CV criteria.
- `case_study/case_utils.py` holds the DAG-DC-ADMM version used for the case study. It works on
  the data matrices directly. Its final clustering step cuts the complete-linkage dendrogram at
  the midpoint of the largest height jump (`_largest_jump_midpoint`) instead of at `tau`.
