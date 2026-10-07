# Reproducing the Experiments

This file lists the command for every experiment in the manuscript. Run all commands from the
repository root. Each command runs one seed. The paper uses 50 seeds (0-49) per setting:

```bash
export OMP_NUM_THREADS=1
for seed in $(seq 0 49); do
  python experiments/run_experiment.py --seed $seed --out_dir results <arguments below>
done
```

Seeds are independent, so they can run in parallel on any machine. One seed needs one CPU core
and at most 6 GB of memory. With the default grid, one seed takes about 1-2 hours for `n = 50`
and much longer for `n = 200`.

---

## Main two-cluster experiments (Section 5, Tables 1-4)

Six settings: `(n, m) in {(50, 300), (200, 50)}` and `sigma in {0.5, 1, 2}`.
The reported results use an 80-combination grid with `lambda2 in {0.1, 0.01, 0.001, 0.0001, 0.00001}`.

```bash
# Small-sample/long-series (repeat with --std 0.5 and --std 2, renaming --setting_name)
--setting_name N50_std1_m300_k06_04 --total_samples 50 --m 300 --std 1 \
  --cluster_proportions 0.6 0.4 --n_vars 5 --s0_list 5 5 \
  --lambda2_grid 0.01 0.001 0.0001 0.00001 0.1

# Large-sample/short-series
--setting_name N200_std1_m50_k06_04 --total_samples 200 --m 50 --std 1 \
  --cluster_proportions 0.6 0.4 --n_vars 5 --s0_list 5 5 \
  --lambda2_grid 0.01 0.001 0.0001 0.00001 0.1
```

This runs DAG-DC-ADMM and the Population, Individual, Oracle, and Two-step baselines.
`--methods` selects a subset, for example `--methods dag_dc_admm two_step`.

## Thresholding robustness (Appendix)

No extra runs are needed. The threshold sweep (0.01-0.10) is saved in the main-experiment
outputs (`skeleton_threshold_sweep` in `dag_dc_admm_results.json`).

## Comparison with a Dirichlet process mixture of Gaussian DAGs (Appendix)

The benchmark uses the R code of Castelletti and Consonni (2023), "Bayesian graphical modeling for
heterogeneous causal effects", *Statistics in Medicine* 42(1):15-32, available at
https://github.com/FedeCastelletti/bnp_mixture_causal_dags (commit `a8018a7`). That code is not
redistributed here. It was run with 25,000 MCMC iterations (5,000 burn-in) on the simulated data
of the main experiment (sigma = 1, seeds 0-24), with all measurements of all subjects stacked.

## Three-cluster scenario (Appendix)

True proportions 0.4/0.4/0.2, five edges per cluster. Settings: `n = 50, m = 300` and
`n = 200, m = 50`, each with `sigma in {0.5, 1, 2}`.

DAG-DC-ADMM uses `experiments/run_three_cluster.py`. The `SETTINGS` table at the top of the
script holds the data configuration and CV grid of each setting:

```bash
python experiments/run_three_cluster.py --setting N50_std1 --seed $seed --out_dir results/three_cluster/N50_std1
```

Settings: `N50_std05`, `N50_std1`, `N50_std2`, `N200_std05`, `N200_std1`, `N200_std2`.

The baselines (Population, Individual, Oracle, Two-step) come from the main runner:

```bash
--setting_name N50_std1_m300_k04_04_02 --total_samples 50 --m 300 --std 1 \
  --cluster_proportions 0.4 0.4 0.2 --n_vars 5 --s0_list 5 5 5 \
  --methods pooled individual oracle two_step
```

## Imbalanced cluster proportions (Appendix)

`n = 50`, `m = 300`, `sigma = 1`, default 64-combination grid:

```bash
--setting_name N50_std1_m300_k08_02 --total_samples 50 --m 300 --std 1 \
  --cluster_proportions 0.8 0.2 --n_vars 5 --s0_list 5 5
--setting_name N50_std1_m300_k09_01 --total_samples 50 --m 300 --std 1 \
  --cluster_proportions 0.9 0.1 --n_vars 5 --s0_list 5 5
```

The 0.6/0.4 row is the main experiment.

## Unequal measurement sizes (Appendix)

`--m 0` draws each `m_i` from `{50, 60, ..., 300}` using the seed:

```bash
--setting_name N50_std1_mlist_k06_04  --total_samples 50 --m 0 --std 1   \
  --cluster_proportions 0.6 0.4 --n_vars 5 --s0_list 5 5
--setting_name N50_std05_mlist_k06_04 --total_samples 50 --m 0 --std 0.5 \
  --cluster_proportions 0.6 0.4 --n_vars 5 --s0_list 5 5
```

## Higher-dimensional setting (Appendix)

`d = 15`, 50 edges per cluster, and the `tau` grid scaled by `d / 5`:

```bash
--setting_name N50_std1_m300_k06_04_p15_s050_tau3 --total_samples 50 --m 300 --std 1 \
  --cluster_proportions 0.6 0.4 --n_vars 15 --s0_list 50 50 --tau_grid 0.15 0.3 1.2 2.1
```

## Case study (Section 6)

The flow cytometry data of Sachs et al. (2005) are in `case_study/data/`
(source: Science 308(5721):523-529, doi:10.1126/science.1105809; see `case_study/data/README.md`).
Start Jupyter from `case_study/` and run `case_study.ipynb`. It

1. splits each perturbation into subpopulations (PCA to 80% variance, K-means with the smallest
   k whose silhouette is within 1% of the maximum, subpopulations with fewer than 20 cells removed),
2. fits the pooled-population and per-perturbation NOTEARS models with 3-fold CV,
3. runs the DAG-DC-ADMM grid search and fits the reported model,
4. builds the reconstruction-error table and the adjacency-matrix figure.

The DAG-DC-ADMM grid search (224 combinations x 3 folds) is the slow step. Set `RUN_CV = False`
in the notebook to skip it and only fit the reported model.

---

## Output files

`experiments/run_experiment.py` writes `results/<setting_name>/seed_<seed>/` with

- `dag_dc_admm_results.json`: CV table, selected `(lambda1, lambda2, tau)`, estimated clusters and
  cluster DAGs, clustering metrics, and graph metrics at thresholds 0.01-0.10;
- `summary_notear_pooled_CV.json`, `summary_notear_individual_CV.json`,
  `summary_notear_cluster_CV.json`: Population, Individual, and Oracle baselines;
- `summary_notear_individual_hierarchical.json`: Two-step baseline;
- `summary_all_methods.json`: data configuration and baseline results.

`experiments/run_three_cluster.py` writes `<out_dir>/seed_<seed>/` with the CV table
(`cv_results.csv`), the five best combinations for each CV criterion (`top5_*.json`), the final fit
for each criterion (`*_full_artifacts.json`, `skeleton_thresholds_*.csv`), and `summary.json`.

## Reproducibility notes

- Data generation, CV fold splits (`KFold(shuffle=True, random_state=seed)`), and NOTEARS
  baselines are seeded by `--seed`.
- `src/dagdc/dc_admm.py` is the exact optimizer version used for the reported two-cluster results.
- Run each seed with one thread (`OMP_NUM_THREADS=1`). The runners set this by default.
