# Example

`two_cluster_example.py` runs DAG-DC-ADMM end to end on one simulated data set. It uses the main
simulation setting of the paper with one seed and a small CV grid, so it finishes in a few minutes
on a laptop (about 7 minutes on one CPU core).

```bash
python examples/two_cluster_example.py
```

## What it does

1. Generates 50 subjects in two clusters (60% / 40%). Each cluster has its own 5-node DAG with
   5 edges. Each subject has 300 measurements from a linear SEM with noise standard deviation 1.
2. Selects `(lambda1, lambda2, tau)` by 3-fold CV over a 2 x 2 x 2 grid. The criterion is the mean
   held-out reconstruction error.
3. Refits DAG-DC-ADMM on all data with the selected values.
4. Prints the true and estimated cluster labels, the clustering metrics, the graph-recovery
   metrics, and the estimated DAG of each cluster.

The solver also prints one line per DC iteration.

## Expected output (seed 0)

```
Selected: {'lambda1': 0.01, 'lambda2': 0.001, 'tau': 0.7} (CV reconstruction error 4.9509)
ARI 0.810, AMI 0.755, clusters found 5
Reconstruction error: estimated 4.9526, true graphs 4.9667
```

All 30 subjects of the first true cluster form one estimated cluster. The 20 subjects of the
second true cluster are split into clusters of 10 and 8 subjects and two singletons. This is one
seed with a reduced grid. The paper's full grid is given in
[`references/reproducing_experiments.md`](../references/reproducing_experiments.md).

## Using your own data

Pass a list of `m_i x d` arrays, one per subject, to `optimize_dc_admm`:

```python
import sys
sys.path.insert(0, "src")
from dagdc.dc_admm import optimize_dc_admm

cluster_W, clusters, W_list, labels, W_by_subject = optimize_dc_admm(
    X_list, lambda1=0.01, lambda2=0.001, tau=0.7, rho1=0.1, rho2=0.05,
    max_dc_iter=10, max_admm_iter=15, thres_value=0.01)
```

`cluster_W` holds one DAG per estimated cluster, `clusters` the subject indices of each cluster,
and `labels` the cluster label of each subject (starting at 1). Standardize each variable first
if the variables are on different scales.
