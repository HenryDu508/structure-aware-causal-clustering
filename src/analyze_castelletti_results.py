#!/usr/bin/env python3
"""Aggregate Castelletti--Consonni outputs without pandas/scikit-learn."""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np


PRIMARY_THRESHOLD = 0.5
SENSITIVITY_THRESHOLDS = np.arange(0.1, 1.0, 0.1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer columns for empty output {path}")
        fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def seed_number(path):
    match = re.fullmatch(r"seed_(\d+)", path.name)
    return int(match.group(1)) if match else None


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def contingency_matrix(true_labels, estimated_labels):
    true_values, true_inverse = np.unique(true_labels, return_inverse=True)
    est_values, est_inverse = np.unique(estimated_labels, return_inverse=True)
    table = np.zeros((len(true_values), len(est_values)), dtype=int)
    np.add.at(table, (true_inverse, est_inverse), 1)
    return table


def mutual_information(table):
    n = table.sum()
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    value = 0.0
    for i, j in zip(*np.nonzero(table)):
        nij = table[i, j]
        value += nij / n * math.log(n * nij / (row_sums[i] * col_sums[j]))
    return value


def expected_mutual_information(table):
    n = int(table.sum())
    row_sums = table.sum(axis=1).astype(int)
    col_sums = table.sum(axis=0).astype(int)
    log_n_choose_b = {
        b: math.lgamma(n + 1) - math.lgamma(b + 1) - math.lgamma(n - b + 1)
        for b in np.unique(col_sums)
    }
    expected = 0.0
    for a in row_sums:
        for b in col_sums:
            lower = max(1, a + b - n)
            upper = min(a, b)
            for nij in range(lower, upper + 1):
                log_probability = (
                    math.lgamma(a + 1)
                    - math.lgamma(nij + 1)
                    - math.lgamma(a - nij + 1)
                    + math.lgamma(n - a + 1)
                    - math.lgamma(b - nij + 1)
                    - math.lgamma(n - a - b + nij + 1)
                    - log_n_choose_b[b]
                )
                expected += (
                    math.exp(log_probability)
                    * nij
                    / n
                    * math.log(n * nij / (a * b))
                )
    return expected


def clustering_metrics(true_labels, estimated_labels):
    table = contingency_matrix(true_labels, estimated_labels)
    n = int(table.sum())
    row_sums = table.sum(axis=1)
    col_sums = table.sum(axis=0)
    mi = mutual_information(table)
    h_true = entropy_from_counts(row_sums)
    h_est = entropy_from_counts(col_sums)
    homogeneity = 1.0 if h_true == 0 else mi / h_true
    completeness = 1.0 if h_est == 0 else mi / h_est
    v_measure = (
        0.0
        if homogeneity + completeness == 0
        else 2 * homogeneity * completeness / (homogeneity + completeness)
    )

    sum_pairs = lambda values: float(np.sum(values * (values - 1) / 2))
    nij_pairs = sum_pairs(table)
    row_pairs = sum_pairs(row_sums)
    col_pairs = sum_pairs(col_sums)
    total_pairs = n * (n - 1) / 2
    expected_pairs = row_pairs * col_pairs / total_pairs
    ari_denominator = 0.5 * (row_pairs + col_pairs) - expected_pairs
    ari = 1.0 if ari_denominator == 0 else (nij_pairs - expected_pairs) / ari_denominator

    emi = expected_mutual_information(table)
    normalizer = 0.5 * (h_true + h_est)
    ami_denominator = normalizer - emi
    ami = 1.0 if abs(ami_denominator) < np.finfo(float).eps else (mi - emi) / ami_denominator
    return {
        "clustering_ari": ari,
        "clustering_ami": ami,
        "clustering_homogeneity": homogeneity,
        "clustering_completeness": completeness,
        "clustering_v_measure": v_measure,
    }


def read_edge_probabilities(path, n_subjects, n_vars):
    probabilities = np.full((n_subjects, n_vars, n_vars), np.nan)
    rows = read_csv_rows(path)
    if len(rows) != n_subjects * n_vars * n_vars:
        raise ValueError(f"{path}: unexpected number of edge rows")
    for row in rows:
        i = int(row["subject_id"]) - 1
        u = int(row["from_node"]) - 1
        v = int(row["to_node"]) - 1
        if np.isfinite(probabilities[i, u, v]):
            raise ValueError(f"{path}: duplicate subject/edge row")
        probabilities[i, u, v] = float(row["posterior_probability"])
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{path}: missing or non-finite probabilities")
    return probabilities


def graph_metrics_one(true_w, edge_probabilities, threshold):
    true_dag = np.abs(true_w) > 0
    estimated_dag = edge_probabilities > threshold
    np.fill_diagonal(true_dag, False)
    np.fill_diagonal(estimated_dag, False)
    n_vars = true_dag.shape[0]

    directed_mask = ~np.eye(n_vars, dtype=bool)
    truth = true_dag[directed_mask]
    estimate = estimated_dag[directed_mask]
    tp = int(np.sum(truth & estimate))
    fp = int(np.sum(~truth & estimate))
    fn = int(np.sum(truth & ~estimate))
    tn = int(np.sum(~truth & ~estimate))
    precision = tp / (tp + fp) if tp + fp else 0.0

    true_skeleton = true_dag | true_dag.T
    estimated_skeleton = estimated_dag | estimated_dag.T
    skeleton_mask = np.triu(np.ones((n_vars, n_vars), dtype=bool), k=1)
    skeleton_truth = true_skeleton[skeleton_mask]
    skeleton_estimate = estimated_skeleton[skeleton_mask]
    stp = int(np.sum(skeleton_truth & skeleton_estimate))
    sfp = int(np.sum(~skeleton_truth & skeleton_estimate))
    sfn = int(np.sum(skeleton_truth & ~skeleton_estimate))
    stn = int(np.sum(~skeleton_truth & ~skeleton_estimate))
    return {
        "dag_fdr": 1.0 - precision,
        "dag_tpr": tp / (tp + fn) if tp + fn else 0.0,
        "dag_fpr": fp / (fp + tn) if fp + tn else 0.0,
        "dag_tnr": tn / (tn + fp) if tn + fp else 0.0,
        "dag_shd": float(fp + fn),
        "dag_nnz": float(tp + fp),
        "skeleton_tpr": stp / (stp + sfn) if stp + sfn else 0.0,
        "skeleton_fdr": sfp / (stp + sfp) if stp + sfp else 0.0,
        "skeleton_tnr": stn / (stn + sfp) if stn + sfp else 0.0,
        "skeleton_shd": float(sfp + sfn),
    }


def macro_graph_metrics(subject_dags, edge_probabilities, threshold):
    true_dags = np.abs(subject_dags) > 0
    estimated_dags = edge_probabilities > threshold
    n_units, n_vars, _ = true_dags.shape
    diagonal = np.arange(n_vars)
    true_dags[:, diagonal, diagonal] = False
    estimated_dags[:, diagonal, diagonal] = False

    directed_mask = ~np.eye(n_vars, dtype=bool)
    truth = true_dags[:, directed_mask]
    estimate = estimated_dags[:, directed_mask]
    tp = np.sum(truth & estimate, axis=1)
    fp = np.sum(~truth & estimate, axis=1)
    fn = np.sum(truth & ~estimate, axis=1)
    tn = np.sum(~truth & ~estimate, axis=1)
    precision = np.divide(
        tp,
        tp + fp,
        out=np.zeros(n_units, dtype=float),
        where=(tp + fp) != 0,
    )

    true_skeletons = true_dags | np.swapaxes(true_dags, 1, 2)
    estimated_skeletons = estimated_dags | np.swapaxes(
        estimated_dags, 1, 2
    )
    skeleton_mask = np.triu(np.ones((n_vars, n_vars), dtype=bool), k=1)
    skeleton_truth = true_skeletons[:, skeleton_mask]
    skeleton_estimate = estimated_skeletons[:, skeleton_mask]
    stp = np.sum(skeleton_truth & skeleton_estimate, axis=1)
    sfp = np.sum(~skeleton_truth & skeleton_estimate, axis=1)
    sfn = np.sum(skeleton_truth & ~skeleton_estimate, axis=1)
    stn = np.sum(~skeleton_truth & ~skeleton_estimate, axis=1)

    safe_ratio = lambda numerator, denominator: np.divide(
        numerator,
        denominator,
        out=np.zeros(n_units, dtype=float),
        where=denominator != 0,
    )
    rows = {
        "dag_fdr": 1.0 - precision,
        "dag_tpr": safe_ratio(tp, tp + fn),
        "dag_fpr": safe_ratio(fp, fp + tn),
        "dag_tnr": safe_ratio(tn, tn + fp),
        "dag_shd": (fp + fn).astype(float),
        "dag_nnz": (tp + fp).astype(float),
        "skeleton_tpr": safe_ratio(stp, stp + sfn),
        "skeleton_fdr": safe_ratio(sfp, stp + sfp),
        "skeleton_tnr": safe_ratio(stn, stn + sfp),
        "skeleton_shd": (sfp + sfn).astype(float),
    }
    return {name: float(np.mean(values)) for name, values in rows.items()}


def rank_average_ties(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def pooled_edge_scores(subject_dags, edge_probabilities):
    n_vars = subject_dags.shape[1]
    mask = ~np.eye(n_vars, dtype=bool)
    truth = (np.abs(subject_dags) > 0)[:, mask].reshape(-1).astype(bool)
    scores = edge_probabilities[:, mask].reshape(-1)
    n_positive = int(truth.sum())
    n_negative = len(truth) - n_positive
    ranks = rank_average_ties(scores)
    auroc = (ranks[truth].sum() - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    )

    order = np.argsort(-scores, kind="mergesort")
    sorted_truth = truth[order]
    sorted_scores = scores[order]
    ap = 0.0
    tp = 0
    previous_tp = 0
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        tp += int(sorted_truth[index:end].sum())
        precision = tp / end
        ap += (tp - previous_tp) / n_positive * precision
        previous_tp = tp
        index = end
    return {"edge_auroc": float(auroc), "edge_average_precision": float(ap)}


def similarity_diagnostics(similarity, labels):
    if similarity.shape != (len(labels), len(labels)):
        raise ValueError("Posterior-similarity matrix has an unexpected shape")
    upper = np.triu_indices(len(labels), k=1)
    same_cluster = labels[upper[0]] == labels[upper[1]]
    values = similarity[upper]
    within = float(np.mean(values[same_cluster]))
    between = float(np.mean(values[~same_cluster]))
    return {
        "posterior_similarity_within": within,
        "posterior_similarity_between": between,
        "posterior_similarity_gap": within - between,
    }


def analyze_seed(result_dir, data_root):
    seed = seed_number(result_dir)
    with np.load(data_root / result_dir.name / "ground_truth.npz") as truth_file:
        if "cc_unit_dags" in truth_file and "cc_unit_cluster_labels" in truth_file:
            subject_dags = np.asarray(truth_file["cc_unit_dags"], dtype=float)
            saved_labels = np.asarray(
                truth_file["cc_unit_cluster_labels"], dtype=int
            )
            evaluation_unit = "CC input row"
        else:
            subject_dags = np.asarray(truth_file["subject_dags"], dtype=float)
            saved_labels = np.asarray(truth_file["cluster_labels"], dtype=int)
            evaluation_unit = "subject"

    assignments = sorted(
        read_csv_rows(result_dir / "cluster_assignments.csv"),
        key=lambda row: int(row["subject_id"]),
    )
    true_labels = np.asarray([int(row["true_cluster_label"]) for row in assignments])
    estimated_labels = np.asarray(
        [int(row["estimated_cluster_label"]) for row in assignments]
    )
    if len(assignments) != len(subject_dags) or not np.array_equal(
        true_labels, saved_labels
    ):
        raise ValueError(
            f"seed {seed}: ground-truth labels or evaluation-unit count mismatch"
        )

    n_subjects, n_vars, _ = subject_dags.shape
    probabilities = read_edge_probabilities(
        result_dir / "subject_edge_probabilities.csv", n_subjects, n_vars
    )
    runtime = read_csv_rows(result_dir / "runtime.csv")[0]
    true_k = int(np.unique(true_labels).size)
    estimated_k = int(np.unique(estimated_labels).size)
    acyclic = [row["estimated_graph_is_acyclic"].strip().upper() == "TRUE" for row in assignments]

    primary = {
        "seed": seed,
        "evaluation_unit": evaluation_unit,
        "n_evaluation_units": n_subjects,
        "n_subjects": n_subjects,
        "n_vars": n_vars,
        "true_k": true_k,
        "estimated_k": estimated_k,
        "correct_k": int(estimated_k == true_k),
        "absolute_k_error": abs(estimated_k - true_k),
        "runtime_elapsed_seconds": float(runtime["elapsed_seconds"]),
        "acyclic_fraction_at_0.5": float(np.mean(acyclic)),
    }
    primary.update(clustering_metrics(true_labels, estimated_labels))
    primary.update(macro_graph_metrics(subject_dags, probabilities, PRIMARY_THRESHOLD))
    primary.update(pooled_edge_scores(subject_dags, probabilities))
    similarity_path = result_dir / "posterior_similarity.csv"
    if similarity_path.is_file():
        similarity = np.genfromtxt(
            similarity_path, delimiter=",", skip_header=1
        )
        primary.update(similarity_diagnostics(similarity, true_labels))

    sensitivity = []
    for threshold in SENSITIVITY_THRESHOLDS:
        row = {"seed": seed, "threshold": float(np.round(threshold, 1))}
        row.update(macro_graph_metrics(subject_dags, probabilities, threshold))
        sensitivity.append(row)
    return primary, sensitivity, evaluation_unit


def summarize_rows(rows, excluded=()):
    summary = []
    for metric in rows[0]:
        if metric in excluded:
            continue
        try:
            values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        except (TypeError, ValueError):
            continue
        summary.append(
            {
                "metric": metric,
                "n_seeds": len(values),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "sem": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan,
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return summary


def main():
    args = parse_args()
    results_root = args.results_dir.resolve(strict=True)
    data_root = args.data_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve() if args.output_dir else results_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(
        [path for path in results_root.iterdir() if path.is_dir() and seed_number(path) is not None],
        key=seed_number,
    )
    primary_rows, sensitivity_rows, failures = [], [], []
    evaluation_units = set()
    for result_dir in seed_dirs:
        try:
            primary, sensitivity, evaluation_unit = analyze_seed(
                result_dir, data_root
            )
            primary_rows.append(primary)
            sensitivity_rows.extend(sensitivity)
            evaluation_units.add(evaluation_unit)
        except Exception as error:
            failures.append({"seed": seed_number(result_dir), "error": str(error)})
    if not primary_rows:
        raise RuntimeError("No seed could be analyzed")

    summary = summarize_rows(
        primary_rows,
        {
            "seed",
            "evaluation_unit",
            "n_evaluation_units",
            "n_subjects",
            "n_vars",
            "true_k",
        },
    )
    threshold_summary = []
    for threshold in SENSITIVITY_THRESHOLDS:
        value = float(np.round(threshold, 1))
        subset = [row for row in sensitivity_rows if row["threshold"] == value]
        for row in summarize_rows(subset, {"seed", "threshold"}):
            threshold_summary.append({"threshold": value, **row})
    frequencies = Counter(row["estimated_k"] for row in primary_rows)
    k_rows = [
        {"estimated_k": k, "n_seeds": count, "proportion": count / len(primary_rows)}
        for k, count in sorted(frequencies.items())
    ]

    write_csv(output_dir / "per_seed_metrics.csv", primary_rows)
    write_csv(output_dir / "summary_metrics.csv", summary)
    write_csv(output_dir / "threshold_sensitivity_per_seed.csv", sensitivity_rows)
    write_csv(output_dir / "threshold_sensitivity_summary.csv", threshold_summary)
    write_csv(output_dir / "estimated_k_frequency.csv", k_rows)
    with (output_dir / "analysis_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "results_dir": str(results_root),
                "data_dir": str(data_root),
                "primary_edge_threshold": PRIMARY_THRESHOLD,
                "threshold_selection_used_ground_truth": False,
                "n_seed_directories": len(seed_dirs),
                "n_seeds_analyzed": len(primary_rows),
                "failures": failures,
                "evaluation_units": sorted(evaluation_units),
                "posterior_similarity_optional": True,
                "graph_aggregation": (
                    "macro average over CC evaluation units within each seed"
                ),
                "edge_score_aggregation": (
                    "pooled CC-evaluation-unit edge pairs within each seed"
                ),
                "edge_orientation": "from_node -> to_node",
            },
            file,
            indent=2,
        )

    selected = {
        "clustering_ari", "clustering_ami", "clustering_v_measure", "correct_k",
        "estimated_k", "dag_tpr", "dag_fdr", "dag_shd", "skeleton_tpr",
        "skeleton_fdr", "skeleton_shd", "edge_auroc", "edge_average_precision",
        "runtime_elapsed_seconds",
    }
    print(f"Analyzed {len(primary_rows)}/{len(seed_dirs)} seeds; failures={len(failures)}")
    for row in summary:
        if row["metric"] in selected:
            print(f"{row['metric']:30s} {row['mean']:.6f} +/- {row['std']:.6f}")
    print(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
