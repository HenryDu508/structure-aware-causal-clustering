#!/usr/bin/env python3
"""Adapt the original experiment data for the Castelletti--Consonni code."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from data_generation import generate_clustered_data


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate data with the repository's original simulation model "
            "and export either one q-vector per subject or all raw rows for "
            "mcmc_mixture_dags()."
        )
    )
    parser.add_argument("--seed", type=int, nargs="+", required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("castelletti_consonni_data"),
    )
    parser.add_argument(
        "--setting-name", type=str, default="N50_std1_m300_k06_04"
    )
    parser.add_argument("--total-samples", type=int, default=50)
    parser.add_argument(
        "--cluster-proportions", type=float, nargs="+", default=[0.6, 0.4]
    )
    parser.add_argument("--n-vars", type=int, default=5)
    parser.add_argument("--m", type=int, default=300)
    parser.add_argument("--std", type=float, default=1.0)
    parser.add_argument("--s0-list", type=int, nargs="+", default=[5, 5])
    parser.add_argument(
        "--input-mode",
        choices=("aggregate", "rows"),
        default="aggregate",
        help=(
            "aggregate: one sum/sqrt(m_i) row per subject (legacy); "
            "rows: vertically stack all m_i raw observations"
        ),
    )
    return parser.parse_args()


def _is_dag(weight_matrix):
    adjacency = np.asarray(weight_matrix) != 0
    indegree = adjacency.sum(axis=0).astype(int)
    available = list(np.flatnonzero(indegree == 0))
    visited = 0

    while available:
        node = available.pop()
        visited += 1
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                available.append(int(child))

    return visited == adjacency.shape[0]


def _write_matrix_csv(path, matrix):
    np.savetxt(path, matrix, delimiter=",", fmt="%.17g")


def _write_cluster_labels(path, cluster_labels):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["subject_id", "cluster_label"])
        for subject_index, label in enumerate(cluster_labels, start=1):
            writer.writerow([subject_index, int(label)])


def _write_row_mapping(
    path, cc_row_ids, original_subject_ids, replicate_ids, cluster_labels
):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["cc_row_id", "original_subject_id", "replicate_id", "cluster_label"]
        )
        writer.writerows(
            zip(cc_row_ids, original_subject_ids, replicate_ids, cluster_labels)
        )


def _to_castelletti_input(X_list):
    """Map each subject's m_i x q sample to one N(0, Sigma_k) row.

    Under the original generator, rows are conditionally iid
    N(0, Sigma_k). Therefore sum_t X_it / sqrt(m_i) has exactly the same
    component distribution required by the Castelletti--Consonni mixture.
    """
    return np.vstack(
        [X_i.sum(axis=0) / np.sqrt(X_i.shape[0]) for X_i in X_list]
    )


def _build_cc_units(X_list, subject_cluster_labels, input_mode):
    observations_per_subject = np.asarray(
        [X_i.shape[0] for X_i in X_list], dtype=np.int64
    )
    n_subjects = len(X_list)

    if input_mode == "aggregate":
        cc_input = _to_castelletti_input(X_list)
        original_subject_ids = np.arange(1, n_subjects + 1, dtype=np.int64)
        replicate_ids = np.zeros(n_subjects, dtype=np.int64)
        cc_cluster_labels = subject_cluster_labels.copy()
    else:
        cc_input = np.vstack(X_list)
        original_subject_ids = np.repeat(
            np.arange(1, n_subjects + 1, dtype=np.int64),
            observations_per_subject,
        )
        replicate_ids = np.concatenate(
            [np.arange(1, m_i + 1, dtype=np.int64) for m_i in observations_per_subject]
        )
        cc_cluster_labels = np.repeat(
            subject_cluster_labels, observations_per_subject
        )

    cc_row_ids = np.arange(1, len(cc_input) + 1, dtype=np.int64)
    return (
        cc_input,
        cc_cluster_labels,
        cc_row_ids,
        original_subject_ids,
        replicate_ids,
        observations_per_subject,
    )


def generate_and_save(seed, args):
    X_list, W_list_gt, W_centers_gt, clusters_gt, label_truth = (
        generate_clustered_data(
            total_samples=args.total_samples,
            cluster_proportions=args.cluster_proportions,
            n_vars=args.n_vars,
            m=args.m,
            W_pos_range=(0.2, 0.5),
            W_neg_range=(-0.5, -0.2),
            mean=0.0,
            std=args.std,
            s0_list=args.s0_list,
            graph_type="UR",
            seed=seed,
            permute=True,
        )
    )

    raw_subject_data = np.stack(X_list, axis=0)
    subject_dags = np.stack(W_list_gt, axis=0)
    cluster_dags = np.stack(W_centers_gt, axis=0)
    subject_cluster_labels = np.asarray(label_truth, dtype=np.int64)
    (
        castelletti_input,
        cc_cluster_labels,
        cc_row_ids,
        cc_original_subject_ids,
        cc_replicate_ids,
        observations_per_subject,
    ) = _build_cc_units(
        X_list, subject_cluster_labels, args.input_mode
    )
    cc_unit_dags = subject_dags[cc_original_subject_ids - 1]

    seed_dir = args.out_dir / args.setting_name / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    input_path = seed_dir / "castelletti_input.csv"
    labels_path = seed_dir / "cluster_labels.csv"
    row_mapping_path = seed_dir / "row_mapping.csv"
    raw_data_path = seed_dir / f"raw_subject_data_m{args.m}.npz"
    truth_path = seed_dir / "ground_truth.npz"
    metadata_path = seed_dir / "metadata.json"

    _write_matrix_csv(input_path, castelletti_input)
    _write_cluster_labels(labels_path, cc_cluster_labels)
    _write_row_mapping(
        row_mapping_path,
        cc_row_ids,
        cc_original_subject_ids,
        cc_replicate_ids,
        cc_cluster_labels,
    )
    np.savez_compressed(raw_data_path, X=raw_subject_data)
    np.savez_compressed(
        truth_path,
        subject_dags=subject_dags,
        cluster_dags=cluster_dags,
        cluster_labels=subject_cluster_labels,
        observations_per_subject=observations_per_subject,
        cc_unit_dags=cc_unit_dags,
        cc_unit_cluster_labels=cc_cluster_labels,
        cc_unit_original_subject_ids=cc_original_subject_ids,
        cc_unit_replicate_ids=cc_replicate_ids,
    )

    cluster_counts = {
        str(cluster_index + 1): int(len(members))
        for cluster_index, members in enumerate(clusters_gt)
    }
    metadata = {
        "schema_version": 2,
        "seed": int(seed),
        "data_generation": {
            "source": "experiments.data_generation.generate_clustered_data",
            "total_samples": args.total_samples,
            "cluster_proportions": args.cluster_proportions,
            "n_vars": args.n_vars,
            "m": args.m,
            "mean": 0.0,
            "std": args.std,
            "s0_list": args.s0_list,
            "graph_type": "UR",
            "weight_positive_range": [0.2, 0.5],
            "weight_negative_range": [-0.5, -0.2],
            "permute": True,
        },
        "castelletti_input": {
            "file": input_path.name,
            "shape": list(castelletti_input.shape),
            "csv_header": False,
            "input_mode": args.input_mode,
            "row_unit": "subject" if args.input_mode == "aggregate" else "observation",
            "column_unit": "DAG variable",
            "transformation": (
                "X_CC[i, :] = sum_t X_i[t, :] / sqrt(m_i)"
                if args.input_mode == "aggregate"
                else "X_CC = vstack(X_1, ..., X_N); no aggregation"
            ),
            "distributional_justification": (
                "Rows are conditionally iid draws from the cluster-specific "
                "Gaussian DAG distribution."
            ),
            "information_limitation": (
                "CC treats each input row as an exchangeable mixture unit and "
                "does not enforce common assignments within original subjects."
                if args.input_mode == "rows"
                else "CC receives one q-vector per original subject."
            ),
            "assumptions": [
                "rows within each subject are conditionally iid",
                "the Gaussian SEM has zero mean",
            ],
        },
        "saved_files": {
            "cluster_labels": labels_path.name,
            "row_mapping": row_mapping_path.name,
            "raw_subject_data": raw_data_path.name,
            "ground_truth": truth_path.name,
        },
        "array_shapes": {
            "raw_subject_data": list(raw_subject_data.shape),
            "subject_dags": list(subject_dags.shape),
            "cluster_dags": list(cluster_dags.shape),
            "subject_cluster_labels": list(subject_cluster_labels.shape),
            "cc_unit_dags": list(cc_unit_dags.shape),
            "cc_unit_cluster_labels": list(cc_cluster_labels.shape),
        },
        "cluster_counts": cluster_counts,
        "cluster_labels_are_one_based": True,
        "cc_row_ids_are_one_based": True,
        "original_subject_ids_are_one_based": True,
        "replicate_ids_are_one_based_in_rows_mode": True,
        "edge_orientation": "W[u, v] != 0 represents u -> v",
        "sanity_checks": {
            "all_subject_dags_acyclic": bool(
                all(_is_dag(W_i) for W_i in subject_dags)
            ),
            "all_cluster_dags_acyclic": bool(
                all(_is_dag(W_k) for W_k in cluster_dags)
            ),
            "finite_castelletti_input": bool(
                np.isfinite(castelletti_input).all()
            ),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(
        f"seed={seed}: wrote {castelletti_input.shape} R input and "
        f"{raw_subject_data.shape} raw data to {seed_dir}"
    )


def main():
    args = _parse_args()
    if args.m <= 0:
        raise ValueError("m must be positive.")
    if len(args.cluster_proportions) != len(args.s0_list):
        raise ValueError("cluster_proportions and s0_list must have equal length.")
    if not np.isclose(sum(args.cluster_proportions), 1.0):
        raise ValueError("cluster_proportions must sum to one.")

    for seed in args.seed:
        generate_and_save(seed, args)


if __name__ == "__main__":
    main()
