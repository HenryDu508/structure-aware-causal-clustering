"""Run the existing two-step method on the verified legacy three-cluster design.

Example: python run_three_cluster_two_step.py --setting N50_std1 --seed 0
Use --dry-run to inspect the exact command without running NOTEARS.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {f"N{n}_std{token}": (n, m, std)
            for n, m in ((50, 300), (200, 50))
            for token, std in (("05", .5), ("1", 1.), ("2", 2.))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", required=True, choices=SETTINGS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--tau", type=float, default=.7,
                        help="Default 0.7 matches the existing two-cluster two-step launchers; not a per-seed CV choice.")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "results_three_cluster_two_step")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.seed < 50:
        parser.error("Formal supplement runs use seed 0–49; do not mix the extra legacy N50 seeds.")
    if not 0 <= args.tau < float("inf"):
        parser.error("tau must be finite and nonnegative")
    n, m, std = SETTINGS[args.setting]
    name = f"{args.setting}_m{m}_k04_04_02"
    script = Path(__file__).with_name("experiment_notear_individual_hierarchical.py")
    output = args.out_dir.resolve() / name / f"seed_{args.seed}" / "summary_notear_individual_hierarchical.json"
    command = [sys.executable, str(script), "--seed", str(args.seed),
               "--out_dir", str(args.out_dir.resolve()), "--setting_name", name,
               "--total_samples", str(n), "--m", str(m), "--std", str(std),
               "--cluster_proportions", "0.4", "0.4", "0.2", "--n_vars", "5",
               "--s0_list", "5", "5", "5", "--lambda1_grid",
               "0.0001", "0.001", "0.01", "0.1", "--n_folds", "3",
               "--tau", str(args.tau)]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if output.exists():
        raise FileExistsError(f"Preserving existing result: {output}")
    subprocess.run(command, check=True)
    data = json.loads(output.read_text())
    assert data["data_config"]["cluster_proportions"] == [.4, .4, .2]
    assert data["hierarchical_clustering"]["clustering_accuracy"]["n_true_clusters"] == 3
    print(f"Verified three-cluster result: {output}")


if __name__ == "__main__":
    main()
