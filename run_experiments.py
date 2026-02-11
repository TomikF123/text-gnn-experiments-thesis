#!/usr/bin/env python3
"""Run all experiments with timeout and resume support."""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import mlflow


def get_completed_runs(experiment_names):
    """Get set of completed run names from MLflow, filtering for success."""
    completed = set()
    for exp_name in experiment_names:
        try:
            exp = mlflow.get_experiment_by_name(exp_name)
            if exp:
                # Filter for runs that are successfully FINISHED
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    filter_string="status = 'FINISHED'"
                )
                completed.update(runs["tags.mlflow.runName"].dropna().tolist())
        except Exception as e:
            print(f"Warning: Could not query MLflow for experiment '{exp_name}': {e}")
            pass
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout per experiment in seconds (default: 1800 = 30min)")
    parser.add_argument("--no-resume", action="store_true", help="Run all experiments, ignoring completed ones")
    parser.add_argument("--filter", type=str, help="Only run configs matching pattern (e.g., 'baseline')")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    experiments_dir = project_root / "runConfigs" / "experiments"
    log_file = project_root / f"experiment_log_{datetime.now():%Y%m%d_%H%M%S}.txt"

    # Setup MLflow
    mlflow.set_tracking_uri(str(project_root / "mlruns"))

    # Find all config files
    configs = sorted(experiments_dir.rglob("*.json"))
    if args.filter:
        configs = [c for c in configs if args.filter in str(c)]

    # Get completed runs for resume
    completed_runs = set()
    if not args.no_resume:
        # Get unique experiment names from configs
        exp_names = set()
        for config in configs:
            with open(config) as f:
                exp_names.add(json.load(f).get("experiment_name", ""))
        completed_runs = get_completed_runs(exp_names)
        print(f"Found {len(completed_runs)} completed runs in MLflow")

    results = {"success": 0, "failed": 0, "skipped": 0, "timeout": 0}

    # Open log file once in append mode for real-time logging
    with open(log_file, "w") as f:
        f.write(f"Experiment Run Log - {datetime.now()}\n")
        f.write("=" * 50 + "\n\n")

        for i, config in enumerate(configs, 1):
            relative_path = config.relative_to(project_root)

            # Check if already completed
            with open(config) as f_json:
                run_name = json.load(f_json).get("run_name", "")

            if run_name in completed_runs:
                log_line = f"[{i}/{len(configs)}] SKIP (completed): {relative_path}\n"
                print(log_line.strip())
                f.write(log_line)
                results["skipped"] += 1
                continue

            log_line = f"[{i}/{len(configs)}] RUNNING: {relative_path}..."
            print(f"\n{log_line}")
            f.write(f"\n{log_line}\n")

            try:
                result = subprocess.run(
                    [sys.executable, "main.py", "--config", str(relative_path)],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout
                )
                if result.returncode != 0:
                    results["failed"] += 1
                    status = "FAILED"
                    error_info = result.stderr[-2000:]
                    print(f"  -> {status}")
                    f.write(f"  -> {status}\n")
                    f.write(f"  ERROR:\n{'-'*20}\n{error_info}\n{'-'*20}\n")
                else:
                    results["success"] += 1
                    status = "SUCCESS"
                    print(f"  -> {status}")
                    f.write(f"  -> {status}\n")
            except subprocess.TimeoutExpired:
                results["timeout"] += 1
                status = f"TIMEOUT ({args.timeout}s)"
                print(f"  -> {status}")
                f.write(f"  -> {status}\n")
            except Exception as e:
                results["failed"] += 1
                status = "ERROR"
                print(f"  -> {status}: {e}")
                f.write(f"  -> {status}: {e}\n")

    print(f"\n{'='*50}")
    print(f"Success: {results['success']}, Failed: {results['failed']}, Timeout: {results['timeout']}, Skipped: {results['skipped']}")
    print(f"Full log saved to: {log_file}")


if __name__ == "__main__":
    main()
