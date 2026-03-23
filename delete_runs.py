"""Delete MLflow runs matching a filter pattern.

Usage:
    python delete_runs.py mr          # delete all MR runs
    python delete_runs.py lstm        # delete all LSTM runs
    python delete_runs.py lstm_mr     # delete LSTM MR runs
    python delete_runs.py --all       # delete everything
"""
import sys
import mlflow

mlflow.set_tracking_uri("mlruns")

pattern = sys.argv[1] if len(sys.argv) > 1 else None
if not pattern:
    print("Usage: python delete_runs.py <pattern|--all>")
    sys.exit(1)

runs = mlflow.search_runs(search_all_experiments=True, output_format="pandas")

if pattern == "--all":
    to_delete = runs
else:
    to_delete = runs[runs["tags.mlflow.runName"].str.contains(pattern, na=False)]

print(f"Will delete {len(to_delete)} runs matching '{pattern}':")
for _, r in to_delete.iterrows():
    name = r.get("tags.mlflow.runName", "?")
    status = r["status"]
    print(f"  {name} ({status})")

confirm = input(f"\nDelete {len(to_delete)} runs? [y/N] ")
if confirm.lower() == "y":
    for _, r in to_delete.iterrows():
        mlflow.delete_run(r["run_id"])
    print("Done.")
else:
    print("Cancelled.")
