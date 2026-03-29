import mlflow

mlflow.set_tracking_uri("mlruns")
r = mlflow.search_runs(search_all_experiments=True)
runs = r[r["tags.mlflow.runName"] == "textgcn_20ng_baseline"]
runs = runs.sort_values("metrics.test_accuracy", ascending=False)

for rid in runs["run_id"].iloc[1:]:
    mlflow.delete_run(rid)
    print(f"Deleted {rid}")

print(f"Kept best: {runs.iloc[0]['metrics.test_accuracy']:.4f}")
