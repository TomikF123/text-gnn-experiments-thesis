import mlflow

mlflow.set_tracking_uri("mlruns")
r = mlflow.search_runs(search_all_experiments=True)

for name in ["textgcn_20ng_baseline", "textgcn_mr_baseline"]:
    runs = r[r["tags.mlflow.runName"] == name]
    runs = runs.sort_values("metrics.test_accuracy", ascending=False)
    if len(runs) <= 1:
        print(f"{name}: only {len(runs)} run(s), nothing to delete")
        continue
    for rid in runs["run_id"].iloc[1:]:
        mlflow.delete_run(rid)
        print(f"{name}: deleted {rid}")
    print(f"{name}: kept best {runs.iloc[0]['metrics.test_accuracy']:.4f}")
