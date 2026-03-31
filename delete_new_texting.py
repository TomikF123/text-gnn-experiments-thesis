import mlflow
mlflow.set_tracking_uri("mlruns")
r = mlflow.search_runs(search_all_experiments=True, output_format="pandas")
f = r[r.status == "FINISHED"]
t = f[f["tags.mlflow.runName"].str.contains("texting", na=False)].sort_values("end_time")
d = 0
kept = 0
for n in sorted(t["tags.mlflow.runName"].unique()):
    runs = t[t["tags.mlflow.runName"] == n].sort_values("end_time")
    if len(runs) < 2:
        kept += 1
        continue
    kept += 1
    print(f"  {n}: keeping {runs.iloc[0]['metrics.test_accuracy']:.4f}, deleting {len(runs)-1} newer")
    for rid in runs["run_id"].iloc[1:]:
        mlflow.delete_run(rid)
        d += 1
print(f"\nDeleted {d} runs. Kept {kept} TextING runs.")
