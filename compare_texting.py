import mlflow

mlflow.set_tracking_uri("mlruns")
r = mlflow.search_runs(search_all_experiments=True, output_format="pandas")
f = r[r["status"] == "FINISHED"]

texting = f[f["tags.mlflow.runName"].str.contains("texting", na=False)]
texting = texting.sort_values("end_time")

for name in sorted(texting["tags.mlflow.runName"].unique()):
    runs = texting[texting["tags.mlflow.runName"] == name].sort_values("end_time")
    if len(runs) < 2:
        continue
    old_acc = runs.iloc[0]["metrics.test_accuracy"]
    new_acc = runs.iloc[-1]["metrics.test_accuracy"]
    diff = new_acc - old_acc
    marker = "+" if diff > 0.005 else "-" if diff < -0.005 else "="
    print(f"{marker} {name:45s} old={old_acc:.4f}  new={new_acc:.4f}  diff={diff:+.4f}")
