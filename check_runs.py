import mlflow, json, glob

mlflow.set_tracking_uri("mlruns")
r = mlflow.search_runs(search_all_experiments=True, output_format="pandas")
f = r[r["status"] == "FINISHED"]
print(f"Total: {len(r)}, Finished: {len(f)}, Failed: {len(r[r.status=='FAILED'])}")

run_names = set(f["tags.mlflow.runName"].dropna())
configs = sorted(glob.glob("runConfigs/experiments/*/*.json"))
missing = []
for c in configs:
    name = json.load(open(c))["run_name"]
    if name not in run_names:
        missing.append(name)

print(f"Configs: {len(configs)}, Missing: {len(missing)}")
for m in missing:
    print(f"  {m}")
