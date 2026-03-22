import mlflow

mlflow.set_tracking_uri("mlruns")
for name in [
    "baseline_comparison",
    "hyperparams_lstm",
    "data_efficiency",
    "preprocessing_impact",
    "ablation_texting",
]:
    exp = mlflow.get_experiment_by_name(name)
    if exp:
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        for _, r in runs.iterrows():
            rn = r.get("tags.mlflow.runName", "")
            if "lstm" in rn:
                mlflow.delete_run(r["run_id"])
                print(f"Deleted {rn}")
