import mlflow


mlflow.set_tracking_uri("http://127.0.0.1:5000")
exp = mlflow.get_experiment_by_name("tp2-registre")
if exp is None:
    raise RuntimeError("L'experience 'tp2-registre' est introuvable.")

runs = mlflow.search_runs([exp.experiment_id], order_by=["metrics.f1 DESC"])
for _, run in runs.head(3).iterrows():
    version = mlflow.register_model(f"runs:/{run['run_id']}/model", "fraude")
    print("version", version.version, "- F1", round(run["metrics.f1"], 4))
