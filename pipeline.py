import json
import subprocess
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml


with open("params.yaml", encoding="utf-8") as stream:
    P = yaml.safe_load(stream)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp2-pipeline")


def lancer(etape: str) -> None:
    subprocess.run([sys.executable, f"src/{etape}.py"], check=True)


refuse = False
with mlflow.start_run(run_name="pipeline"):
    with mlflow.start_run(run_name="preparer", nested=True):
        lancer("preparer")
        mlflow.log_params(P["preparation"])

    with mlflow.start_run(run_name="entrainer", nested=True):
        lancer("entrainer")
        mlflow.log_params(P["entrainement"])

    with mlflow.start_run(run_name="evaluer", nested=True):
        lancer("evaluer")
        with open("reports/metrics.json", encoding="utf-8") as stream:
            mets = json.load(stream)
        mlflow.log_metrics(mets)
        mlflow.log_artifact("reports/confusion.png")

    seuil = P["validation"]["seuil_f1"]
    mlflow.log_param("seuil_f1", seuil)
    mlflow.log_metrics(mets)
    if mets["f1"] >= seuil:
        exemple = pd.read_csv("data/prepare/test.csv").drop(columns="fraude").head(3)
        mlflow.sklearn.log_model(
            joblib.load("models/model.pkl"),
            name="model",
            input_example=exemple,
            registered_model_name="fraude-tp2",
        )
        mlflow.set_tag("statut", "accepte")
        print(f"modele accepte (F1 {mets['f1']} >= {seuil}) et enregistre")
    else:
        mlflow.set_tag("statut", "refuse")
        refuse = True

if refuse:
    raise SystemExit(f"modele refuse : F1 {mets['f1']} < seuil {seuil}")
