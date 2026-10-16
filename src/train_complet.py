import json
import subprocess
import sys
from pathlib import Path

import matplotlib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt


n_estimators = int(sys.argv[1])
max_depth = int(sys.argv[2])

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp2-registre")
Path("reports").mkdir(exist_ok=True)

df = pd.read_csv("data/transactions.csv")
X, y = df.drop(columns="fraude"), df["fraude"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

with mlflow.start_run(run_name=f"rf-{n_estimators}-{max_depth}"):
    m = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    ).fit(Xtr, ytr)
    pred = m.predict(Xte)
    proba = m.predict_proba(Xte)[:, 1]

    mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
    mlflow.log_metrics(
        {
            "f1": f1_score(yte, pred),
            "precision": precision_score(yte, pred),
            "recall": recall_score(yte, pred),
            "roc_auc": roc_auc_score(yte, proba),
        }
    )
    commit = subprocess.getoutput("git rev-parse --short HEAD")
    mlflow.set_tags(
        {"auteur": "Yassinevlg", "commit": commit, "jeu_de_donnees": "transactions v1"}
    )

    ConfusionMatrixDisplay.from_predictions(yte, pred)
    plt.savefig("reports/confusion.png", dpi=110, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact("reports/confusion.png")

    imp = {c: float(round(v, 4)) for c, v in zip(X.columns, m.feature_importances_)}
    with open("reports/importances.json", "w", encoding="utf-8") as stream:
        json.dump(imp, stream, indent=2)
    mlflow.log_artifact("reports/importances.json")

    mlflow.sklearn.log_model(m, name="model", input_example=Xte.head(3))
    print("F1 =", round(f1_score(yte, pred), 4))
