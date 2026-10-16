import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score, precision_score, recall_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


test = pd.read_csv("data/prepare/test.csv")
X, y = test.drop(columns="fraude"), test["fraude"]
modele = joblib.load("models/model.pkl")
pred = modele.predict(X)

mets = {
    "f1": round(f1_score(y, pred), 4),
    "precision": round(precision_score(y, pred), 4),
    "recall": round(recall_score(y, pred), 4),
}
Path("reports").mkdir(exist_ok=True)
with open("reports/metrics.json", "w", encoding="utf-8") as stream:
    json.dump(mets, stream, indent=2)
ConfusionMatrixDisplay.from_predictions(y, pred)
plt.savefig("reports/confusion.png", dpi=110, bbox_inches="tight")
plt.close()
print("evaluation :", mets)
