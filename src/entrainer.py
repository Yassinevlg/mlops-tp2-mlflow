from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


with open("params.yaml", encoding="utf-8") as stream:
    p = yaml.safe_load(stream)["entrainement"]

train = pd.read_csv("data/prepare/train.csv")
X, y = train.drop(columns="fraude"), train["fraude"]
modele = RandomForestClassifier(
    n_estimators=p["n_estimators"], max_depth=p["max_depth"], random_state=p["random_state"]
).fit(X, y)
Path("models").mkdir(exist_ok=True)
joblib.dump(modele, "models/model.pkl")
print("entrainement : modele ecrit dans models/model.pkl")
