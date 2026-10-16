from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


with open("params.yaml", encoding="utf-8") as stream:
    p = yaml.safe_load(stream)["preparation"]

df = pd.read_csv("data/transactions.csv")
train, test = train_test_split(df, test_size=p["test_size"], random_state=p["random_state"])
Path("data/prepare").mkdir(parents=True, exist_ok=True)
train.to_csv("data/prepare/train.csv", index=False)
test.to_csv("data/prepare/test.csv", index=False)
print("preparation :", len(train), "lignes train /", len(test), "lignes test")
