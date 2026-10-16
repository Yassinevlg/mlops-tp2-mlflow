import mlflow
import pandas as pd
from mlflow import MlflowClient


mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()
X = pd.DataFrame([{"montant": 8500.0, "anciennete": 3, "freq_24h": 9, "nuit": 1}])


def servi() -> None:
    version = client.get_model_version_by_alias("fraude", "production").version
    model = mlflow.pyfunc.load_model("models:/fraude@production")
    print(
        "alias production -> version",
        version,
        "| run",
        str(model.metadata.run_id)[:8],
        "| prediction",
        model.predict(X),
    )


print("--- etat initial")
servi()
client.set_registered_model_alias("fraude", "production", 2)
print("--- apres promotion de la v2")
servi()
client.set_registered_model_alias("fraude", "production", 1)
print("--- apres retour arriere vers la v1")
servi()
