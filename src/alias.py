import mlflow
from mlflow import MlflowClient


mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()
client.set_registered_model_alias("fraude", "staging", 2)
client.set_registered_model_alias("fraude", "production", 1)
client.update_model_version(
    "fraude", 1, description="Validee par l equipe : meilleur F1 sur le jeu de test"
)
for alias in ["staging", "production"]:
    version = client.get_model_version_by_alias("fraude", alias)
    print(alias, "-> version", version.version)
