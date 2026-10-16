# TP2 MLflow - Pipeline, suivi et registre de modeles

Projet de detection de fraude bancaire synthetique realise avec MLflow 3.

## Demarrage

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python donnees.py
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Dans un second terminal active dans le meme environnement :

```powershell
python pipeline.py
foreach ($p in '200 8', '300 12', '500 16') {
    $a = $p -split ' '
    python src/train_complet.py $a[0] $a[1]
}
python src/enregistrer.py
python src/alias.py
python src/rollback.py
```

Le serveur HTTP du modele promu est lance avec :

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow models serve -m "models:/fraude@production" -p 8000 --env-manager local
```
