"""
Script para guardar el modelo de regresión en BentoML.
"""

import pickle
import bentoml
from pathlib import Path


def save_regression_model_to_bentoml():
    """Guarda el modelo de regresión y scaler en BentoML."""
    model_path = Path(__file__).parent / "trained_model"

    print("[INFO] Cargando modelo de regresión...")

    # Cargar modelo
    with open(model_path / "random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Cargar scaler
    with open(model_path / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Cargar features
    with open(model_path / "features.txt", "r") as f:
        features = [line.strip() for line in f.readlines()]

    print("[INFO] Guardando modelo en BentoML...")

    # Guardar modelo en BentoML
    bentoml.sklearn.save_model(
        "scrap_regressor",
        model,
        labels={
            "framework": "sklearn",
            "model_type": "RandomForestRegressor",
            "task": "regression"
        },
        metadata={
            "features": features,
            "target": "scrap_rate"
        }
    )

    # Guardar scaler en BentoML
    bentoml.sklearn.save_model(
        "scrap_scaler",
        scaler,
        labels={
            "framework": "sklearn",
            "model_type": "StandardScaler",
            "task": "preprocessing"
        },
        metadata={
            "features": features
        }
    )

    print("[INFO] ✅ Modelo guardado exitosamente en BentoML!")
    print("[INFO] Modelo: scrap_regressor:latest")
    print("[INFO] Scaler: scrap_scaler:latest")
    print("\nPara servir el modelo, ejecuta:")
    print("  bentoml serve models/regression/service.py:svc")


if __name__ == "__main__":
    save_regression_model_to_bentoml()
