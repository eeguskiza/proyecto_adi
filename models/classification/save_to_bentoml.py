"""
Script para guardar el modelo de clasificación en BentoML.
"""

import pickle
import bentoml
from pathlib import Path


def save_classification_model_to_bentoml():
    """Guarda el modelo de clasificación y scaler en BentoML."""
    model_path = Path(__file__).parent / "trained_model"

    print("[INFO] Cargando modelo de clasificación...")

    # Cargar modelo
    with open(model_path / "random_forest_classifier.pkl", "rb") as f:
        model = pickle.load(f)

    # Cargar scaler
    with open(model_path / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Cargar features
    with open(model_path / "features.txt", "r") as f:
        features = [line.strip() for line in f.readlines()]

    # Cargar classes
    with open(model_path / "classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    print("[INFO] Guardando modelo en BentoML...")

    # Guardar modelo en BentoML
    bentoml.sklearn.save_model(
        "machine_classifier",
        model,
        labels={
            "framework": "sklearn",
            "model_type": "RandomForestClassifier",
            "task": "classification"
        },
        metadata={
            "features": features,
            "classes": classes,
            "n_classes": len(classes)
        }
    )

    # Guardar scaler en BentoML
    bentoml.sklearn.save_model(
        "machine_scaler",
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
    print("[INFO] Modelo: machine_classifier:latest")
    print("[INFO] Scaler: machine_scaler:latest")
    print("\nPara servir el modelo, ejecuta:")
    print("  bentoml serve models/classification/service.py:svc")


if __name__ == "__main__":
    save_classification_model_to_bentoml()
