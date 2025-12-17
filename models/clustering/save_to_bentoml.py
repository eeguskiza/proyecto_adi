"""
Script para guardar el modelo de clustering en BentoML.
"""

import pickle
import bentoml
from pathlib import Path


def save_clustering_model_to_bentoml():
    """Guarda el modelo de clustering y scaler en BentoML."""
    model_path = Path(__file__).parent / "trained_model"

    print("[INFO] Cargando modelo de clustering...")

    # Cargar modelo
    with open(model_path / "kmeans_model.pkl", "rb") as f:
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
        "machine_clustering",
        model,
        labels={
            "framework": "sklearn",
            "model_type": "KMeans",
            "task": "clustering"
        },
        metadata={
            "features": features,
            "n_clusters": model.n_clusters
        }
    )

    # Guardar scaler en BentoML
    bentoml.sklearn.save_model(
        "clustering_scaler",
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
    print("[INFO] Modelo: machine_clustering:latest")
    print("[INFO] Scaler: clustering_scaler:latest")
    print("\nPara servir el modelo, ejecuta:")
    print("  bentoml serve models/clustering/service.py:svc")


if __name__ == "__main__":
    save_clustering_model_to_bentoml()
