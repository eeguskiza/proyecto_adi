"""
Servicio BentoML para el modelo de clustering de máquinas.
"""

import bentoml
import numpy as np
from bentoml.io import JSON
from pydantic import BaseModel
from typing import List, Dict


class ClusteringInput(BaseModel):
    """Esquema de entrada para clustering."""
    disponibilidad: float
    scrap_rate: float
    uph_real: float
    dur_prod: float


class ClusteringOutput(BaseModel):
    """Esquema de salida para clustering."""
    cluster: int
    distancia_al_centro: float
    caracteristicas_cluster: Dict[str, float]


# Cargar modelo y scaler
clustering_runner = bentoml.sklearn.get("machine_clustering:latest").to_runner()
scaler_runner = bentoml.sklearn.get("clustering_scaler:latest").to_runner()

# Crear servicio
svc = bentoml.Service("machine_clustering_service", runners=[clustering_runner, scaler_runner])


@svc.api(input=JSON(pydantic_model=ClusteringInput), output=JSON(pydantic_model=ClusteringOutput))
async def predict(input_data: ClusteringInput) -> ClusteringOutput:
    """
    Asigna una máquina a un cluster.

    Args:
        input_data: Métricas de la máquina

    Returns:
        Cluster asignado e información del cluster
    """
    # Preparar features
    features = np.array([[
        input_data.disponibilidad,
        input_data.scrap_rate,
        input_data.uph_real,
        input_data.dur_prod
    ]])

    # Escalar features
    features_scaled = await scaler_runner.transform.async_run(features)

    # Predecir cluster
    cluster = await clustering_runner.predict.async_run(features_scaled)

    # Calcular distancia al centro del cluster
    cluster_centers = clustering_runner.cluster_centers_
    assigned_cluster = int(cluster[0])
    center = cluster_centers[assigned_cluster]
    distance = float(np.linalg.norm(features_scaled[0] - center))

    # Obtener características del centro del cluster (desnormalizado)
    center_denorm = await scaler_runner.inverse_transform.async_run(center.reshape(1, -1))

    return ClusteringOutput(
        cluster=assigned_cluster,
        distancia_al_centro=distance,
        caracteristicas_cluster={
            "disponibilidad": float(center_denorm[0][0]),
            "scrap_rate": float(center_denorm[0][1]),
            "uph_real": float(center_denorm[0][2]),
            "dur_prod": float(center_denorm[0][3])
        }
    )


@svc.api(input=JSON(), output=JSON())
async def batch_predict(input_data: List[Dict]) -> List[Dict]:
    """
    Asigna múltiples máquinas a clusters.

    Args:
        input_data: Lista de métricas de máquinas

    Returns:
        Lista de asignaciones de clusters
    """
    # Convertir a array
    features = np.array([[
        item["disponibilidad"],
        item["scrap_rate"],
        item["uph_real"],
        item["dur_prod"]
    ] for item in input_data])

    # Escalar features
    features_scaled = await scaler_runner.transform.async_run(features)

    # Predecir clusters
    clusters = await clustering_runner.predict.async_run(features_scaled)

    # Obtener centros
    cluster_centers = clustering_runner.cluster_centers_

    # Formatear salida
    results = []
    for i, cluster_id in enumerate(clusters):
        cluster_id = int(cluster_id)
        center = cluster_centers[cluster_id]
        distance = float(np.linalg.norm(features_scaled[i] - center))

        # Desnormalizar centro
        center_denorm = await scaler_runner.inverse_transform.async_run(center.reshape(1, -1))

        results.append({
            "cluster": cluster_id,
            "distancia_al_centro": distance,
            "caracteristicas_cluster": {
                "disponibilidad": float(center_denorm[0][0]),
                "scrap_rate": float(center_denorm[0][1]),
                "uph_real": float(center_denorm[0][2]),
                "dur_prod": float(center_denorm[0][3])
            }
        })

    return results


@svc.api(input=JSON(), output=JSON())
async def get_cluster_info() -> Dict:
    """
    Obtiene información de todos los clusters.

    Returns:
        Información de los clusters
    """
    cluster_centers = clustering_runner.cluster_centers_
    n_clusters = len(cluster_centers)

    # Desnormalizar centros
    centers_denorm = await scaler_runner.inverse_transform.async_run(cluster_centers)

    clusters_info = {}
    for i in range(n_clusters):
        clusters_info[f"cluster_{i}"] = {
            "disponibilidad": float(centers_denorm[i][0]),
            "scrap_rate": float(centers_denorm[i][1]),
            "uph_real": float(centers_denorm[i][2]),
            "dur_prod": float(centers_denorm[i][3])
        }

    return {
        "n_clusters": n_clusters,
        "clusters": clusters_info
    }
