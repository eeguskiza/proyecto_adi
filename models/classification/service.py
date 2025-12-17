"""
Servicio BentoML para el modelo de clasificación de máquinas.
"""

import bentoml
import numpy as np
import pandas as pd
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel
from typing import List, Dict


class ClassificationInput(BaseModel):
    """Esquema de entrada para predicción de clasificación."""
    disponibilidad: float
    scrap_rate: float
    uph_real: float
    dur_prod: float
    prep_ratio: float
    inci_ratio: float


class ClassificationOutput(BaseModel):
    """Esquema de salida para predicción de clasificación."""
    estado_predicho: str
    probabilidades: Dict[str, float]
    confianza: float


# Cargar modelo y scaler
classifier_runner = bentoml.sklearn.get("machine_classifier:latest").to_runner()
scaler_runner = bentoml.sklearn.get("machine_scaler:latest").to_runner()

# Crear servicio
svc = bentoml.Service("machine_classification_service", runners=[classifier_runner, scaler_runner])


@svc.api(input=JSON(pydantic_model=ClassificationInput), output=JSON(pydantic_model=ClassificationOutput))
async def predict(input_data: ClassificationInput) -> ClassificationOutput:
    """
    Predice el estado de una máquina basándose en sus métricas.

    Args:
        input_data: Métricas de la máquina

    Returns:
        Predicción de estado y probabilidades
    """
    # Preparar features
    features = np.array([[
        input_data.disponibilidad,
        input_data.scrap_rate,
        input_data.uph_real,
        input_data.dur_prod,
        input_data.prep_ratio,
        input_data.inci_ratio
    ]])

    # Escalar features
    features_scaled = await scaler_runner.transform.async_run(features)

    # Predecir
    prediction = await classifier_runner.predict.async_run(features_scaled)
    probabilities = await classifier_runner.predict_proba.async_run(features_scaled)

    # Formatear salida
    classes = ["CRITICA", "BUENA", "EXCELENTE", "REQUIERE_ATENCION"]  # Ajustar según tus clases
    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities[0])}

    return ClassificationOutput(
        estado_predicho=str(prediction[0]),
        probabilidades=prob_dict,
        confianza=float(probabilities[0].max())
    )


@svc.api(input=JSON(), output=JSON())
async def batch_predict(input_data: List[Dict]) -> List[Dict]:
    """
    Predice el estado de múltiples máquinas.

    Args:
        input_data: Lista de métricas de máquinas

    Returns:
        Lista de predicciones
    """
    # Convertir a array
    features = np.array([[
        item["disponibilidad"],
        item["scrap_rate"],
        item["uph_real"],
        item["dur_prod"],
        item["prep_ratio"],
        item["inci_ratio"]
    ] for item in input_data])

    # Escalar features
    features_scaled = await scaler_runner.transform.async_run(features)

    # Predecir
    predictions = await classifier_runner.predict.async_run(features_scaled)
    probabilities = await classifier_runner.predict_proba.async_run(features_scaled)

    # Formatear salida
    classes = ["CRITICA", "BUENA", "EXCELENTE", "REQUIERE_ATENCION"]

    results = []
    for i, pred in enumerate(predictions):
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities[i])}
        results.append({
            "estado_predicho": str(pred),
            "probabilidades": prob_dict,
            "confianza": float(probabilities[i].max())
        })

    return results
