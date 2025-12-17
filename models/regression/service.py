"""
Servicio BentoML para el modelo de regresión de scrap.
"""

import bentoml
import numpy as np
from bentoml.io import JSON
from pydantic import BaseModel
from typing import List, Dict


class RegressionInput(BaseModel):
    """Esquema de entrada para predicción de scrap."""
    duracion_min: float
    hora_del_dia: int
    dia_semana: int
    ref_frequency: int
    estado_incidencia: int = 0
    estado_preparacion: int = 0
    estado_produccion: int = 0


class RegressionOutput(BaseModel):
    """Esquema de salida para predicción de scrap."""
    scrap_rate_predicho: float
    confianza_intervalo: Dict[str, float]


# Cargar modelo y scaler
regressor_runner = bentoml.sklearn.get("scrap_regressor:latest").to_runner()
scaler_runner = bentoml.sklearn.get("scrap_scaler:latest").to_runner()

# Crear servicio
svc = bentoml.Service("scrap_regression_service", runners=[regressor_runner, scaler_runner])


@svc.api(input=JSON(pydantic_model=RegressionInput), output=JSON(pydantic_model=RegressionOutput))
async def predict(input_data: RegressionInput) -> RegressionOutput:
    """
    Predice el scrap rate de una operación.

    Args:
        input_data: Características de la operación

    Returns:
        Predicción de scrap rate
    """
    # Preparar features (ajustar según tus features reales)
    features = np.array([[
        input_data.duracion_min,
        input_data.hora_del_dia,
        input_data.dia_semana,
        input_data.ref_frequency,
        input_data.estado_incidencia,
        input_data.estado_preparacion,
        input_data.estado_produccion
    ]])

    # Escalar features
    features_scaled = await scaler_runner.transform.async_run(features)

    # Predecir
    prediction = await regressor_runner.predict.async_run(features_scaled)

    # Calcular intervalo de confianza (simplificado)
    # En producción, usa predict con return_std si el modelo lo soporta
    scrap_pred = float(prediction[0])
    std_dev = 0.02  # Estimación, deberías calcularla del modelo

    return RegressionOutput(
        scrap_rate_predicho=scrap_pred,
        confianza_intervalo={
            "lower": max(0.0, scrap_pred - 1.96 * std_dev),
            "upper": min(1.0, scrap_pred + 1.96 * std_dev)
        }
    )


@svc.api(input=JSON(), output=JSON())
async def batch_predict(input_data: List[Dict]) -> List[Dict]:
    """
    Predice el scrap rate de múltiples operaciones.

    Args:
        input_data: Lista de características de operaciones

    Returns:
        Lista de predicciones
    """
    # Convertir a array
    features = np.array([[
        item["duracion_min"],
        item["hora_del_dia"],
        item["dia_semana"],
        item["ref_frequency"],
        item.get("estado_incidencia", 0),
        item.get("estado_preparacion", 0),
        item.get("estado_produccion", 0)
    ] for item in input_data])

    # Escalar features
    features_scaled = await scaler_runner.transform.async_run(features)

    # Predecir
    predictions = await regressor_runner.predict.async_run(features_scaled)

    # Formatear salida
    std_dev = 0.02
    results = []
    for pred in predictions:
        scrap_pred = float(pred)
        results.append({
            "scrap_rate_predicho": scrap_pred,
            "confianza_intervalo": {
                "lower": max(0.0, scrap_pred - 1.96 * std_dev),
                "upper": min(1.0, scrap_pred + 1.96 * std_dev)
            }
        })

    return results
