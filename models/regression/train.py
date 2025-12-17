"""
Script de entrenamiento del modelo de regresion para prediccion de scrap.

Este script:
1. Carga los datos historicos de produccion
2. Calcula features por operacion de produccion
3. Entrena un modelo de Random Forest para predecir scrap rate
4. Guarda el modelo y el scaler
5. Genera un reporte de resultados con metricas de regresion
"""

import os
import sys
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Agregar el directorio raiz al path
sys.path.append(str(Path(__file__).parent.parent.parent))


def load_production_data():
    """Carga los datos de produccion."""
    print("[INFO] Cargando datos de produccion...")

    data_path = Path(__file__).parent.parent.parent / "data" / "raw"
    prod = pd.read_csv(data_path / "produccion_operaciones.csv", parse_dates=["ts_ini", "ts_fin"])

    print(f"[INFO] Cargados {len(prod):,} registros de produccion")
    return prod


def prepare_regression_features(prod: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara las features para el modelo de regresion de scrap.

    Features consideradas:
    - duracion_min: Duracion de la operacion
    - machine_name: Maquina (one-hot encoded)
    - ref_id_str: Referencia del producto (agregada por frecuencia)
    - hora_del_dia: Hora de inicio de produccion
    - dia_semana: Dia de la semana
    - evento: Tipo de evento (produccion, preparacion, etc.)

    Target:
    - scrap_rate: Tasa de scrap (piezas_scrap / total_piezas)

    Returns:
        DataFrame con features y target
    """
    print("[INFO] Preparando features para regresion...")

    # Preparar datos
    df = prod.copy()

    # Calcular target: scrap rate
    df["total_piezas"] = df["piezas_ok"] + df["piezas_scrap"]
    df["scrap_rate"] = np.where(
        df["total_piezas"] > 0,
        df["piezas_scrap"] / df["total_piezas"],
        np.nan
    )

    # Filtrar registros con produccion valida
    df = df[df["total_piezas"] > 0].copy()
    df = df.dropna(subset=["scrap_rate"])

    # Features temporales
    df["hora_del_dia"] = df["ts_ini"].dt.hour
    df["dia_semana"] = df["ts_ini"].dt.dayofweek

    # Preparar ref_id_str
    df["ref_id_str"] = df["ref_id"].astype(str).str.replace(r'\.0$', '', regex=True)

    # Frecuencia de referencias (para encoding)
    ref_freq = df["ref_id_str"].value_counts().to_dict()
    df["ref_frequency"] = df["ref_id_str"].map(ref_freq)

    # Estado OEE
    df["estado_oee"] = df["evento"].str.lower().map({
        "produccion": "produccion",
        "producción": "produccion",
        "preparacion": "preparacion",
        "preparación": "preparacion",
    })
    df["estado_oee"] = df["estado_oee"].fillna("incidencia")

    # One-hot encoding para estado_oee
    estado_dummies = pd.get_dummies(df["estado_oee"], prefix="estado")

    # One-hot encoding para maquinas (top 10 mas frecuentes)
    top_machines = df["machine_name"].value_counts().head(10).index
    df["machine_group"] = df["machine_name"].apply(lambda x: x if x in top_machines else "otras")
    machine_dummies = pd.get_dummies(df["machine_group"], prefix="machine")

    # Combinar todas las features
    features_df = pd.concat([
        df[["duracion_min", "hora_del_dia", "dia_semana", "ref_frequency", "scrap_rate"]],
        estado_dummies,
        machine_dummies
    ], axis=1)

    features_df = features_df.dropna()

    print(f"[INFO] Dataset preparado: {len(features_df):,} registros, {len(features_df.columns)-1} features")
    return features_df


def train_regression_model(df: pd.DataFrame) -> tuple:
    """
    Entrena el modelo de regresion.

    Args:
        df: DataFrame con las features y target

    Returns:
        (modelo, scaler, X_test, y_test, y_pred, feature_names)
    """
    print("[INFO] Entrenando modelo Random Forest Regressor...")

    # Separar features y target
    feature_cols = [col for col in df.columns if col != "scrap_rate"]
    X = df[feature_cols]
    y = df["scrap_rate"]

    print(f"[INFO] Features: {len(feature_cols)}")
    print(f"[INFO] Samples: {len(X)}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"[INFO] Train set: {len(X_train):,} | Test set: {len(X_test):,}")

    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Predicciones
    y_pred = model.predict(X_test_scaled)

    # Metricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"[INFO] Metricas del modelo:")
    print(f"  - MAE (Mean Absolute Error): {mae:.4f}")
    print(f"  - RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  - R² Score: {r2:.4f}")

    return model, scaler, X_test, y_test, y_pred, feature_cols


def save_model(model, scaler, features, output_dir: Path):
    """Guarda el modelo y el scaler."""
    print(f"[INFO] Guardando modelo en {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar modelo
    with open(output_dir / "random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Guardar scaler
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Guardar nombres de features
    with open(output_dir / "features.txt", "w") as f:
        f.write("\n".join(features))

    # Guardar metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "model_type": "RandomForestRegressor",
        "n_features": len(features),
        "target": "scrap_rate",
    }

    with open(output_dir / "metadata.txt", "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print("[INFO] Modelo guardado exitosamente")


def generate_report(model, X_test, y_test, y_pred, features, output_dir: Path):
    """Genera un reporte de resultados."""
    print("[INFO] Generando reporte...")

    # Metricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    report_path = output_dir / "training_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE DE ENTRENAMIENTO - REGRESION DE SCRAP\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modelo: Random Forest Regressor\n")
        f.write(f"Target: Scrap Rate (tasa de piezas defectuosas)\n\n")

        f.write("METRICAS DE EVALUACION\n")
        f.write("-" * 60 + "\n")
        f.write(f"MAE (Mean Absolute Error):     {mae:.4f}\n")
        f.write(f"RMSE (Root Mean Squared Error): {rmse:.4f}\n")
        f.write(f"R² Score:                       {r2:.4f}\n\n")

        f.write("INTERPRETACION:\n")
        f.write(f"- En promedio, el modelo se equivoca en {mae*100:.2f}% al predecir scrap\n")
        f.write(f"- El modelo explica {r2*100:.1f}% de la varianza en scrap rate\n\n")

        f.write("TOP 15 FEATURES MAS IMPORTANTES\n")
        f.write("-" * 60 + "\n")
        f.write(feature_importance.head(15).to_string(index=False))
        f.write("\n\n")

        f.write("ESTADISTICAS DE PREDICCIONES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Scrap real promedio:      {y_test.mean():.4f} ({y_test.mean()*100:.2f}%)\n")
        f.write(f"Scrap predicho promedio:  {y_pred.mean():.4f} ({y_pred.mean()*100:.2f}%)\n")
        f.write(f"Scrap real min/max:       {y_test.min():.4f} / {y_test.max():.4f}\n")
        f.write(f"Scrap predicho min/max:   {y_pred.min():.4f} / {y_pred.max():.4f}\n")

    print(f"[INFO] Reporte guardado en {report_path}")


def main():
    """Funcion principal de entrenamiento."""
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO DE MODELO DE REGRESION - PREDICCION DE SCRAP")
    print("=" * 60 + "\n")

    # Cargar datos
    prod = load_production_data()

    # Preparar features
    features_df = prepare_regression_features(prod)

    # Entrenar modelo
    model, scaler, X_test, y_test, y_pred, features = train_regression_model(features_df)

    # Guardar modelo
    output_dir = Path(__file__).parent / "trained_model"
    save_model(model, scaler, features, output_dir)

    # Generar reporte
    generate_report(model, X_test, y_test, y_pred, features, output_dir)

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
