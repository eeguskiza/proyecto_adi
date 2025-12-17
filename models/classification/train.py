"""
Script de entrenamiento del modelo de clasificacion de estado de maquinas.

Este script:
1. Carga los datos historicos de produccion
2. Calcula metricas agregadas por maquina y periodo
3. Entrena un modelo de clasificacion para determinar el estado de salud de maquinas
4. Guarda el modelo y el scaler
5. Genera un reporte de resultados con metricas de clasificacion

Categorias de clasificacion:
- EXCELENTE: Alta disponibilidad, bajo scrap, alto rendimiento
- BUENA: Metricas en rangos aceptables
- REQUIERE_ATENCION: Metricas por debajo del objetivo
- CRITICA: Metricas muy por debajo del objetivo
"""

import os
import sys
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Agregar el directorio raiz al path
sys.path.append(str(Path(__file__).parent.parent.parent))


def load_production_data():
    """Carga los datos de produccion."""
    print("[INFO] Cargando datos de produccion...")

    data_path = Path(__file__).parent.parent.parent / "data" / "raw"
    prod = pd.read_csv(data_path / "produccion_operaciones.csv", parse_dates=["ts_ini", "ts_fin"])

    print(f"[INFO] Cargados {len(prod):,} registros de produccion")
    return prod


def prepare_classification_features(prod: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara las features para clasificacion de estado de maquinas.

    Agrega datos por maquina y semana, calculando metricas clave.

    Returns:
        DataFrame con features y clasificacion
    """
    print("[INFO] Preparando features para clasificacion...")

    # Preparar datos
    df = prod.copy()
    df["estado_oee"] = df["evento"].str.lower().map({
        "produccion": "produccion",
        "producción": "produccion",
        "preparacion": "preparacion",
        "preparación": "preparacion",
    })
    df["estado_oee"] = df["estado_oee"].fillna("incidencia")

    # Calcular totales
    df["total_piezas"] = df["piezas_ok"] + df["piezas_scrap"]

    # Evitar sobrecontar: ultima operacion con piezas>0 por OF
    prod_valid = df[df["total_piezas"] > 0].sort_values("ts_fin")
    of_final = prod_valid.groupby("work_order_id").tail(1)

    # Agregar por fecha para crear periodos semanales
    df["fecha"] = df["ts_ini"].dt.date
    df["semana"] = df["ts_ini"].dt.isocalendar().week
    df["anio"] = df["ts_ini"].dt.year
    df["periodo"] = df["anio"].astype(str) + "-W" + df["semana"].astype(str)

    # Agregar por maquina y periodo
    machine_period_agg = df.groupby(["machine_name", "periodo"]).agg(
        dur_prod=("duracion_min", lambda x: x[df.loc[x.index, "estado_oee"] == "produccion"].sum()),
        dur_prep=("duracion_min", lambda x: x[df.loc[x.index, "estado_oee"] == "preparacion"].sum()),
        dur_inci=("duracion_min", lambda x: x[df.loc[x.index, "estado_oee"] == "incidencia"].sum()),
        n_operaciones=("work_order_id", "count"),
    ).reset_index()

    # Agregar piezas (usando of_final para evitar duplicados)
    of_final["fecha"] = of_final["ts_ini"].dt.date
    of_final["semana"] = of_final["ts_ini"].dt.isocalendar().week
    of_final["anio"] = of_final["ts_ini"].dt.year
    of_final["periodo"] = of_final["anio"].astype(str) + "-W" + of_final["semana"].astype(str)

    piezas_agg = of_final.groupby(["machine_name", "periodo"]).agg(
        piezas_ok=("piezas_ok", "sum"),
        piezas_scrap=("piezas_scrap", "sum"),
    ).reset_index()

    # Merge
    machine_metrics = machine_period_agg.merge(piezas_agg, on=["machine_name", "periodo"], how="left").fillna(0)

    # Calcular metricas derivadas
    machine_metrics["total_dur"] = machine_metrics["dur_prod"] + machine_metrics["dur_prep"] + machine_metrics["dur_inci"]
    machine_metrics["disponibilidad"] = np.where(
        machine_metrics["total_dur"] > 0,
        machine_metrics["dur_prod"] / machine_metrics["total_dur"],
        np.nan
    )

    machine_metrics["total_piezas"] = machine_metrics["piezas_ok"] + machine_metrics["piezas_scrap"]
    machine_metrics["scrap_rate"] = np.where(
        machine_metrics["total_piezas"] > 0,
        machine_metrics["piezas_scrap"] / machine_metrics["total_piezas"],
        np.nan
    )

    machine_metrics["uph_real"] = np.where(
        machine_metrics["dur_prod"] > 0,
        60 * machine_metrics["piezas_ok"] / machine_metrics["dur_prod"],
        np.nan
    )

    # Ratio de preparacion vs produccion
    machine_metrics["prep_ratio"] = np.where(
        machine_metrics["dur_prod"] > 0,
        machine_metrics["dur_prep"] / machine_metrics["dur_prod"],
        np.nan
    )

    # Ratio de incidencias vs produccion
    machine_metrics["inci_ratio"] = np.where(
        machine_metrics["dur_prod"] > 0,
        machine_metrics["dur_inci"] / machine_metrics["dur_prod"],
        np.nan
    )

    # Filtrar registros con datos suficientes
    machine_metrics = machine_metrics[machine_metrics["total_dur"] > 60].copy()  # Al menos 1 hora de actividad
    machine_metrics = machine_metrics.dropna(subset=["disponibilidad", "scrap_rate", "uph_real"])

    # CLASIFICACION: Determinar estado de la maquina
    def classify_machine_state(row):
        """Clasifica el estado de una maquina basandose en sus metricas."""
        disp = row["disponibilidad"]
        scrap = row["scrap_rate"]
        uph = row["uph_real"]

        # Criterios de clasificacion
        # EXCELENTE: Disp >= 0.85, Scrap <= 0.02, UPH >= 100
        if disp >= 0.85 and scrap <= 0.02 and uph >= 100:
            return "EXCELENTE"

        # BUENA: Disp >= 0.70, Scrap <= 0.05, UPH >= 60
        elif disp >= 0.70 and scrap <= 0.05 and uph >= 60:
            return "BUENA"

        # CRITICA: Disp < 0.50 or Scrap > 0.10
        elif disp < 0.50 or scrap > 0.10:
            return "CRITICA"

        # REQUIERE_ATENCION: Todo lo demas
        else:
            return "REQUIERE_ATENCION"

    machine_metrics["estado"] = machine_metrics.apply(classify_machine_state, axis=1)

    print(f"[INFO] Dataset preparado: {len(machine_metrics):,} registros")
    print(f"[INFO] Distribucion de clases:")
    print(machine_metrics["estado"].value_counts().to_string())

    return machine_metrics


def train_classification_model(df: pd.DataFrame) -> tuple:
    """
    Entrena el modelo de clasificacion.

    Args:
        df: DataFrame con las features y clasificacion

    Returns:
        (modelo, scaler, X_test, y_test, y_pred, feature_names)
    """
    print(f"\n[INFO] Entrenando modelo Random Forest Classifier...")

    # Seleccionar features
    features = [
        "disponibilidad", "scrap_rate", "uph_real",
        "dur_prod", "dur_prep", "dur_inci",
        "prep_ratio", "inci_ratio", "n_operaciones"
    ]

    X = df[features]
    y = df["estado"]

    print(f"[INFO] Features: {len(features)}")
    print(f"[INFO] Samples: {len(X)}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Train set: {len(X_train):,} | Test set: {len(X_test):,}")

    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenar Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Predicciones
    y_pred = model.predict(X_test_scaled)

    # Metricas
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[INFO] Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    return model, scaler, X_test, y_test, y_pred, features


def save_model(model, scaler, features, output_dir: Path):
    """Guarda el modelo y el scaler."""
    print(f"\n[INFO] Guardando modelo en {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar modelo
    with open(output_dir / "random_forest_classifier.pkl", "wb") as f:
        pickle.dump(model, f)

    # Guardar scaler
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Guardar nombres de features
    with open(output_dir / "features.txt", "w") as f:
        f.write("\n".join(features))

    # Guardar clases
    with open(output_dir / "classes.txt", "w") as f:
        f.write("\n".join(model.classes_))

    # Guardar metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "model_type": "RandomForestClassifier",
        "n_features": len(features),
        "n_classes": len(model.classes_),
        "classes": ", ".join(model.classes_),
    }

    with open(output_dir / "metadata.txt", "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print("[INFO] Modelo guardado exitosamente")


def generate_report(model, X_test, y_test, y_pred, features, output_dir: Path):
    """Genera un reporte de resultados."""
    print("[INFO] Generando reporte...")

    # Metricas
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    report_path = output_dir / "training_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE DE ENTRENAMIENTO - CLASIFICACION DE ESTADO DE MAQUINAS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modelo: Random Forest Classifier\n")
        f.write(f"Target: Estado de Maquina (EXCELENTE / BUENA / REQUIERE_ATENCION / CRITICA)\n\n")

        f.write("METRICAS DE EVALUACION\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")

        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 60 + "\n")
        f.write(class_report)
        f.write("\n")

        f.write("CONFUSION MATRIX\n")
        f.write("-" * 60 + "\n")
        f.write(f"Clases: {model.classes_}\n\n")
        f.write(str(conf_matrix))
        f.write("\n\n")

        f.write("TOP 10 FEATURES MAS IMPORTANTES\n")
        f.write("-" * 60 + "\n")
        f.write(feature_importance.head(10).to_string(index=False))
        f.write("\n\n")

        f.write("INTERPRETACION DE CLASES\n")
        f.write("-" * 60 + "\n")
        f.write("EXCELENTE: Disponibilidad >= 85%, Scrap <= 2%, UPH >= 100\n")
        f.write("BUENA: Disponibilidad >= 70%, Scrap <= 5%, UPH >= 60\n")
        f.write("CRITICA: Disponibilidad < 50% o Scrap > 10%\n")
        f.write("REQUIERE_ATENCION: Metricas por debajo de los objetivos\n")

    print(f"[INFO] Reporte guardado en {report_path}")


def main():
    """Funcion principal de entrenamiento."""
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO DE MODELO DE CLASIFICACION - ESTADO DE MAQUINAS")
    print("=" * 60 + "\n")

    # Cargar datos
    prod = load_production_data()

    # Preparar features
    machine_metrics = prepare_classification_features(prod)

    # Entrenar modelo
    model, scaler, X_test, y_test, y_pred, features = train_classification_model(machine_metrics)

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
