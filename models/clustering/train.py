"""
Script de entrenamiento del modelo de clustering de maquinas.

Este script:
1. Carga los datos historicos de produccion
2. Calcula metricas agregadas por maquina
3. Entrena un modelo K-Means
4. Guarda el modelo y el scaler
5. Genera un reporte de resultados
"""

import os
import sys
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Agregar el directorio raiz al path
sys.path.append(str(Path(__file__).parent.parent.parent))


def load_production_data():
    """Carga los datos de produccion."""
    print("[INFO] Cargando datos de produccion...")

    data_path = Path(__file__).parent.parent.parent / "data" / "raw"
    prod = pd.read_csv(data_path / "produccion_operaciones.csv", parse_dates=["ts_ini", "ts_fin"])

    print(f"[INFO] Cargados {len(prod):,} registros de produccion")
    return prod


def prepare_machine_features(prod: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara las features por maquina para clustering.

    Returns:
        DataFrame con las features por maquina
    """
    print("[INFO] Calculando metricas por maquina...")

    # Preparar datos
    prod = prod.copy()
    prod["estado_oee"] = prod["evento"].str.lower().map({
        "produccion": "produccion",
        "producción": "produccion",
        "preparacion": "preparacion",
        "preparación": "preparacion",
    })
    prod["estado_oee"] = prod["estado_oee"].fillna("incidencia")

    # Evitar sobrecontar: ultima operacion con piezas>0 por OF
    prod["total_piezas"] = prod["piezas_ok"] + prod["piezas_scrap"]
    prod_valid = prod[prod["total_piezas"] > 0].sort_values("ts_fin")
    of_final = prod_valid.groupby("work_order_id").tail(1)

    # Agregar por maquina
    machine_agg = prod.groupby("machine_name").agg(
        dur_prod=("duracion_min", lambda x: x[prod.loc[x.index, "estado_oee"] == "produccion"].sum()),
        dur_prep=("duracion_min", lambda x: x[prod.loc[x.index, "estado_oee"] == "preparacion"].sum()),
        dur_inci=("duracion_min", lambda x: x[prod.loc[x.index, "estado_oee"] == "incidencia"].sum()),
    ).reset_index()

    machine_piezas = of_final.groupby("machine_name").agg(
        piezas_ok=("piezas_ok", "sum"),
        piezas_scrap=("piezas_scrap", "sum"),
    ).reset_index()

    machine_metrics = machine_agg.merge(machine_piezas, on="machine_name", how="left").fillna(0)
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

    # Filtrar maquinas con datos suficientes
    machine_metrics = machine_metrics[machine_metrics["total_dur"] > 0].copy()

    print(f"[INFO] Preparadas metricas para {len(machine_metrics)} maquinas")
    return machine_metrics


def train_clustering_model(df: pd.DataFrame, n_clusters: int = 3) -> tuple:
    """
    Entrena el modelo de clustering.

    Args:
        df: DataFrame con las metricas por maquina
        n_clusters: Numero de clusters

    Returns:
        (modelo, scaler, features_df, feature_names)
    """
    print(f"[INFO] Entrenando modelo K-Means con {n_clusters} clusters...")

    # Seleccionar features
    features = ["disponibilidad", "scrap_rate", "uph_real", "dur_prod"]
    df_features = df[["machine_name"] + features].copy()
    df_features = df_features.dropna()

    print(f"[INFO] Maquinas validas para clustering: {len(df_features)}")

    # Normalizar features
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features[features])

    # Entrenar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_features["cluster"] = kmeans.fit_predict(X)

    # Metricas de calidad del clustering
    silhouette = silhouette_score(X, df_features["cluster"])
    davies_bouldin = davies_bouldin_score(X, df_features["cluster"])

    print(f"[INFO] Metricas de calidad:")
    print(f"  - Silhouette Score: {silhouette:.3f} (mayor es mejor, rango [-1, 1])")
    print(f"  - Davies-Bouldin Index: {davies_bouldin:.3f} (menor es mejor)")

    return kmeans, scaler, df_features, features


def save_model(kmeans, scaler, features, output_dir: Path):
    """Guarda el modelo y el scaler."""
    print(f"[INFO] Guardando modelo en {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar modelo
    with open(output_dir / "kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)

    # Guardar scaler
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Guardar nombres de features
    with open(output_dir / "features.txt", "w") as f:
        f.write("\n".join(features))

    # Guardar metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "n_clusters": kmeans.n_clusters,
        "features": features,
    }

    with open(output_dir / "metadata.txt", "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print("[INFO] Modelo guardado exitosamente")


def generate_report(df_features: pd.DataFrame, output_dir: Path):
    """Genera un reporte de resultados."""
    print("[INFO] Generando reporte...")

    # Resumen por cluster
    cluster_summary = df_features.groupby("cluster").agg(
        n_maquinas=("machine_name", "count"),
        disp_media=("disponibilidad", "mean"),
        scrap_medio=("scrap_rate", "mean"),
        uph_media=("uph_real", "mean"),
    ).reset_index()

    report_path = output_dir / "training_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE DE ENTRENAMIENTO - CLUSTERING DE MAQUINAS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total de maquinas: {len(df_features)}\n")
        f.write(f"Numero de clusters: {df_features['cluster'].nunique()}\n\n")

        f.write("RESUMEN POR CLUSTER\n")
        f.write("-" * 60 + "\n")
        f.write(cluster_summary.to_string(index=False))
        f.write("\n\n")

        f.write("MAQUINAS POR CLUSTER\n")
        f.write("-" * 60 + "\n")
        for cluster in sorted(df_features["cluster"].unique()):
            machines = df_features[df_features["cluster"] == cluster]["machine_name"].tolist()
            f.write(f"\nCluster {cluster} ({len(machines)} maquinas):\n")
            f.write("  " + ", ".join(machines) + "\n")

    print(f"[INFO] Reporte guardado en {report_path}")


def main():
    """Funcion principal de entrenamiento."""
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO DE MODELO DE CLUSTERING")
    print("=" * 60 + "\n")

    # Cargar datos
    prod = load_production_data()

    # Preparar features
    machine_metrics = prepare_machine_features(prod)

    # Entrenar modelo
    n_clusters = 3  # Puedes ajustar este parametro
    kmeans, scaler, df_features, features = train_clustering_model(machine_metrics, n_clusters)

    # Guardar modelo
    output_dir = Path(__file__).parent / "trained_model"
    save_model(kmeans, scaler, features, output_dir)

    # Generar reporte
    generate_report(df_features, output_dir)

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
