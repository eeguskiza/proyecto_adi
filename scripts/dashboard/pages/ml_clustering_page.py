"""
Pagina del dashboard para el modelo de Clustering ML.
Carga el modelo entrenado y permite hacer predicciones sobre maquinas.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@st.cache_resource
def load_clustering_model():
    """Carga el modelo de clustering entrenado."""
    model_path = Path(__file__).parent.parent.parent.parent / "models" / "clustering" / "trained_model"

    try:
        with open(model_path / "kmeans_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open(model_path / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open(model_path / "features.txt", "r") as f:
            features = [line.strip() for line in f.readlines()]

        return model, scaler, features
    except FileNotFoundError:
        st.error("Modelo no encontrado. Ejecuta: python models/clustering/train.py")
        return None, None, None


def page_ml_clustering(filtered: dict, ciclos: pd.DataFrame) -> None:
    st.title("Clustering ML - Agrupacion de Maquinas")

    st.markdown("""
    Este modelo utiliza **K-Means** para agrupar maquinas con caracteristicas de rendimiento similares.
    El modelo ha sido entrenado con datos historicos de produccion.
    """)

    # Cargar modelo
    model, scaler, features = load_clustering_model()

    if model is None:
        st.warning("No se pudo cargar el modelo. Asegurate de haberlo entrenado primero.")
        return

    # Info del modelo
    col1, col2, col3 = st.columns(3)
    col1.metric("Clusters", model.n_clusters)
    col2.metric("Features", len(features))
    col3.metric("Algoritmo", "K-Means")

    st.markdown("---")

    # Preparar datos actuales
    prod = filtered.get("produccion", pd.DataFrame())

    if prod.empty:
        st.info("Sin datos de produccion en el rango seleccionado.")
        return

    # Calcular metricas por maquina (misma logica que el entrenamiento)
    prod = prod.copy()
    prod["estado_oee"] = prod["evento"].str.lower().map({
        "produccion": "produccion",
        "producción": "produccion",
        "preparacion": "preparacion",
        "preparación": "preparacion",
    })
    prod["estado_oee"] = prod["estado_oee"].fillna("incidencia")

    prod["total_piezas"] = prod["piezas_ok"] + prod["piezas_scrap"]
    prod_valid = prod[prod["total_piezas"] > 0].sort_values("ts_fin")
    of_final = prod_valid.groupby("work_order_id").tail(1)

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

    machine_metrics = machine_metrics[machine_metrics["total_dur"] > 0].copy()
    machine_metrics = machine_metrics.dropna(subset=["disponibilidad", "scrap_rate", "uph_real"])

    if len(machine_metrics) < 1:
        st.warning("No hay suficientes datos para hacer predicciones.")
        return

    st.markdown(f"### Predicciones para {len(machine_metrics)} maquinas")

    # Hacer predicciones
    X = machine_metrics[features].values
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    machine_metrics["cluster"] = predictions
    machine_metrics["cluster_str"] = "Cluster " + machine_metrics["cluster"].astype(str)

    # Resumen por cluster
    st.markdown("#### Distribucion por Cluster")

    cluster_summary = machine_metrics.groupby("cluster_str").agg(
        n_maquinas=("machine_name", "count"),
        disp_media=("disponibilidad", "mean"),
        scrap_medio=("scrap_rate", "mean"),
        uph_media=("uph_real", "mean"),
    ).reset_index()

    col_sum1, col_sum2 = st.columns(2)

    col_sum1.dataframe(
        cluster_summary.style.format({
            "disp_media": "{:.1%}",
            "scrap_medio": "{:.1%}",
            "uph_media": "{:.1f}",
        }),
        width='stretch',
        hide_index=True
    )

    # Grafico de barras
    fig_dist = px.bar(
        cluster_summary,
        x="cluster_str",
        y="n_maquinas",
        title="Distribucion de Maquinas por Cluster",
        labels={"n_maquinas": "Numero de Maquinas", "cluster_str": "Cluster"},
        text="n_maquinas"
    )
    fig_dist.update_traces(textposition="outside")
    col_sum2.plotly_chart(fig_dist, use_container_width=True)

    # Visualizacion 3D
    st.markdown("#### Visualizacion 3D de Clusters")

    fig_3d = px.scatter_3d(
        machine_metrics,
        x="disponibilidad",
        y="scrap_rate",
        z="uph_real",
        color="cluster_str",
        hover_data=["machine_name"],
        title="Clusters Predichos (3D)",
        labels={
            "disponibilidad": "Disponibilidad",
            "scrap_rate": "Scrap %",
            "uph_real": "UPH Real",
            "cluster_str": "Cluster"
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_3d.update_traces(marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey")))
    st.plotly_chart(fig_3d, use_container_width=True)

    # Scatter 2D
    st.markdown("#### Graficos 2D de Clusters")

    col_2d1, col_2d2 = st.columns(2)

    fig_disp_scrap = px.scatter(
        machine_metrics,
        x="disponibilidad",
        y="scrap_rate",
        color="cluster_str",
        hover_data=["machine_name"],
        title="Disponibilidad vs Scrap %",
        labels={"disponibilidad": "Disponibilidad", "scrap_rate": "Scrap %", "cluster_str": "Cluster"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_disp_scrap.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig_disp_scrap.update_xaxes(tickformat=".0%")
    fig_disp_scrap.update_yaxes(tickformat=".0%")
    col_2d1.plotly_chart(fig_disp_scrap, use_container_width=True)

    fig_uph_scrap = px.scatter(
        machine_metrics,
        x="uph_real",
        y="scrap_rate",
        color="cluster_str",
        hover_data=["machine_name"],
        title="UPH Real vs Scrap %",
        labels={"uph_real": "UPH Real", "scrap_rate": "Scrap %", "cluster_str": "Cluster"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_uph_scrap.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig_uph_scrap.update_yaxes(tickformat=".0%")
    col_2d2.plotly_chart(fig_uph_scrap, use_container_width=True)

    # Tabla detallada
    st.markdown("#### Detalle de Maquinas y Clusters Predichos")

    df_display = machine_metrics[["machine_name", "disponibilidad", "scrap_rate", "uph_real", "dur_prod", "cluster_str"]].copy()
    df_display = df_display.sort_values(["cluster_str", "machine_name"])

    st.dataframe(
        df_display.style.format({
            "disponibilidad": "{:.1%}",
            "scrap_rate": "{:.1%}",
            "uph_real": "{:.1f}",
            "dur_prod": "{:.0f}",
        }),
        width='stretch',
        hide_index=True
    )

    # Centros de los clusters
    with st.expander("Ver centros de los clusters (valores normalizados)"):
        centers = pd.DataFrame(
            scaler.inverse_transform(model.cluster_centers_),
            columns=features
        )
        centers.index = [f"Cluster {i}" for i in range(model.n_clusters)]
        st.dataframe(
            centers.style.format({
                "disponibilidad": "{:.1%}",
                "scrap_rate": "{:.1%}",
                "uph_real": "{:.1f}",
                "dur_prod": "{:.0f}",
            }),
            width='stretch'
        )

    # Interpretacion
    st.markdown("### Interpretacion")
    st.info(
        "**Como usar estos clusters:**\n\n"
        "- **Cluster con alta disponibilidad y bajo scrap:** Maquinas de referencia (best performers).\n"
        "- **Cluster con baja disponibilidad:** Investigar paradas y mantenimiento.\n"
        "- **Cluster con alto scrap:** Revisar ajustes, tooling, o calidad de materia prima.\n"
        "- **Outliers (clusters pequenos):** Maquinas con comportamiento atipico que requieren atencion especial."
    )
