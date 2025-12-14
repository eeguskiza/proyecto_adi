import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def page_clustering(filtered: dict, ciclos: pd.DataFrame) -> None:
    st.subheader("Clustering de Máquinas")
    st.markdown(
        "Agrupamos las máquinas en clusters según sus métricas de rendimiento: "
        "disponibilidad, % de scrap, UPH real y duraciones. "
        "Esto ayuda a identificar perfiles de máquinas similares y detectar outliers."
    )

    prod = filtered.get("produccion", pd.DataFrame())
    
    if prod.empty:
        st.info("Sin datos de producción en el rango seleccionado para clustering.")
        return

    # Preparar métricas por máquina
    prod = prod.copy()
    prod["estado_oee"] = prod["evento"].str.lower().map({
        "producción": "produccion",
        "produccion": "produccion",
        "preparación": "preparacion",
        "preparacion": "preparacion",
    })
    prod["estado_oee"] = prod["estado_oee"].fillna("incidencia")
    
    # Evitar sobrecontar: última operación con piezas>0 por OF
    prod["total_piezas"] = prod["piezas_ok"] + prod["piezas_scrap"]
    prod_valid = prod[prod["total_piezas"] > 0].sort_values("ts_fin")
    of_final = prod_valid.groupby("work_order_id").tail(1)

    # Agregar por máquina
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

    # Filtrar máquinas con datos suficientes
    machine_metrics = machine_metrics[machine_metrics["total_dur"] > 0].copy()
    
    if len(machine_metrics) < 3:
        st.warning("No hay suficientes máquinas con datos para clustering (mínimo 3).")
        st.dataframe(machine_metrics, width='stretch')
        return

    st.markdown(f"**Máquinas en el análisis:** {len(machine_metrics)}")

    # Seleccionar features para clustering
    features = ["disponibilidad", "scrap_rate", "uph_real", "dur_prod"]
    df_features = machine_metrics[["machine_name"] + features].copy()
    df_features = df_features.dropna()

    if len(df_features) < 3:
        st.warning("No hay suficientes máquinas con métricas completas para clustering.")
        st.dataframe(machine_metrics, width='stretch')
        return

    # Parámetros de clustering
    col1, col2 = st.columns([1, 3])
    n_clusters = col1.slider("Número de clusters", min_value=2, max_value=min(6, len(df_features)), value=3, key="n_clusters")

    # Normalizar features
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features[features])

    # Aplicar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_features["cluster"] = kmeans.fit_predict(X)
    df_features["cluster"] = df_features["cluster"].astype(str)

    # Mostrar métricas por cluster
    st.markdown("---")
    st.markdown("### 📊 Métricas por Cluster")
    
    cluster_summary = df_features.groupby("cluster").agg(
        n_maquinas=("machine_name", "count"),
        disp_media=("disponibilidad", "mean"),
        scrap_medio=("scrap_rate", "mean"),
        uph_media=("uph_real", "mean"),
    ).reset_index()
    cluster_summary["cluster"] = "Cluster " + cluster_summary["cluster"]
    
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

    # Gráfico de barras: distribución de máquinas por cluster
    fig_dist = px.bar(
        cluster_summary,
        x="cluster",
        y="n_maquinas",
        title="Distribución de Máquinas por Cluster",
        labels={"n_maquinas": "Número de Máquinas", "cluster": "Cluster"},
        text="n_maquinas"
    )
    fig_dist.update_traces(textposition="outside")
    col_sum2.plotly_chart(fig_dist, width='stretch')

    # Visualización 3D (disponibilidad, scrap, UPH)
    st.markdown("---")
    st.markdown("### 🔍 Visualización 3D de Clusters")
    
    fig_3d = px.scatter_3d(
        df_features,
        x="disponibilidad",
        y="scrap_rate",
        z="uph_real",
        color="cluster",
        hover_data=["machine_name"],
        title="Clusters de Máquinas (3D)",
        labels={
            "disponibilidad": "Disponibilidad",
            "scrap_rate": "Scrap %",
            "uph_real": "UPH Real",
            "cluster": "Cluster"
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_3d.update_traces(marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey")))
    st.plotly_chart(fig_3d, width='stretch')

    # Scatter 2D: Disponibilidad vs Scrap
    st.markdown("---")
    st.markdown("### 📈 Gráficos 2D de Clusters")
    
    col_2d1, col_2d2 = st.columns(2)
    
    fig_disp_scrap = px.scatter(
        df_features,
        x="disponibilidad",
        y="scrap_rate",
        color="cluster",
        hover_data=["machine_name"],
        title="Disponibilidad vs Scrap %",
        labels={"disponibilidad": "Disponibilidad", "scrap_rate": "Scrap %", "cluster": "Cluster"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_disp_scrap.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig_disp_scrap.update_xaxes(tickformat=".0%")
    fig_disp_scrap.update_yaxes(tickformat=".0%")
    col_2d1.plotly_chart(fig_disp_scrap, width='stretch')

    # Scatter 2D: UPH vs Scrap
    fig_uph_scrap = px.scatter(
        df_features,
        x="uph_real",
        y="scrap_rate",
        color="cluster",
        hover_data=["machine_name"],
        title="UPH Real vs Scrap %",
        labels={"uph_real": "UPH Real", "scrap_rate": "Scrap %", "cluster": "Cluster"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_uph_scrap.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig_uph_scrap.update_yaxes(tickformat=".0%")
    col_2d2.plotly_chart(fig_uph_scrap, width='stretch')

    # Tabla detallada con asignación de clusters
    st.markdown("---")
    st.markdown("### 🗂️ Detalle de Máquinas y Clusters")
    
    df_display = df_features.copy()
    df_display["cluster"] = "Cluster " + df_display["cluster"]
    df_display = df_display.sort_values(["cluster", "machine_name"])
    
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

    # Interpretación y recomendaciones
    st.markdown("---")
    st.markdown("### 💡 Interpretación")
    st.info(
        "**Cómo usar estos clusters:**\n\n"
        "- **Cluster con alta disponibilidad y bajo scrap:** Máquinas de referencia (best performers).\n"
        "- **Cluster con baja disponibilidad:** Investigar paradas y mantenimiento.\n"
        "- **Cluster con alto scrap:** Revisar ajustes, tooling, o calidad de materia prima.\n"
        "- **Outliers (clusters pequeños):** Máquinas con comportamiento atípico que requieren atención especial."
    )

    # Centros de los clusters (opcional, para análisis avanzado)
    with st.expander("Ver centros de los clusters (valores normalizados)"):
        centers = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=features
        )
        centers.index = [f"Cluster {i}" for i in range(n_clusters)]
        st.dataframe(
            centers.style.format({
                "disponibilidad": "{:.1%}",
                "scrap_rate": "{:.1%}",
                "uph_real": "{:.1f}",
                "dur_prod": "{:.0f}",
            }),
            width='stretch'
        )
