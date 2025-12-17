"""
Pagina del dashboard para clustering interactivo de maquinas.
Permite entrenar modelos con diferentes algoritmos y parametros.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def prepare_features(prod):
    """Prepara features para clustering."""
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

    return machine_metrics


def train_clustering_model(X, algorithm, params):
    """Entrena modelo de clustering según el algoritmo seleccionado."""
    if algorithm == "K-Means":
        model = KMeans(n_clusters=params["n_clusters"], random_state=42, n_init=10)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    elif algorithm == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=params["n_clusters"], linkage=params["linkage"])

    clusters = model.fit_predict(X)

    return model, clusters


def evaluate_clustering(X, clusters):
    """Evalúa el clustering con métricas."""
    # Filtrar outliers de DBSCAN
    valid_mask = clusters != -1
    X_valid = X[valid_mask]
    clusters_valid = clusters[valid_mask]

    if len(np.unique(clusters_valid)) < 2:
        return None, None, None

    silhouette = silhouette_score(X_valid, clusters_valid)
    davies_bouldin = davies_bouldin_score(X_valid, clusters_valid)
    calinski = calinski_harabasz_score(X_valid, clusters_valid)

    return silhouette, davies_bouldin, calinski


def page_clustering(filtered: dict, ciclos: pd.DataFrame) -> None:
    st.title("Clustering Interactivo de Máquinas")

    st.markdown("""
    **Agrupa máquinas** con comportamiento similar usando diferentes algoritmos de clustering.
    Experimenta con features, algoritmos e hiperparámetros para encontrar los mejores agrupamientos.
    """)

    with st.expander("ℹ️ ¿Qué es el Clustering?"):
        st.markdown("""
        El clustering es una técnica de ML no supervisado que agrupa datos similares.

        **Métricas analizadas:**
        - **Disponibilidad**: % tiempo en producción
        - **Scrap Rate**: % piezas defectuosas
        - **UPH Real**: Unidades por hora
        - **Duración Producción**: Minutos totales

        **Métricas de evaluación:**
        - **Silhouette Score** (0-1): Mayor es mejor. Mide qué tan similares son los puntos dentro de su cluster vs otros.
        - **Davies-Bouldin Index**: Menor es mejor. Mide la separación entre clusters.
        - **Calinski-Harabasz**: Mayor es mejor. Ratio de dispersión entre clusters vs dentro de clusters.
        """)

    prod = filtered.get("produccion", pd.DataFrame())

    if prod.empty:
        st.info("Sin datos de producción en el rango seleccionado.")
        return

    machine_metrics = prepare_features(prod)

    if len(machine_metrics) < 3:
        st.warning("No hay suficientes máquinas con datos para clustering (mínimo 3).")
        return

    st.markdown(f"**📊 Máquinas disponibles:** {len(machine_metrics)}")

    st.markdown("---")
    st.markdown("### 1️⃣ Configuración del Clustering")

    col_config1, col_config2 = st.columns([2, 1])

    with col_config1:
        st.markdown("#### Selección de Features")

        all_features = ["disponibilidad", "scrap_rate", "uph_real", "dur_prod"]

        selected_features = st.multiselect(
            "Selecciona features:",
            all_features,
            default=all_features,
            key="features_clustering"
        )

        if len(selected_features) == 0:
            st.warning("⚠️ Selecciona al menos 1 feature")
            return

        st.success(f"✅ {len(selected_features)} features seleccionadas")

    with col_config2:
        st.markdown("#### Algoritmo")

        algorithm = st.selectbox(
            "Elige algoritmo:",
            ["K-Means", "DBSCAN", "Agglomerative"],
            help="K-Means: rápido, requiere n_clusters. DBSCAN: detecta outliers. Agglomerative: jerárquico."
        )

    st.markdown("---")
    st.markdown("#### Hiperparámetros")

    col_param1, col_param2, col_param3 = st.columns(3)

    params = {}

    if algorithm == "K-Means":
        with col_param1:
            params["n_clusters"] = st.slider(
                "Número de clusters",
                min_value=2,
                max_value=min(10, len(machine_metrics)),
                value=3,
                help="Cuántos grupos crear"
            )

    elif algorithm == "DBSCAN":
        with col_param1:
            params["eps"] = st.slider(
                "Epsilon (distancia)",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Máxima distancia entre puntos del mismo cluster"
            )
        with col_param2:
            params["min_samples"] = st.slider(
                "Mínimo de muestras",
                min_value=2,
                max_value=10,
                value=3,
                help="Mínimo de puntos para formar un cluster"
            )

    elif algorithm == "Agglomerative":
        with col_param1:
            params["n_clusters"] = st.slider(
                "Número de clusters",
                min_value=2,
                max_value=min(10, len(machine_metrics)),
                value=3
            )
        with col_param2:
            params["linkage"] = st.selectbox(
                "Método de enlace",
                ["ward", "complete", "average", "single"],
                help="Cómo medir distancia entre clusters"
            )

    st.markdown("---")

    # Botón entrenar
    if st.button("🚀 Ejecutar Clustering", type="primary", use_container_width=True):
        with st.spinner("Ejecutando clustering..."):
            # Preparar datos
            df_features = machine_metrics[["machine_name"] + selected_features].copy()
            df_features = df_features.dropna()

            if len(df_features) < 3:
                st.error("No hay suficientes máquinas con métricas completas.")
                return

            # Normalizar
            scaler = StandardScaler()
            X = scaler.fit_transform(df_features[selected_features])

            # Entrenar
            model, clusters = train_clustering_model(X, algorithm, params)

            # Evaluar
            silhouette, davies_bouldin, calinski = evaluate_clustering(X, clusters)

            # Guardar en session_state
            st.session_state['clustering_model'] = model
            st.session_state['clustering_scaler'] = scaler
            st.session_state['clustering_features'] = selected_features
            st.session_state['clustering_algorithm'] = algorithm
            st.session_state['clustering_X'] = X
            st.session_state['clustering_clusters'] = clusters
            st.session_state['clustering_metrics'] = (silhouette, davies_bouldin, calinski)
            st.session_state['clustering_df'] = df_features

            st.success("✅ Clustering completado!")
            st.rerun()

    # Mostrar resultados si hay clustering ejecutado
    if 'clustering_model' in st.session_state:
        model = st.session_state['clustering_model']
        scaler = st.session_state['clustering_scaler']
        selected_features = st.session_state['clustering_features']
        algorithm = st.session_state['clustering_algorithm']
        X = st.session_state['clustering_X']
        clusters = st.session_state['clustering_clusters']
        silhouette, davies_bouldin, calinski = st.session_state['clustering_metrics']
        df_features = st.session_state['clustering_df']

        df_features["cluster"] = clusters
        df_features["cluster_str"] = df_features["cluster"].apply(
            lambda x: f"Cluster {x}" if x != -1 else "Outlier"
        )

        st.markdown("---")
        st.markdown("### 2️⃣ Resultados del Clustering")

        # Métricas de evaluación
        if silhouette is not None:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Silhouette Score", f"{silhouette:.3f}", help="0-1, mayor es mejor")
            col_m2.metric("Davies-Bouldin", f"{davies_bouldin:.3f}", help="Menor es mejor")
            col_m3.metric("Calinski-Harabasz", f"{calinski:.1f}", help="Mayor es mejor")

            # Interpretación
            if silhouette > 0.5:
                st.success("✅ Excelente separación entre clusters!")
            elif silhouette > 0.3:
                st.info("ℹ️ Separación aceptable. Prueba ajustar parámetros.")
            else:
                st.warning("⚠️ Baja separación. Los clusters se solapan. Ajusta parámetros o features.")
        else:
            st.warning("No se pudieron calcular métricas (clusters insuficientes o muchos outliers)")

        st.markdown("---")
        st.markdown("### 3️⃣ Visualización de Clusters")

        # Resumen por cluster
        cluster_summary = df_features.groupby("cluster_str").agg(
            n_maquinas=("machine_name", "count"),
            disp_media=("disponibilidad", "mean"),
            scrap_medio=("scrap_rate", "mean"),
            uph_media=("uph_real", "mean") if "uph_real" in selected_features else ("disponibilidad", "count"),
        ).reset_index()

        col_sum1, col_sum2 = st.columns(2)

        col_sum1.dataframe(
            cluster_summary.style.format({
                "disp_media": "{:.1%}",
                "scrap_medio": "{:.1%}",
                "uph_media": "{:.1f}",
            }),
            hide_index=True,
            use_container_width=True
        )

        fig_dist = px.bar(
            cluster_summary,
            x="cluster_str",
            y="n_maquinas",
            title="Distribución de Máquinas por Cluster",
            labels={"n_maquinas": "Número", "cluster_str": "Cluster"},
            text="n_maquinas"
        )
        fig_dist.update_traces(textposition="outside")
        col_sum2.plotly_chart(fig_dist, use_container_width=True)

        # Visualización 3D o 2D según features disponibles
        if len(selected_features) >= 3:
            st.markdown("#### Visualización 3D")

            fig_3d = px.scatter_3d(
                df_features,
                x=selected_features[0],
                y=selected_features[1],
                z=selected_features[2],
                color="cluster_str",
                hover_data=["machine_name"],
                title="Clusters en 3D",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_3d.update_traces(marker=dict(size=8, line=dict(width=1, color="DarkSlateGrey")))
            st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("#### Gráficos 2D")

        col_2d1, col_2d2 = st.columns(2)

        if "disponibilidad" in selected_features and "scrap_rate" in selected_features:
            fig_disp_scrap = px.scatter(
                df_features,
                x="disponibilidad",
                y="scrap_rate",
                color="cluster_str",
                hover_data=["machine_name"],
                title="Disponibilidad vs Scrap %",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_disp_scrap.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
            fig_disp_scrap.update_xaxes(tickformat=".0%")
            fig_disp_scrap.update_yaxes(tickformat=".0%")
            col_2d1.plotly_chart(fig_disp_scrap, use_container_width=True)

        if "uph_real" in selected_features and "scrap_rate" in selected_features:
            fig_uph_scrap = px.scatter(
                df_features,
                x="uph_real",
                y="scrap_rate",
                color="cluster_str",
                hover_data=["machine_name"],
                title="UPH Real vs Scrap %",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_uph_scrap.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
            fig_uph_scrap.update_yaxes(tickformat=".0%")
            col_2d2.plotly_chart(fig_uph_scrap, use_container_width=True)

        st.markdown("---")
        st.markdown("### 4️⃣ Detalle de Máquinas y Clusters")

        df_display = df_features.copy()
        df_display = df_display.sort_values(["cluster_str", "machine_name"])

        st.dataframe(
            df_display.style.format({
                "disponibilidad": "{:.1%}",
                "scrap_rate": "{:.1%}",
                "uph_real": "{:.1f}",
                "dur_prod": "{:.0f}",
            }),
            hide_index=True,
            use_container_width=True
        )

        # Comparación con modelo pre-entrenado
        st.markdown("---")
        st.markdown("### 5️⃣ Comparación con Modelo Pre-entrenado")

        pretrained_path = Path(__file__).parent.parent.parent.parent / "models" / "clustering" / "trained_model"

        try:
            with open(pretrained_path / "kmeans_model.pkl", "rb") as f:
                pretrained_model = pickle.load(f)
            with open(pretrained_path / "scaler.pkl", "rb") as f:
                pretrained_scaler = pickle.load(f)
            with open(pretrained_path / "features.txt", "r") as f:
                pretrained_features = [line.strip() for line in f.readlines()]

            # Hacer predicciones con modelo pre-entrenado
            X_pretrained = df_features[pretrained_features].fillna(0)
            X_pretrained_scaled = pretrained_scaler.transform(X_pretrained)
            pretrained_clusters = pretrained_model.predict(X_pretrained_scaled)

            pretrained_silhouette, pretrained_davies, pretrained_calinski = evaluate_clustering(
                X_pretrained_scaled, pretrained_clusters
            )

            if pretrained_silhouette is not None and silhouette is not None:
                col_comp1, col_comp2 = st.columns(2)

                with col_comp1:
                    delta_silhouette = silhouette - pretrained_silhouette
                    st.metric(
                        "Silhouette Score",
                        f"{silhouette:.3f}",
                        delta=f"{delta_silhouette:.3f}",
                        help=f"Pre-entrenado: {pretrained_silhouette:.3f}"
                    )

                with col_comp2:
                    if silhouette > pretrained_silhouette:
                        st.success("🎉 ¡Tu clustering supera al pre-entrenado!")
                    else:
                        st.info("El modelo pre-entrenado tiene mejor separación.")
        except FileNotFoundError:
            st.info("No hay modelo pre-entrenado para comparar.")

        # Opción de guardar
        st.markdown("---")
        st.markdown("### 6️⃣ Guardar Modelo")

        if st.button("💾 Guardar este modelo", help="Guarda el modelo entrenado para uso futuro"):
            model_path = Path(__file__).parent.parent.parent.parent / "models" / "clustering" / "trained_model"
            model_path.mkdir(parents=True, exist_ok=True)

            with open(model_path / "interactive_clustering.pkl", "wb") as f:
                pickle.dump(model, f)
            with open(model_path / "interactive_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            with open(model_path / "interactive_features.txt", "w") as f:
                f.write("\n".join(selected_features))

            st.success(f"✅ Modelo guardado en: models/clustering/trained_model/interactive_clustering.pkl")

        # Interpretación
        st.markdown("---")
        st.markdown("### 💡 Interpretación")

        st.info(
            "**Cómo usar estos clusters:**\n\n"
            "- **Cluster verde (alta disp, bajo scrap)**: Best performers, replica sus prácticas\n"
            "- **Cluster rojo (baja disp)**: Problemas de paradas, mantenimiento urgente\n"
            "- **Cluster naranja (alto scrap)**: Problemas de calidad, revisa ajustes\n"
            "- **Outliers**: Máquinas atípicas, requieren análisis individual\n\n"
            "**Métricas:**\n"
            "- Silhouette > 0.5 = Excelente separación\n"
            "- Davies-Bouldin < 1.0 = Buena compacidad\n"
            "- Calinski-Harabasz alto = Clusters bien definidos"
        )
