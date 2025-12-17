"""
Pagina del dashboard para el modelo de Clustering ML.
Permite usar modelo pre-entrenado Y entrenar modelos interactivamente.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score


def list_available_clustering_models():
    """
    Lista todos los modelos de clustering disponibles.

    Solo lista:
    - Modelo base pre-entrenado
    - Modelos guardados en BentoML

    Returns:
        dict: Diccionario con nombre_modelo -> ruta_archivos
    """
    model_dir = Path(__file__).parent.parent.parent.parent / "models" / "clustering" / "trained_model"

    if not model_dir.exists():
        return {}

    available_models = {}

    # Buscar modelo base pre-entrenado
    if (model_dir / "kmeans_model.pkl").exists():
        available_models["Modelo Base"] = {
            "model": model_dir / "kmeans_model.pkl",
            "scaler": model_dir / "scaler.pkl",
            "features": model_dir / "features.txt"
        }

    # Buscar modelos de BentoML
    try:
        import bentoml
        bento_models = bentoml.models.list()
        for bm in bento_models:
            if ("clustering" in bm.tag.name.lower() or "kmeans" in bm.tag.name.lower()) and "scaler" not in bm.tag.name.lower():
                available_models[f"{bm.tag.name} (v{bm.tag.version})"] = {
                    "bentoml_tag": str(bm.tag),
                    "type": "bentoml"
                }
    except:
        pass

    return available_models


@st.cache_resource
def load_clustering_model(model_name=None):
    """
    Carga el modelo de clustering especificado.

    Args:
        model_name: Nombre del modelo a cargar. Si es None, carga el principal.

    Returns:
        tuple: (modelo, scaler, lista_features) o (None, None, None) si no existe
    """
    available_models = list_available_clustering_models()

    if not available_models:
        return None, None, None

    if model_name is None or model_name not in available_models:
        model_name = list(available_models.keys())[0]

    model_info = available_models[model_name]

    try:
        # Cargar desde BentoML
        if model_info.get("type") == "bentoml":
            import bentoml
            bento_model = bentoml.models.get(model_info["bentoml_tag"])
            metadata = bento_model.info.metadata
            features = metadata.get("features", [])

            model = bentoml.sklearn.load_model(model_info["bentoml_tag"])

            scaler_tag = metadata.get("scaler_tag")
            if scaler_tag:
                scaler = bentoml.sklearn.load_model(scaler_tag)
            else:
                tag_parts = model_info["bentoml_tag"].split(":")
                model_base_name = tag_parts[0]
                scaler_name = f"{model_base_name}_scaler"
                scaler = bentoml.sklearn.load_model(scaler_name)

            return model, scaler, features

        # Cargar desde pickle
        with open(model_info["model"], "rb") as f:
            model = pickle.load(f)

        with open(model_info["scaler"], "rb") as f:
            scaler = pickle.load(f)

        with open(model_info["features"], "r") as f:
            features = [line.strip() for line in f.readlines()]

        return model, scaler, features
    except Exception as e:
        st.error(f"Error al cargar modelo {model_name}: {str(e)}")
        return None, None, None


def train_interactive_clustering(X, n_clusters, max_iter, n_init, random_state=42):
    """Entrena un modelo K-Means con hiperparametros especificados."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state
    )

    clusters = model.fit_predict(X_scaled)

    return model, scaler, clusters, X_scaled


def evaluate_clustering(X_scaled, clusters):
    """Evalua la calidad del clustering."""
    if len(np.unique(clusters)) < 2:
        return 0.0, float('inf')

    silhouette = silhouette_score(X_scaled, clusters)
    davies_bouldin = davies_bouldin_score(X_scaled, clusters)

    return silhouette, davies_bouldin


def find_optimal_clusters(X, max_clusters=10):
    """Encuentra el numero optimo de clusters usando el metodo del codo y silhouette."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    silhouettes = []
    k_range = range(2, min(max_clusters + 1, len(X)))

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, clusters))

    return list(k_range), inertias, silhouettes


def page_ml_clustering(filtered: dict, ciclos: pd.DataFrame) -> None:
    st.title("Clustering ML - Agrupacion de Maquinas")

    # Selector de modo
    st.markdown("### Modo de Operacion")
    modo = st.radio(
        "Elige el modo:",
        ["Modelo Pre-entrenado", "Entrenar Modelo Interactivo"],
        horizontal=True,
        help="Pre-entrenado: usa el modelo ya guardado. Interactivo: entrena tu propio modelo."
    )

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

    if len(machine_metrics) < 2:
        st.warning("No hay suficientes datos para hacer clustering (mínimo 2 máquinas).")
        return

    # ========== MODO PRE-ENTRENADO ==========
    if modo == "Modelo Pre-entrenado":
        st.markdown("### Seleccion de Modelo Pre-entrenado")

        # Listar modelos disponibles
        available_models = list_available_clustering_models()

        if not available_models:
            st.error("No hay modelos pre-entrenados disponibles. Ejecuta: python models/clustering/train.py")
            st.info("O entrena un modelo en modo interactivo y guardalo.")
            return

        # Selector de modelo
        st.markdown(f"**{len(available_models)} modelo(s) disponible(s)**")

        selected_model = st.selectbox(
            "Selecciona el modelo a usar:",
            options=list(available_models.keys()),
            help="Elige entre los modelos entrenados disponibles"
        )

        # Mostrar informacion del modelo seleccionado
        model_info = available_models[selected_model]
        if model_info.get("type") == "bentoml":
            st.info(f"Modelo de BentoML: {model_info['bentoml_tag']}")
        else:
            st.info(f"Modelo en disco: {model_info['model'].name}")

        # Cargar modelo seleccionado
        model, scaler, features = load_clustering_model(selected_model)

        if model is None:
            st.error(f"Error al cargar el modelo: {selected_model}")
            return

        st.markdown("---")
        st.markdown("""
        Usando **K-Means** para agrupar maquinas con caracteristicas similares.

        **Features**: Disponibilidad, Scrap Rate, UPH Real, Duracion Produccion.
        """)

        # Info del modelo
        col1, col2, col3 = st.columns(3)
        col1.metric("Clusters", model.n_clusters)
        col2.metric("Features", len(features))
        col3.metric("Algoritmo", "K-Means")

        st.markdown(f"### Predicciones para {len(machine_metrics)} maquinas")

        # Hacer predicciones
        X = machine_metrics[features].values
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        machine_metrics["cluster"] = predictions
        machine_metrics["cluster_str"] = "Cluster " + machine_metrics["cluster"].astype(str)

    # ========== MODO INTERACTIVO ==========
    else:
        st.markdown("""
        **Entrena tu propio modelo** de K-Means eligiendo features e hiperparámetros.
        """)

        st.markdown("---")
        st.markdown("### 1️⃣ Configuración del Modelo")

        col_config1, col_config2 = st.columns([2, 1])

        with col_config1:
            st.markdown("#### Selección de Features")

            all_features = ["disponibilidad", "scrap_rate", "uph_real", "dur_prod", "dur_prep", "dur_inci"]

            selected_features = st.multiselect(
                "Selecciona features:",
                all_features,
                default=["disponibilidad", "scrap_rate", "uph_real", "dur_prod"],
                key="features_clustering"
            )

            if len(selected_features) < 2:
                st.warning("⚠️ Selecciona al menos 2 features")
                return

            st.success(f"✅ {len(selected_features)} features seleccionadas")

            # Análisis del número óptimo de clusters
            st.markdown("#### Análisis de Clusters Óptimos")

            if st.button("🔍 Analizar número óptimo de clusters"):
                with st.spinner("Analizando..."):
                    X_analysis = machine_metrics[selected_features].values
                    k_range, inertias, silhouettes = find_optimal_clusters(X_analysis, max_clusters=8)

                    col_elbow, col_sil = st.columns(2)

                    with col_elbow:
                        fig_elbow = go.Figure()
                        fig_elbow.add_trace(go.Scatter(
                            x=k_range,
                            y=inertias,
                            mode='lines+markers',
                            name='Inercia'
                        ))
                        fig_elbow.update_layout(
                            title="Método del Codo",
                            xaxis_title="Número de Clusters (k)",
                            yaxis_title="Inercia"
                        )
                        st.plotly_chart(fig_elbow, use_container_width=True)

                    with col_sil:
                        fig_sil = go.Figure()
                        fig_sil.add_trace(go.Scatter(
                            x=k_range,
                            y=silhouettes,
                            mode='lines+markers',
                            name='Silhouette Score',
                            line=dict(color='green')
                        ))
                        fig_sil.update_layout(
                            title="Silhouette Score",
                            xaxis_title="Número de Clusters (k)",
                            yaxis_title="Silhouette Score"
                        )
                        st.plotly_chart(fig_sil, use_container_width=True)

                    best_k = k_range[silhouettes.index(max(silhouettes))]
                    st.info(f"💡 **Recomendación**: El número óptimo de clusters según Silhouette Score es **{best_k}**")

        with col_config2:
            st.markdown("#### Hiperparámetros")

            n_clusters = st.slider(
                "Número de clusters",
                min_value=2,
                max_value=min(10, len(machine_metrics) - 1),
                value=3,
                help="Número de grupos a formar"
            )

            max_iter = st.slider(
                "Iteraciones máximas",
                min_value=100,
                max_value=1000,
                value=300,
                step=100,
                help="Máximo de iteraciones del algoritmo"
            )

            n_init = st.slider(
                "Número de inicializaciones",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="Número de veces que se ejecuta el algoritmo con diferentes centroides"
            )

        st.markdown("---")

        # Botón entrenar
        if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
            with st.spinner("Entrenando modelo... Esto puede tardar unos segundos."):
                # Preparar datos
                X = machine_metrics[selected_features].values

                # Entrenar
                model, scaler, clusters, X_scaled = train_interactive_clustering(
                    X, n_clusters, max_iter, n_init
                )

                # Evaluar
                silhouette, davies_bouldin = evaluate_clustering(X_scaled, clusters)

                # Guardar en session_state
                st.session_state['trained_clustering_model'] = model
                st.session_state['trained_clustering_scaler'] = scaler
                st.session_state['trained_clustering_features'] = selected_features
                st.session_state['clustering_metrics'] = (silhouette, davies_bouldin)
                st.session_state['clustering_clusters'] = clusters

                st.success("✅ Modelo entrenado exitosamente!")
                st.rerun()

        # Mostrar resultados si hay modelo entrenado
        if 'trained_clustering_model' in st.session_state:
            model = st.session_state['trained_clustering_model']
            scaler = st.session_state['trained_clustering_scaler']
            selected_features = st.session_state['trained_clustering_features']
            silhouette, davies_bouldin = st.session_state['clustering_metrics']
            clusters = st.session_state['clustering_clusters']

            st.markdown("### 2️⃣ Resultados del Entrenamiento")

            # Métricas
            col_m1, col_m2, col_m3 = st.columns(3)

            col_m1.metric("Número de Clusters", model.n_clusters)
            col_m2.metric(
                "Silhouette Score",
                f"{silhouette:.3f}",
                help="Rango: [-1, 1]. Mayor es mejor. >0.5 indica buenos clusters."
            )
            col_m3.metric(
                "Davies-Bouldin Index",
                f"{davies_bouldin:.3f}",
                help="Menor es mejor. <1.0 indica buenos clusters."
            )

            # Interpretación
            if silhouette > 0.5:
                st.success("✅ Excelente calidad de clustering! Los clusters están bien definidos.")
            elif silhouette > 0.3:
                st.info("ℹ️ Calidad aceptable. Los clusters tienen cierta separación.")
            else:
                st.warning("⚠️ Baja calidad de clustering. Prueba ajustar el número de clusters o features.")

            # Comparación con modelo pre-entrenado
            st.markdown("---")
            st.markdown("### 3️⃣ Comparación con Modelo Pre-entrenado")

            pretrained_model, pretrained_scaler, pretrained_features = load_clustering_model()

            if pretrained_model is not None:
                X_pretrained = machine_metrics[pretrained_features].values
                X_pretrained_scaled = pretrained_scaler.transform(X_pretrained)
                pretrained_clusters = pretrained_model.predict(X_pretrained_scaled)

                pretrained_silhouette, pretrained_davies_bouldin = evaluate_clustering(
                    X_pretrained_scaled, pretrained_clusters
                )

                col_comp1, col_comp2 = st.columns(2)

                with col_comp1:
                    delta_sil = silhouette - pretrained_silhouette
                    st.metric(
                        "Silhouette Score",
                        f"{silhouette:.3f}",
                        delta=f"{delta_sil:.3f}",
                        help=f"Pre-entrenado: {pretrained_silhouette:.3f}"
                    )

                with col_comp2:
                    delta_db = pretrained_davies_bouldin - davies_bouldin
                    st.metric(
                        "Davies-Bouldin Index",
                        f"{davies_bouldin:.3f}",
                        delta=f"{delta_db:.3f}",
                        help=f"Pre-entrenado: {pretrained_davies_bouldin:.3f}"
                    )

                if silhouette > pretrained_silhouette:
                    st.success("🎉 ¡Tu modelo interactivo supera al pre-entrenado!")
                else:
                    st.info("El modelo pre-entrenado sigue siendo mejor. Prueba ajustar hiperparámetros.")

            # Guardar modelo
            st.markdown("---")
            st.markdown("### Guardar Modelo")

            st.info("Tu modelo ha sido entrenado. Guardalo en BentoML para uso futuro y despliegue como API.")

            # Campo para nombre del modelo
            model_name = st.text_input(
                "Nombre del modelo:",
                value="machine_clustering",
                help="Dale un nombre descriptivo a tu modelo"
            )

            if st.button("Guardar Modelo", help="Guarda el modelo en BentoML", use_container_width=True, type="primary"):
                if not model_name or model_name.strip() == "":
                    st.error("Por favor ingresa un nombre para el modelo")
                else:
                    try:
                        import bentoml

                        scaler_name = f"{model_name}_scaler"
                        saved_scaler = bentoml.sklearn.save_model(
                            scaler_name,
                            scaler,
                            labels={
                                "framework": "sklearn",
                                "model_type": "StandardScaler",
                                "task": "preprocessing"
                            },
                            metadata={
                                "features": selected_features
                            }
                        )

                        bentoml.sklearn.save_model(
                            model_name,
                            model,
                            labels={
                                "framework": "sklearn",
                                "model_type": "KMeans",
                                "task": "clustering"
                            },
                            metadata={
                                "features": selected_features,
                                "n_clusters": model.n_clusters,
                                "silhouette_score": silhouette,
                                "davies_bouldin_index": davies_bouldin,
                                "scaler_tag": str(saved_scaler.tag)
                            }
                        )

                        st.success(f"Modelo guardado exitosamente en BentoML: {model_name}:latest")

                    except ImportError:
                        st.error("BentoML no esta instalado. Ejecuta: pip install bentoml")
                    except Exception as e:
                        st.error(f"Error al guardar el modelo: {str(e)}")

            # Hacer predicciones en todos los datos
            X_all = machine_metrics[selected_features].values
            X_all_scaled = scaler.transform(X_all)
            all_predictions = model.predict(X_all_scaled)

            machine_metrics["cluster"] = all_predictions
            machine_metrics["cluster_str"] = "Cluster " + machine_metrics["cluster"].astype(str)

    # ========== VISUALIZACIONES COMUNES (para ambos modos) ==========
    if modo == "Modelo Pre-entrenado" or 'trained_clustering_model' in st.session_state:
        st.markdown("---")
        st.markdown("### Analisis de Clustering")

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

        # Determinar features disponibles (puede variar entre pre-entrenado e interactivo)
        available_features = machine_metrics.columns.tolist()
        display_columns = ["machine_name", "disponibilidad", "scrap_rate", "uph_real", "dur_prod", "cluster_str"]
        display_columns = [col for col in display_columns if col in available_features]

        df_display = machine_metrics[display_columns].copy()
        df_display = df_display.sort_values(["cluster_str", "machine_name"])

        format_dict = {}
        if "disponibilidad" in display_columns:
            format_dict["disponibilidad"] = "{:.1%}"
        if "scrap_rate" in display_columns:
            format_dict["scrap_rate"] = "{:.1%}"
        if "uph_real" in display_columns:
            format_dict["uph_real"] = "{:.1f}"
        if "dur_prod" in display_columns:
            format_dict["dur_prod"] = "{:.0f}"

        st.dataframe(
            df_display.style.format(format_dict),
            width='stretch',
            hide_index=True
        )

        # Centros de los clusters
        with st.expander("Ver centros de los clusters (valores normalizados)"):
            # Obtener features del modelo actual
            if modo == "Modelo Pre-entrenado":
                current_features = features
            else:
                current_features = selected_features

            centers = pd.DataFrame(
                scaler.inverse_transform(model.cluster_centers_),
                columns=current_features
            )
            centers.index = [f"Cluster {i}" for i in range(model.n_clusters)]

            format_dict_centers = {}
            if "disponibilidad" in current_features:
                format_dict_centers["disponibilidad"] = "{:.1%}"
            if "scrap_rate" in current_features:
                format_dict_centers["scrap_rate"] = "{:.1%}"
            if "uph_real" in current_features:
                format_dict_centers["uph_real"] = "{:.1f}"
            if "dur_prod" in current_features:
                format_dict_centers["dur_prod"] = "{:.0f}"
            if "dur_prep" in current_features:
                format_dict_centers["dur_prep"] = "{:.0f}"
            if "dur_inci" in current_features:
                format_dict_centers["dur_inci"] = "{:.0f}"

            st.dataframe(
                centers.style.format(format_dict_centers),
                width='stretch'
            )

        # Interpretacion
        st.markdown("### Interpretacion")

        # Análisis automático de clusters
        interpretations = []
        for idx, row in cluster_summary.iterrows():
            cluster_name = row["cluster_str"]
            n_machines = row["n_maquinas"]
            disp = row["disp_media"]
            scrap = row["scrap_medio"]
            uph = row["uph_media"]

            if disp >= 0.80 and scrap <= 0.03:
                interpretations.append(f"**{cluster_name}** ({n_machines} máquinas): 🌟 **Best Performers** - Alta disponibilidad ({disp:.1%}) y bajo scrap ({scrap:.1%}). Estas son las máquinas de referencia.")
            elif disp < 0.60:
                interpretations.append(f"**{cluster_name}** ({n_machines} máquinas): ⚠️ **Baja Disponibilidad** - Solo {disp:.1%} de disponibilidad. Requiere investigación de paradas y mantenimiento.")
            elif scrap > 0.08:
                interpretations.append(f"**{cluster_name}** ({n_machines} máquinas): 🔴 **Alto Scrap** - {scrap:.1%} de scrap. Revisar ajustes, tooling y calidad de materia prima.")
            elif n_machines <= 2:
                interpretations.append(f"**{cluster_name}** ({n_machines} máquinas): 🎯 **Outliers** - Comportamiento atípico que requiere atención especial.")
            else:
                interpretations.append(f"**{cluster_name}** ({n_machines} máquinas): ✅ **Rendimiento Normal** - Disponibilidad: {disp:.1%}, Scrap: {scrap:.1%}, UPH: {uph:.1f}")

        for interpretation in interpretations:
            st.markdown(interpretation)

        st.info(
            "**Recomendaciones generales:**\n\n"
            "- Estudiar las mejores prácticas de los clusters de alto rendimiento.\n"
            "- Priorizar mejoras en clusters con baja disponibilidad o alto scrap.\n"
            "- Investigar outliers individualmente para entender causas específicas."
        )
