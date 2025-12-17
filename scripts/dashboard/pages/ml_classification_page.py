"""
Pagina del dashboard para el modelo de Clasificacion ML.
Carga el modelo entrenado y permite clasificar el estado de maquinas.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report


@st.cache_resource
def load_classification_model():
    """Carga el modelo de clasificacion entrenado."""
    model_path = Path(__file__).parent.parent.parent.parent / "models" / "classification" / "trained_model"

    try:
        with open(model_path / "random_forest_classifier.pkl", "rb") as f:
            model = pickle.load(f)

        with open(model_path / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open(model_path / "features.txt", "r") as f:
            features = [line.strip() for line in f.readlines()]

        with open(model_path / "classes.txt", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        return model, scaler, features, classes
    except FileNotFoundError:
        st.error("Modelo no encontrado. Ejecuta: python models/classification/train.py")
        return None, None, None, None


def page_ml_classification(filtered: dict, ciclos: pd.DataFrame) -> None:
    st.title("Clasificacion ML - Estado de Maquinas")

    st.markdown("""
    Este modelo utiliza **Random Forest Classifier** para clasificar el estado de salud de las maquinas
    basandose en metricas de rendimiento.

    **Clases:**
    - **EXCELENTE**: Disponibilidad >= 85%, Scrap <= 2%, UPH >= 100
    - **BUENA**: Disponibilidad >= 70%, Scrap <= 5%, UPH >= 60
    - **REQUIERE_ATENCION**: Metricas por debajo de objetivos
    - **CRITICA**: Disponibilidad < 50% o Scrap > 10%
    """)

    # Cargar modelo
    model, scaler, features, classes = load_classification_model()

    if model is None:
        st.warning("No se pudo cargar el modelo. Asegurate de haberlo entrenado primero.")
        return

    # Info del modelo
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Features", len(features))
    col2.metric("Clases", len(classes))
    col3.metric("Algoritmo", "Random Forest")
    col4.metric("Accuracy", "92.77%", help="Accuracy del modelo en test set")

    st.markdown("---")

    # Preparar datos actuales
    prod = filtered.get("produccion", pd.DataFrame())

    if prod.empty:
        st.info("Sin datos de produccion en el rango seleccionado.")
        return

    # Preparar features (misma logica que el entrenamiento)
    df = prod.copy()
    df["estado_oee"] = df["evento"].str.lower().map({
        "produccion": "produccion",
        "producción": "produccion",
        "preparacion": "preparacion",
        "preparación": "preparacion",
    })
    df["estado_oee"] = df["estado_oee"].fillna("incidencia")

    df["total_piezas"] = df["piezas_ok"] + df["piezas_scrap"]
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

    # Piezas
    of_final = of_final.copy()
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

    machine_metrics["prep_ratio"] = np.where(
        machine_metrics["dur_prod"] > 0,
        machine_metrics["dur_prep"] / machine_metrics["dur_prod"],
        np.nan
    )

    machine_metrics["inci_ratio"] = np.where(
        machine_metrics["dur_prod"] > 0,
        machine_metrics["dur_inci"] / machine_metrics["dur_prod"],
        np.nan
    )

    # Filtrar registros con datos suficientes
    machine_metrics = machine_metrics[machine_metrics["total_dur"] > 60].copy()
    machine_metrics = machine_metrics.dropna(subset=["disponibilidad", "scrap_rate", "uph_real"])

    if len(machine_metrics) < 1:
        st.warning("No hay suficientes datos para hacer predicciones.")
        return

    st.markdown(f"### Clasificacion de {len(machine_metrics)} registros (maquina+periodo)")

    # Hacer predicciones
    X = machine_metrics[features].values
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    machine_metrics["estado_predicho"] = predictions
    machine_metrics["probabilidad_max"] = probabilities.max(axis=1)

    # Distribucion de estados predichos
    st.markdown("#### Distribucion de Estados Predichos")

    estado_counts = machine_metrics["estado_predicho"].value_counts().reset_index()
    estado_counts.columns = ["estado", "count"]

    # Mapear colores por estado
    color_map = {
        "EXCELENTE": "#22c55e",
        "BUENA": "#3b82f6",
        "REQUIERE_ATENCION": "#f59e0b",
        "CRITICA": "#ef4444"
    }

    col_dist1, col_dist2 = st.columns(2)

    # Tabla
    col_dist1.dataframe(estado_counts, width='stretch', hide_index=True)

    # Grafico de pie
    fig_pie = px.pie(
        estado_counts,
        values="count",
        names="estado",
        title="Distribucion de Estados",
        color="estado",
        color_discrete_map=color_map
    )
    col_dist2.plotly_chart(fig_pie, width='stretch')

    # Maquinas por estado
    st.markdown("#### Maquinas por Estado")

    # Agrupar por maquina (tomar estado mas frecuente o mas reciente)
    machine_latest = machine_metrics.sort_values("periodo", ascending=False).groupby("machine_name").first().reset_index()

    tabs = st.tabs(["EXCELENTE", "BUENA", "REQUIERE_ATENCION", "CRITICA"])

    for i, estado in enumerate(["EXCELENTE", "BUENA", "REQUIERE_ATENCION", "CRITICA"]):
        with tabs[i]:
            machines_in_state = machine_latest[machine_latest["estado_predicho"] == estado]

            if len(machines_in_state) == 0:
                st.info(f"No hay maquinas en estado {estado}.")
                continue

            st.markdown(f"**{len(machines_in_state)} maquinas en estado {estado}**")

            st.dataframe(
                machines_in_state[[
                    "machine_name", "disponibilidad", "scrap_rate", "uph_real",
                    "dur_prod", "probabilidad_max"
                ]].style.format({
                    "disponibilidad": "{:.1%}",
                    "scrap_rate": "{:.1%}",
                    "uph_real": "{:.1f}",
                    "dur_prod": "{:.0f}",
                    "probabilidad_max": "{:.1%}",
                }),
                width='stretch',
                hide_index=True
            )

    # Scatter plot: Disponibilidad vs Scrap coloreado por estado
    st.markdown("#### Visualizacion: Disponibilidad vs Scrap por Estado")

    fig_scatter = px.scatter(
        machine_metrics,
        x="disponibilidad",
        y="scrap_rate",
        color="estado_predicho",
        size="probabilidad_max",
        hover_data=["machine_name", "periodo", "uph_real"],
        title="Clasificacion de Maquinas (Disponibilidad vs Scrap)",
        labels={
            "disponibilidad": "Disponibilidad",
            "scrap_rate": "Scrap %",
            "estado_predicho": "Estado",
            "probabilidad_max": "Probabilidad"
        },
        color_discrete_map=color_map
    )
    fig_scatter.update_xaxes(tickformat=".0%")
    fig_scatter.update_yaxes(tickformat=".0%")
    fig_scatter.update_traces(marker=dict(line=dict(width=1, color="white")))
    st.plotly_chart(fig_scatter, width='stretch')

    # Timeline de estados por maquina
    st.markdown("#### Timeline de Estados por Maquina")

    # Seleccionar maquina
    machines_list = sorted(machine_metrics["machine_name"].unique())
    selected_machine = st.selectbox("Seleccionar maquina", machines_list, key="machine_selector")

    machine_timeline = machine_metrics[machine_metrics["machine_name"] == selected_machine].sort_values("periodo")

    if len(machine_timeline) > 0:
        fig_timeline = px.line(
            machine_timeline,
            x="periodo",
            y="probabilidad_max",
            color="estado_predicho",
            markers=True,
            title=f"Timeline de Estados - {selected_machine}",
            labels={"probabilidad_max": "Probabilidad", "periodo": "Periodo", "estado_predicho": "Estado"},
            color_discrete_map=color_map
        )
        fig_timeline.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_timeline, width='stretch')

        # Metricas de la maquina
        col_mach1, col_mach2, col_mach3, col_mach4 = st.columns(4)
        col_mach1.metric("Disponibilidad Media", f"{machine_timeline['disponibilidad'].mean():.1%}")
        col_mach2.metric("Scrap Medio", f"{machine_timeline['scrap_rate'].mean():.1%}")
        col_mach3.metric("UPH Medio", f"{machine_timeline['uph_real'].mean():.1f}")
        col_mach4.metric("Estado Actual", machine_timeline.iloc[-1]["estado_predicho"])
    else:
        st.info("No hay datos suficientes para esta maquina.")

    # Feature importance
    with st.expander("Ver importancia de features"):
        feature_importance = pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        fig_importance = px.bar(
            feature_importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Importancia de Features"
        )
        st.plotly_chart(fig_importance, width='stretch')

    # Probabilidades por clase
    with st.expander("Ver probabilidades de clasificacion"):
        st.markdown("**Primeros 20 registros con probabilidades por clase:**")

        prob_df = pd.DataFrame(probabilities, columns=classes)
        prob_display = pd.concat([
            machine_metrics[["machine_name", "periodo", "estado_predicho"]].reset_index(drop=True),
            prob_df
        ], axis=1).head(20)

        st.dataframe(
            prob_display.style.format({
                **{clase: "{:.1%}" for clase in classes}
            }),
            width='stretch',
            hide_index=True
        )

    # Interpretacion
    st.markdown("### Interpretacion y Recomendaciones")

    critical_machines = machine_latest[machine_latest["estado_predicho"] == "CRITICA"]["machine_name"].tolist()
    attention_machines = machine_latest[machine_latest["estado_predicho"] == "REQUIERE_ATENCION"]["machine_name"].tolist()

    if critical_machines:
        st.error(
            f"**ATENCION URGENTE:** {len(critical_machines)} maquinas en estado CRITICO:\n\n"
            f"{', '.join(critical_machines)}\n\n"
            f"Requieren intervencion inmediata de mantenimiento."
        )

    if attention_machines:
        st.warning(
            f"**ATENCION:** {len(attention_machines)} maquinas requieren atencion:\n\n"
            f"{', '.join(attention_machines[:5])}{'...' if len(attention_machines) > 5 else ''}\n\n"
            f"Programar revision preventiva."
        )

    st.info(
        "**Como usar este modelo:**\n\n"
        "- Monitorear maquinas en estado CRITICO diariamente.\n"
        "- Programar mantenimiento preventivo para REQUIERE_ATENCION.\n"
        "- Usar maquinas EXCELENTES como referencia para mejores practicas.\n"
        "- Analizar timeline de maquinas para detectar degradacion progresiva."
    )
