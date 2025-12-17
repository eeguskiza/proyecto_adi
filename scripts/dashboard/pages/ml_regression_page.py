"""
Pagina del dashboard para el modelo de Regresion ML.
Carga el modelo entrenado y permite predecir scrap rate para operaciones.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

@st.cache_resource
def load_regression_model():
    """Carga el modelo de regresion entrenado."""
    model_path = Path(__file__).parent.parent.parent.parent / "models" / "regression" / "trained_model"

    try:
        with open(model_path / "random_forest_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open(model_path / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open(model_path / "features.txt", "r") as f:
            features = [line.strip() for line in f.readlines()]

        return model, scaler, features
    except FileNotFoundError:
        st.error("Modelo no encontrado. Ejecuta: python models/regression/train.py")
        return None, None, None


def page_ml_regression(filtered: dict, ciclos: pd.DataFrame) -> None:
    st.title("Regresion ML - Prediccion de Scrap")

    st.markdown("""
    Este modelo utiliza **Random Forest Regressor** para predecir la tasa de scrap esperada
    en operaciones de produccion basandose en caracteristicas historicas.

    **Features consideradas**: Duracion, Maquina, Referencia, Hora del dia, Dia de la semana, Estado OEE.
    """)

    # Cargar modelo
    model, scaler, features = load_regression_model()

    if model is None:
        st.warning("No se pudo cargar el modelo. Asegurate de haberlo entrenado primero.")
        return

    # Info del modelo
    col1, col2, col3 = st.columns(3)
    col1.metric("Features", len(features))
    col2.metric("Algoritmo", "Random Forest")
    col3.metric("Target", "Scrap Rate")

    st.markdown("---")

    # Preparar datos actuales
    prod = filtered.get("produccion", pd.DataFrame())

    if prod.empty:
        st.info("Sin datos de produccion en el rango seleccionado.")
        return

    # Preparar features (misma logica que el entrenamiento)
    df = prod.copy()

    df["total_piezas"] = df["piezas_ok"] + df["piezas_scrap"]
    df["scrap_rate_real"] = np.where(
        df["total_piezas"] > 0,
        df["piezas_scrap"] / df["total_piezas"],
        np.nan
    )

    df = df[df["total_piezas"] > 0].copy()
    df = df.dropna(subset=["scrap_rate_real"])

    if len(df) < 1:
        st.warning("No hay suficientes datos para hacer predicciones.")
        return

    # Features temporales
    df["hora_del_dia"] = df["ts_ini"].dt.hour
    df["dia_semana"] = df["ts_ini"].dt.dayofweek

    # Preparar ref_id_str
    df["ref_id_str"] = df["ref_id"].astype(str).str.replace(r'\.0$', '', regex=True)

    # Frecuencia de referencias
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

    # One-hot encoding para maquinas (top 10)
    top_machines = df["machine_name"].value_counts().head(10).index
    df["machine_group"] = df["machine_name"].apply(lambda x: x if x in top_machines else "otras")
    machine_dummies = pd.get_dummies(df["machine_group"], prefix="machine")

    # Combinar features
    features_df = pd.concat([
        df[["duracion_min", "hora_del_dia", "dia_semana", "ref_frequency"]],
        estado_dummies,
        machine_dummies
    ], axis=1)

    # Asegurar que todas las features del modelo esten presentes
    for feature in features:
        if feature not in features_df.columns:
            features_df[feature] = 0

    features_df = features_df[features]
    features_df = features_df.fillna(0)

    st.markdown(f"### Predicciones para {len(features_df):,} operaciones")

    # --- CORRECCIÓN 1: Pasar DataFrame al scaler ---
    # Al pasar el DataFrame con nombres de columnas, sklearn valida los nombres
    # contra los usados en el fit(), eliminando el warning.
    X_scaled = scaler.transform(features_df) 
    
    predictions = model.predict(X_scaled)

    df["scrap_rate_predicho"] = predictions
    df["error_absoluto"] = np.abs(df["scrap_rate_real"] - df["scrap_rate_predicho"])

    # Metricas globales
    st.markdown("#### Metricas de Prediccion")

    mae = df["error_absoluto"].mean()
    rmse = np.sqrt((df["error_absoluto"] ** 2).mean())
    # R2 calculation manual is correct here
    r2 = 1 - (((df["scrap_rate_real"] - df["scrap_rate_predicho"]) ** 2).sum() /
              ((df["scrap_rate_real"] - df["scrap_rate_real"].mean()) ** 2).sum())

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("MAE (Error Medio)", f"{mae:.2%}", help="Error promedio en puntos porcentuales")
    col_m2.metric("RMSE", f"{rmse:.2%}", help="Raiz del error cuadratico medio")
    col_m3.metric("R² Score", f"{r2:.3f}", help="Proporcion de varianza explicada")

    st.markdown("---")

    # Grafico: Real vs Predicho
    st.markdown("#### Real vs Predicho")

    col_chart1, col_chart2 = st.columns(2)

    # Scatter plot
    fig_scatter = px.scatter(
        df.sample(min(1000, len(df))),  # Limitar a 1000 puntos para rendimiento
        x="scrap_rate_real",
        y="scrap_rate_predicho",
        title="Scrap Real vs Scrap Predicho",
        labels={"scrap_rate_real": "Scrap Real", "scrap_rate_predicho": "Scrap Predicho"},
        opacity=0.6
    )
    # Linea diagonal (prediccion perfecta)
    fig_scatter.add_trace(
        go.Scatter(
            x=[0, df["scrap_rate_real"].max()],
            y=[0, df["scrap_rate_real"].max()],
            mode="lines",
            name="Prediccion Perfecta",
            line=dict(color="red", dash="dash")
        )
    )
    fig_scatter.update_xaxes(tickformat=".0%")
    fig_scatter.update_yaxes(tickformat=".0%")
    
    # --- CORRECCIÓN 2: use_container_width=True ---
    col_chart1.plotly_chart(fig_scatter, use_container_width=True)

    # Distribucion de errores
    fig_error = px.histogram(
        df,
        x="error_absoluto",
        title="Distribucion de Errores Absolutos",
        labels={"error_absoluto": "Error Absoluto"},
        nbins=50
    )
    fig_error.update_xaxes(tickformat=".0%")
    col_chart2.plotly_chart(fig_error, use_container_width=True)

    # Top errores por maquina
    st.markdown("#### Errores por Maquina")

    machine_error = df.groupby("machine_name").agg(
        error_medio=("error_absoluto", "mean"),
        n_operaciones=("error_absoluto", "count"),
        scrap_real_medio=("scrap_rate_real", "mean"),
        scrap_predicho_medio=("scrap_rate_predicho", "mean"),
    ).reset_index()

    machine_error = machine_error.sort_values("error_medio", ascending=False).head(10)

    fig_machine_error = px.bar(
        machine_error,
        x="machine_name",
        y="error_medio",
        title="Top 10 Maquinas con Mayor Error de Prediccion",
        labels={"error_medio": "Error Medio", "machine_name": "Maquina"},
        text="error_medio"
    )
    fig_machine_error.update_traces(texttemplate='%{text:.2%}', textposition="outside")
    fig_machine_error.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig_machine_error, use_container_width=True)

    # Scrap por hora del dia
    st.markdown("#### Predicciones por Hora del Dia")

    hourly = df.groupby("hora_del_dia").agg(
        scrap_real=("scrap_rate_real", "mean"),
        scrap_predicho=("scrap_rate_predicho", "mean"),
    ).reset_index()

    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Scatter(
        x=hourly["hora_del_dia"],
        y=hourly["scrap_real"],
        mode="lines+markers",
        name="Scrap Real",
        line=dict(color="#ef4444", width=2)
    ))
    fig_hourly.add_trace(go.Scatter(
        x=hourly["hora_del_dia"],
        y=hourly["scrap_predicho"],
        mode="lines+markers",
        name="Scrap Predicho",
        line=dict(color="#3b82f6", width=2)
    ))
    fig_hourly.update_layout(
        title="Scrap Real vs Predicho por Hora del Dia",
        xaxis_title="Hora del Dia",
        yaxis_title="Scrap Rate",
        yaxis_tickformat=".0%"
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

    # Feature importance
    with st.expander("Ver importancia de features"):
        feature_importance = pd.DataFrame({
            "feature": features,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False).head(15)

        fig_importance = px.bar(
            feature_importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Features Mas Importantes"
        )
        st.plotly_chart(fig_importance, use_container_width=True)

    # Tabla de ejemplos
    st.markdown("#### Ejemplos de Predicciones")

    df_sample = df[[
        "machine_name", "ref_id_str", "duracion_min",
        "scrap_rate_real", "scrap_rate_predicho", "error_absoluto"
    ]].sample(min(20, len(df)))

    st.dataframe(
        df_sample.style.format({
            "duracion_min": "{:.1f}",
            "scrap_rate_real": "{:.2%}",
            "scrap_rate_predicho": "{:.2%}",
            "error_absoluto": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True
    )

    # Interpretacion
    st.markdown("### Interpretacion")
    st.info(
        f"**Rendimiento del modelo:**\n\n"
        f"- El modelo tiene un error promedio de **{mae:.2%}** al predecir scrap.\n"
        f"- Explica **{r2*100:.1f}%** de la varianza en la tasa de scrap.\n"
        f"- Las predicciones pueden usarse para identificar operaciones de alto riesgo.\n\n"
        f"**Recomendaciones:**\n"
        f"- Revisar maquinas con mayor error de prediccion (comportamiento inestable).\n"
        f"- Usar predicciones para planificar inspecciones de calidad mas frecuentes."
    )