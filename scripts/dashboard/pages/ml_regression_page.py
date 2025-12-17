"""
Pagina del dashboard para el modelo de Regresion ML.
Permite usar modelo pre-entrenado Y entrenar modelos interactivamente.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
        return None, None, None


def prepare_features(df):
    """Prepara features para el modelo."""
    df = df.copy()

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

    # One-hot encoding
    estado_dummies = pd.get_dummies(df["estado_oee"], prefix="estado")

    top_machines = df["machine_name"].value_counts().head(10).index
    df["machine_group"] = df["machine_name"].apply(lambda x: x if x in top_machines else "otras")
    machine_dummies = pd.get_dummies(df["machine_group"], prefix="machine")

    # Combinar features
    features_df = pd.concat([
        df[["duracion_min", "hora_del_dia", "dia_semana", "ref_frequency"]],
        estado_dummies,
        machine_dummies
    ], axis=1)

    return df, features_df


def train_interactive_model(X_train, y_train, n_estimators, max_depth, min_samples_split):
    """Entrena un modelo con hiperparametros especificados."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model.fit(X_train_scaled, y_train)

    return model, scaler


def evaluate_model(model, scaler, X, y):
    """Evalua el modelo y retorna metricas."""
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    return predictions, mae, rmse, r2


def page_ml_regression(filtered: dict, ciclos: pd.DataFrame) -> None:
    st.title("Regresion ML - Prediccion de Scrap")

    # Selector de modo
    st.markdown("### Modo de Operacion")
    modo = st.radio(
        "Elige el modo:",
        ["📊 Modelo Pre-entrenado", "🔧 Entrenar Modelo Interactivo"],
        horizontal=True,
        help="Pre-entrenado: usa el modelo ya guardado. Interactivo: entrena tu propio modelo."
    )

    # Preparar datos
    prod = filtered.get("produccion", pd.DataFrame())

    if prod.empty:
        st.info("Sin datos de produccion en el rango seleccionado.")
        return

    df = prod.copy()
    df["total_piezas"] = df["piezas_ok"] + df["piezas_scrap"]
    df["scrap_rate_real"] = np.where(
        df["total_piezas"] > 0,
        df["piezas_scrap"] / df["total_piezas"],
        np.nan
    )

    df = df[df["total_piezas"] > 0].copy()
    df = df.dropna(subset=["scrap_rate_real"])

    if len(df) < 10:
        st.warning("No hay suficientes datos para entrenar (mínimo 10 registros).")
        return

    df, features_df_all = prepare_features(df)

    # ========== MODO PRE-ENTRENADO ==========
    if modo == "📊 Modelo Pre-entrenado":
        model, scaler, features = load_regression_model()

        if model is None:
            st.error("Modelo pre-entrenado no encontrado. Ejecuta: python models/regression/train.py")
            return

        st.markdown("""
        Usando el **modelo pre-entrenado** con Random Forest.

        **Features**: Duración, Máquina, Referencia, Hora del día, Día de la semana, Estado OEE.
        """)

        col1, col2, col3 = st.columns(3)
        col1.metric("Features", len(features))
        col2.metric("Algoritmo", "Random Forest")
        col3.metric("Target", "Scrap Rate")

        # Asegurar features del modelo
        features_df = features_df_all.copy()
        for feature in features:
            if feature not in features_df.columns:
                features_df[feature] = 0
        features_df = features_df[features].fillna(0)

        # Predicciones
        X_scaled = scaler.transform(features_df)
        predictions = model.predict(X_scaled)

        df["scrap_rate_predicho"] = predictions
        df["error_absoluto"] = np.abs(df["scrap_rate_real"] - df["scrap_rate_predicho"])

        # Métricas
        mae = df["error_absoluto"].mean()
        rmse = np.sqrt((df["error_absoluto"] ** 2).mean())
        r2 = 1 - (((df["scrap_rate_real"] - df["scrap_rate_predicho"]) ** 2).sum() /
                  ((df["scrap_rate_real"] - df["scrap_rate_real"].mean()) ** 2).sum())

        st.markdown("---")
        st.markdown("### Métricas del Modelo Pre-entrenado")

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("MAE (Error Medio)", f"{mae:.2%}")
        col_m2.metric("RMSE", f"{rmse:.2%}")
        col_m3.metric("R² Score", f"{r2:.3f}")

    # ========== MODO INTERACTIVO ==========
    else:
        st.markdown("""
        **Entrena tu propio modelo** de Random Forest eligiendo features e hiperparámetros.
        """)

        st.markdown("---")
        st.markdown("### 1️⃣ Configuración del Modelo")

        col_config1, col_config2 = st.columns([2, 1])

        with col_config1:
            st.markdown("#### Selección de Features")

            # Categorías de features
            all_features = list(features_df_all.columns)

            features_basicas = ["duracion_min", "hora_del_dia", "dia_semana", "ref_frequency"]
            features_estado = [f for f in all_features if f.startswith("estado_")]
            features_maquina = [f for f in all_features if f.startswith("machine_")]

            st.markdown("**Features Básicas:**")
            use_basicas = st.multiselect(
                "Selecciona features básicas:",
                features_basicas,
                default=features_basicas,
                key="features_basicas"
            )

            st.markdown("**Features de Estado:**")
            use_estado_all = st.checkbox("Incluir todos los estados OEE", value=True, key="estado_all")
            if not use_estado_all:
                use_estado = st.multiselect(
                    "Selecciona estados:",
                    features_estado,
                    default=[],
                    key="features_estado"
                )
            else:
                use_estado = features_estado

            st.markdown("**Features de Máquina:**")
            use_maquina_all = st.checkbox("Incluir todas las máquinas", value=True, key="maquina_all")
            if not use_maquina_all:
                use_maquina = st.multiselect(
                    "Selecciona máquinas:",
                    features_maquina,
                    default=[],
                    key="features_maquina"
                )
            else:
                use_maquina = features_maquina

            selected_features = use_basicas + use_estado + use_maquina

            if len(selected_features) == 0:
                st.warning("⚠️ Selecciona al menos 1 feature")
                return

            st.success(f"✅ {len(selected_features)} features seleccionadas")

        with col_config2:
            st.markdown("#### Hiperparámetros")

            n_estimators = st.slider(
                "Número de árboles",
                min_value=10,
                max_value=300,
                value=100,
                step=10,
                help="Más árboles = mejor precisión pero más lento"
            )

            max_depth = st.slider(
                "Profundidad máxima",
                min_value=3,
                max_value=30,
                value=15,
                help="None = sin límite. Menor = menos overfitting"
            )

            min_samples_split = st.slider(
                "Mínimo para dividir",
                min_value=2,
                max_value=20,
                value=5,
                help="Mínimo de muestras para dividir un nodo"
            )

            test_size = st.slider(
                "% Test",
                min_value=10,
                max_value=40,
                value=20,
                help="Porcentaje de datos para test"
            ) / 100

        st.markdown("---")

        # Botón entrenar
        if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
            with st.spinner("Entrenando modelo... Esto puede tardar unos segundos."):
                # Preparar datos
                features_df = features_df_all[selected_features].fillna(0)
                y = df["scrap_rate_real"].values

                # Split train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    features_df, y, test_size=test_size, random_state=42
                )

                # Entrenar
                model, scaler = train_interactive_model(
                    X_train, y_train, n_estimators, max_depth, min_samples_split
                )

                # Evaluar
                train_pred, train_mae, train_rmse, train_r2 = evaluate_model(model, scaler, X_train, y_train)
                test_pred, test_mae, test_rmse, test_r2 = evaluate_model(model, scaler, X_test, y_test)

                # Guardar en session_state
                st.session_state['trained_model'] = model
                st.session_state['trained_scaler'] = scaler
                st.session_state['trained_features'] = selected_features
                st.session_state['train_metrics'] = (train_mae, train_rmse, train_r2)
                st.session_state['test_metrics'] = (test_mae, test_rmse, test_r2)
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['test_pred'] = test_pred

                st.success("✅ Modelo entrenado exitosamente!")
                st.rerun()

        # Mostrar resultados si hay modelo entrenado
        if 'trained_model' in st.session_state:
            model = st.session_state['trained_model']
            scaler = st.session_state['trained_scaler']
            selected_features = st.session_state['trained_features']
            train_mae, train_rmse, train_r2 = st.session_state['train_metrics']
            test_mae, test_rmse, test_r2 = st.session_state['test_metrics']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']
            test_pred = st.session_state['test_pred']

            st.markdown("### 2️⃣ Resultados del Entrenamiento")

            # Comparar train vs test
            col_train, col_test = st.columns(2)

            with col_train:
                st.markdown("#### 📘 Conjunto de Entrenamiento")
                st.metric("MAE", f"{train_mae:.2%}")
                st.metric("RMSE", f"{train_rmse:.2%}")
                st.metric("R²", f"{train_r2:.3f}")

            with col_test:
                st.markdown("#### 📗 Conjunto de Test")
                st.metric("MAE", f"{test_mae:.2%}")
                st.metric("RMSE", f"{test_rmse:.2%}")
                st.metric("R²", f"{test_r2:.3f}")

            # Interpretación
            if abs(train_r2 - test_r2) > 0.1:
                st.warning("⚠️ Diferencia grande entre train y test. Posible overfitting. Reduce max_depth o aumenta min_samples_split.")
            elif test_r2 > 0.7:
                st.success("✅ Excelente rendimiento del modelo!")
            elif test_r2 > 0.5:
                st.info("ℹ️ Rendimiento aceptable. Prueba ajustar hiperparámetros.")
            else:
                st.error("⚠️ Bajo rendimiento. Prueba seleccionar más features o ajustar hiperparámetros.")

            st.markdown("---")
            st.markdown("### 3️⃣ Visualización de Resultados")

            # Gráfico Real vs Predicho
            test_df = pd.DataFrame({
                'real': y_test,
                'predicho': test_pred
            })

            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                fig_scatter = px.scatter(
                    test_df,
                    x="real",
                    y="predicho",
                    title="Test Set: Real vs Predicho",
                    labels={"real": "Scrap Real", "predicho": "Scrap Predicho"},
                    opacity=0.6
                )
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[0, test_df["real"].max()],
                        y=[0, test_df["real"].max()],
                        mode="lines",
                        name="Predicción Perfecta",
                        line=dict(color="red", dash="dash")
                    )
                )
                fig_scatter.update_xaxes(tickformat=".0%")
                fig_scatter.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col_chart2:
                errors = np.abs(test_df['real'] - test_df['predicho'])
                fig_error = px.histogram(
                    x=errors,
                    title="Distribución de Errores (Test)",
                    labels={"x": "Error Absoluto"},
                    nbins=30
                )
                fig_error.update_xaxes(tickformat=".0%")
                st.plotly_chart(fig_error, use_container_width=True)

            # Feature importance
            st.markdown("### 4️⃣ Importancia de Features")

            feature_importance = pd.DataFrame({
                "feature": selected_features,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False).head(15)

            fig_importance = px.bar(
                feature_importance,
                x="importance",
                y="feature",
                orientation="h",
                title="Top 15 Features Más Importantes"
            )
            st.plotly_chart(fig_importance, use_container_width=True)

            # Comparación con modelo pre-entrenado
            st.markdown("---")
            st.markdown("### 5️⃣ Comparación con Modelo Pre-entrenado")

            pretrained_model, pretrained_scaler, pretrained_features = load_regression_model()

            if pretrained_model is not None:
                # Hacer predicciones con modelo pre-entrenado en mismo test set
                features_df_pretrained = features_df_all.copy()
                for feature in pretrained_features:
                    if feature not in features_df_pretrained.columns:
                        features_df_pretrained[feature] = 0
                features_df_pretrained = features_df_pretrained[pretrained_features].fillna(0)

                # Obtener solo las filas de test
                test_indices = X_test.index
                X_test_pretrained = features_df_pretrained.loc[test_indices]

                pretrained_pred, pretrained_mae, pretrained_rmse, pretrained_r2 = evaluate_model(
                    pretrained_model, pretrained_scaler, X_test_pretrained, y_test
                )

                col_comp1, col_comp2, col_comp3 = st.columns(3)

                with col_comp1:
                    delta_mae = pretrained_mae - test_mae
                    col_comp1.metric(
                        "MAE",
                        f"{test_mae:.2%}",
                        delta=f"{-delta_mae:.2%}",
                        delta_color="inverse",
                        help=f"Pre-entrenado: {pretrained_mae:.2%}"
                    )

                with col_comp2:
                    delta_rmse = pretrained_rmse - test_rmse
                    col_comp2.metric(
                        "RMSE",
                        f"{test_rmse:.2%}",
                        delta=f"{-delta_rmse:.2%}",
                        delta_color="inverse",
                        help=f"Pre-entrenado: {pretrained_rmse:.2%}"
                    )

                with col_comp3:
                    delta_r2 = test_r2 - pretrained_r2
                    col_comp3.metric(
                        "R²",
                        f"{test_r2:.3f}",
                        delta=f"{delta_r2:.3f}",
                        help=f"Pre-entrenado: {pretrained_r2:.3f}"
                    )

                if test_r2 > pretrained_r2:
                    st.success("🎉 ¡Tu modelo interactivo supera al pre-entrenado!")
                else:
                    st.info("El modelo pre-entrenado sigue siendo mejor. Prueba ajustar hiperparámetros.")

            # Opción de guardar
            st.markdown("---")
            st.markdown("### 6️⃣ Guardar Modelo")

            if st.button("💾 Guardar este modelo", help="Guarda el modelo entrenado para uso futuro"):
                model_path = Path(__file__).parent.parent.parent.parent / "models" / "regression" / "trained_model"
                model_path.mkdir(parents=True, exist_ok=True)

                with open(model_path / "interactive_model.pkl", "wb") as f:
                    pickle.dump(model, f)
                with open(model_path / "interactive_scaler.pkl", "wb") as f:
                    pickle.dump(scaler, f)
                with open(model_path / "interactive_features.txt", "w") as f:
                    f.write("\n".join(selected_features))

                st.success(f"✅ Modelo guardado en: models/regression/trained_model/interactive_model.pkl")

    # ========== VISUALIZACIONES COMUNES (para ambos modos) ==========
    if modo == "📊 Modelo Pre-entrenado" or 'trained_model' in st.session_state:
        st.markdown("---")
        st.markdown("### Análisis de Predicciones")

        # Usar el modelo correspondiente para predicciones completas
        if modo == "📊 Modelo Pre-entrenado":
            features_df = features_df_all.copy()
            for feature in features:
                if feature not in features_df.columns:
                    features_df[feature] = 0
            features_df = features_df[features].fillna(0)
            X_scaled = scaler.transform(features_df)
            predictions = model.predict(X_scaled)
        else:
            features_df = features_df_all[selected_features].fillna(0)
            X_scaled = scaler.transform(features_df)
            predictions = model.predict(X_scaled)

        df["scrap_rate_predicho"] = predictions
        df["error_absoluto"] = np.abs(df["scrap_rate_real"] - df["scrap_rate_predicho"])

        # Scrap por hora del día
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
            title="Predicciones por Hora del Día",
            xaxis_title="Hora",
            yaxis_title="Scrap Rate",
            yaxis_tickformat=".0%"
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

        # Errores por máquina
        machine_error = df.groupby("machine_name").agg(
            error_medio=("error_absoluto", "mean"),
            n_operaciones=("error_absoluto", "count"),
        ).reset_index()
        machine_error = machine_error.sort_values("error_medio", ascending=False).head(10)

        fig_machine_error = px.bar(
            machine_error,
            x="machine_name",
            y="error_medio",
            title="Top 10 Máquinas con Mayor Error",
            labels={"error_medio": "Error Medio", "machine_name": "Máquina"},
            text="error_medio"
        )
        fig_machine_error.update_traces(texttemplate='%{text:.2%}', textposition="outside")
        fig_machine_error.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_machine_error, use_container_width=True)
