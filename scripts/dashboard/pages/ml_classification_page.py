"""
Pagina del dashboard para el modelo de Clasificacion ML.
Permite usar modelo pre-entrenado Y entrenar modelos interactivamente.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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
        return None, None, None, None


def classify_estado(row, umbrales):
    """Clasifica el estado de una maquina basandose en umbrales."""
    disp = row["disponibilidad"]
    scrap = row["scrap_rate"]
    uph = row["uph_real"]

    if disp >= umbrales["excelente"]["disp"] and scrap <= umbrales["excelente"]["scrap"] and uph >= umbrales["excelente"]["uph"]:
        return "EXCELENTE"
    elif disp >= umbrales["buena"]["disp"] and scrap <= umbrales["buena"]["scrap"] and uph >= umbrales["buena"]["uph"]:
        return "BUENA"
    elif disp < umbrales["critica"]["disp"] or scrap > umbrales["critica"]["scrap"]:
        return "CRITICA"
    else:
        return "REQUIERE_ATENCION"


def prepare_features(df):
    """Prepara features para el modelo."""
    df = df.copy()
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

    return machine_metrics


def train_interactive_model(X_train, y_train, n_estimators, max_depth, min_samples_split):
    """Entrena un modelo con hiperparametros especificados."""
    model = RandomForestClassifier(
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


def evaluate_classification_model(model, scaler, X, y):
    """Evalua el modelo de clasificacion."""
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    accuracy = accuracy_score(y, predictions)
    conf_matrix = confusion_matrix(y, predictions)
    report = classification_report(y, predictions, output_dict=True)

    return predictions, probabilities, accuracy, conf_matrix, report


def page_ml_classification(filtered: dict, ciclos: pd.DataFrame) -> None:
    st.title("Clasificacion ML - Estado de Maquinas")

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

    machine_metrics = prepare_features(prod)

    if len(machine_metrics) < 10:
        st.warning("No hay suficientes datos para entrenar (mínimo 10 registros).")
        return

    # Color map para estados
    color_map = {
        "EXCELENTE": "#22c55e",
        "BUENA": "#3b82f6",
        "REQUIERE_ATENCION": "#f59e0b",
        "CRITICA": "#ef4444"
    }

    # ========== MODO PRE-ENTRENADO ==========
    if modo == "📊 Modelo Pre-entrenado":
        model, scaler, features, classes = load_classification_model()

        if model is None:
            st.error("Modelo pre-entrenado no encontrado. Ejecuta: python models/classification/train.py")
            return

        st.markdown("""
        Usando el **modelo pre-entrenado** con Random Forest Classifier.

        **Clases:**
        - **EXCELENTE**: Disp ≥ 85%, Scrap ≤ 2%, UPH ≥ 100
        - **BUENA**: Disp ≥ 70%, Scrap ≤ 5%, UPH ≥ 60
        - **REQUIERE_ATENCION**: Por debajo de objetivos
        - **CRITICA**: Disp < 50% o Scrap > 10%
        """)

        col1, col2, col3 = st.columns(3)
        col1.metric("Features", len(features))
        col2.metric("Clases", len(classes))
        col3.metric("Algoritmo", "Random Forest")

        # Hacer predicciones
        X = machine_metrics[features].values
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        machine_metrics["estado_predicho"] = predictions
        machine_metrics["probabilidad_max"] = probabilities.max(axis=1)

    # ========== MODO INTERACTIVO ==========
    else:
        st.markdown("""
        **Entrena tu propio modelo** de clasificación eligiendo features, hiperparámetros y umbrales de las clases.
        """)

        st.markdown("---")
        st.markdown("### 1️⃣ Configuración del Modelo")

        col_config1, col_config2 = st.columns([2, 1])

        with col_config1:
            st.markdown("#### Selección de Features")

            all_features = ["disponibilidad", "scrap_rate", "uph_real", "dur_prod", "dur_prep",
                           "dur_inci", "prep_ratio", "inci_ratio", "n_operaciones"]

            selected_features = st.multiselect(
                "Selecciona features:",
                all_features,
                default=["disponibilidad", "scrap_rate", "uph_real", "dur_prod", "prep_ratio", "inci_ratio"],
                key="features_classification"
            )

            if len(selected_features) == 0:
                st.warning("⚠️ Selecciona al menos 1 feature")
                return

            st.success(f"✅ {len(selected_features)} features seleccionadas")

            st.markdown("#### Umbrales de Clasificación")
            st.markdown("Define los umbrales para cada clase:")

            col_umb1, col_umb2 = st.columns(2)

            with col_umb1:
                st.markdown("**EXCELENTE:**")
                exc_disp = st.slider("Disp mín", 70, 100, 85, key="exc_disp") / 100
                exc_scrap = st.slider("Scrap máx", 0, 10, 2, key="exc_scrap") / 100
                exc_uph = st.slider("UPH mín", 50, 200, 100, key="exc_uph")

            with col_umb2:
                st.markdown("**BUENA:**")
                bue_disp = st.slider("Disp mín", 50, 90, 70, key="bue_disp") / 100
                bue_scrap = st.slider("Scrap máx", 0, 15, 5, key="bue_scrap") / 100
                bue_uph = st.slider("UPH mín", 30, 150, 60, key="bue_uph")

            st.markdown("**CRÍTICA:**")
            col_cri1, col_cri2 = st.columns(2)
            cri_disp = col_cri1.slider("Disp < ", 20, 70, 50, key="cri_disp") / 100
            cri_scrap = col_cri2.slider("Scrap >", 5, 20, 10, key="cri_scrap") / 100

            umbrales = {
                "excelente": {"disp": exc_disp, "scrap": exc_scrap, "uph": exc_uph},
                "buena": {"disp": bue_disp, "scrap": bue_scrap, "uph": bue_uph},
                "critica": {"disp": cri_disp, "scrap": cri_scrap}
            }

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
                help="Menor = menos overfitting"
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
                # Generar labels basados en umbrales
                machine_metrics["estado_real"] = machine_metrics.apply(lambda row: classify_estado(row, umbrales), axis=1)

                # Preparar datos
                X = machine_metrics[selected_features].fillna(0)
                y = machine_metrics["estado_real"].values

                # Split train/test
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )

                # Entrenar
                model, scaler = train_interactive_model(
                    X_train, y_train, n_estimators, max_depth, min_samples_split
                )

                # Evaluar
                train_pred, train_prob, train_acc, train_cm, train_report = evaluate_classification_model(
                    model, scaler, X_train, y_train
                )
                test_pred, test_prob, test_acc, test_cm, test_report = evaluate_classification_model(
                    model, scaler, X_test, y_test
                )

                # Guardar en session_state
                st.session_state['trained_clf_model'] = model
                st.session_state['trained_clf_scaler'] = scaler
                st.session_state['trained_clf_features'] = selected_features
                st.session_state['train_clf_metrics'] = (train_acc, train_cm, train_report)
                st.session_state['test_clf_metrics'] = (test_acc, test_cm, test_report)
                st.session_state['X_test_clf'] = X_test
                st.session_state['y_test_clf'] = y_test
                st.session_state['test_pred_clf'] = test_pred
                st.session_state['test_prob_clf'] = test_prob
                st.session_state['classes_clf'] = sorted(machine_metrics["estado_real"].unique())

                st.success("✅ Modelo entrenado exitosamente!")
                st.rerun()

        # Mostrar resultados si hay modelo entrenado
        if 'trained_clf_model' in st.session_state:
            model = st.session_state['trained_clf_model']
            scaler = st.session_state['trained_clf_scaler']
            selected_features = st.session_state['trained_clf_features']
            train_acc, train_cm, train_report = st.session_state['train_clf_metrics']
            test_acc, test_cm, test_report = st.session_state['test_clf_metrics']
            X_test = st.session_state['X_test_clf']
            y_test = st.session_state['y_test_clf']
            test_pred = st.session_state['test_pred_clf']
            test_prob = st.session_state['test_prob_clf']
            classes = st.session_state['classes_clf']

            st.markdown("### 2️⃣ Resultados del Entrenamiento")

            # Métricas de accuracy
            col_train, col_test = st.columns(2)

            with col_train:
                st.markdown("#### 📘 Conjunto de Entrenamiento")
                st.metric("Accuracy", f"{train_acc:.1%}")

            with col_test:
                st.markdown("#### 📗 Conjunto de Test")
                st.metric("Accuracy", f"{test_acc:.1%}")

            # Interpretación
            if abs(train_acc - test_acc) > 0.1:
                st.warning("⚠️ Diferencia grande entre train y test. Posible overfitting.")
            elif test_acc > 0.85:
                st.success("✅ Excelente rendimiento del modelo!")
            elif test_acc > 0.70:
                st.info("ℹ️ Rendimiento aceptable. Prueba ajustar hiperparámetros.")
            else:
                st.error("⚠️ Bajo rendimiento. Prueba seleccionar más features o ajustar umbrales.")

            st.markdown("---")
            st.markdown("### 3️⃣ Matriz de Confusión")

            col_cm1, col_cm2 = st.columns(2)

            with col_cm1:
                st.markdown("#### Conjunto de Test")
                fig_cm = px.imshow(
                    test_cm,
                    labels=dict(x="Predicho", y="Real", color="Count"),
                    x=classes,
                    y=classes,
                    text_auto=True,
                    title="Matriz de Confusión"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_cm2:
                st.markdown("#### Métricas por Clase")
                metrics_df = pd.DataFrame({
                    "Clase": classes,
                    "Precision": [test_report[c]["precision"] for c in classes],
                    "Recall": [test_report[c]["recall"] for c in classes],
                    "F1-Score": [test_report[c]["f1-score"] for c in classes],
                })
                st.dataframe(
                    metrics_df.style.format({
                        "Precision": "{:.1%}",
                        "Recall": "{:.1%}",
                        "F1-Score": "{:.1%}",
                    }),
                    hide_index=True,
                    use_container_width=True
                )

            st.markdown("---")
            st.markdown("### 4️⃣ Feature Importance")

            feature_importance = pd.DataFrame({
                "feature": selected_features,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)

            fig_importance = px.bar(
                feature_importance,
                x="importance",
                y="feature",
                orientation="h",
                title="Importancia de Features"
            )
            st.plotly_chart(fig_importance, use_container_width=True)

            # Comparación con pre-entrenado
            st.markdown("---")
            st.markdown("### 5️⃣ Comparación con Modelo Pre-entrenado")

            pretrained_model, pretrained_scaler, pretrained_features, pretrained_classes = load_classification_model()

            if pretrained_model is not None:
                # Hacer predicciones con modelo pre-entrenado en mismo test set
                X_test_pretrained = machine_metrics.loc[X_test.index][pretrained_features].fillna(0)

                pretrained_pred, pretrained_prob, pretrained_acc, pretrained_cm, pretrained_report = evaluate_classification_model(
                    pretrained_model, pretrained_scaler, X_test_pretrained, y_test
                )

                col_comp1, col_comp2 = st.columns(2)

                with col_comp1:
                    delta_acc = test_acc - pretrained_acc
                    st.metric(
                        "Accuracy",
                        f"{test_acc:.1%}",
                        delta=f"{delta_acc:.1%}",
                        help=f"Pre-entrenado: {pretrained_acc:.1%}"
                    )

                with col_comp2:
                    if test_acc > pretrained_acc:
                        st.success("🎉 ¡Tu modelo interactivo supera al pre-entrenado!")
                    else:
                        st.info("El modelo pre-entrenado sigue siendo mejor. Prueba ajustar hiperparámetros.")

            # Opción de guardar
            st.markdown("---")
            st.markdown("### 6️⃣ Guardar Modelo")

            if st.button("💾 Guardar este modelo", help="Guarda el modelo entrenado para uso futuro"):
                model_path = Path(__file__).parent.parent.parent.parent / "models" / "classification" / "trained_model"
                model_path.mkdir(parents=True, exist_ok=True)

                with open(model_path / "interactive_classifier.pkl", "wb") as f:
                    pickle.dump(model, f)
                with open(model_path / "interactive_scaler.pkl", "wb") as f:
                    pickle.dump(scaler, f)
                with open(model_path / "interactive_features.txt", "w") as f:
                    f.write("\n".join(selected_features))
                with open(model_path / "interactive_classes.txt", "w") as f:
                    f.write("\n".join(classes))

                st.success(f"✅ Modelo guardado en: models/classification/trained_model/interactive_classifier.pkl")

            # Hacer predicciones en todos los datos
            X_all = machine_metrics[selected_features].fillna(0)
            X_all_scaled = scaler.transform(X_all)
            all_predictions = model.predict(X_all_scaled)
            all_probabilities = model.predict_proba(X_all_scaled)

            machine_metrics["estado_predicho"] = all_predictions
            machine_metrics["probabilidad_max"] = all_probabilities.max(axis=1)

    # ========== VISUALIZACIONES COMUNES (para ambos modos) ==========
    if modo == "📊 Modelo Pre-entrenado" or 'trained_clf_model' in st.session_state:
        st.markdown("---")
        st.markdown("### Análisis de Clasificación")

        # Distribución de estados
        estado_counts = machine_metrics["estado_predicho"].value_counts().reset_index()
        estado_counts.columns = ["estado", "count"]

        col_dist1, col_dist2 = st.columns(2)

        col_dist1.dataframe(estado_counts, hide_index=True, use_container_width=True)

        fig_pie = px.pie(
            estado_counts,
            values="count",
            names="estado",
            title="Distribución de Estados",
            color="estado",
            color_discrete_map=color_map
        )
        col_dist2.plotly_chart(fig_pie, use_container_width=True)

        # Scatter plot
        st.markdown("#### Visualización: Disponibilidad vs Scrap")

        fig_scatter = px.scatter(
            machine_metrics,
            x="disponibilidad",
            y="scrap_rate",
            color="estado_predicho",
            size="probabilidad_max",
            hover_data=["machine_name", "periodo", "uph_real"],
            title="Clasificación de Máquinas",
            labels={
                "disponibilidad": "Disponibilidad",
                "scrap_rate": "Scrap %",
                "estado_predicho": "Estado",
            },
            color_discrete_map=color_map
        )
        fig_scatter.update_xaxes(tickformat=".0%")
        fig_scatter.update_yaxes(tickformat=".0%")
        fig_scatter.update_traces(marker=dict(line=dict(width=1, color="white")))
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Máquinas por estado
        st.markdown("#### Máquinas por Estado")

        machine_latest = machine_metrics.sort_values("periodo", ascending=False).groupby("machine_name").first().reset_index()

        tabs = st.tabs(["EXCELENTE", "BUENA", "REQUIERE_ATENCION", "CRITICA"])

        for i, estado in enumerate(["EXCELENTE", "BUENA", "REQUIERE_ATENCION", "CRITICA"]):
            with tabs[i]:
                machines_in_state = machine_latest[machine_latest["estado_predicho"] == estado]

                if len(machines_in_state) == 0:
                    st.info(f"No hay máquinas en estado {estado}.")
                    continue

                st.markdown(f"**{len(machines_in_state)} máquinas en estado {estado}**")

                st.dataframe(
                    machines_in_state[[
                        "machine_name", "disponibilidad", "scrap_rate", "uph_real", "probabilidad_max"
                    ]].style.format({
                        "disponibilidad": "{:.1%}",
                        "scrap_rate": "{:.1%}",
                        "uph_real": "{:.1f}",
                        "probabilidad_max": "{:.1%}",
                    }),
                    hide_index=True,
                    use_container_width=True
                )
