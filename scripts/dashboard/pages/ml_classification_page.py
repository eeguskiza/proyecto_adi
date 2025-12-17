"""
Pagina del dashboard para el modelo de Clasificacion ML.

Este modulo implementa una interfaz interactiva en Streamlit para:
- Usar modelos pre-entrenados de clasificacion
- Entrenar nuevos modelos de forma interactiva
- Evaluar y comparar modelos
- Empaquetar modelos con BentoML para despliegue

Autor: Sistema de ML Industrial
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def list_available_classification_models():
    """
    Lista todos los modelos de clasificacion disponibles.

    Returns:
        dict: Diccionario con nombre_modelo -> ruta_archivos
    """
    model_dir = Path(__file__).parent.parent.parent.parent / "models" / "classification" / "trained_model"

    available_models = {}

    # Buscar modelo principal base (pre-entrenado)
    if model_dir.exists() and (model_dir / "random_forest_classifier.pkl").exists():
        available_models["Modelo Base"] = {
            "model": model_dir / "random_forest_classifier.pkl",
            "scaler": model_dir / "scaler.pkl",
            "features": model_dir / "features.txt",
            "classes": model_dir / "classes.txt"
        }

    # Buscar modelos guardados en BentoML
    try:
        import bentoml
        bento_models = bentoml.models.list()
        for bm in bento_models:
            if "classifier" in bm.tag.name.lower() and "scaler" not in bm.tag.name.lower():
                available_models[f"{bm.tag.name} (v{bm.tag.version})"] = {
                    "bentoml_tag": str(bm.tag),
                    "type": "bentoml"
                }
    except:
        pass  # BentoML no disponible o sin modelos

    return available_models


@st.cache_resource
def load_classification_model(model_name=None):
    """
    Carga el modelo de clasificacion especificado desde disco.

    Args:
        model_name: Nombre del modelo a cargar. Si es None, carga el principal.

    Returns:
        tuple: (modelo, scaler, lista_features, lista_clases) o (None, None, None, None) si no existe
    """
    available_models = list_available_classification_models()

    if not available_models:
        return None, None, None, None

    # Si no se especifica modelo, usar el primero disponible
    if model_name is None or model_name not in available_models:
        model_name = list(available_models.keys())[0]

    model_info = available_models[model_name]

    try:
        # Cargar desde BentoML si es ese tipo
        if model_info.get("type") == "bentoml":
            import bentoml
            bento_model = bentoml.models.get(model_info["bentoml_tag"])
            metadata = bento_model.info.metadata
            features = metadata.get("features", [])
            classes = metadata.get("classes", [])

            model = bentoml.sklearn.load_model(model_info["bentoml_tag"])

            scaler_tag = metadata.get("scaler_tag")
            if scaler_tag:
                scaler = bentoml.sklearn.load_model(scaler_tag)
            else:
                tag_parts = model_info["bentoml_tag"].split(":")
                model_base_name = tag_parts[0]
                scaler_name = f"{model_base_name}_scaler"
                scaler = bentoml.sklearn.load_model(scaler_name)

            return model, scaler, features, classes

        # Cargar desde archivos pickle
        with open(model_info["model"], "rb") as f:
            model = pickle.load(f)

        with open(model_info["scaler"], "rb") as f:
            scaler = pickle.load(f)

        with open(model_info["features"], "r") as f:
            features = [line.strip() for line in f.readlines()]

        with open(model_info["classes"], "r") as f:
            classes = [line.strip() for line in f.readlines()]

        return model, scaler, features, classes
    except Exception as e:
        st.error(f"Error al cargar modelo {model_name}: {str(e)}")
        return None, None, None, None


def classify_estado(row, umbrales):
    """
    Clasifica el estado de una maquina basandose en umbrales definidos.

    Args:
        row: Fila de DataFrame con metricas de la maquina
        umbrales: Diccionario con umbrales para cada clase

    Returns:
        str: Estado clasificado (EXCELENTE, BUENA, CRITICA, REQUIERE_ATENCION)
    """
    disp = row["disponibilidad"]
    scrap = row["scrap_rate"]
    uph = row["uph_real"]

    # Clasificacion jerarquica basada en umbrales
    if disp >= umbrales["excelente"]["disp"] and scrap <= umbrales["excelente"]["scrap"] and uph >= umbrales["excelente"]["uph"]:
        return "EXCELENTE"
    elif disp >= umbrales["buena"]["disp"] and scrap <= umbrales["buena"]["scrap"] and uph >= umbrales["buena"]["uph"]:
        return "BUENA"
    elif disp < umbrales["critica"]["disp"] or scrap > umbrales["critica"]["scrap"]:
        return "CRITICA"
    else:
        return "REQUIERE_ATENCION"


def prepare_features(df):
    """
    Prepara las features necesarias para el modelo de clasificacion.

    Calcula metricas agregadas por maquina y periodo, incluyendo:
    - Duraciones de produccion, preparacion e incidencias
    - Disponibilidad, scrap rate, UPH
    - Ratios de preparacion e incidencias

    Args:
        df: DataFrame con datos de produccion

    Returns:
        DataFrame: Metricas agregadas por maquina y periodo
    """
    df = df.copy()

    # Mapear eventos a estados OEE estandarizados
    df["estado_oee"] = df["evento"].str.lower().map({
        "produccion": "produccion",
        "producción": "produccion",
        "preparacion": "preparacion",
        "preparación": "preparacion",
    })
    df["estado_oee"] = df["estado_oee"].fillna("incidencia")

    # Calcular total de piezas y obtener registro final por orden
    df["total_piezas"] = df["piezas_ok"] + df["piezas_scrap"]
    prod_valid = df[df["total_piezas"] > 0].sort_values("ts_fin")
    of_final = prod_valid.groupby("work_order_id").tail(1)

    # Crear periodos semanales para agregacion
    df["fecha"] = df["ts_ini"].dt.date
    df["semana"] = df["ts_ini"].dt.isocalendar().week
    df["anio"] = df["ts_ini"].dt.year
    df["periodo"] = df["anio"].astype(str) + "-W" + df["semana"].astype(str)

    # Agregar duraciones por maquina y periodo
    machine_period_agg = df.groupby(["machine_name", "periodo"]).agg(
        dur_prod=("duracion_min", lambda x: x[df.loc[x.index, "estado_oee"] == "produccion"].sum()),
        dur_prep=("duracion_min", lambda x: x[df.loc[x.index, "estado_oee"] == "preparacion"].sum()),
        dur_inci=("duracion_min", lambda x: x[df.loc[x.index, "estado_oee"] == "incidencia"].sum()),
        n_operaciones=("work_order_id", "count"),
    ).reset_index()

    # Agregar piezas producidas
    of_final = of_final.copy()
    of_final["fecha"] = of_final["ts_ini"].dt.date
    of_final["semana"] = of_final["ts_ini"].dt.isocalendar().week
    of_final["anio"] = of_final["ts_ini"].dt.year
    of_final["periodo"] = of_final["anio"].astype(str) + "-W" + of_final["semana"].astype(str)

    piezas_agg = of_final.groupby(["machine_name", "periodo"]).agg(
        piezas_ok=("piezas_ok", "sum"),
        piezas_scrap=("piezas_scrap", "sum"),
    ).reset_index()

    # Combinar metricas
    machine_metrics = machine_period_agg.merge(piezas_agg, on=["machine_name", "periodo"], how="left").fillna(0)

    # Calcular metricas derivadas
    machine_metrics["total_dur"] = machine_metrics["dur_prod"] + machine_metrics["dur_prep"] + machine_metrics["dur_inci"]

    # Disponibilidad: tiempo productivo / tiempo total
    machine_metrics["disponibilidad"] = np.where(
        machine_metrics["total_dur"] > 0,
        machine_metrics["dur_prod"] / machine_metrics["total_dur"],
        np.nan
    )

    # Scrap rate: piezas defectuosas / piezas totales
    machine_metrics["total_piezas"] = machine_metrics["piezas_ok"] + machine_metrics["piezas_scrap"]
    machine_metrics["scrap_rate"] = np.where(
        machine_metrics["total_piezas"] > 0,
        machine_metrics["piezas_scrap"] / machine_metrics["total_piezas"],
        np.nan
    )

    # UPH: unidades por hora
    machine_metrics["uph_real"] = np.where(
        machine_metrics["dur_prod"] > 0,
        60 * machine_metrics["piezas_ok"] / machine_metrics["dur_prod"],
        np.nan
    )

    # Ratios de preparacion e incidencias respecto a produccion
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

    # Filtrar registros con datos suficientes (minimo 60 minutos)
    machine_metrics = machine_metrics[machine_metrics["total_dur"] > 60].copy()
    machine_metrics = machine_metrics.dropna(subset=["disponibilidad", "scrap_rate", "uph_real"])

    return machine_metrics


def train_interactive_model(X_train, y_train, n_estimators, max_depth, min_samples_split):
    """
    Entrena un modelo Random Forest con hiperparametros especificados.

    Args:
        X_train: Features de entrenamiento
        y_train: Etiquetas de entrenamiento
        n_estimators: Numero de arboles en el bosque
        max_depth: Profundidad maxima de cada arbol
        min_samples_split: Minimo de muestras para dividir un nodo

    Returns:
        tuple: (modelo_entrenado, scaler_ajustado)
    """
    # Inicializar modelo con hiperparametros
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1  # Usar todos los cores disponibles
    )

    # Escalar features para mejorar convergencia
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Entrenar modelo
    model.fit(X_train_scaled, y_train)

    return model, scaler


def evaluate_classification_model(model, scaler, X, y):
    """
    Evalua el rendimiento del modelo de clasificacion.

    Args:
        model: Modelo entrenado
        scaler: Scaler ajustado
        X: Features de evaluacion
        y: Etiquetas reales

    Returns:
        tuple: (predicciones, probabilidades, accuracy, matriz_confusion, reporte)
    """
    # Escalar features
    X_scaled = scaler.transform(X)

    # Realizar predicciones
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    # Calcular metricas
    accuracy = accuracy_score(y, predictions)
    conf_matrix = confusion_matrix(y, predictions)
    report = classification_report(y, predictions, output_dict=True)

    return predictions, probabilities, accuracy, conf_matrix, report


def plot_roc_curves(model, scaler, X, y, classes):
    """
    Genera curvas ROC para clasificacion multiclase.

    Args:
        model: Modelo entrenado
        scaler: Scaler ajustado
        X: Features
        y: Etiquetas reales
        classes: Lista de nombres de clases

    Returns:
        plotly.graph_objects.Figure: Grafico de curvas ROC
    """
    # Escalar features
    X_scaled = scaler.transform(X)

    # Obtener probabilidades
    y_score = model.predict_proba(X_scaled)

    # Binarizar etiquetas para ROC multiclase
    y_bin = label_binarize(y, classes=classes)
    n_classes = len(classes)

    # Crear figura
    fig = go.Figure()

    # Calcular ROC para cada clase
    for i, class_name in enumerate(classes):
        if n_classes == 2 and i == 1:
            # Para clasificacion binaria, solo mostrar una curva
            break

        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{class_name} (AUC = {roc_auc:.2f})',
            line=dict(width=2)
        ))

    # Linea diagonal de referencia
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Aleatorio',
        line=dict(dash='dash', color='gray', width=1)
    ))

    fig.update_layout(
        title='Curvas ROC por Clase',
        xaxis_title='Tasa de Falsos Positivos',
        yaxis_title='Tasa de Verdaderos Positivos',
        hovermode='closest',
        legend=dict(x=0.6, y=0.1)
    )

    return fig


def save_model_with_bentoml(model, scaler, features, classes):
    """
    Empaqueta el modelo con BentoML para despliegue como API.

    Args:
        model: Modelo entrenado
        scaler: Scaler ajustado
        features: Lista de nombres de features
        classes: Lista de nombres de clases

    Returns:
        bool: True si se guardo correctamente, False si hubo error
    """
    try:
        import bentoml

        # Guardar modelo en BentoML con metadatos
        bentoml.sklearn.save_model(
            "machine_classifier",
            model,
            labels={
                "framework": "sklearn",
                "model_type": "RandomForestClassifier",
                "task": "classification"
            },
            metadata={
                "features": features,
                "classes": classes,
                "n_classes": len(classes)
            }
        )

        # Guardar scaler en BentoML
        bentoml.sklearn.save_model(
            "machine_scaler",
            scaler,
            labels={
                "framework": "sklearn",
                "model_type": "StandardScaler",
                "task": "preprocessing"
            },
            metadata={
                "features": features
            }
        )

        return True
    except ImportError:
        return False
    except Exception as e:
        st.error(f"Error al empaquetar: {str(e)}")
        return False


def page_ml_classification(filtered: dict, ciclos: pd.DataFrame) -> None:
    """
    Funcion principal que renderiza la pagina de clasificacion ML.

    Args:
        filtered: Diccionario con datos filtrados
        ciclos: DataFrame con ciclos de produccion
    """
    st.title("Clasificacion ML - Estado de Maquinas")

    # Selector de modo de operacion
    st.markdown("### Modo de Operacion")
    modo = st.radio(
        "Elige el modo:",
        ["Modelo Pre-entrenado", "Entrenar Modelo Interactivo"],
        horizontal=True,
        help="Pre-entrenado: usa el modelo ya guardado. Interactivo: entrena tu propio modelo."
    )

    # Preparar datos de produccion
    prod = filtered.get("produccion", pd.DataFrame())

    if prod.empty:
        st.info("Sin datos de produccion en el rango seleccionado.")
        return

    # Calcular metricas por maquina
    machine_metrics = prepare_features(prod)

    if len(machine_metrics) < 10:
        st.warning("No hay suficientes datos para entrenar (minimo 10 registros).")
        return

    # Mapa de colores para visualizaciones
    color_map = {
        "EXCELENTE": "#22c55e",
        "BUENA": "#3b82f6",
        "REQUIERE_ATENCION": "#f59e0b",
        "CRITICA": "#ef4444"
    }

    # ========== MODO PRE-ENTRENADO ==========
    if modo == "Modelo Pre-entrenado":
        st.markdown("### Seleccion de Modelo Pre-entrenado")

        # Listar modelos disponibles
        available_models = list_available_classification_models()

        if not available_models:
            st.error("No hay modelos pre-entrenados disponibles. Ejecuta: python models/classification/train.py")
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
        model, scaler, features, classes = load_classification_model(selected_model)

        if model is None:
            st.error(f"Error al cargar el modelo: {selected_model}")
            return

        st.markdown("---")
        st.markdown("""
        Usando **Random Forest Classifier** para clasificacion de estado de maquinas.

        **Clases:**
        - **EXCELENTE**: Alta disponibilidad, bajo scrap, alto UPH
        - **BUENA**: Disponibilidad aceptable, scrap controlado
        - **REQUIERE_ATENCION**: Por debajo de objetivos
        - **CRITICA**: Disponibilidad muy baja o scrap muy alto
        """)

        col1, col2, col3 = st.columns(3)
        col1.metric("Features", len(features))
        col2.metric("Clases", len(classes))
        col3.metric("Algoritmo", "Random Forest")

        # Realizar predicciones
        X = machine_metrics[features].values
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        machine_metrics["estado_predicho"] = predictions
        machine_metrics["probabilidad_max"] = probabilities.max(axis=1)

    # ========== MODO INTERACTIVO ==========
    else:
        st.markdown("""
        **Entrena tu propio modelo** de clasificacion eligiendo features, hiperparametros y umbrales de las clases.
        """)

        st.markdown("---")

        # SECCION DE GUARDADO PROMINENTE (MOVIDA ARRIBA)
        if 'trained_clf_model' in st.session_state:
            st.markdown("### Guardar Modelo")
            st.info("Tu modelo ha sido entrenado. Guardalo en BentoML para uso futuro y despliegue como API.")

            model = st.session_state['trained_clf_model']
            scaler = st.session_state['trained_clf_scaler']
            selected_features = st.session_state['trained_clf_features']
            classes = st.session_state['classes_clf']

            # Campo para nombre del modelo
            model_name = st.text_input(
                "Nombre del modelo:",
                value="machine_classifier",
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
                                "model_type": "RandomForestClassifier",
                                "task": "classification"
                            },
                            metadata={
                                "features": selected_features,
                                "classes": classes,
                                "n_classes": len(classes),
                                "scaler_tag": str(saved_scaler.tag)
                            }
                        )

                        st.success(f"Modelo guardado exitosamente en BentoML: {model_name}")

                    except ImportError:
                        st.error("BentoML no esta instalado. Ejecuta: pip install bentoml")
                    except Exception as e:
                        st.error(f"Error al guardar: {str(e)}")

            st.markdown("---")

        # Configuracion del modelo
        st.markdown("### Configuracion del Modelo")

        col_config1, col_config2 = st.columns([2, 1])

        with col_config1:
            st.markdown("#### Seleccion de Features")

            all_features = ["disponibilidad", "scrap_rate", "uph_real", "dur_prod", "dur_prep",
                           "dur_inci", "prep_ratio", "inci_ratio", "n_operaciones"]

            selected_features = st.multiselect(
                "Selecciona features:",
                all_features,
                default=["disponibilidad", "scrap_rate", "uph_real", "dur_prod", "prep_ratio", "inci_ratio"],
                key="features_classification"
            )

            if len(selected_features) == 0:
                st.warning("Selecciona al menos 1 feature")
                return

            st.success(f"{len(selected_features)} features seleccionadas")

            st.markdown("#### Umbrales de Clasificacion")
            st.markdown("Define los umbrales para cada clase:")

            col_umb1, col_umb2 = st.columns(2)

            with col_umb1:
                st.markdown("**EXCELENTE:**")
                exc_disp = st.slider("Disp min", 70, 100, 85, key="exc_disp") / 100
                exc_scrap = st.slider("Scrap max", 0, 10, 2, key="exc_scrap") / 100
                exc_uph = st.slider("UPH min", 50, 200, 100, key="exc_uph")

            with col_umb2:
                st.markdown("**BUENA:**")
                bue_disp = st.slider("Disp min", 50, 90, 70, key="bue_disp") / 100
                bue_scrap = st.slider("Scrap max", 0, 15, 5, key="bue_scrap") / 100
                bue_uph = st.slider("UPH min", 30, 150, 60, key="bue_uph")

            st.markdown("**CRITICA:**")
            col_cri1, col_cri2 = st.columns(2)
            cri_disp = col_cri1.slider("Disp < ", 20, 70, 50, key="cri_disp") / 100
            cri_scrap = col_cri2.slider("Scrap >", 5, 20, 10, key="cri_scrap") / 100

            umbrales = {
                "excelente": {"disp": exc_disp, "scrap": exc_scrap, "uph": exc_uph},
                "buena": {"disp": bue_disp, "scrap": bue_scrap, "uph": bue_uph},
                "critica": {"disp": cri_disp, "scrap": cri_scrap}
            }

        with col_config2:
            st.markdown("#### Hiperparametros")

            n_estimators = st.slider(
                "Numero de arboles",
                min_value=10,
                max_value=300,
                value=100,
                step=10,
                help="Mas arboles = mejor precision pero mas lento"
            )

            max_depth = st.slider(
                "Profundidad maxima",
                min_value=3,
                max_value=30,
                value=15,
                help="Menor = menos overfitting"
            )

            min_samples_split = st.slider(
                "Minimo para dividir",
                min_value=2,
                max_value=20,
                value=5,
                help="Minimo de muestras para dividir un nodo"
            )

            test_size = st.slider(
                "% Test",
                min_value=10,
                max_value=40,
                value=20,
                help="Porcentaje de datos para test"
            ) / 100

        st.markdown("---")

        # Boton entrenar
        if st.button("Entrenar Modelo", type="primary", use_container_width=True):
            with st.spinner("Entrenando modelo... Esto puede tardar unos segundos."):
                # Generar labels basados en umbrales
                machine_metrics["estado_real"] = machine_metrics.apply(lambda row: classify_estado(row, umbrales), axis=1)

                # Preparar datos
                X = machine_metrics[selected_features].fillna(0)
                y = machine_metrics["estado_real"].values

                # Split train/test con estratificacion
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )

                # Entrenar
                model, scaler = train_interactive_model(
                    X_train, y_train, n_estimators, max_depth, min_samples_split
                )

                # Evaluar en train y test
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

                st.success("Modelo entrenado exitosamente!")
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

            st.markdown("### Resultados del Entrenamiento")

            # Metricas de accuracy
            col_train, col_test = st.columns(2)

            with col_train:
                st.markdown("#### Conjunto de Entrenamiento")
                st.metric("Accuracy", f"{train_acc:.1%}")

            with col_test:
                st.markdown("#### Conjunto de Test")
                st.metric("Accuracy", f"{test_acc:.1%}")

            # Interpretacion de resultados
            if abs(train_acc - test_acc) > 0.1:
                st.warning("Diferencia grande entre train y test. Posible overfitting. Considera reducir max_depth o aumentar min_samples_split.")
            elif test_acc > 0.85:
                st.success("Excelente rendimiento del modelo!")
            elif test_acc > 0.70:
                st.info("Rendimiento aceptable. Prueba ajustar hiperparametros para mejorar.")
            else:
                st.error("Bajo rendimiento. Prueba seleccionar mas features o ajustar umbrales.")

            st.markdown("---")
            st.markdown("### Matriz de Confusion y Metricas")

            col_cm1, col_cm2 = st.columns(2)

            with col_cm1:
                st.markdown("#### Conjunto de Test")
                fig_cm = px.imshow(
                    test_cm,
                    labels=dict(x="Predicho", y="Real", color="Count"),
                    x=classes,
                    y=classes,
                    text_auto=True,
                    title="Matriz de Confusion"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_cm2:
                st.markdown("#### Metricas por Clase")
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

            # Curvas ROC
            st.markdown("---")
            st.markdown("### Curvas ROC (Receiver Operating Characteristic)")
            st.markdown("""
            Las curvas ROC muestran el rendimiento del clasificador variando el umbral de decision.
            Un area bajo la curva (AUC) cercana a 1.0 indica excelente capacidad de discriminacion.
            """)

            fig_roc = plot_roc_curves(model, scaler, X_test, y_test, classes)
            st.plotly_chart(fig_roc, use_container_width=True)

            st.markdown("---")
            st.markdown("### Feature Importance")
            st.markdown("""
            Muestra la importancia relativa de cada feature en las decisiones del modelo.
            Features con mayor importancia tienen mas influencia en las predicciones.
            """)

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

            # Comparacion con pre-entrenado
            st.markdown("---")
            st.markdown("### Comparacion con Modelo Pre-entrenado")

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
                        st.success("Tu modelo interactivo supera al pre-entrenado!")
                    else:
                        st.info("El modelo pre-entrenado sigue siendo mejor. Prueba ajustar hiperparametros.")

            # Hacer predicciones en todos los datos
            X_all = machine_metrics[selected_features].fillna(0)
            X_all_scaled = scaler.transform(X_all)
            all_predictions = model.predict(X_all_scaled)
            all_probabilities = model.predict_proba(X_all_scaled)

            machine_metrics["estado_predicho"] = all_predictions
            machine_metrics["probabilidad_max"] = all_probabilities.max(axis=1)

    # ========== VISUALIZACIONES COMUNES (para ambos modos) ==========
    if modo == "Modelo Pre-entrenado" or 'trained_clf_model' in st.session_state:
        st.markdown("---")
        st.markdown("### Analisis de Clasificacion")

        # Distribucion de estados
        estado_counts = machine_metrics["estado_predicho"].value_counts().reset_index()
        estado_counts.columns = ["estado", "count"]

        col_dist1, col_dist2 = st.columns(2)

        col_dist1.dataframe(estado_counts, hide_index=True, use_container_width=True)

        fig_pie = px.pie(
            estado_counts,
            values="count",
            names="estado",
            title="Distribucion de Estados",
            color="estado",
            color_discrete_map=color_map
        )
        col_dist2.plotly_chart(fig_pie, use_container_width=True)

        # Scatter plot
        st.markdown("#### Visualizacion: Disponibilidad vs Scrap")

        fig_scatter = px.scatter(
            machine_metrics,
            x="disponibilidad",
            y="scrap_rate",
            color="estado_predicho",
            size="probabilidad_max",
            hover_data=["machine_name", "periodo", "uph_real"],
            title="Clasificacion de Maquinas",
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

        # Maquinas por estado
        st.markdown("#### Maquinas por Estado")

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
