import streamlit as st

from scripts.dashboard.logger import DashboardLogger
from scripts.dashboard.data import load_data
from scripts.dashboard.filters import apply_filters, get_filters
from scripts.dashboard.oee import inject_metric_styles
from scripts.dashboard.pages.almacen_page import page_almacen
from scripts.dashboard.pages.clustering_page import page_clustering
from scripts.dashboard.pages.dashboard_page import page_dashboard
from scripts.dashboard.pages.produccion_page import page_produccion
from scripts.dashboard.pages.rrhh_page import page_rrhh
from scripts.dashboard.pages.ml_clustering_page import page_ml_clustering
from scripts.dashboard.pages.ml_regression_page import page_ml_regression
from scripts.dashboard.pages.ml_classification_page import page_ml_classification

# Logger
logger = DashboardLogger.get_logger()

# Banner de inicio
logger.info("=" * 60)
logger.info("DASHBOARD DE PLANTA - Sistema de Monitorizacion")
logger.info("=" * 60)

st.set_page_config(
    page_title="Dashboard planta",
    page_icon=":bar_chart:",
    layout="wide",
)
inject_metric_styles()

# Logging de módulos cargados
DashboardLogger.log_module_load("Configuración Streamlit")
DashboardLogger.log_module_load("Estilos métricos OEE")


def main() -> None:
    # Carga de datos
    logger.info("Iniciando carga de datos principales...")
    data = load_data()
    DashboardLogger.log_module_load("Datos principales")

    # Aplicar filtros
    logger.info("Aplicando filtros...")
    filtros = get_filters(data)
    filtered = apply_filters(data, filtros)
    DashboardLogger.log_module_load("Sistema de filtros")

    with st.sidebar:
        page = st.selectbox(
            "Páginas",
            [
                "Cuadro de mando general",
                "Producción",
                "Almacén MP",
                "RRHH",
                "Clustering ML",
                "ML - Clustering",
                "ML - Regresión Scrap",
                "ML - Clasificación Estado",
            ],
            key="page_selector",
        )

        # Opción para habilitar/deshabilitar chatbot
        enable_chatbot = st.checkbox("Habilitar Chatbot IA", value=False, help="Requiere Ollama ejecutandose")

    # Logging de página seleccionada
    logger.info(f"Cargando pagina: {page}")

    if page == "Cuadro de mando general":
        page_dashboard(filtered, data["ciclos"], filtros.get("recurso_oee", "(Todos)"))
        DashboardLogger.log_module_load(f"Página: {page}")
    elif page == "Producción":
        page_produccion(filtered)
        DashboardLogger.log_module_load(f"Página: {page}")
    elif page == "Almacén MP":
        page_almacen(filtered, data["refs"])
        DashboardLogger.log_module_load(f"Página: {page}")
    elif page == "RRHH":
        page_rrhh(filtered, filtered["produccion"])
        DashboardLogger.log_module_load(f"Página: {page}")
    elif page == "Clustering ML":
        page_clustering(filtered, data["ciclos"])
        DashboardLogger.log_module_load(f"Página: {page}")
    elif page == "ML - Clustering":
        page_ml_clustering(filtered, data["ciclos"])
        DashboardLogger.log_module_load(f"Página: {page}")
    elif page == "ML - Regresión Scrap":
        page_ml_regression(filtered, data["ciclos"])
        DashboardLogger.log_module_load(f"Página: {page}")
    elif page == "ML - Clasificación Estado":
        page_ml_classification(filtered, data["ciclos"])
        DashboardLogger.log_module_load(f"Página: {page}")

    # Renderizar chatbot solo si está habilitado (carga lazy)
    if enable_chatbot:
        logger.info("Cargando modulo de chatbot IA...")
        try:
            from scripts.dashboard.chatbot import render_chatbot_bubble
            render_chatbot_bubble(filtered, page, filtros)
            DashboardLogger.log_module_load("Chatbot IA (Ollama)")
        except Exception as e:
            logger.error(f"Error al cargar chatbot: {str(e)}")
            st.sidebar.warning("Error al cargar el chatbot. Verifica que Ollama este ejecutandose.")


if __name__ == "__main__":
    main()
