import streamlit as st

from scripts.dashboard.data import load_data
from scripts.dashboard.filters import apply_filters, get_filters
from scripts.dashboard.oee import inject_metric_styles
from scripts.dashboard.pages.almacen_page import page_almacen
from scripts.dashboard.pages.dashboard_page import page_dashboard
from scripts.dashboard.pages.modelos_page import page_modelos
from scripts.dashboard.pages.produccion_page import page_produccion
from scripts.dashboard.pages.rrhh_page import page_rrhh


st.set_page_config(
    page_title="Dashboard planta",
    page_icon="📊",
    layout="wide",
)
inject_metric_styles()


def main() -> None:
    data = load_data()
    filtros = get_filters(data)
    filtered = apply_filters(data, filtros)

    with st.sidebar:
        page = st.selectbox(
            "Páginas",
            [
                "Cuadro de mando general",
                "Producción",
                "Almacén MP",
                "RRHH",
                "Modelos IA / BentoML",
            ],
            key="page_selector",
        )

    if page == "Cuadro de mando general":
        page_dashboard(filtered, data["ciclos"], filtros.get("recurso_oee", "(Todos)"))
    elif page == "Producción":
        page_produccion(filtered)
    elif page == "Almacén MP":
        page_almacen(filtered, data["refs"])
    elif page == "RRHH":
        page_rrhh(filtered, filtered["produccion"])
    else:
        page_modelos(filtered, filtered["produccion"])


if __name__ == "__main__":
    main()
