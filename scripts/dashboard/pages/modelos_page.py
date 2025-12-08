import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ..models import call_bentoml_scrap
from ..utils import calcular_riesgo_cambio


def page_modelos(filtered: dict, prod: pd.DataFrame) -> None:
    st.subheader("Modelos IA / BentoML")
    st.markdown("#### Scrap")
    col1, col2, col3, col4, col5 = st.columns(5)
    machine = col1.selectbox("Máquina", sorted(prod["machine_name"].dropna().unique()))
    ref = col2.selectbox("Referencia", sorted(prod["ref_id_str"].dropna().unique()))
    familia = col3.selectbox("Familia", sorted(prod["familia"].dropna().unique()))
    qty_plan = col4.number_input("Cantidad planificada", min_value=0, value=int(prod["piezas_ok"].median() if not prod.empty else 0))
    turno = col5.selectbox("Turno", ["mañana", "tarde", "noche"])
    endpoint = st.text_input("Endpoint BentoML", value="http://localhost:3000/predict")

    if st.button("Predecir scrap (Bento)"):
        payload = {
            "machine_name": machine,
            "ref_id": ref,
            "familia": familia,
            "qty_plan": qty_plan,
            "turno": turno,
        }
        resultado = call_bentoml_scrap(endpoint, payload)
        if "error" in resultado:
            st.error(f"Error llamando al modelo: {resultado['error']}")
        else:
            esperado = resultado.get("scrap_esperado", np.nan)
            tasa = resultado.get("scrap_rate", np.nan)
            riesgo = resultado.get("riesgo", "desconocido")
            st.success(f"Scrap esperado: {esperado:.1f} uds | Scrap% estimado: {tasa:.2%} | Riesgo: {riesgo}")

    st.markdown("#### Sistema de cambio de fresa")
    fresa = filtered["fresa"]
    if fresa.empty:
        st.info("Sin eventos de cambio de fresa en el rango.")
        return

    talladoras = fresa[fresa["machine_name"].str.contains("talladora", case=False, na=False)]
    if talladoras.empty:
        st.info("No hay talladoras en los datos filtrados.")
        return

    resumen = (
        talladoras.groupby("machine_name")
        .agg(
            ultimo_cambio=("ts_cambio", "max"),
            piezas_med=("piezas_hasta_cambio", "median"),
            piezas_ult=("piezas_hasta_cambio", "last"),
        )
        .reset_index()
    )
    resumen["ratio_sobre_tipico"] = resumen["piezas_ult"] / resumen["piezas_med"]
    resumen["riesgo"] = resumen["ratio_sobre_tipico"].apply(calcular_riesgo_cambio)
    st.dataframe(resumen, width="stretch", hide_index=True)

    hist = px.histogram(
        talladoras,
        x="piezas_hasta_cambio",
        color="machine_name",
        barmode="overlay",
        nbins=40,
        title="Distribución de piezas entre cambios",
    )
    st.plotly_chart(hist, width="stretch")
