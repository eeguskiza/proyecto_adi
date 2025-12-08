import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ..models import call_bentoml_scrap


def page_produccion(filtered: dict) -> None:
    st.subheader("Producción")
    prod = filtered["produccion"]
    if prod.empty:
        st.info("Sin datos en el rango seleccionado.")
        return

    cols_tabla = [
        "work_order_id",
        "op_id",
        "ref_id_str",
        "familia",
        "cliente",
        "machine_id",
        "machine_name",
        "op_text",
        "planta",
        "ts_ini",
        "ts_fin",
        "duracion_min",
        "piezas_ok",
        "piezas_scrap",
        "scrap_rate",
        "turno",
    ]
    st.dataframe(prod[cols_tabla], width="stretch", hide_index=True)

    c1, c2, c3 = st.columns(3)
    agg_machine = (
        prod.groupby("machine_name")
        .agg(
            piezas_ok=("piezas_ok", "sum"),
            piezas_scrap=("piezas_scrap", "sum"),
            duracion_min=("duracion_min", "sum"),
        )
        .reset_index()
    )
    agg_machine["scrap_rate"] = np.where(
        agg_machine["piezas_ok"] + agg_machine["piezas_scrap"] > 0,
        agg_machine["piezas_scrap"] / (agg_machine["piezas_ok"] + agg_machine["piezas_scrap"]),
        np.nan,
    )
    agg_machine["piezas_hora"] = np.where(agg_machine["duracion_min"] > 0, 60 * agg_machine["piezas_ok"] / agg_machine["duracion_min"], np.nan)
    c1.subheader("Por máquina")
    c1.dataframe(agg_machine, width="stretch", hide_index=True)

    agg_ref = (
        prod.groupby("ref_id_str")
        .agg(piezas_ok=("piezas_ok", "sum"), piezas_scrap=("piezas_scrap", "sum"), duracion_min=("duracion_min", "sum"))
        .reset_index()
    )
    agg_ref["scrap_rate"] = np.where(
        agg_ref["piezas_ok"] + agg_ref["piezas_scrap"] > 0,
        agg_ref["piezas_scrap"] / (agg_ref["piezas_ok"] + agg_ref["piezas_scrap"]),
        np.nan,
    )
    c2.subheader("Por referencia")
    c2.dataframe(agg_ref, width="stretch", hide_index=True)

    agg_turno = (
        prod.groupby("turno")
        .agg(piezas_ok=("piezas_ok", "sum"), piezas_scrap=("piezas_scrap", "sum"), duracion_min=("duracion_min", "sum"))
        .reset_index()
    )
    agg_turno["scrap_rate"] = np.where(
        agg_turno["piezas_ok"] + agg_turno["piezas_scrap"] > 0,
        agg_turno["piezas_scrap"] / (agg_turno["piezas_ok"] + agg_turno["piezas_scrap"]),
        np.nan,
    )
    c3.subheader("Por turno")
    c3.dataframe(agg_turno, width="stretch", hide_index=True)

    c4, c5 = st.columns(2)
    heat = (
        prod.groupby(["machine_name", "ref_id_str"])
        .agg(piezas_scrap=("piezas_scrap", "sum"), total=("total_piezas", "sum"))
        .reset_index()
    )
    heat["scrap_rate"] = np.where(heat["total"] > 0, heat["piezas_scrap"] / heat["total"], np.nan)
    fig_heat = px.density_heatmap(
        heat,
        x="machine_name",
        y="ref_id_str",
        z="scrap_rate",
        color_continuous_scale="Reds",
        title="Heatmap scrap% máquina vs referencia",
    )
    c4.plotly_chart(fig_heat, width="stretch")

    fig_hist = px.histogram(prod, x="scrap_rate", nbins=50, title="Distribución scrap por operación")
    c5.plotly_chart(fig_hist, width="stretch")

    st.markdown("### Modelo de scrap (BentoML)")
    col1, col2, col3, col4, col5 = st.columns(5)
    machine = col1.selectbox("Máquina", sorted(prod["machine_name"].dropna().unique()))
    ref = col2.selectbox("Referencia", sorted(prod["ref_id_str"].dropna().unique()))
    familia = col3.selectbox("Familia", sorted(prod["familia"].dropna().unique()))
    qty_plan = col4.number_input("Cantidad planificada", min_value=0, value=int(prod["piezas_ok"].median() if not prod.empty else 0))
    turno = col5.selectbox("Turno", ["mañana", "tarde", "noche"])
    endpoint = st.text_input("Endpoint BentoML", value="http://localhost:3000/predict")

    if st.button("Predecir scrap"):
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
