import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ..oee import compute_oee, render_kpi_cards


def page_dashboard(filtered: dict, ciclos: pd.DataFrame, recurso_sel: str) -> None:
    st.subheader("Cuadro de mando general")
    prod_all = filtered["produccion"]
    if prod_all.empty:
        st.info("Sin datos en el rango seleccionado.")
        return

    prod = prod_all if recurso_sel == "(Todos)" else prod_all[prod_all["machine_name"] == recurso_sel]
    if "estado_oee" not in prod.columns:
        prod = prod.copy()
        prod["estado_oee"] = prod["evento"].str.lower().map(
            {"producción": "produccion", "produccion": "produccion", "preparación": "preparacion", "preparacion": "preparacion"}
        )
        prod["estado_oee"] = prod["estado_oee"].fillna("incidencia")

    oee_data = compute_oee(prod, ciclos)
    with st.container():
        render_kpi_cards(oee_data)

    dur = oee_data["durations"]
    disp_df = pd.DataFrame(
        {
            "estado": ["Producción", "Preparación", "Incidencias"],
            "minutos": [dur.get("produccion", 0), dur.get("preparacion", 0), dur.get("incidencia", 0)],
        }
    )
    disp_df["barra"] = "Disponibilidad"

    col1, col2 = st.columns((2, 1))
    fig_disp = px.bar(
        disp_df,
        x="minutos",
        y="barra",
        color="estado",
        orientation="h",
        text="minutos",
        title="Distribución del tiempo",
        color_discrete_map={"Producción": "#22c55e", "Preparación": "#fbbf24", "Incidencias": "#ef4444"},
    )
    fig_disp.update_layout(barmode="stack", xaxis_title="Minutos", yaxis_title="")
    fig_disp.update_traces(texttemplate="%{text:.0f} min", textposition="inside")
    col1.plotly_chart(fig_disp, use_container_width=True)

    calidad_df = pd.DataFrame(
        {
            "tipo": ["OK", "Scrap"],
            "piezas": [oee_data["total_ok"], oee_data["total_scrap"]],
        }
    )
    fig_calidad = px.pie(calidad_df, values="piezas", names="tipo", title="Distribución de calidad", color="tipo")
    col2.plotly_chart(fig_calidad, use_container_width=True)

    prod_perf = prod[prod["estado_oee"] == "produccion"].copy()
    ciclos_ref_machine = ciclos[["ref_id_str", "machine_name", "piezas_hora_teorico"]].drop_duplicates()
    prod_perf = prod_perf.merge(ciclos_ref_machine, on=["ref_id_str", "machine_name"], how="left")

    hist_perf = (
        prod_perf.groupby(["machine_name", "ref_id_str"])
        .agg(piezas=("piezas_ok", "sum"), dur_min=("duracion_min", "sum"))
        .reset_index()
    )
    hist_perf = hist_perf[hist_perf["dur_min"] > 0]
    hist_perf["uph_hist"] = hist_perf["piezas"] / (hist_perf["dur_min"] / 60)
    hist_perf["uph_hist"] = hist_perf["uph_hist"].clip(30, 180)
    prod_perf = prod_perf.merge(hist_perf[["machine_name", "ref_id_str", "uph_hist"]], on=["machine_name", "ref_id_str"], how="left")

    prod_perf["piezas_hora_teorico"] = prod_perf["piezas_hora_teorico"].replace(0, pd.NA)
    prod_perf["piezas_hora_teorico"] = prod_perf["piezas_hora_teorico"].fillna(prod_perf["uph_hist"])
    prod_perf["piezas_hora_teorico"] = prod_perf["piezas_hora_teorico"].fillna(80)
    prod_perf = prod_perf.drop(columns=["uph_hist"])

    prod_perf["ideal_piezas"] = prod_perf["piezas_hora_teorico"] * (prod_perf["duracion_min"] / 60)
    prod_perf["fecha"] = prod_perf["ts_ini"].dt.date

    perf_diaria = (
        prod_perf.groupby("fecha")
        .agg(
            piezas_ok=("piezas_ok", "sum"),
            duracion_min=("duracion_min", "sum"),
            ideal_piezas=("ideal_piezas", "sum"),
        )
        .reset_index()
    )
    perf_diaria = perf_diaria[perf_diaria["duracion_min"] > 0]
    perf_diaria["uph_real"] = perf_diaria["piezas_ok"] / (perf_diaria["duracion_min"] / 60)
    perf_diaria["uph_ideal"] = np.where(
        perf_diaria["duracion_min"] > 0, perf_diaria["ideal_piezas"] / (perf_diaria["duracion_min"] / 60), np.nan
    )
    perf_diaria["uph_real"] = pd.to_numeric(perf_diaria["uph_real"], errors="coerce")
    perf_diaria["uph_ideal"] = pd.to_numeric(perf_diaria["uph_ideal"], errors="coerce")

    prod_inci = prod.copy()
    prod_inci["estado_oee"] = prod_inci["evento"].str.lower().map(
        {"producción": "produccion", "produccion": "produccion", "preparación": "preparacion", "preparacion": "preparacion"}
    )
    prod_inci["estado_oee"] = prod_inci["estado_oee"].fillna("incidencia")
    incidencias = prod_inci[prod_inci["estado_oee"] == "incidencia"]
    fig_inci = None
    if not incidencias.empty:
        top_inci = (
            incidencias.groupby("tipo_incidencia")["duracion_min"]
            .sum()
            .sort_values(ascending=False)
            .head(8)
            .reset_index()
        )
        fig_inci = px.bar(
            top_inci,
            x="duracion_min",
            y="tipo_incidencia",
            orientation="h",
            title="Top incidencias por tiempo",
            labels={"duracion_min": "Minutos"},
        )

    st.markdown("—")
    c3, c4 = st.columns((2, 1))
    if not perf_diaria.empty:
        c3.plotly_chart(
            px.line(
                perf_diaria,
                x="fecha",
                y=["uph_real", "uph_ideal"],
                markers=True,
                labels={"value": "Piezas/hora", "variable": "Serie"},
                title="Rendimiento real vs. ideal",
            ),
            use_container_width=True,
        )
    if fig_inci:
        c4.plotly_chart(fig_inci, use_container_width=True)
