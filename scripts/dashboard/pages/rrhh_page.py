import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def page_rrhh(filtered: dict, prod: pd.DataFrame) -> None:
    st.subheader("RRHH")
    rrhh = filtered["rrhh"]
    if rrhh.empty:
        st.info("Sin datos de RRHH en el rango seleccionado.")
        return

    rrhh_display = rrhh.copy()
    rrhh_display["año_mes"] = rrhh_display["año_mes"].dt.strftime("%Y-%m")
    st.dataframe(rrhh_display, width="stretch", hide_index=True)

    fig_netas = px.line(rrhh, x="año_mes", y="horas_netas", markers=True, title="Horas netas por mes")
    st.plotly_chart(fig_netas, width="stretch")

    ausencias = rrhh.melt(
        id_vars=["año_mes"],
        value_vars=["horas_enfermedad", "horas_accidente", "horas_permiso"],
        var_name="tipo",
        value_name="horas",
    )
    fig_aus = px.bar(ausencias, x="año_mes", y="horas", color="tipo", title="Ausencias por mes", barmode="stack")
    st.plotly_chart(fig_aus, width="stretch")

    if not prod.empty:
        prod_month = prod.copy()
        prod_month["año_mes"] = prod_month["ts_ini"].dt.to_period("M").dt.to_timestamp()
        prod_month = prod_month.groupby("año_mes")["piezas_ok"].sum().reset_index()
        prod_rrhh = prod_month.merge(rrhh[["año_mes", "horas_netas"]], on="año_mes", how="left")
        prod_rrhh["productividad"] = np.where(prod_rrhh["horas_netas"] > 0, prod_rrhh["piezas_ok"] / prod_rrhh["horas_netas"], np.nan)
        fig_prod = px.bar(prod_rrhh, x="año_mes", y="productividad", title="Productividad (piezas OK / horas netas)")
        st.plotly_chart(fig_prod, width="stretch")
