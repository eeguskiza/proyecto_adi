import pandas as pd
import plotly.express as px
import streamlit as st


def page_almacen(filtered: dict, refs: pd.DataFrame) -> None:
    st.subheader("Almacén de materia prima")
    compras = filtered["compras"]
    almacen = filtered["almacen"]
    prod = filtered["produccion"]

    if compras.empty and almacen.empty:
        st.info("Sin datos de almacén en el rango seleccionado.")
        return

    col1, col2, col3 = st.columns(3)
    kg_recibidos = compras["qty_recibida"].sum()
    col1.metric("Kg recibidos", f"{kg_recibidos:,.0f}")
    lotes = compras["material_lot_id"].nunique()
    col2.metric("Lotes recibidos", f"{lotes}")

    prod_ref = prod.copy()
    if "peso_neto_kg" not in prod_ref.columns:
        prod_ref = prod_ref.merge(refs[["ref_id_str", "peso_neto_kg"]], on="ref_id_str", how="left")
    prod_ref["kg_consumidos"] = (prod_ref["piezas_ok"] + prod_ref["piezas_scrap"]) * prod_ref.get("peso_neto_kg", 0).fillna(0)
    kg_consumidos = prod_ref["kg_consumidos"].sum()
    col3.metric("Kg consumo teórico", f"{kg_consumidos:,.0f}")

    c1, c2 = st.columns(2)
    if not compras.empty:
        top_ref = compras.groupby("ref_materia_str")["qty_recibida"].sum().reset_index()
        fig_top = px.bar(top_ref.sort_values("qty_recibida", ascending=False).head(5), x="ref_materia_str", y="qty_recibida", title="Top referencias por kg recibidos")
        c1.plotly_chart(fig_top, width="stretch")

        compras["fecha"] = compras["fecha_recepcion_ts"].dt.date
        ts = compras.groupby(["fecha", "ref_materia_str"])["qty_recibida"].sum().reset_index()
        fig_ts = px.area(ts, x="fecha", y="qty_recibida", color="ref_materia_str", title="Serie de kg recibidos")
        c2.plotly_chart(fig_ts, width="stretch")

    if not prod_ref.empty:
        cons_ref = prod_ref.groupby("ref_id_str")["kg_consumidos"].sum().reset_index()
        fig_cons = px.bar(cons_ref.sort_values("kg_consumidos", ascending=False).head(10), x="ref_id_str", y="kg_consumidos", title="MP más consumida (teórico)")
        st.plotly_chart(fig_cons, width="stretch")

        entradas = compras.groupby("ref_materia_str")["qty_recibida"].sum()
        stock_teorico = pd.DataFrame({"ref_id_str": cons_ref["ref_id_str"]})
        stock_teorico["kg_recibidos"] = stock_teorico["ref_id_str"].map(entradas).fillna(0)
        stock_teorico = stock_teorico.merge(cons_ref, on="ref_id_str", how="left")
        stock_teorico["stock_teorico"] = stock_teorico["kg_recibidos"] - stock_teorico["kg_consumidos"]
        st.dataframe(stock_teorico, width="stretch", hide_index=True)
