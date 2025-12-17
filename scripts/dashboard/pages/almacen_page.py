import pandas as pd
import plotly.express as px
import streamlit as st


def page_almacen(filtered: dict, refs: pd.DataFrame) -> None:
    st.subheader("Gestión de Almacén")
    
    # Carga y limpieza de datos
    compras = filtered.get("compras", pd.DataFrame())
    prod = filtered.get("produccion", pd.DataFrame())

    if not compras.empty:
        compras["fecha_recepcion_ts"] = pd.to_datetime(compras["fecha_recepcion_ts"])
        compras["qty_recibida"] = pd.to_numeric(compras["qty_recibida"], errors="coerce").fillna(0)
        
        if "ref_materia" in compras.columns and "ref_materia_str" not in compras.columns:
            compras["ref_materia_str"] = compras["ref_materia"].astype(str)
    
    if not prod.empty:
        prod["ref_id_str"] = prod["ref_id"].astype(str).str.replace(r'\.0$', '', regex=True)
        prod["ts_fin"] = pd.to_datetime(prod["ts_fin"]) 

    if not refs.empty:
        refs["ref_id_str"] = refs["ref_id"].astype(str)
        refs["peso_neto_kg"] = pd.to_numeric(refs["peso_neto_kg"], errors="coerce").fillna(0)
    
    # Cruce de datos para obtener pesos
    prod_ref = prod.copy()
    if not prod_ref.empty and not refs.empty:
        if "peso_neto_kg" not in prod_ref.columns:
            prod_ref = prod_ref.merge(refs[["ref_id_str", "peso_neto_kg"]], on="ref_id_str", how="left")
        
        prod_ref["kg_piezas_ok"] = prod_ref["piezas_ok"] * prod_ref.get("peso_neto_kg", 0).fillna(0)
    
    if compras.empty and prod.empty:
        st.info("Sin datos en el rango seleccionado.")
        return

    # Seccion 1: Recepcion de materia prima (compras)
    st.markdown("### Recepción de Materia Prima")
    
    mp1, mp2, mp3 = st.columns(3)
    
    kg_recibidos = compras["qty_recibida"].sum() if not compras.empty else 0
    lotes = compras["material_lot_id"].nunique() if not compras.empty else 0
    avg_lote = kg_recibidos / lotes if lotes > 0 else 0

    mp1.metric("Total MP Recibida", f"{kg_recibidos:,.0f} kg")
    mp2.metric("Lotes Recibidos", f"{lotes}")
    mp3.metric("Tamaño medio Lote", f"{avg_lote:,.0f} kg/lote")

    c1, c2 = st.columns(2)
    if not compras.empty:
        top_ref = compras.groupby("ref_materia_str")["qty_recibida"].sum().reset_index()
        fig_top = px.bar(
            top_ref.sort_values("qty_recibida", ascending=False).head(5),
            x="ref_materia_str",
            y="qty_recibida",
            title="Top Materia Prima Recibida (Kg)"
        )
        fig_top.update_xaxes(type='category')
        c1.plotly_chart(fig_top, use_container_width=True)

        compras["fecha"] = compras["fecha_recepcion_ts"].dt.date
        ts = compras.groupby(["fecha", "ref_materia_str"])["qty_recibida"].sum().reset_index()
        fig_ts = px.area(
            ts,
            x="fecha",
            y="qty_recibida",
            color="ref_materia_str",
            title="Cronología de Entradas MP"
        )
        c2.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")

    # Seccion 2: Recepcion de producto terminado (fabrica)
    st.markdown("### Recepción de Producto Terminado")
    
    p1, p2, p3 = st.columns(3)
    
    total_piezas_ok = prod_ref["piezas_ok"].sum() if not prod_ref.empty else 0
    total_kg_piezas = prod_ref["kg_piezas_ok"].sum() if not prod_ref.empty else 0
    refs_activas = prod_ref["ref_id_str"].nunique() if not prod_ref.empty else 0

    p1.metric("Piezas Ingresadas (OK)", f"{total_piezas_ok:,.0f} uds")
    p2.metric("Peso Ingresado (Stock)", f"{total_kg_piezas:,.0f} kg", help="Peso neto de las piezas buenas que han entrado al almacén")
    p3.metric("Referencias distintas", f"{refs_activas}", help="Variedad de productos recibidos en el periodo")

    c3, c4 = st.columns(2)
    
    if not prod_ref.empty:
        piezas_por_ref = prod_ref.groupby("ref_id_str")["piezas_ok"].sum().reset_index()
        fig_piezas = px.bar(
            piezas_por_ref.sort_values("piezas_ok", ascending=False).head(10),
            x="ref_id_str",
            y="piezas_ok",
            title="Top 10 Referencias Ingresadas (Unidades)"
        )
        fig_piezas.update_xaxes(type='category')
        c3.plotly_chart(fig_piezas, use_container_width=True)

        prod_ref["fecha"] = prod_ref["ts_fin"].dt.date
        ts_pt = prod_ref.groupby("fecha")["piezas_ok"].sum().reset_index()

        fig_ts_pt = px.line(
            ts_pt,
            x="fecha",
            y="piezas_ok",
            markers=True,
            title="Cronología de Entradas PT",
            labels={"piezas_ok": "Piezas OK", "fecha": "Fecha"}
        )
        fig_ts_pt.update_traces(line_color="#22c55e")
        c4.plotly_chart(fig_ts_pt, use_container_width=True)

    # Seccion 3: Detalle de lotes
    if not compras.empty:
        st.markdown("#### Detalle de recepciones")
        compras_val = compras[compras["qty_recibida"] > 0].copy()
        if not compras_val.empty:
            fig_scatter = px.scatter(
                compras_val,
                x="fecha_recepcion_ts",
                y="qty_recibida",
                size="qty_recibida",
                color="ref_materia_str",
                title="Mapa de Recepciones: Tamaño de Lote vs Fecha",
                labels={"qty_recibida": "Kg Lote", "fecha_recepcion_ts": "Fecha"}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)