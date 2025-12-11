import pandas as pd
import plotly.express as px
import streamlit as st

def page_almacen(filtered: dict, refs: pd.DataFrame) -> None:
    st.subheader("Almacén de materia prima")
    
    # 1. Carga de datos
    compras = filtered.get("compras", pd.DataFrame())
    almacen = filtered.get("almacen", pd.DataFrame())
    prod = filtered.get("produccion", pd.DataFrame())

    # Correciones
    if not compras.empty:
        compras["fecha_recepcion_ts"] = pd.to_datetime(compras["fecha_recepcion_ts"])
        compras["qty_recibida"] = pd.to_numeric(compras["qty_recibida"], errors="coerce").fillna(0)
        if "ref_materia" in compras.columns and "ref_materia_str" not in compras.columns:
            compras["ref_materia_str"] = compras["ref_materia"].astype(str)
    
    if not prod.empty:
        prod["ref_id_str"] = prod["ref_id"].astype(str).str.replace(r'\.0$', '', regex=True)

    if not refs.empty:
        refs["ref_id_str"] = refs["ref_id"].astype(str)
        refs["peso_neto_kg"] = pd.to_numeric(refs["peso_neto_kg"], errors="coerce").fillna(0)

    if compras.empty and almacen.empty:
        st.info("Sin datos de almacén en el rango seleccionado.")
        return

    # KPI SUPERIORES 
    col1, col2, col3 = st.columns(3)
    
    kg_recibidos = compras["qty_recibida"].sum() if not compras.empty else 0
    col1.metric("Kg recibidos", f"{kg_recibidos:,.0f}")
    
    lotes = compras["material_lot_id"].nunique() if not compras.empty else 0
    col2.metric("Lotes recibidos", f"{lotes}")

    prod_ref = prod.copy()
    if "peso_neto_kg" not in prod_ref.columns and not refs.empty:
        prod_ref = prod_ref.merge(refs[["ref_id_str", "peso_neto_kg"]], on="ref_id_str", how="left")
    
    prod_ref["kg_consumidos"] = (prod_ref["piezas_ok"] + prod_ref["piezas_scrap"]) * prod_ref.get("peso_neto_kg", 0).fillna(0)
    kg_consumidos = prod_ref["kg_consumidos"].sum()
    col3.metric("Kg consumo teórico", f"{kg_consumidos:,.0f}")

    # Graficos de compras
    c1, c2 = st.columns(2)
    if not compras.empty:
        top_ref = compras.groupby("ref_materia_str")["qty_recibida"].sum().reset_index()
        fig_top = px.bar(top_ref.sort_values("qty_recibida", ascending=False).head(5), x="ref_materia_str", y="qty_recibida", title="Top referencias por kg recibidos")
        c1.plotly_chart(fig_top, use_container_width=True)

        compras["fecha"] = compras["fecha_recepcion_ts"].dt.date
        ts = compras.groupby(["fecha", "ref_materia_str"])["qty_recibida"].sum().reset_index()
        fig_ts = px.area(ts, x="fecha", y="qty_recibida", color="ref_materia_str", title="Serie de kg recibidos")
        c2.plotly_chart(fig_ts, use_container_width=True)

    # Grafico original de consumo y tabla
    if not prod_ref.empty:
        cons_ref = prod_ref.groupby("ref_id_str")["kg_consumidos"].sum().reset_index()
        # Aqui ordenamos por consumo para el grafico de barras original
        fig_cons = px.bar(cons_ref.sort_values("kg_consumidos", ascending=False).head(10), x="ref_id_str", y="kg_consumidos", title="MP más consumida (teórico)")
        # Forzamos eje categórico también aquí por consistencia
        fig_cons.update_xaxes(type='category')
        st.plotly_chart(fig_cons, use_container_width=True)

        if not compras.empty:
            entradas = compras.groupby("ref_materia_str")["qty_recibida"].sum()
            stock_teorico = pd.DataFrame({"ref_id_str": cons_ref["ref_id_str"]})
            stock_teorico["kg_recibidos"] = stock_teorico["ref_id_str"].map(entradas).fillna(0)
            stock_teorico = stock_teorico.merge(cons_ref, on="ref_id_str", how="left")
            stock_teorico["stock_teorico"] = stock_teorico["kg_recibidos"] - stock_teorico["kg_consumidos"]
            st.dataframe(stock_teorico, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Eficiencia y Estrategia de Compras")

    # Graficos de scrap y scatter
    c3, c4 = st.columns(2)

    # Gráfico de scrap
    if not prod_ref.empty:
        prod_ref["kg_scrap"] = prod_ref["piezas_scrap"] * prod_ref.get("peso_neto_kg", 0).fillna(0)
        scrap_by_ref = prod_ref.groupby("ref_id_str")["kg_scrap"].sum().reset_index()
        # Ordenamos descendente para ver los peores casos primero
        scrap_by_ref = scrap_by_ref[scrap_by_ref["kg_scrap"] > 0].sort_values("kg_scrap", ascending=False).head(10)
        
        if not scrap_by_ref.empty:
            fig_scrap = px.bar(
                scrap_by_ref, 
                x="ref_id_str", 
                y="kg_scrap",   
                title="Top 10: Desperdicio de material (Kg Scrap)",
                labels={"kg_scrap": "Kg Desperdiciados", "ref_id_str": "Referencia PT"}
            )
            fig_scrap.update_xaxes(type='category')
            c3.plotly_chart(fig_scrap, use_container_width=True)

    # Grafico de scatter
    if not compras.empty:
        compras_validas = compras[compras["qty_recibida"] > 0].copy()
        if not compras_validas.empty:
            fig_scatter = px.scatter(
                compras_validas,
                x="fecha_recepcion_ts",
                y="qty_recibida",
                size="qty_recibida",
                color="ref_materia_str",
                title="Distribución de Recepciones (Tamaño de lote)",
                labels={"qty_recibida": "Kg del Lote", "fecha_recepcion_ts": "Fecha Recepción"}
            )
            c4.plotly_chart(fig_scatter, use_container_width=True)