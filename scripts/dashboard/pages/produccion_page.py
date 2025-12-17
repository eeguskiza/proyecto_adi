import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def page_produccion(filtered: dict) -> None:
    st.subheader("Producción")
    prod = filtered["produccion"]
    if prod.empty:
        st.info("Sin datos en el rango seleccionado.")
        return

    prod = prod.copy()
    prod["kg_ok"] = prod["piezas_ok"] * prod["peso_neto_kg"].fillna(0)

    # Evita sobrecontar piezas al sumar todas las operaciones: tomamos la última operación con piezas>0 por OF.
    prod_valid = prod[prod["total_piezas"] > 0].sort_values("ts_fin")
    of_final = prod_valid.groupby("work_order_id").tail(1)

    total_ok = int(of_final["piezas_ok"].sum())
    total_scrap = int(of_final["piezas_scrap"].sum())
    total_piezas = total_ok + total_scrap
    scrap_rate_total = total_scrap / total_piezas if total_piezas > 0 else np.nan

    dur_total_min = prod[prod["evento"].str.lower() == "producción"]["duracion_min"].sum()
    uph_real = total_ok / (dur_total_min / 60) if dur_total_min > 0 else np.nan

    ordenes_activas = prod["work_order_id"].nunique()
    ops_activas = prod["op_id"].nunique()
    kg_ok = (of_final["piezas_ok"] * of_final["peso_neto_kg"].fillna(0)).sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Piezas OK", f"{total_ok:,}")
    k2.metric("Scrap%", f"{scrap_rate_total:.1%}" if pd.notna(scrap_rate_total) else "—", delta=f"{total_scrap:,} uds scrap")
    k3.metric("UPH real", f"{uph_real:,.1f}" if pd.notna(uph_real) else "—", help="Piezas OK / hora en el rango")
    k4.metric("Órdenes y ops.", f"{ordenes_activas} OF", delta=f"{ops_activas} operaciones")
    st.caption(f"Kg OK estimados: {kg_ok:,.1f} kg")

    st.markdown("#### Volumen en el rango")
    serie = (
        prod.groupby(prod["ts_ini"].dt.date)
        .agg(piezas_ok=("piezas_ok", "sum"), piezas_scrap=("piezas_scrap", "sum"), duracion_min=("duracion_min", "sum"))
        .reset_index()
    )
    serie = serie.rename(columns={serie.columns[0]: "fecha"})
    serie["fecha"] = serie["fecha"].astype(str)
    serie["scrap_rate"] = np.where(
        (serie["piezas_ok"] + serie["piezas_scrap"]) > 0,
        serie["piezas_scrap"] / (serie["piezas_ok"] + serie["piezas_scrap"]),
        np.nan,
    )
    c1, c2 = st.columns((2, 1))
    fig_vol = px.bar(
        serie,
        x="fecha",
        y=["piezas_ok", "piezas_scrap"],
        labels={"value": "Piezas", "variable": "Tipo"},
        title="Piezas fabricadas por día (OK vs scrap)",
    )
    c1.plotly_chart(fig_vol, width='stretch')
    fig_scrap = px.line(serie, x="fecha", y="scrap_rate", markers=True, title="Scrap% diario")
    fig_scrap.update_yaxes(tickformat=".0%")
    c2.plotly_chart(fig_scrap, width='stretch')

    st.markdown("#### Mezcla y rendimiento de fabricación")
    mix_ref = (
        prod.groupby(["ref_id_str", "familia"])
        .agg(piezas_ok=("piezas_ok", "sum"), piezas_scrap=("piezas_scrap", "sum"))
        .reset_index()
        .sort_values("piezas_ok", ascending=False)
        .head(12)
    )
    mix_ref["scrap_rate"] = np.where(
        mix_ref["piezas_ok"] + mix_ref["piezas_scrap"] > 0, mix_ref["piezas_scrap"] / (mix_ref["piezas_ok"] + mix_ref["piezas_scrap"]), np.nan
    )
    mix_ref["ref_label"] = mix_ref["ref_id_str"].fillna("") + " (" + mix_ref["familia"].fillna("sin familia") + ")"

    perf_machine = (
        prod.groupby("machine_name")
        .agg(piezas_ok=("piezas_ok", "sum"), piezas_scrap=("piezas_scrap", "sum"), duracion_min=("duracion_min", "sum"))
        .reset_index()
    )
    perf_machine["uph_real"] = np.where(perf_machine["duracion_min"] > 0, 60 * perf_machine["piezas_ok"] / perf_machine["duracion_min"], np.nan)
    perf_machine["scrap_rate"] = np.where(
        perf_machine["piezas_ok"] + perf_machine["piezas_scrap"] > 0,
        perf_machine["piezas_scrap"] / (perf_machine["piezas_ok"] + perf_machine["piezas_scrap"]),
        np.nan,
    )

    c3, c4 = st.columns(2)
    if not mix_ref.empty:
        fig_mix = px.bar(
            mix_ref,
            x="piezas_ok",
            y="ref_label",
            orientation="h",
            title="Top referencias por piezas OK",
            labels={"piezas_ok": "Piezas OK", "ref_label": "Referencia"},
        )
        fig_mix.update_layout(yaxis={"categoryorder": "total ascending"})
        c3.plotly_chart(fig_mix, width='stretch')
    if not perf_machine.empty:
        fig_uph = px.bar(
            perf_machine.sort_values("uph_real", ascending=False),
            x="machine_name",
            y="uph_real",
            title="Ritmo real (UPH) por máquina",
            labels={"uph_real": "Piezas/hora", "machine_name": "Máquina"},
        )
        c4.plotly_chart(fig_uph, width='stretch')

    st.markdown("#### Scrap por recurso / referencia")
    c5, c6 = st.columns(2)
    scrap_machine = perf_machine.sort_values("scrap_rate", ascending=False).head(10)
    if not scrap_machine.empty:
        fig_scrap_m = px.bar(
            scrap_machine,
            x="scrap_rate",
            y="machine_name",
            orientation="h",
            title="Scrap% por máquina",
            labels={"scrap_rate": "Scrap%", "machine_name": "Máquina"},
        )
        fig_scrap_m.update_xaxes(tickformat=".0%")
        c5.plotly_chart(fig_scrap_m, width='stretch')

    scrap_ref = (
        prod.groupby("ref_id_str")
        .agg(piezas_scrap=("piezas_scrap", "sum"), total=("total_piezas", "sum"))
        .reset_index()
    )
    scrap_ref["scrap_rate"] = np.where(scrap_ref["total"] > 0, scrap_ref["piezas_scrap"] / scrap_ref["total"], np.nan)
    scrap_ref = scrap_ref.sort_values("scrap_rate", ascending=False).head(12)
    if not scrap_ref.empty:
        fig_scrap_r = px.bar(
            scrap_ref,
            x="scrap_rate",
            y="ref_id_str",
            orientation="h",
            title="Scrap% por referencia",
            labels={"scrap_rate": "Scrap%", "ref_id_str": "Referencia"},
        )
        fig_scrap_r.update_xaxes(tickformat=".0%")
        c6.plotly_chart(fig_scrap_r, width='stretch')

    st.markdown("#### Avance de órdenes del rango")
    ordenes = (
        prod.groupby(["work_order_id", "cliente", "ref_id_str"])
        .agg(
            qty_plan=("qty_plan", "max"),
            piezas_ok=("piezas_ok", "sum"),
            piezas_scrap=("piezas_scrap", "sum"),
            ts_ini=("ts_ini", "min"),
            ts_fin=("ts_fin", "max"),
        )
        .reset_index()
    )
    ordenes["qty_plan"] = pd.to_numeric(ordenes["qty_plan"], errors="coerce")
    ordenes["progreso"] = np.where(ordenes["qty_plan"] > 0, ordenes["piezas_ok"] / ordenes["qty_plan"], np.nan)
    ordenes["scrap_rate"] = np.where(
        ordenes["piezas_ok"] + ordenes["piezas_scrap"] > 0,
        ordenes["piezas_scrap"] / (ordenes["piezas_ok"] + ordenes["piezas_scrap"]),
        np.nan,
    )
    ordenes = ordenes.sort_values("ts_ini", ascending=False)
    cols_ordenes = ["work_order_id", "cliente", "ref_id_str", "qty_plan", "piezas_ok", "piezas_scrap", "progreso", "scrap_rate", "ts_ini", "ts_fin"]
    st.dataframe(ordenes[cols_ordenes].head(50), width='stretch', hide_index=True)
    st.caption("Top 50 órdenes del rango por fecha de inicio.")

    st.markdown("#### Detalle de operaciones en rango")
    cols_tabla = [
        "work_order_id",
        "op_id",
        "ref_id_str",
        "familia",
        "cliente",
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
    st.dataframe(prod[cols_tabla], width='stretch', hide_index=True)
