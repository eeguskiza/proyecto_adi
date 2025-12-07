import datetime as dt
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


st.set_page_config(
    page_title="Dashboard planta",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data() -> Dict[str, pd.DataFrame]:
    refs = pd.read_csv("data/raw/datos_referencias.csv")
    refs["ref_id_str"] = refs["ref_id"].astype(str).str.zfill(6)

    ordenes = pd.read_csv("data/raw/ordenes_header.csv", parse_dates=["fecha_lanzamiento", "due_date"])
    ordenes["ref_id_str"] = ordenes["ref_id"].astype(str).str.zfill(6)
    ordenes = ordenes.rename(columns={"familia": "familia_of"})

    prod = pd.read_csv("data/raw/produccion_operaciones.csv", parse_dates=["ts_ini", "ts_fin"])
    prod["ref_id_str"] = prod["ref_id"].astype(str).str.zfill(6)
    prod["piezas_ok"] = prod["piezas_ok"].fillna(0)
    prod["piezas_scrap"] = prod["piezas_scrap"].fillna(0)
    prod["duracion_min"] = prod["duracion_min"].fillna(0)
    prod = prod.merge(refs[["ref_id_str", "familia", "peso_neto_kg"]], on="ref_id_str", how="left")
    prod = prod.merge(
        ordenes[["work_order_id", "ref_id_str", "cliente", "qty_plan", "planta_inicio", "familia_of"]],
        on=["work_order_id", "ref_id_str"],
        how="left",
    )
    prod["familia"] = prod["familia"].combine_first(prod["familia_of"])
    prod["total_piezas"] = prod["piezas_ok"] + prod["piezas_scrap"]
    prod["scrap_rate"] = np.where(prod["total_piezas"] > 0, prod["piezas_scrap"] / prod["total_piezas"], np.nan)
    prod["throughput_uph"] = np.where(
        prod["duracion_min"] > 0, 60 * prod["piezas_ok"] / prod["duracion_min"], np.nan
    )
    prod["fecha"] = prod["ts_ini"].dt.date
    prod["turno"] = prod["ts_ini"].apply(map_turno)

    compras = pd.read_csv("data/raw/compras_lotes.csv", parse_dates=["fecha_recepcion_ts"])
    compras["ref_materia_str"] = compras["ref_materia"].astype(str).str.zfill(6)

    almacen = pd.read_csv("data/raw/almacen_movimientos.csv", parse_dates=["fecha_ts"])
    almacen["item_ref_id_str"] = almacen["item_ref_id"].astype(str).str.zfill(6)

    rrhh = pd.read_csv("data/raw/rrhh_turno.csv")
    rrhh.columns = rrhh.columns.str.strip()
    rrhh["año_mes"] = pd.to_datetime(rrhh["año_mes"])

    fresa = pd.read_csv("data/processed/fresa_cambios.csv", parse_dates=["ts_cambio"])

    return {
        "refs": refs,
        "ordenes": ordenes,
        "produccion": prod,
        "compras": compras,
        "almacen": almacen,
        "rrhh": rrhh,
        "fresa": fresa,
    }


def map_turno(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "desconocido"
    hour = ts.hour
    if 6 <= hour < 14:
        return "mañana"
    if 14 <= hour < 22:
        return "tarde"
    return "noche"


def get_date_bounds(data: Dict[str, pd.DataFrame]) -> Tuple[dt.date, dt.date]:
    fechas = []
    for col, df in [
        ("ts_ini", data["produccion"]),
        ("ts_fin", data["produccion"]),
        ("fecha_ts", data["almacen"]),
        ("fecha_recepcion_ts", data["compras"]),
    ]:
        fechas.append(df[col].dropna().min())
        fechas.append(df[col].dropna().max())
    fechas = [f for f in fechas if pd.notna(f)]
    min_date, max_date = min(fechas).date(), max(fechas).date()
    return min_date, max_date


def get_filters(data: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    min_date, max_date = get_date_bounds(data)
    default_range = (min_date, max_date)
    if "filtros" not in st.session_state:
        st.session_state.filtros = {
            "date_range": default_range,
            "planta": [],
            "planta_inicio": [],
            "familia": [],
            "machine": [],
            "referencia": [],
        }

    st.sidebar.header("Filtros globales")
    date_range = st.sidebar.date_input(
        "Rango de fechas",
        value=st.session_state.filtros["date_range"],
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, dt.date):
        date_range = (date_range, date_range)
    # Si el widget devuelve un solo valor o nada, forzamos a rango válido
    if not isinstance(date_range, (list, tuple)) or len(date_range) == 0:
        date_range = default_range
    if len(date_range) == 1:
        date_range = (date_range[0], date_range[0])

    plantas = sorted(data["produccion"]["planta"].dropna().unique())
    plantas_inicio = sorted(data["ordenes"]["planta_inicio"].dropna().unique())
    familias = sorted(data["produccion"]["familia"].dropna().unique())
    machines = sorted(data["produccion"]["machine_name"].dropna().unique())
    referencias = sorted(data["produccion"]["ref_id_str"].dropna().unique())

    planta_sel = st.sidebar.multiselect("Planta", plantas, default=st.session_state.filtros["planta"])
    planta_inicio_sel = st.sidebar.multiselect(
        "Planta inicio OF", plantas_inicio, default=st.session_state.filtros["planta_inicio"]
    )
    familia_sel = st.sidebar.multiselect("Familia", familias, default=st.session_state.filtros["familia"])
    machine_sel = st.sidebar.multiselect("Máquina", machines, default=st.session_state.filtros["machine"])
    ref_sel = st.sidebar.multiselect("Referencia", referencias, default=st.session_state.filtros["referencia"])

    filtros = {
        "date_range": date_range,
        "planta": planta_sel,
        "planta_inicio": planta_inicio_sel,
        "familia": familia_sel,
        "machine": machine_sel,
        "referencia": ref_sel,
    }
    st.session_state.filtros = filtros
    return filtros


def apply_filters(data: Dict[str, pd.DataFrame], filtros: Dict[str, object]) -> Dict[str, pd.DataFrame]:
    start = pd.to_datetime(filtros["date_range"][0])
    end = pd.to_datetime(filtros["date_range"][1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    prod = data["produccion"].copy()
    mask = prod["ts_ini"].between(start, end)
    if filtros["planta"]:
        mask &= prod["planta"].isin(filtros["planta"])
    if filtros["planta_inicio"]:
        mask &= prod["planta_inicio"].isin(filtros["planta_inicio"])
    if filtros["familia"]:
        mask &= prod["familia"].isin(filtros["familia"])
    if filtros["machine"]:
        mask &= prod["machine_name"].isin(filtros["machine"])
    if filtros["referencia"]:
        mask &= prod["ref_id_str"].isin(filtros["referencia"])
    prod = prod.loc[mask]

    ordenes = data["ordenes"].copy()
    ordenes_mask = ordenes["fecha_lanzamiento"].between(start, end, inclusive="both")
    if filtros["planta_inicio"]:
        ordenes_mask &= ordenes["planta_inicio"].isin(filtros["planta_inicio"])
    if filtros["familia"]:
        ordenes_mask &= ordenes["familia_of"].isin(filtros["familia"])
    if filtros["referencia"]:
        ordenes_mask &= ordenes["ref_id_str"].isin(filtros["referencia"])
    ordenes = ordenes.loc[ordenes_mask]

    compras = data["compras"].copy()
    compras_mask = compras["fecha_recepcion_ts"].between(start, end, inclusive="both")
    compras = compras.loc[compras_mask]
    if filtros["referencia"]:
        compras = compras[compras["ref_materia_str"].isin(filtros["referencia"])]

    almacen = data["almacen"].copy()
    almacen_mask = almacen["fecha_ts"].between(start, end, inclusive="both")
    almacen = almacen.loc[almacen_mask]
    if filtros["referencia"]:
        almacen = almacen[almacen["item_ref_id_str"].isin(filtros["referencia"])]

    rrhh = data["rrhh"].copy()
    inicio_mes = start.to_period("M")
    fin_mes = end.to_period("M")
    rrhh["periodo"] = rrhh["año_mes"].dt.to_period("M")
    rrhh = rrhh[(rrhh["periodo"] >= inicio_mes) & (rrhh["periodo"] <= fin_mes)]

    fresa = data["fresa"].copy()
    fresa_mask = fresa["ts_cambio"].between(start, end, inclusive="both")
    fresa = fresa.loc[fresa_mask]

    return {
        "produccion": prod,
        "ordenes": ordenes,
        "compras": compras,
        "almacen": almacen,
        "rrhh": rrhh,
        "fresa": fresa,
    }


def render_kpi_cards(filtered: Dict[str, pd.DataFrame]) -> None:
    prod = filtered["produccion"]
    ordenes = filtered["ordenes"]
    compras = filtered["compras"]
    rrhh = filtered["rrhh"]

    total_ok = int(prod["piezas_ok"].sum())
    total_scrap = int(prod["piezas_scrap"].sum())
    scrap_rate = total_scrap / (total_ok + total_scrap) if (total_ok + total_scrap) > 0 else 0
    num_ordenes = ordenes["work_order_id"].nunique()
    kg_recibidos = compras["qty_recibida"].sum()
    horas_teoricas = rrhh["horas_teoricas"].sum() if "horas_teoricas" in rrhh else 0
    horas_netas = rrhh["horas_netas"].sum() if "horas_netas" in rrhh else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Piezas OK", f"{total_ok:,}")
    c2.metric("Piezas scrap", f"{total_scrap:,}")
    c3.metric("Tasa scrap", f"{scrap_rate:.1%}")
    c4.metric("Órdenes lanzadas", f"{num_ordenes}")
    c5.metric("Kg MP recibida", f"{kg_recibidos:,.0f}")
    c6.metric("Horas teóricas / netas", f"{horas_teoricas:,.0f} / {horas_netas:,.0f}")


def page_dashboard(filtered: Dict[str, pd.DataFrame]) -> None:
    st.subheader("Cuadro de mando general")
    render_kpi_cards(filtered)

    prod = filtered["produccion"]
    compras = filtered["compras"]
    rrhh = filtered["rrhh"]

    c1, c2 = st.columns(2)
    if not prod.empty:
        prod_daily = prod.copy()
        prod_daily["fecha"] = prod_daily["ts_ini"].dt.date
        prod_daily = prod_daily.groupby("fecha")[["piezas_ok", "piezas_scrap"]].sum().reset_index()
        fig_prod = px.line(prod_daily, x="fecha", y=["piezas_ok", "piezas_scrap"], markers=True)
        fig_prod.update_layout(title="Producción vs scrap en el tiempo", legend_title="")
        c1.plotly_chart(fig_prod, width="stretch")

        scrap_ref = prod.groupby("ref_id_str").agg(piezas_scrap=("piezas_scrap", "sum"), total=("total_piezas", "sum"))
        scrap_ref["scrap_rate"] = np.where(scrap_ref["total"] > 0, scrap_ref["piezas_scrap"] / scrap_ref["total"], 0)
        scrap_ref = scrap_ref.sort_values("piezas_scrap", ascending=False).head(10).reset_index()
        fig_scrap_ref = px.bar(scrap_ref, x="ref_id_str", y="piezas_scrap", color="scrap_rate", title="Top referencias por scrap")
        fig_scrap_ref.update_layout(coloraxis_colorbar_title="% scrap")
        c2.plotly_chart(fig_scrap_ref, width="stretch")

    c3, c4 = st.columns(2)
    if not compras.empty:
        compras_ref = compras.groupby("ref_materia_str")["qty_recibida"].sum().reset_index()
        fig_compras = px.bar(compras_ref.sort_values("qty_recibida", ascending=False).head(10), x="ref_materia_str", y="qty_recibida", title="Entradas de MP por referencia")
        c3.plotly_chart(fig_compras, width="stretch")

        compras["fecha"] = compras["fecha_recepcion_ts"].dt.date
        compras_ts = compras.groupby(["fecha", "ref_materia_str"])["qty_recibida"].sum().reset_index()
        fig_comp_ts = px.area(compras_ts, x="fecha", y="qty_recibida", color="ref_materia_str", title="Serie de kg recibidos por referencia")
        c4.plotly_chart(fig_comp_ts, width="stretch")

    if not prod.empty and not rrhh.empty:
        prod_month = prod.copy()
        prod_month["año_mes"] = prod_month["ts_ini"].dt.to_period("M").dt.to_timestamp()
        prod_month = prod_month.groupby("año_mes")["piezas_ok"].sum().reset_index()
        rrhh_month = rrhh.copy()
        rrhh_month = rrhh_month.rename(columns={"año_mes": "año_mes_ts"})
        rrhh_month["año_mes"] = rrhh_month["año_mes_ts"]
        prod_rrhh = prod_month.merge(rrhh_month[["año_mes", "horas_netas"]], on="año_mes", how="left")
        prod_rrhh["productividad"] = np.where(prod_rrhh["horas_netas"] > 0, prod_rrhh["piezas_ok"] / prod_rrhh["horas_netas"], np.nan)
        fig_prod_rrhh = px.bar(prod_rrhh, x="año_mes", y="productividad", title="Productividad (piezas OK / horas netas)")
        st.plotly_chart(fig_prod_rrhh, width="stretch")


def page_produccion(filtered: Dict[str, pd.DataFrame]) -> None:
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


def call_bentoml_scrap(endpoint: str, payload: Dict[str, object]) -> Dict[str, object]:
    try:
        resp = requests.post(endpoint, json=payload, timeout=5)
        if resp.ok:
            return resp.json()
        return {"error": f"Status {resp.status_code}"}
    except Exception as exc:  # pragma: no cover - network call
        return {"error": str(exc)}


def page_almacen(filtered: Dict[str, pd.DataFrame], refs: pd.DataFrame) -> None:
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


def page_rrhh(filtered: Dict[str, pd.DataFrame], prod: pd.DataFrame) -> None:
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


def page_modelos(filtered: Dict[str, pd.DataFrame], prod: pd.DataFrame) -> None:
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


def calcular_riesgo_cambio(ratio: float) -> str:
    if ratio < 0.7:
        return "verde"
    if ratio < 0.9:
        return "amarillo"
    return "rojo"


def main() -> None:
    data = load_data()
    filtros = get_filters(data)
    filtered = apply_filters(data, filtros)

    page = st.sidebar.radio(
        "Páginas",
        [
            "Cuadro de mando general",
            "Producción",
            "Almacén MP",
            "RRHH",
            "Modelos IA / BentoML",
        ],
    )

    if page == "Cuadro de mando general":
        page_dashboard(filtered)
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
