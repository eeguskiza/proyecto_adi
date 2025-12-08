import datetime as dt
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .data import get_date_bounds


def build_week_presets(start: dt.date, end: dt.date):
    days_to_sunday = (start.weekday() + 1) % 7  # domingo=6 -> 0
    current_start = start - dt.timedelta(days=days_to_sunday)
    semanas = []
    idx = 1
    while current_start <= end:
        current_end = current_start + dt.timedelta(days=5)  # domingo a viernes
        label = f"Semana {idx} ({current_start} a {current_end})"
        semanas.append((label, current_start, min(current_end, end)))
        current_start = current_start + dt.timedelta(days=7)
        idx += 1
    return semanas


def get_filters(data: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    min_date, max_date = get_date_bounds(data)
    default_range = (min_date, max_date)
    if "filtros" not in st.session_state:
        st.session_state.filtros = {
            "date_range": default_range,
            "planta": [],
            "recurso_oee": "(Todos)",
            "week_label": "(Rango personalizado)",
        }

    week_presets = build_week_presets(min_date, max_date)

    st.sidebar.header("Filtros globales")
    date_range = st.sidebar.date_input(
        "Rango de fechas",
        value=st.session_state.filtros["date_range"],
        min_value=min_date,
        max_value=max_date,
        key="date_range_input",
    )
    if isinstance(date_range, dt.date):
        date_range = (date_range, date_range)
    if not isinstance(date_range, (list, tuple)) or len(date_range) == 0:
        date_range = default_range
    if len(date_range) == 1:
        date_range = (date_range[0], date_range[0])

    week_match = "(Rango personalizado)"
    for label, w_start, w_end in week_presets:
        if date_range[0] == w_start and date_range[1] == w_end:
            week_match = label
            break

    week_options = ["(Rango personalizado)"] + [w[0] for w in week_presets]
    initial_week_label = st.session_state.filtros.get("week_label", "(Rango personalizado)")
    if initial_week_label == "(Rango personalizado)" and week_match != "(Rango personalizado)":
        initial_week_label = week_match
    week_label = st.sidebar.selectbox(
        "Semana (domingo a viernes)",
        week_options,
        index=week_options.index(initial_week_label)
        if initial_week_label in week_options
        else 0,
    )
    if week_label != "(Rango personalizado)":
        selected = next((w for w in week_presets if w[0] == week_label), None)
        if selected:
            date_range = (selected[1], selected[2])
    elif week_match != "(Rango personalizado)":
        week_label = week_match

    plantas = sorted(data["produccion"]["planta"].dropna().unique())
    machines = sorted(data["produccion"]["machine_name"].dropna().unique())

    recurso_oee = st.sidebar.selectbox(
        "Recurso / máquina (OEE)",
        options=["(Todos)"] + machines,
        index=(["(Todos)"] + machines).index(st.session_state.filtros.get("recurso_oee", "(Todos)"))
        if st.session_state.filtros.get("recurso_oee", "(Todos)") in ["(Todos)"] + machines
        else 0,
    )

    st.sidebar.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)

    # Planta como badges según recurso
    plantas_por_maquina = (
        data["produccion"].groupby("machine_name")["planta"].agg(lambda s: sorted(set(s.dropna()))).to_dict()
    )
    if recurso_oee != "(Todos)":
        planta_sel = plantas_por_maquina.get(recurso_oee, [])
    else:
        planta_sel = []
    palette = ["#22c55e", "#0ea5e9", "#f97316", "#a855f7", "#eab308", "#ef4444"]
    badges = []
    plants_to_show = planta_sel if planta_sel else plantas
    for idx, p in enumerate(plants_to_show):
        color = palette[idx % len(palette)]
        badges.append(f"<span style='background:{color};color:#0b0f17;padding:2px 8px;border-radius:12px;margin-right:6px;font-weight:700;'>{p}</span>")
    st.sidebar.markdown(
        "<div style='font-weight:700;margin-bottom:4px;'>Planta</div>" + (" ".join(badges) if badges else "—"),
        unsafe_allow_html=True,
    )

    # Resumen en el rango filtrado
    resumen_mask = (data["produccion"]["ts_ini"].dt.date >= date_range[0]) & (data["produccion"]["ts_ini"].dt.date <= date_range[1])
    resumen_df = data["produccion"].loc[resumen_mask].copy()
    if planta_sel:
        resumen_df = resumen_df[resumen_df["planta"].isin(planta_sel)]
    if recurso_oee != "(Todos)":
        resumen_df = resumen_df[resumen_df["machine_name"] == recurso_oee]
    resumen_df["ref_id_str"] = resumen_df["ref_id_str"].astype(str)
    resumen_df["total_piezas"] = resumen_df[["piezas_ok", "piezas_scrap"]].sum(axis=1)
    refs_clean = resumen_df["ref_id_str"].dropna().astype(str).str.zfill(6)
    refs_in_range = sorted([r for r in refs_clean if r.isdigit()])
    piezas_range = int(resumen_df["total_piezas"].sum())
    st.sidebar.markdown(
        f"<div style='font-weight:700;'>Referencias en rango:</div><div style='font-size:0.95rem;'>{', '.join(refs_in_range[:8])}{'…' if len(refs_in_range)>8 else ''}</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(f"<div style='font-size:1.4rem;font-weight:900; margin-top:6px;'>Piezas en rango: {piezas_range:,}</div>", unsafe_allow_html=True)

    st.sidebar.markdown("<div class='sidebar-grow'></div>", unsafe_allow_html=True)

    filtros = {
        "date_range": date_range,
        "planta": planta_sel,
        "recurso_oee": recurso_oee,
        "week_label": week_label,
    }
    st.session_state.filtros = filtros
    return filtros


def apply_filters(data: Dict[str, pd.DataFrame], filtros: Dict[str, object]) -> Dict[str, pd.DataFrame]:
    start = pd.to_datetime(filtros["date_range"][0])
    end = pd.to_datetime(filtros["date_range"][1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    prod = data["produccion"].copy()
    mask = prod["ts_ini"].between(start, end)
    if filtros.get("planta"):
        mask &= prod["planta"].isin(filtros["planta"])
    if filtros.get("recurso_oee") and filtros.get("recurso_oee") != "(Todos)":
        mask &= prod["machine_name"] == filtros["recurso_oee"]
    prod = prod.loc[mask]

    ordenes = data["ordenes"].copy()
    ordenes_mask = ordenes["fecha_lanzamiento"].between(start, end, inclusive="both")
    if filtros.get("planta"):
        ordenes_mask &= ordenes["planta_inicio"].isin(filtros["planta"])
    if filtros.get("recurso_oee") and filtros.get("recurso_oee") != "(Todos)":
        ordenes_mask &= ordenes["machine_name"] == filtros["recurso_oee"] if "machine_name" in ordenes else True
    ordenes = ordenes.loc[ordenes_mask]

    compras = data["compras"].copy()
    compras_mask = compras["fecha_recepcion_ts"].between(start, end, inclusive="both")
    compras = compras.loc[compras_mask]

    almacen = data["almacen"].copy()
    almacen_mask = almacen["fecha_ts"].between(start, end, inclusive="both")
    almacen = almacen.loc[almacen_mask]

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
