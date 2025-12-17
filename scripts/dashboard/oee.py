from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st


def inject_metric_styles() -> None:
    st.markdown(
        """
        <style>
        /* Sidebar tweaks - mejor responsive */
        [data-testid="stSidebar"] > div:first-child {
            max-width: 360px;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow-y: auto;
            overflow-x: hidden;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            max-width: 100%;
            overflow-wrap: break-word;
        }
        .sidebar-spacer {flex:1;}
        .sidebar-grow {flex:1 1 auto;}
        .sidebar-nav {background:#0f172a; border:1px solid #1f2937; padding:10px; border-radius:12px;}
        .sidebar-nav label {font-weight:700;}
        .metric-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem;}
        .metric-card {background-color: #0e1117; border: 1px solid #222; border-radius: 10px; padding: 0.75rem 0.9rem;}
        .metric-label {font-size: 0.8rem; letter-spacing: 0.05em; text-transform: uppercase; color: #9ca3af;}
        .metric-value {font-size: 1.9rem; font-weight: 800; line-height: 1.2;}
        .metric-sub {font-size: 0.95rem; color: #cbd5e1;}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=60)
def compute_oee(prod: pd.DataFrame, ciclos: pd.DataFrame) -> Dict[str, object]:
    if prod.empty:
        return {
            "availability": np.nan,
            "performance": np.nan,
            "quality": np.nan,
            "oee": np.nan,
            "durations": {"produccion": 0, "preparacion": 0, "incidencia": 0},
            "total_ok": 0,
            "total_scrap": 0,
            "total_piezas": 0,
            "uph_real": np.nan,
        }

    prod = prod.copy()
    prod["estado_oee"] = prod["evento"].str.lower().map(
        {
            "producción": "produccion",
            "produccion": "produccion",
            "preparación": "preparacion",
            "preparacion": "preparacion",
        }
    )
    prod["estado_oee"] = prod["estado_oee"].fillna("incidencia")
    ciclos_ref_machine = ciclos[["ref_id_str", "machine_name", "piezas_hora_teorico"]].drop_duplicates()
    prod = prod.merge(
        ciclos_ref_machine,
        on=["ref_id_str", "machine_name"],
        how="left",
    )
    prod["total_piezas"] = prod["piezas_ok"] + prod["piezas_scrap"]

    # NUEVA ESTRATEGIA: Calcular tiempo de ciclo (segundos/pieza) en lugar de UPH
    # Esto evita el problema de comparar referencias de diferentes complejidades
    prod_prod_hist = prod[(prod["estado_oee"] == "produccion") & (prod["duracion_min"] > 0) & (prod["total_piezas"] > 0)]

    # Calcular tiempo de ciclo real por operación (segundos por pieza)
    prod_prod_hist_ops = prod_prod_hist.copy()
    prod_prod_hist_ops["cycle_time_sec"] = (prod_prod_hist_ops["duracion_min"] * 60) / prod_prod_hist_ops["total_piezas"]

    # Filtrar outliers extremos en tiempo de ciclo
    # Menos de 1 segundo/pieza (3600 UPH) o más de 600 segundos/pieza (6 UPH) son probablemente errores
    prod_prod_hist_ops = prod_prod_hist_ops[
        (prod_prod_hist_ops["cycle_time_sec"] >= 1) &
        (prod_prod_hist_ops["cycle_time_sec"] <= 600)
    ]

    # Calcular percentil 25 del tiempo de ciclo (los mejores desempeños, tiempos más rápidos)
    # Percentil 25 = tiempos rápidos pero alcanzables (no el mínimo absoluto que puede ser un outlier)
    cycle_time_p25 = (
        prod_prod_hist_ops.groupby(["machine_name", "ref_id_str"])["cycle_time_sec"]
        .quantile(0.25)
        .reset_index()
        .rename(columns={"cycle_time_sec": "cycle_time_objetivo"})
    )

    # También calcular mediana como fallback
    cycle_time_median = (
        prod_prod_hist_ops.groupby(["machine_name", "ref_id_str"])["cycle_time_sec"]
        .median()
        .reset_index()
        .rename(columns={"cycle_time_sec": "cycle_time_mediana"})
    )

    # Merge de los tiempos de ciclo calculados
    prod = prod.merge(cycle_time_p25, on=["machine_name", "ref_id_str"], how="left")
    prod = prod.merge(cycle_time_median, on=["machine_name", "ref_id_str"], how="left")

    # Convertir piezas_hora_teorico de ciclos a tiempo de ciclo (si existe)
    prod["cycle_time_from_ciclos"] = np.where(
        prod["piezas_hora_teorico"] > 0,
        3600 / prod["piezas_hora_teorico"],  # Convertir UPH a segundos/pieza
        np.nan
    )

    # Estrategia en cascada para elegir el tiempo de ciclo objetivo:
    # 1. Si existe en ciclos_teoricos.csv → usar ese
    # 2. Si no, usar el percentil 25 del histórico (objetivo ambicioso pero alcanzable)
    # 3. Si no hay suficientes datos, usar la mediana histórica
    # 4. Como último recurso, usar mediana global

    prod["cycle_time_teorico"] = prod["cycle_time_from_ciclos"]
    prod["cycle_time_teorico"] = prod["cycle_time_teorico"].fillna(prod["cycle_time_objetivo"])
    prod["cycle_time_teorico"] = prod["cycle_time_teorico"].fillna(prod["cycle_time_mediana"])

    # Calcular mediana global como último recurso
    if prod["cycle_time_teorico"].isna().any() and len(cycle_time_median) > 0:
        mediana_global = cycle_time_median["cycle_time_mediana"].median()
        if pd.notna(mediana_global) and mediana_global > 0:
            prod["cycle_time_teorico"] = prod["cycle_time_teorico"].fillna(mediana_global)
        else:
            # Último recurso: 45 segundos/pieza (80 UPH)
            prod["cycle_time_teorico"] = prod["cycle_time_teorico"].fillna(45)

    # Convertir el tiempo de ciclo teórico de vuelta a UPH para mantener compatibilidad
    prod["piezas_hora_teorico"] = 3600 / prod["cycle_time_teorico"]

    # Limpiar columnas auxiliares
    prod = prod.drop(columns=["cycle_time_from_ciclos", "cycle_time_objetivo", "cycle_time_mediana", "cycle_time_teorico"], errors='ignore')

    duraciones = prod.groupby("estado_oee")["duracion_min"].sum().to_dict()
    dur_prod = duraciones.get("produccion", 0)
    dur_prep = duraciones.get("preparacion", 0)
    dur_inci = duraciones.get("incidencia", 0)
    tiempo_plan_min = dur_prod + dur_prep + dur_inci
    availability = dur_prod / tiempo_plan_min if tiempo_plan_min > 0 else np.nan

    prod_prod = prod[prod["estado_oee"] == "produccion"].copy()
    prod_prod = prod_prod[prod_prod["piezas_hora_teorico"] > 0]
    prod_prod.loc[:, "ideal_piezas"] = prod_prod["piezas_hora_teorico"] * (prod_prod["duracion_min"] / 60)
    ideal_output = prod_prod["ideal_piezas"].sum()

    # Evita sobrecontar piezas: usamos la última operación con piezas>0 por orden.
    prod_valid = prod_prod[prod_prod["total_piezas"] > 0].sort_values("ts_fin")
    prod_last = prod_valid.groupby("work_order_id").tail(1)

    actual_output = prod_last["total_piezas"].sum()
    performance = actual_output / ideal_output if ideal_output > 0 and actual_output > 0 else np.nan
    uph_real = actual_output / (dur_prod / 60) if dur_prod > 0 else np.nan

    # Calcular UPH teórico medio usado (para info)
    uph_teorico_medio = prod_prod["piezas_hora_teorico"].mean() if len(prod_prod) > 0 else np.nan

    total_ok = int(prod_last["piezas_ok"].sum())
    total_scrap = int(prod_last["piezas_scrap"].sum())
    total_piezas = total_ok + total_scrap
    quality = total_ok / total_piezas if total_piezas > 0 else np.nan

    oee = availability * performance * quality if all(
        pd.notna(val) for val in [availability, performance, quality]
    ) else np.nan

    return {
        "availability": availability,
        "performance": performance,
        "quality": quality,
        "oee": oee,
        "durations": {"produccion": dur_prod, "preparacion": dur_prep, "incidencia": dur_inci},
        "total_ok": total_ok,
        "total_scrap": total_scrap,
        "total_piezas": total_piezas,
        "uph_real": uph_real,
        "uph_teorico": uph_teorico_medio,
        "actual_output": actual_output,
        "ideal_output": ideal_output,
    }


@st.cache_data(show_spinner=False, ttl=60)
def compute_oee_daily(prod: pd.DataFrame, ciclos: pd.DataFrame) -> pd.DataFrame:
    """Calcula OEE día a día para análisis de tendencias."""
    if prod.empty:
        return pd.DataFrame()

    prod = prod.copy()
    prod["estado_oee"] = prod["evento"].str.lower().map(
        {"producción": "produccion", "produccion": "produccion", "preparación": "preparacion", "preparacion": "preparacion"}
    )
    prod["estado_oee"] = prod["estado_oee"].fillna("incidencia")

    # Asegurar que tenemos solo la fecha (sin hora)
    prod["fecha_day"] = pd.to_datetime(prod["ts_ini"].dt.date)

    daily_oee = []
    # Ordenar las fechas únicas para procesamiento ordenado
    fechas_unicas = sorted(prod["fecha_day"].unique())

    for fecha in fechas_unicas:
        prod_day = prod[prod["fecha_day"] == fecha]
        oee_day = compute_oee(prod_day, ciclos)
        daily_oee.append({
            "fecha": fecha,
            "oee": oee_day["oee"],
            "availability": oee_day["availability"],
            "performance": oee_day["performance"],
            "quality": oee_day["quality"],
        })

    result = pd.DataFrame(daily_oee)
    if not result.empty:
        result["fecha"] = pd.to_datetime(result["fecha"])
        result = result.sort_values("fecha")

    return result


def get_kpi_color(value: float, thresholds: Dict[str, float]) -> str:
    """Determina el color del KPI según thresholds (verde/amarillo/rojo)."""
    if pd.isna(value):
        return "#4b5563"  # gris
    if value >= thresholds.get("good", 0.85):
        return "#22c55e"  # verde
    elif value >= thresholds.get("warning", 0.75):
        return "#fbbf24"  # amarillo
    else:
        return "#ef4444"  # rojo


def render_kpi_cards(oee_data: Dict[str, object]) -> None:
    inject_metric_styles()
    availability = oee_data["availability"]
    performance = oee_data["performance"]
    quality = oee_data["quality"]
    oee = oee_data["oee"]
    piezas_total = oee_data.get("total_piezas", 0)
    uph_real = oee_data.get("uph_real", np.nan)

    # Thresholds para cada KPI
    oee_color = get_kpi_color(oee, {"good": 0.75, "warning": 0.60})
    disp_color = get_kpi_color(availability, {"good": 0.85, "warning": 0.75})
    perf_color = get_kpi_color(performance, {"good": 0.90, "warning": 0.80})
    qual_color = get_kpi_color(quality, {"good": 0.97, "warning": 0.95})

    st.markdown(
        """
        <div class="metric-grid">
            <div class="metric-card" style="border-left: 4px solid {oee_color};">
                <div class="metric-label">OEE</div>
                <div class="metric-value" style="color: {oee_color};">{oee_val}</div>
                <div class="metric-sub">Objetivo: ≥75%</div>
            </div>
            <div class="metric-card" style="border-left: 4px solid {disp_color};">
                <div class="metric-label">Disponibilidad</div>
                <div class="metric-value" style="color: {disp_color};">{disp_val}</div>
                <div class="metric-sub">Objetivo: ≥85%</div>
            </div>
            <div class="metric-card" style="border-left: 4px solid {perf_color};">
                <div class="metric-label">Rendimiento</div>
                <div class="metric-value" style="color: {perf_color};">{perf_val}</div>
                <div class="metric-sub">Objetivo: ≥90%</div>
            </div>
            <div class="metric-card" style="border-left: 4px solid {qual_color};">
                <div class="metric-label">Calidad</div>
                <div class="metric-value" style="color: {qual_color};">{qual_val}</div>
                <div class="metric-sub">Objetivo: ≥97%</div>
            </div>
        </div>
        """.format(
            oee_val=f"{oee:.1%}" if pd.notna(oee) else "—",
            disp_val=f"{availability:.1%}" if pd.notna(availability) else "—",
            perf_val=f"{performance:.1%}" if pd.notna(performance) else "—",
            qual_val=f"{quality:.1%}" if pd.notna(quality) else "—",
            oee_color=oee_color,
            disp_color=disp_color,
            perf_color=perf_color,
            qual_color=qual_color,
        ),
        unsafe_allow_html=True,
    )
