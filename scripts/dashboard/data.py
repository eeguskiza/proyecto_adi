import datetime as dt
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .logger import DashboardLogger


def map_turno(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "desconocido"
    hour = ts.hour
    if 6 <= hour < 14:
        return "mañana"
    if 14 <= hour < 22:
        return "tarde"
    return "noche"


@st.cache_data(ttl=300, show_spinner="Cargando datos...")
def load_data() -> Dict[str, pd.DataFrame]:
    logger = DashboardLogger.get_logger()
    logger.info("Iniciando carga de datos del dashboard...")

    total_steps = 8
    current_step = 0

    # 1. Referencias
    start_time = time.time()
    refs = pd.read_csv("data/raw/datos_referencias.csv")
    refs["ref_id_str"] = refs["ref_id"].astype(str).str.zfill(6)
    elapsed = (time.time() - start_time) * 1000
    current_step += 1
    DashboardLogger.log_progress("Carga de datos", current_step, total_steps)
    DashboardLogger.log_data_load("referencias", len(refs), elapsed)

    # 2. Ciclos teóricos
    start_time = time.time()
    try:
        ciclos = pd.read_csv("data/raw/ciclos_teoricos.csv")
        if "ref_id_str" not in ciclos and "ref_id" in ciclos:
            ciclos["ref_id_str"] = ciclos["ref_id"].astype(str).str.zfill(6)
        if "machine_name" not in ciclos:
            ciclos["machine_name"] = ""
        if "piezas_hora_teorico" not in ciclos:
            if "ciclo_teorico_seg" in ciclos:
                ciclos["piezas_hora_teorico"] = np.where(
                    ciclos["ciclo_teorico_seg"] > 0, 3600 / ciclos["ciclo_teorico_seg"], 0
                )
            else:
                ciclos["piezas_hora_teorico"] = 0
    except FileNotFoundError:
        logger.warning("Archivo ciclos_teoricos.csv no encontrado, generando datos por defecto")
        ciclos = pd.DataFrame(
            {
                "ref_id_str": refs["ref_id_str"].unique(),
                "machine_name": "",
                "piezas_hora_teorico": 0,
            }
        )
    elapsed = (time.time() - start_time) * 1000
    current_step += 1
    DashboardLogger.log_progress("Carga de datos", current_step, total_steps)
    DashboardLogger.log_data_load("ciclos teóricos", len(ciclos), elapsed)

    # 3. Órdenes
    start_time = time.time()
    ordenes = pd.read_csv("data/raw/ordenes_header.csv", parse_dates=["fecha_lanzamiento", "due_date"])
    ordenes["ref_id_str"] = ordenes["ref_id"].astype(str).str.zfill(6)
    ordenes = ordenes.rename(columns={"familia": "familia_of"})
    elapsed = (time.time() - start_time) * 1000
    current_step += 1
    DashboardLogger.log_progress("Carga de datos", current_step, total_steps)
    DashboardLogger.log_data_load("órdenes", len(ordenes), elapsed)

    # 4. Producción (el más grande)
    start_time = time.time()
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
    elapsed = (time.time() - start_time) * 1000
    current_step += 1
    DashboardLogger.log_progress("Carga de datos", current_step, total_steps)
    DashboardLogger.log_data_load("producción", len(prod), elapsed)

    if ciclos.empty or ciclos["machine_name"].eq("").all():
        combos = prod[["ref_id_str", "machine_name"]].dropna().drop_duplicates()
        ciclos = combos.copy()
        ciclos["piezas_hora_teorico"] = 0

    # 5. Compras
    start_time = time.time()
    compras = pd.read_csv("data/raw/compras_lotes.csv", parse_dates=["fecha_recepcion_ts"])
    compras["ref_materia_str"] = compras["ref_materia"].astype(str).str.zfill(6)
    elapsed = (time.time() - start_time) * 1000
    current_step += 1
    DashboardLogger.log_progress("Carga de datos", current_step, total_steps)
    DashboardLogger.log_data_load("compras", len(compras), elapsed)

    # 6. Almacén
    start_time = time.time()
    almacen = pd.read_csv("data/raw/almacen_movimientos.csv", parse_dates=["fecha_ts"])
    almacen["item_ref_id_str"] = almacen["item_ref_id"].astype(str).str.zfill(6)
    elapsed = (time.time() - start_time) * 1000
    current_step += 1
    DashboardLogger.log_progress("Carga de datos", current_step, total_steps)
    DashboardLogger.log_data_load("almacén", len(almacen), elapsed)

    # 7. RRHH
    start_time = time.time()
    rrhh = pd.read_csv("data/raw/rrhh_turno.csv")
    rrhh.columns = rrhh.columns.str.strip()
    rrhh["año_mes"] = pd.to_datetime(rrhh["año_mes"])
    elapsed = (time.time() - start_time) * 1000
    current_step += 1
    DashboardLogger.log_progress("Carga de datos", current_step, total_steps)
    DashboardLogger.log_data_load("RRHH", len(rrhh), elapsed)

    # 8. Fresa
    start_time = time.time()
    fresa = pd.read_csv("data/processed/fresa_cambios.csv", parse_dates=["ts_cambio"])
    elapsed = (time.time() - start_time) * 1000
    current_step += 1
    DashboardLogger.log_progress("Carga de datos", current_step, total_steps)
    DashboardLogger.log_data_load("fresa", len(fresa), elapsed)

    logger.info("Carga de datos completada exitosamente")

    return {
        "refs": refs,
        "ordenes": ordenes,
        "produccion": prod,
        "compras": compras,
        "almacen": almacen,
        "rrhh": rrhh,
        "fresa": fresa,
        "ciclos": ciclos,
    }


@st.cache_data(show_spinner=False)
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
