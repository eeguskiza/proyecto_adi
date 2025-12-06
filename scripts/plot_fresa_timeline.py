#!/usr/bin/env python3
"""CLI para visualizar la vida de una talladora y sus cambios de fresa (matplotlib).

Usa el dataset `data/processed/fresa_cambios.csv` generado en el notebook.
Permite elegir la talladora y abre un gráfico temporal con las piezas y scrap acumulados.
"""
from pathlib import Path
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).resolve().parents[1] / "data/processed/fresa_cambios.csv"


def load_cambios() -> pd.DataFrame:
    if not DATA_PATH.exists():
        sys.exit(f"No se encuentra el dataset: {DATA_PATH}. Genera primero fresa_cambios.csv desde el notebook.")
    df = pd.read_csv(DATA_PATH, parse_dates=["ts_cambio"])
    # Asegurar orden temporal y tipos amigables
    df = df.sort_values(["machine_id", "ts_cambio"]).reset_index(drop=True)
    df["machine_id"] = df["machine_id"].astype(str)
    df["machine_name"] = df["machine_name"].fillna("")
    df["piezas_hasta_cambio"] = df["piezas_hasta_cambio"].fillna(0)
    df["scrap_hasta_cambio"] = df["scrap_hasta_cambio"].fillna(0)
    df["duracion_cambio_min"] = df["duracion_cambio_min"].fillna(0)
    df["ops_hasta_cambio"] = df["ops_hasta_cambio"].fillna(0)
    return df


def elegir_maquina(df: pd.DataFrame) -> str:
    machines = df[["machine_id", "machine_name"]].drop_duplicates().sort_values("machine_id").reset_index(drop=True)
    print("Talladoras disponibles:")
    for idx, row in machines.iterrows():
        name = f" - {row.machine_name}" if row.machine_name else ""
        print(f"[{idx}] {row.machine_id}{name}")

    prompt = "Elige índice o id (Enter para 0): "
    raw = ""
    if sys.stdin.isatty():
        raw = input(prompt).strip()
    else:
        print("Sin TTY para leer; usando índice 0 por defecto.")
        raw = "0"

    if raw == "":
        raw = "0"

    # Si el usuario pega '48 - Talladora', nos quedamos con el primer token numérico o id
    token = raw.split()[0]

    if token.isdigit() and int(token) < len(machines):
        machine_id = str(machines.loc[int(token), "machine_id"])
    else:
        token = token.strip()
        if token not in set(machines["machine_id"].astype(str)):
            sys.exit(f"Máquina no encontrada: {raw}")
        machine_id = token
    return machine_id


def plot_maquina(df: pd.DataFrame, machine_id: str):
    subset = df[df["machine_id"] == machine_id].copy()
    if subset.empty:
        sys.exit(f"No hay datos para la máquina {machine_id}")

    subset = subset.sort_values("ts_cambio")
    subset["piezas_cumuladas"] = subset["piezas_hasta_cambio"].cumsum()
    subset["scrap_cumulado"] = subset["scrap_hasta_cambio"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(subset["ts_cambio"], subset["piezas_cumuladas"], marker="o", label="Piezas acumuladas")
    ax.plot(subset["ts_cambio"], subset["scrap_cumulado"], marker="s", label="Scrap acumulado")
    ax.set_title(f"Vida de fresa - Máquina {machine_id} ({subset['machine_name'].iloc[0]})")
    ax.set_xlabel("Fecha cambio")
    ax.set_ylabel("Unidades")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Etiquetas al pasar ratón no existen en matplotlib básico, pero añadimos anotaciones básicas
    for _, row in subset.iterrows():
        ax.annotate(
            f"{int(row['piezas_hasta_cambio'])} uds\n{row['duracion_cambio_min']:.0f} min",
            (row["ts_cambio"], row["piezas_cumuladas"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            alpha=0.7,
        )

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualizar timeline de cambios de fresa por talladora")
    parser.add_argument("--machine", "-m", help="ID de talladora o índice mostrado en la lista")
    args = parser.parse_args()

    df = load_cambios()

    if args.machine is not None:
        raw = args.machine.strip()
        machines = df[["machine_id", "machine_name"]].drop_duplicates().sort_values("machine_id").reset_index(drop=True)
        if raw.isdigit() and int(raw) < len(machines):
            machine_id = str(machines.loc[int(raw), "machine_id"])
        else:
            machine_id = raw
    else:
        machine_id = elegir_maquina(df)

    plot_maquina(df, machine_id)


if __name__ == "__main__":
    main()
