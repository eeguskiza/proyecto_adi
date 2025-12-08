def calcular_riesgo_cambio(ratio: float) -> str:
    if ratio < 0.7:
        return "verde"
    if ratio < 0.9:
        return "amarillo"
    return "rojo"
