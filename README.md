# Dashboard de planta (Streamlit)

Aplicación multipágina para ver rendimiento, disponibilidad y calidad de la planta con datos de producción, almacén, RRHH y compras.

## Cómo ejecutarlo
1. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Lanza el dashboard desde la raíz:
   ```bash
   streamlit run app.py
   ```
   (opcional: `--server.port 8501`)

## Qué ofrece cada menú
- **Cuadro de mando general**: KPIs OEE (OEE, disponibilidad, rendimiento, calidad), distribución de tiempos, calidad OK/scrap, rendimiento real vs. ideal, top incidencias.
- **Producción**: tabla de operaciones, agregados por máquina/referencia/turno, heatmap de scrap y distribución, prueba del modelo de scrap (BentoML).
- **Almacén MP**: kg y lotes recibidos, serie temporal por referencia, consumo y stock teórico.
- **RRHH**: horas netas y ausencias por mes, productividad piezas/hora-hombre.
- **Modelos IA / BentoML**: formulario de scrap y módulo de cambios de fresa (riesgo y distribución de piezas entre cambios).

## Estructura de scripts
La lógica está dividida por módulos en `scripts/`. Consulta la descripción completa aquí: [scripts/README.md](scripts/README.md).
