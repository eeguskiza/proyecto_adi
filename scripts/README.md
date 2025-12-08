# Scripts del dashboard

Estructura modular de la app Streamlit.

- `dashboard/data.py`: carga de datos (producción, órdenes, compras, almacén, RRHH, cambios de fresa) y utilidades de fechas/turnos.
- `dashboard/filters.py`: construcción de filtros del sidebar (fechas, semanas, recurso, badges de planta) y aplicación de filtros a los dataframes.
- `dashboard/oee.py`: estilos comunes, cálculo de OEE y renderizado de KPIs.
- `dashboard/models.py`: llamada al endpoint BentoML de scrap.
- `dashboard/utils.py`: utilidades compartidas (p. ej. riesgo cambio de fresa).
- `dashboard/pages/dashboard_page.py`: página “Cuadro de mando general” (OEE, disponibilidad, rendimiento, calidad, incidencias).
- `dashboard/pages/produccion_page.py`: página de Producción (tabla, agregados, heatmap, histograma, prueba de modelo).
- `dashboard/pages/almacen_page.py`: página de Almacén MP (entradas, kg, consumo/stock teórico).
- `dashboard/pages/rrhh_page.py`: página de RRHH (horas, ausencias, productividad).
- `dashboard/pages/modelos_page.py`: página de Modelos IA / BentoML (formulario de scrap, módulo de cambio de fresa).
