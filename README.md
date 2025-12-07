# Dataset integrado de operaciones 2025

## 1) Propósito

Consolidar datos de planta, materiales y capacidad para analizar **rendimiento**, **calidad**, **paradas** y **flujo a stock**.

## 2) Granularidad

Una fila = **registro de operación** en una máquina y periodo concreto. Incluye eventos de **Producción** e **Incidencia**.

## 3) Origen y claves

* **Histórico de operaciones** → núcleo (claves: `work_order_id`, tiempos, `machine_id`, `ref_id`).
* **Ingeniería** → maestro de pieza (`ref_id_str` → `familia`, `peso_neto_kg`).
* **Work orders** → planificación (`work_order_id`, `qty_plan`, `due_date`, `cliente`).
* **Movimientos almacén (IN)** → stock diario por referencia (`ref_id_str`, `fecha`).
* **Compras (lotes de materia)** → info de lote si existe (`material_lot_id`).
* **RRHH mensual** → capacidad (`año_mes`, horas netas/ajustadas).

**Claves de unión**

* `ref_id_str` = `ref_id` normalizado a 6 dígitos.
* `work_order_id` entre operaciones y OF.
* `material_lot_id` si no es nulo.
* `fecha` (día) para cruzar con entradas a almacén; `año_mes` para RRHH.

## 4) Convenciones de calidad

* `material_lot_id` “0” → **NaN**.
* `PENDING` en fechas o planta → **NaT/NaN**.
* `ref_id` numérico o con decimales se normaliza a string de 6 dígitos.
* Solo hay **movimientos IN**; no existen OUT en este dataset.
* Posibles solapes de tiempos en el origen; se usan tal cual.

## 5) Diccionario de datos

### Identificación de orden y operación

* **work_order_id** · *string* · OF del ERP/MES. Ej: `24/0674`.
* **op_id** · *string* · Código de operación. Ej: `RECTIFICADO`, `SOLDADURA-RE`.
* **op_text** · *string* · Texto descriptivo de la operación.
* **machine_id** · *string/int* · Identificador de máquina/centro. Ej: `1001`, `515`.
* **machine_name** · *string* · Nombre de máquina. Ej: `Linea Luk 1`, `SOLDADORA`.
* **planta** · *string* · Planta física. Ej: `Abadiño`, `Zaldibar`.

### Pieza / referencia

* **ref_id_str** · *string* · Referencia de pieza normalizada a 6 dígitos. Ej: `081000`, `124203`.
* **familia** · *string* · Familia de producto desde Ingeniería. Ej: `CORONA DE ARRANQUE`.
* **peso_neto_kg** · *float (kg)* · Peso unitario neto de pieza.

### Tiempos

* **ts_ini** · *datetime* · Inicio de registro.
* **ts_fin** · *datetime* · Fin de registro.
* **fecha** · *date* · Día de `ts_fin` para agregados diarios.
* **duracion_min** · *float (min)* · Duración del registro.

### Evento y calidad

* **evento** · *string* · `Producción` o `Incidencia`.
* **tipo_incidencia** · *string/NaN* · Categoría de la incidencia. Ej: `CALENTAR HIDRÁU`, `LIMPIEZA`.
* **piezas_ok** · *int* · Unidades buenas en el registro.
* **piezas_scrap** · *int* · Unidades rechazadas en el registro.

### Planificación OF

* **qty_plan** · *float/int* · Cantidad planificada de la OF.
* **cliente** · *string/NaN* · Cliente asociado a la OF (si disponible).
* **fecha_lanzamiento** · *datetime/NaT* · Lanzamiento de OF.
* **due_date** · *datetime/NaT* · Fecha objetivo.

### Almacén (solo entradas)

* **qty_in_almacen_dia** · *float/int (uds)* · Unidades de `ref_id_str` entradas a stock el **día** `fecha`.

### Compras / lote de materia prima

* **material_lot_id** · *string/NaN* · Lote de materia. `NaN` si desconocido.
* **ref_materia_str** · *string/NaN* · Referencia de materia normalizada.
* **qty_recibida** · *float (kg u otra UDM)* · Cantidad recibida.
* **udm** · *string* · Unidad de medida original, p. ej. `kg`.
* **peso_bruto** · *float* · Peso bruto del lote si aplica.
* **uds** · *float/int* · Unidades en el lote si aplica.
* **fecha_recepcion_ts** · *datetime/NaT* · Fecha de recepción del lote.

### RRHH mensual

* **año_mes** · *string `YYYY-MM`* · Periodo.
* **horas_teoricas** · *float (h)* · Horas calendario del periodo.
* **reduccion_tco** · *float (h)* · Reducciones TCO aplicadas.
* **horas_ajustadas** · *float (h)* · Teóricas menos reducciones.
* **horas_enfermedad** · *float (h)* · Bajas por enfermedad.
* **horas_accidente** · *float (h)* · Bajas por accidente.
* **horas_permiso** · *float (h)* · Permisos.
* **horas_netas** · *float (h)* · Horas útiles netas del periodo.

### Métricas derivadas

* **throughput_uph** · *float (uds/h)*
  Fórmula: `60 * piezas_ok / duracion_min` si `duracion_min > 0`.
* **scrap_rate** · *float [0,1]*
  Fórmula: `piezas_scrap / (piezas_ok + piezas_scrap)` si total > 0.
* **downtime_min** · *float (min)*
  Igual a `duracion_min` **solo** cuando `evento = Incidencia`; 0 en producción.
* **consumo_materia_kg** · *float (kg)*
  Fórmula: `peso_neto_kg * piezas_ok`. Aproximación.
* **lead_time_al_almacen_dias** · *float (días)*
  Días desde `end_date = ts_fin.normalize()` hasta la **primera** entrada IN disponible para la misma `ref_id_str` en o después de esa fecha. Si no hay IN posterior, `NaN`.

## 6) Ejemplos de interpretación

* Un registro con `evento=Incidencia`, `duracion_min=35` y `tipo_incidencia=LIMPIEZA` suma **35 min** a `downtime_min`. No produce piezas.
* Si en el mismo día `fecha` hay `qty_in_almacen_dia=360` para `ref_id_str=124203`, indica 360 uds que han **entrado en stock** ese día.
* `consumo_materia_kg=0.774 * 1 000 = 774 kg` para 1 000 piezas buenas de una corona de 0.774 kg.

## 7) Limitaciones conocidas

* Sin **salidas** de almacén. Solo entradas.
* `lead_time_al_almacen_dias` no es tiempo a cliente, solo a stock.
* Ciclo teórico no disponible; usar `throughput_uph` histórico como proxy.
* Operario no incluido para evitar PII.

## Dashboard Streamlit

Panel interactivo multi-página (`app.py`) que usa los datos `data/raw/` para exploración rápida.

### Qué muestra
- **Cuadro de mando general:** KPIs (piezas OK/scrap, tasa scrap, órdenes, kg MP, horas) y series de producción/scrap, entradas de MP, productividad.
- **Producción:** tabla por orden+operación, agregados por máquina/referencia/turno, heatmap scrap%, histograma, y formulario para probar el modelo de scrap (BentoML).
- **Almacén MP:** entradas por referencia/tiempo, kg y lotes, consumo teórico y stock teórico aproximado.
- **RRHH:** tabla mensual, horas netas, ausencias y productividad piezas OK/horas.
- **Modelos IA / BentoML:** formulario de scrap y sección de cambio de fresa con riesgos y distribución de piezas entre cambios.

### Cómo lanzarlo
1. Instala dependencias:
   ```
   pip install -r requirements.txt
   ```
2. Ejecuta Streamlit desde la raíz del repo:
   ```
   streamlit run app.py
   ```
   Opcional: `--server.port 8501` para fijar puerto.

### Notas de uso
- Los filtros globales del sidebar (rango de fechas, planta, familia, máquina, referencia) se aplican en todas las páginas.
- El endpoint del modelo de scrap se puede configurar en el formulario (por defecto `http://localhost:3000/predict`).
