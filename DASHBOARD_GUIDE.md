# 📖 Guía del Dashboard

Esta guía te explica qué hace cada menú del dashboard y cómo usarlo.

---

## 📋 Los 8 Menús

1. [Cuadro de Mando General](#1-cuadro-de-mando-general) - OEE y análisis de rendimiento
2. [Producción](#2-producción) - Volumen, scrap, órdenes
3. [Almacén MP](#3-almacén-mp) - Entradas de material y producto
4. [RRHH](#4-rrhh) - Horas, absentismo, productividad
5. [Clustering ML](#5-clustering-ml) - Agrupa máquinas (interactivo)
6. [ML - Clustering](#6-ml---clustering) - Agrupa con modelo pre-entrenado
7. [ML - Regresión Scrap](#7-ml---regresión-scrap) - Predice desperdicio
8. [ML - Clasificación Estado](#8-ml---clasificación-estado) - Estado de salud de máquinas

---

## 1. Cuadro de Mando General

**Para qué:** Ver el OEE global y sus componentes, identificar pérdidas.

### Métricas Clave

- **OEE**: Eficiencia global (objetivo: > 75%)
- **Disponibilidad**: % tiempo produciendo (objetivo: > 85%)
- **Rendimiento**: Velocidad real vs teórica (objetivo: > 90%)
- **Calidad**: % piezas buenas (objetivo: > 98%)

`OEE = Disponibilidad × Rendimiento × Calidad`

### Gráficos Principales

**Evolución OEE:**
- Línea temporal del OEE diario
- Componentes (disponibilidad, rendimiento, calidad) por separado
- Detecta días problemáticos de un vistazo

**Cascada de Pérdidas:**
- Empieza en 100%, va restando pérdidas
- Muestra dónde se va la eficiencia

**Pareto de Incidencias:**
- Top causas de paradas ordenadas por impacto
- Línea del 80% → enfócate en lo que está antes de esa línea

**Heatmap de Disponibilidad:**
- Máquinas (filas) × Días (columnas)
- Color verde = bien, rojo = mal
- Identifica patrones rápidamente

**Por Turno:**
- Compara mañana, tarde, noche
- Replica las buenas prácticas del mejor turno

### Cómo Usarlo

- **OEE bajo + disponibilidad baja** → Hay muchas paradas
- **OEE bajo + rendimiento bajo** → Máquinas lentas
- **OEE bajo + calidad baja** → Mucho scrap

---

## 2. Producción

**Para qué:** Ver qué se fabricó, cuánto scrap hay, seguir órdenes.

### Métricas Clave

- **Piezas OK**: Total producidas
- **Scrap %**: Porcentaje de defectos
- **UPH Real**: Unidades por hora
- **Órdenes**: Número de OFs en curso

### Gráficos Principales

**Volumen Diario:**
- Barras de piezas OK + scrap por día
- Línea de scrap % diario

**Top Referencias:**
- Productos más fabricados
- Identifica el mix de producción

**Scrap por Máquina/Referencia:**
- Barras con las peores máquinas/productos
- Prioriza mejoras de calidad

**Tabla de Órdenes:**
- Progreso de cada OF
- Detecta retrasos y problemas

### Cómo Usarlo

- **Día con scrap alto** → Investiga qué pasó ese día
- **Máquina con UPH bajo** → Candidata para mejora
- **Producto con scrap alto** → Revisa especificaciones

---

## 3. Almacén MP

**Para qué:** Controlar entradas de materia prima y producto terminado.

### Métricas Clave

**Materia Prima:**
- Kg recibidos
- Número de lotes
- Tamaño medio de lote

**Producto Terminado:**
- Piezas ingresadas al almacén
- Kg de stock

### Gráficos Principales

**Top Materiales:**
- Referencias de MP más recibidas

**Cronología de Entradas:**
- Serie temporal de recepciones
- Detecta irregularidades en suministro

**Mapa de Recepciones:**
- Scatter plot: fecha × tamaño de lote
- Lotes muy pequeños → ineficiencia
- Lotes muy grandes → riesgo de obsolescencia

### Cómo Usarlo

- Verifica que hay MP suficiente para producir
- Correlaciona entradas PT con demanda
- Detecta problemas con proveedores

---

## 4. RRHH

**Para qué:** Analizar disponibilidad de personal, absentismo y productividad.

### Métricas Clave

- **Horas Netas**: Horas realmente trabajadas
- **Absentismo**: Horas perdidas por enfermedad, accidente, permisos
- **Tasa Absentismo**: % de horas perdidas
- **Saturación**: % de horas usadas en producción (objetivo: ~85%)

### Gráficos Principales

**Cascada de Disponibilidad:**
- Teóricas → ajustes → absentismo → Netas
- Identifica la mayor fuente de pérdida

**Evolución Absentismo:**
- Barras apiladas por mes (enfermedad, accidente, permisos)

**Productividad:**
- Piezas producidas vs piezas/hora
- Mide eficiencia laboral

**Saturación:**
- Horas disponibles vs horas usadas en producción
- < 70% → sobrecapacidad
- > 95% → riesgo de burnout

### Cómo Usarlo

- Absentismo alto → revisa condiciones laborales
- Saturación baja → redistribuye o reduce plantilla
- Saturación alta → contrata o planifica horas extra

---

## 5. Clustering ML

**Para qué:** Agrupar máquinas con comportamiento similar. Entrena el modelo en vivo con tus datos.

### Qué Hace

Usa K-Means para agrupar máquinas según:
- Disponibilidad
- Scrap rate
- UPH real
- Duración de producción

### Visualizaciones

**Gráfico 3D:**
- Rota para ver clusters desde diferentes ángulos
- Cada color es un grupo

**Gráficos 2D:**
- Disponibilidad vs Scrap
- UPH vs Scrap

**Tabla de Clusters:**
- Cuántas máquinas hay en cada grupo
- Métricas promedio del grupo

### Cómo Interpretar

- **Cluster verde (alta disponibilidad, bajo scrap)** → Best performers, replica sus prácticas
- **Cluster rojo (baja disponibilidad)** → Problemas de paradas, mantenimiento urgente
- **Cluster naranja (alto scrap)** → Problemas de calidad, revisa ajustes

### Ajustes

Usa el slider "Número de clusters" para agrupar más o menos.
- 2-3 clusters → grupos generales
- 5-6 clusters → más detalle

---

## 6. ML - Clustering

**Para qué:** Lo mismo que "Clustering ML" pero usa el modelo pre-entrenado.

### Diferencias

| Clustering ML | ML - Clustering |
|---------------|-----------------|
| Entrena nuevo cada vez | Usa modelo pre-entrenado |
| Más lento | Más rápido |
| Clusters ajustables | Clusters fijos |

### Cuándo Usar Cada Uno

- **Clustering ML**: Para explorar, experimentar con diferentes números de clusters
- **ML - Clustering**: Para monitorización diaria, comparar con históricos

---

## 7. ML - Regresión Scrap

**Para qué:** Predecir cuánto scrap tendrá una operación antes de hacerla.

### Qué Predice

**Entrada:** Duración, máquina, producto, hora, día
**Salida:** Scrap rate esperado (0-100%)

### Métricas del Modelo

- **MAE**: Error promedio (ej: 2% significa ±2% error)
- **R²**: Qué tan bueno es el modelo (> 0.7 es bueno)

### Gráficos Principales

**Real vs Predicho:**
- Puntos cerca de la línea diagonal → buenas predicciones
- Puntos alejados → errores del modelo

**Errores por Máquina:**
- Top 10 máquinas donde el modelo falla más
- Puede indicar comportamiento impredecible

**Por Hora del Día:**
- Detecta si el scrap aumenta en ciertas horas (ej: turno noche)

**Feature Importance:**
- Qué factores influyen más en el scrap

### Cómo Usarlo

- Planifica inspecciones más frecuentes en operaciones de alto riesgo
- Asigna las mejores máquinas a productos críticos
- Identifica máquinas inestables (alto error de predicción)

---

## 8. ML - Clasificación Estado

**Para qué:** Clasificar máquinas en 4 estados de salud.

### Los 4 Estados

| Estado | Criterios | Acción |
|--------|-----------|--------|
| 🟢 **EXCELENTE** | Disp ≥ 85%, Scrap ≤ 2%, UPH ≥ 100 | Mantener, replicar |
| 🔵 **BUENA** | Disp ≥ 70%, Scrap ≤ 5%, UPH ≥ 60 | Monitorizar |
| 🟠 **REQUIERE ATENCIÓN** | Por debajo de objetivos | Planificar mejora |
| 🔴 **CRÍTICA** | Disp < 50% o Scrap > 10% | Intervención urgente |

### Visualizaciones

**Distribución de Estados:**
- Cuántas máquinas hay en cada estado (tabla + pie chart)

**Por Estado (4 pestañas):**
- Lista de máquinas en cada categoría
- Métricas detalladas

**Timeline por Máquina:**
- Selecciona una máquina
- Ve cómo evoluciona su estado semana a semana
- Detecta degradación progresiva

**Scatter Plot:**
- Disponibilidad vs Scrap coloreado por estado
- Verifica que los clusters tienen sentido

### Cómo Usarlo

**Dashboard de Mantenimiento:**
1. Abre pestaña "CRÍTICA"
2. Crea tickets para esas máquinas
3. Monitoriza después de intervenir

**Priorización:**
- Cuenta máquinas por estado
- Invierte primero en las críticas

**Predicción:**
- Usa timeline para detectar máquinas que empeoran
- Actúa antes de que lleguen a crítico

---

## 🎨 Filtros (Aplican a Todas las Páginas)

En la barra lateral puedes filtrar por:

- **Fechas**: Rango de análisis
- **Máquina**: Ver una específica o todas
- **Cliente**: Filtrar por cliente
- **Referencia**: Filtrar por producto
- **Turno**: Mañana, tarde, noche

**Tip:** Empieza con todos en "(Todos)" para ver el panorama general, luego filtra para profundizar.

---

## 💬 Chatbot IA

Todas las páginas tienen un botón 💬 (si habilitas el chatbot en la barra lateral).

### Qué Puede Hacer

- Explicarte qué significa una métrica
- Interpretar si un valor es bueno o malo
- Sugerir acciones
- Responder preguntas específicas

### Ejemplos de Preguntas

- "¿Por qué el OEE es bajo?"
- "¿Qué máquina tiene más problemas?"
- "Explícame el gráfico de Pareto"
- "¿Hay tendencias en los datos?"

---

## 💡 Tips de Uso

### Si Eres Nuevo

1. Empieza en "Cuadro de Mando General"
2. Si ves algo raro, ve a "Producción" o "RRHH" para detalles
3. Usa el chatbot si no entiendes algo

### Si Ya Conoces el Dashboard

- Combina filtros para análisis complejos
- Usa los modelos ML para planificación proactiva
- Exporta gráficos clave para reportes

### Rutinas Recomendadas

**Diario (5 min):**
- Cuadro de Mando → ver OEE de ayer
- ML - Clasificación → revisar máquinas críticas

**Semanal (20 min):**
- Todos los menús con filtro de última semana
- Identificar tendencias

**Mensual (1 hora):**
- Revisión completa
- Comparar vs mes anterior
- Re-entrenar modelos ML si es necesario

---

**¿Dudas?** Haz clic en 💬 y pregunta al chatbot, o vuelve al [README.md](README.md) principal.
