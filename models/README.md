# Modelos de Machine Learning

Este directorio contiene los modelos de Machine Learning para el sistema de monitorizacion de planta.

## Estructura

```
models/
├── clustering/          # Modelo de clustering de maquinas
│   ├── train.py        # Script de entrenamiento
│   └── trained_model/  # Modelo entrenado (generado)
├── regression/         # Modelo de prediccion de scrap
│   ├── train.py       # Script de entrenamiento
│   └── trained_model/ # Modelo entrenado (generado)
├── classification/     # Modelo de clasificacion de estado de maquinas
│   ├── train.py       # Script de entrenamiento
│   └── trained_model/ # Modelo entrenado (generado)
└── requirements.txt   # Dependencias ML
```

## Modelos Disponibles

### 1. Clustering de Maquinas

**Objetivo**: Agrupar maquinas con caracteristicas de rendimiento similares.

**Algoritmo**: K-Means

**Features utilizadas**:
- Disponibilidad (% tiempo en produccion)
- Scrap rate (% piezas defectuosas)
- UPH real (unidades por hora)
- Duracion total de produccion

**Salida**: Asignacion de cluster (0, 1, 2...) para cada maquina

**Uso**:
```bash
cd models/clustering
python train.py
```

**Metricas de calidad**:
- Silhouette Score (mayor es mejor, rango [-1, 1])
- Davies-Bouldin Index (menor es mejor)

### 2. Regresion - Prediccion de Scrap

**Objetivo**: Predecir la tasa de scrap esperada para una operacion de produccion.

**Algoritmo**: Random Forest Regressor

**Features utilizadas**:
- Duracion de la operacion
- Maquina (one-hot encoded)
- Frecuencia de referencia del producto
- Hora del dia
- Dia de la semana
- Estado de operacion (produccion, preparacion, incidencia)

**Salida**: Scrap rate predicho (valor entre 0 y 1)

**Uso**:
```bash
cd models/regression
python train.py
```

**Metricas de evaluacion**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

### 3. Clasificacion - Estado de Maquinas

**Objetivo**: Clasificar el estado de salud de las maquinas basandose en metricas de rendimiento.

**Algoritmo**: Random Forest Classifier

**Features utilizadas**:
- Disponibilidad
- Scrap rate
- UPH real
- Duraciones (produccion, preparacion, incidencias)
- Ratios de preparacion e incidencias
- Numero de operaciones

**Clases**:
- `EXCELENTE`: Disponibilidad >= 85%, Scrap <= 2%, UPH >= 100
- `BUENA`: Disponibilidad >= 70%, Scrap <= 5%, UPH >= 60
- `REQUIERE_ATENCION`: Metricas por debajo de objetivos
- `CRITICA`: Disponibilidad < 50% o Scrap > 10%

**Uso**:
```bash
cd models/classification
python train.py
```

**Metricas de evaluacion**:
- Accuracy
- Precision, Recall, F1-Score por clase
- Confusion Matrix

## Entrenamiento

Cada modelo tiene su propio script de entrenamiento que:

1. Carga los datos historicos de produccion
2. Prepara las features necesarias
3. Entrena el modelo
4. Evalua el rendimiento
5. Guarda el modelo entrenado y metadata
6. Genera un reporte detallado

Para entrenar todos los modelos:

```bash
# Clustering
python models/clustering/train.py

# Regresion
python models/regression/train.py

# Clasificacion
python models/classification/train.py
```

## Modelos Entrenados

Cada modelo entrenado se guarda en su carpeta `trained_model/` con:

- `*.pkl`: Modelo serializado con pickle
- `scaler.pkl`: StandardScaler para normalizacion de features
- `features.txt`: Lista de features utilizadas
- `metadata.txt`: Informacion del entrenamiento
- `training_report.txt`: Reporte detallado con metricas

## Despliegue con BentoML

Los modelos estan preparados para ser empaquetados y desplegados con BentoML.

Para empaquetar un modelo:

```python
import bentoml
import pickle

# Cargar modelo
with open("trained_model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Guardar en BentoML
bentoml.sklearn.save_model(
    "machine_clustering",
    model,
    metadata={
        "trained_at": "2025-12-17",
        "accuracy": 0.95
    }
)
```

## Requisitos

Instalar dependencias ML:

```bash
pip install -r models/requirements.txt
```

## Notas

- Los modelos se entrenan con datos historicos completos
- Para predicciones en tiempo real, los modelos deben cargarse en el dashboard
- Los modelos se pueden reentrenar periodicamente con datos actualizados
- Las metricas de calidad se registran en los reportes de entrenamiento
