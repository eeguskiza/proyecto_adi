# Modelos de Machine Learning - Guía Completa

Este directorio contiene tres tipos de modelos de machine learning para análisis de producción industrial:

## 📊 Tipos de Modelos

### 1. **Clasificación de Estado de Máquinas** (`classification/`)
- **Objetivo**: Clasificar máquinas en diferentes estados de rendimiento
- **Clases**: EXCELENTE, BUENA, REQUIERE_ATENCION, CRITICA
- **Algoritmo**: Random Forest Classifier
- **Features**: disponibilidad, scrap_rate, uph_real, dur_prod, prep_ratio, inci_ratio

### 2. **Regresión de Scrap** (`regression/`)
- **Objetivo**: Predecir el porcentaje de scrap de operaciones
- **Target**: scrap_rate
- **Algoritmo**: Random Forest Regressor
- **Features**: duracion_min, hora_del_dia, dia_semana, ref_frequency, estados (one-hot), máquinas (one-hot)

### 3. **Clustering de Máquinas** (`clustering/`)
- **Objetivo**: Agrupar máquinas con características similares
- **Algoritmo**: K-Means
- **Features**: disponibilidad, scrap_rate, uph_real, dur_prod
- **Métricas**: Silhouette Score, Davies-Bouldin Index

## 🚀 Uso desde el Dashboard

### Modo 1: Modelo Pre-entrenado
1. Selecciona "📊 Modelo Pre-entrenado"
2. El modelo cargará automáticamente
3. Visualiza las predicciones en los datos actuales

### Modo 2: Entrenamiento Interactivo
1. Selecciona "🔧 Entrenar Modelo Interactivo"
2. **Configura Features**: Elige las características a usar
3. **Ajusta Hiperparámetros**:
   - **Clasificación/Regresión**: n_estimators, max_depth, min_samples_split
   - **Clustering**: n_clusters, max_iter, n_init
4. **Analiza Resultados**:
   - **Clasificación**: Accuracy, Matriz de Confusión, F1-Score
   - **Regresión**: MAE, RMSE, R²
   - **Clustering**: Silhouette Score, Davies-Bouldin Index
5. **Compara** con el modelo pre-entrenado
6. **Guarda** el modelo:
   - 💾 **Guardar como Pickle**: Para uso local
   - 📦 **Empaquetar con BentoML**: Para servir como API

## 📦 BentoML: Servir Modelos como API

### ¿Qué es BentoML?

BentoML permite empaquetar modelos de ML y servirlos como APIs REST de forma sencilla y escalable.

### Instalación

```bash
pip install bentoml
```

### Empaquetar un Modelo

#### Desde el Dashboard
1. Entrena un modelo interactivo
2. Click en "📦 Empaquetar con BentoML"
3. El modelo se guardará en BentoML automáticamente

#### Desde la Terminal

```bash
# Clasificación
python models/classification/save_to_bentoml.py

# Regresión
python models/regression/save_to_bentoml.py

# Clustering
python models/clustering/save_to_bentoml.py
```

### Servir un Modelo como API

```bash
# Clasificación
bentoml serve models/classification/service.py:svc

# Regresión
bentoml serve models/regression/service.py:svc

# Clustering
bentoml serve models/clustering/service.py:svc
```

El servicio estará disponible en: `http://localhost:3000`

### Probar la API

#### Usando curl

**Clasificación:**
```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "disponibilidad": 0.85,
    "scrap_rate": 0.02,
    "uph_real": 120,
    "dur_prod": 480,
    "prep_ratio": 0.1,
    "inci_ratio": 0.05
  }'
```

**Regresión:**
```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duracion_min": 60,
    "hora_del_dia": 14,
    "dia_semana": 2,
    "ref_frequency": 100,
    "estado_produccion": 1
  }'
```

**Clustering:**
```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "disponibilidad": 0.85,
    "scrap_rate": 0.02,
    "uph_real": 120,
    "dur_prod": 480
  }'
```

#### Usando Python

```python
import requests

# Clasificación
response = requests.post(
    "http://localhost:3000/predict",
    json={
        "disponibilidad": 0.85,
        "scrap_rate": 0.02,
        "uph_real": 120,
        "dur_prod": 480,
        "prep_ratio": 0.1,
        "inci_ratio": 0.05
    }
)
print(response.json())
# Output: {"estado_predicho": "EXCELENTE", "probabilidades": {...}, "confianza": 0.95}

# Predicciones en batch
response = requests.post(
    "http://localhost:3000/batch_predict",
    json=[
        {"disponibilidad": 0.85, "scrap_rate": 0.02, ...},
        {"disponibilidad": 0.60, "scrap_rate": 0.10, ...}
    ]
)
```

### Listar Modelos Guardados

```bash
bentoml models list
```

### Eliminar un Modelo

```bash
bentoml models delete <model_name>:<version>
```

## 🏋️ Entrenamiento desde Terminal

### 1. Entrenar Modelos Pre-entrenados

```bash
# Clasificación
python models/classification/train.py

# Regresión
python models/regression/train.py

# Clustering
python models/clustering/train.py
```

Los modelos se guardarán en `models/<tipo>/trained_model/`

## 📈 Métricas e Interpretación

### Clasificación
- **Accuracy**: % de predicciones correctas. >85% es excelente
- **Precision**: De las predicciones positivas, cuántas son correctas
- **Recall**: De los casos positivos reales, cuántos se detectaron
- **F1-Score**: Media armónica de precision y recall
- **Interpretación automática**: El dashboard analiza cada clase y proporciona recomendaciones

### Regresión
- **MAE (Error Absoluto Medio)**: Error promedio en las predicciones
- **RMSE**: Penaliza más los errores grandes
- **R²**: % de varianza explicada por el modelo. >0.7 es bueno
- **Interpretación**: Muestra errores por máquina y hora del día

### Clustering
- **Silhouette Score**: [-1, 1]. >0.5 indica buenos clusters
- **Davies-Bouldin Index**: Menor es mejor. <1.0 es bueno
- **Interpretación automática**:
  - 🌟 Best Performers: Alta disponibilidad + bajo scrap
  - ⚠️ Baja Disponibilidad: Requiere mantenimiento
  - 🔴 Alto Scrap: Revisar procesos
  - 🎯 Outliers: Casos especiales

## 🔧 Configuración Avanzada

### Hiperparámetros Recomendados

**Random Forest (Clasificación/Regresión):**
- `n_estimators`: 100-300 (más árboles = mejor precisión pero más lento)
- `max_depth`: 10-20 (menor = menos overfitting)
- `min_samples_split`: 5-10 (mayor = menos overfitting)

**K-Means (Clustering):**
- `n_clusters`: Usar análisis de codo y silhouette
- `max_iter`: 300 es suficiente
- `n_init`: 10-20 (más inicializaciones = mejor resultado)

### Evitar Overfitting

1. **Divide datos correctamente**: 80% train, 20% test
2. **Usa validación cruzada** para evaluar
3. **Compara métricas train vs test**: Diferencia >10% indica overfitting
4. **Reduce max_depth** o aumenta min_samples_split
5. **Selecciona features relevantes**: Menos features = menos overfitting

## 📚 Estructura de Archivos

```
models/
├── ML_README.md                    # Este archivo
├── classification/
│   ├── train.py                    # Script de entrenamiento
│   ├── service.py                  # Servicio BentoML
│   ├── save_to_bentoml.py         # Empaquetar en BentoML
│   └── trained_model/
│       ├── random_forest_classifier.pkl
│       ├── scaler.pkl
│       ├── features.txt
│       └── classes.txt
├── regression/
│   ├── train.py
│   ├── service.py
│   ├── save_to_bentoml.py
│   └── trained_model/
│       ├── random_forest_model.pkl
│       ├── scaler.pkl
│       └── features.txt
└── clustering/
    ├── train.py
    ├── service.py
    ├── save_to_bentoml.py
    └── trained_model/
        ├── kmeans_model.pkl
        ├── scaler.pkl
        └── features.txt
```

## 🐳 Despliegue con Docker (Opcional)

### Construir imagen

```bash
bentoml build
```

### Contenedorizar

```bash
bentoml containerize <bento_tag>
```

### Ejecutar contenedor

```bash
docker run -p 3000:3000 <image_name>
```

## 🔍 Troubleshooting

### Error: "Modelo no encontrado"
- Asegúrate de entrenar el modelo primero: `python models/<tipo>/train.py`

### Error: "BentoML no está instalado"
- Ejecuta: `pip install bentoml pydantic`

### Predicciones incorrectas
- Verifica que las features de entrada coincidan con las del entrenamiento
- Revisa que los valores estén normalizados (el scaler se aplica automáticamente)

### Bajo rendimiento del modelo
- Recolecta más datos de entrenamiento
- Ajusta hiperparámetros usando modo interactivo
- Prueba seleccionar diferentes features

## 📞 Soporte

Para más información sobre:
- **Streamlit**: Ver `DASHBOARD_GUIDE.md`
- **BentoML**: https://docs.bentoml.org/
- **Scikit-learn**: https://scikit-learn.org/stable/documentation.html
