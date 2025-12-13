# Chatbot IA para Dashboard

## Descripción

El dashboard ahora incluye un **asistente de IA flotante** que te ayuda a comprender y analizar los datos visualizados. El chatbot utiliza **Ollama** (ejecución local de modelos de lenguaje) para proporcionar:

- 📊 **Explicaciones de métricas**: Comprende qué significa cada KPI (OEE, UPH, scrap%, etc.)
- 💡 **Insights sobre datos**: Detecta tendencias, anomalías y patrones interesantes
- ❓ **Respuestas a preguntas**: Pregunta lo que quieras sobre los datos actuales
- 🎯 **Sugerencias de acciones**: Recibe recomendaciones basadas en el análisis

## Instalación de Ollama

### 1. Instalar Ollama

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**
Descarga desde: https://ollama.ai/download

### 2. Iniciar Ollama

```bash
ollama serve
```

Esto iniciará Ollama en `http://localhost:11434`

### 3. Descargar un modelo

Recomendamos comenzar con **Llama 3**:

```bash
ollama pull llama3
```

Otros modelos disponibles:
- `llama3` - Recomendado, buen equilibrio entre calidad y velocidad (4.7GB)
- `mistral` - Más rápido pero menos potente (4.1GB)
- `llama2` - Versión anterior, también funcional (3.8GB)
- `codellama` - Especializado en código (3.8GB)

Puedes ver todos los modelos disponibles en: https://ollama.ai/library

### 4. Verificar instalación

```bash
ollama list
```

Deberías ver el modelo descargado listado.

## Uso del Chatbot

### Acceso

1. Ejecuta el dashboard normalmente: `streamlit run app.py`
2. En cualquier página del dashboard, verás un botón **💬** en la esquina superior derecha
3. Haz clic para abrir el panel del chatbot

### Funcionalidades

El chatbot tiene acceso a **todos los datos del dashboard**, incluyendo:

- **Producción**: Piezas OK, scrap, máquinas, referencias, OEE
- **Órdenes**: Planificación, avance de órdenes
- **RRHH**: Horas disponibles, absentismo, productividad laboral
- **Almacén**: Materia prima, producto terminado, movimientos

### Ejemplos de preguntas

**Análisis general:**
- "¿Cuál es el estado general de la producción?"
- "¿Hay algún problema importante que deba atender?"
- "Dame un resumen de los datos actuales"

**Preguntas específicas:**
- "¿Qué máquina tiene el peor rendimiento?"
- "¿Cuál es el porcentaje de scrap de la referencia X?"
- "¿Cómo está el absentismo este mes?"
- "¿Qué referencias tienen más problemas de calidad?"

**Explicaciones:**
- "Explícame qué es el OEE"
- "¿Cómo se calcula el rendimiento?"
- "¿Qué significa UPH?"

**Insights y recomendaciones:**
- "¿Hay tendencias preocupantes en los datos?"
- "¿Qué acciones me recomiendas tomar?"
- "¿Qué debería mejorar primero?"

### Selector de modelo

En el panel del chatbot puedes cambiar entre los modelos de Ollama que tengas instalados. Cada modelo tiene diferentes características:

- **Modelos más grandes** (7B+): Mejores respuestas, más lentos
- **Modelos más pequeños** (3B-4B): Respuestas más rápidas, menos precisas

## Configuración avanzada

### Cambiar el puerto de Ollama

Si Ollama está ejecutándose en un puerto diferente, puedes modificar la configuración en `scripts/dashboard/chatbot.py`:

```python
chatbot = OllamaChatbot(base_url="http://localhost:PUERTO")
```

### Personalizar el modelo por defecto

En `scripts/dashboard/chatbot.py`, línea 14:

```python
def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
```

Cambia `"llama3"` por el modelo que prefieras.

## Solución de problemas

### "Ollama no está disponible"

**Causa:** Ollama no está ejecutándose o no está accesible.

**Solución:**
1. Verifica que Ollama esté instalado: `ollama --version`
2. Inicia Ollama: `ollama serve`
3. Verifica que esté escuchando en el puerto correcto: `curl http://localhost:11434/api/tags`

### "No hay modelos disponibles"

**Causa:** No has descargado ningún modelo.

**Solución:**
```bash
ollama pull llama3
```

### El chatbot es muy lento

**Causa:** El modelo es demasiado grande para tu hardware.

**Solución:**
1. Prueba un modelo más pequeño: `ollama pull mistral`
2. Cambia al modelo más pequeño en el selector del chatbot

### El chatbot da respuestas incorrectas

**Causa:** El modelo necesita más contexto o es limitado.

**Solución:**
1. Reformula la pregunta de manera más específica
2. Prueba con un modelo más potente (llama3 en lugar de mistral)
3. Reinicia la conversación con el botón "🗑️ Limpiar"

## Arquitectura técnica

### Componentes

1. **OllamaChatbot** (`scripts/dashboard/chatbot.py`):
   - Clase que maneja la comunicación con Ollama
   - Construye el contexto con los datos del dashboard
   - Mantiene el historial de conversación

2. **render_chatbot_bubble** (`scripts/dashboard/chatbot.py`):
   - Renderiza la UI del chatbot en Streamlit
   - Gestiona el estado de la conversación
   - Proporciona la interfaz de usuario

3. **Integración en app.py**:
   - El chatbot se renderiza en todas las páginas
   - Tiene acceso a todos los datos cargados

### Flujo de datos

```
Dashboard Data (DataFrames)
        ↓
build_context_prompt() - Resume datos clave
        ↓
User Message + Context
        ↓
Ollama API (localhost:11434)
        ↓
AI Response
        ↓
Display in Chat UI
```

### Privacidad

- **Todos los datos se procesan localmente**: Ollama se ejecuta en tu máquina
- **No hay envío a servicios cloud**: A diferencia de ChatGPT/Claude API
- **Control total**: Tú controlas qué modelos usar y cómo se procesan los datos

## Mejoras futuras

Posibles extensiones del chatbot:

- [ ] Generación de gráficos personalizados bajo demanda
- [ ] Exportación de insights a PDF/Excel
- [ ] Alertas proactivas basadas en anomalías detectadas
- [ ] Integración con notificaciones (email, Slack)
- [ ] Memoria a largo plazo de conversaciones anteriores
- [ ] Fine-tuning del modelo con terminología específica de tu planta

## Soporte

Si tienes problemas o sugerencias, puedes:

1. Revisar los logs de Ollama: `journalctl -u ollama -f` (Linux)
2. Revisar la documentación de Ollama: https://github.com/ollama/ollama
3. Contactar al equipo de desarrollo del dashboard
