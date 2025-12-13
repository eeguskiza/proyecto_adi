# Dashboard de planta (Streamlit)

Aplicación multipágina para ver rendimiento, disponibilidad y calidad de la planta con datos de producción, almacén, RRHH y compras.

**NUEVO:** Incluye un **asistente de IA flotante** que te ayuda a comprender los datos en tiempo real. [Ver documentación del chatbot](CHATBOT_README.md)

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
- **Chatbot IA**: Asistente flotante disponible en todas las páginas (botón 💬) que explica métricas, da insights y responde preguntas sobre los datos.

## Chatbot IA (Nuevo)

El dashboard ahora incluye un asistente de IA que te ayuda a comprender los datos. Para usarlo:

1. **Instala Ollama** (motor de IA local):
   ```bash
   # Linux/macOS
   curl -fsSL https://ollama.ai/install.sh | sh

   # O descarga desde: https://ollama.ai/download
   ```

2. **Inicia Ollama y descarga un modelo**:
   ```bash
   ollama serve
   ollama pull llama3
   ```

3. **Usa el chatbot**: Haz clic en el botón 💬 en cualquier página del dashboard.

Para más detalles, consulta: [CHATBOT_README.md](CHATBOT_README.md)

## Estructura de scripts
La lógica está dividida por módulos en `scripts/`. Consulta la descripción completa aquí: [scripts/README.md](scripts/README.md).
