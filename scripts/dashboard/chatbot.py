"""
Módulo de chatbot con Ollama para asistir en la comprensión de datos del dashboard.
"""

import requests
import json
import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional


class OllamaChatbot:
    """Chatbot que utiliza Ollama para proporcionar insights sobre los datos del dashboard."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        """
        Inicializa el chatbot con Ollama.

        Args:
            base_url: URL base de Ollama (por defecto localhost:11434)
            model: Modelo a utilizar (llama3, mistral, etc.)
        """
        self.base_url = base_url
        self.model = model
        self.conversation_history = []

    def check_ollama_available(self) -> bool:
        """Verifica si Ollama está disponible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_available_models(self) -> List[str]:
        """Obtiene la lista de modelos disponibles en Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'] for model in models_data.get('models', [])]
            return []
        except:
            return []

    def build_context_prompt(self, data_dict: Dict[str, Any], current_page: str, filtros: Dict[str, Any] = None) -> str:
        """
        Construye el prompt de contexto basado en los datos actuales del dashboard.

        Args:
            data_dict: Diccionario con los dataframes FILTRADOS del dashboard
            current_page: Nombre de la página actual
            filtros: Diccionario con los filtros activos

        Returns:
            Prompt de contexto formateado
        """
        context_parts = [
            "Eres un asistente experto en análisis de datos de manufactura.",
            "Tu objetivo es ayudar a los usuarios a comprender los datos del dashboard.",
            "Debes proporcionar insights, explicar métricas, responder preguntas y sugerir acciones.",
            "",
            f"**PÁGINA ACTUAL:** {current_page}",
            ""
        ]

        # Añadir información de filtros activos
        if filtros:
            context_parts.append("**FILTROS ACTIVOS:**")

            if 'date_range' in filtros and filtros['date_range']:
                fecha_inicio, fecha_fin = filtros['date_range']
                context_parts.append(f"- Rango de fechas: {fecha_inicio} a {fecha_fin}")

            if 'week_label' in filtros and filtros['week_label'] != "(Rango personalizado)":
                context_parts.append(f"- Período: {filtros['week_label']}")

            if 'recurso_oee' in filtros and filtros['recurso_oee'] != "(Todos)":
                context_parts.append(f"- Máquina seleccionada: {filtros['recurso_oee']}")

            if 'planta' in filtros and filtros['planta']:
                plantas_str = ", ".join(filtros['planta'])
                context_parts.append(f"- Plantas: {plantas_str}")

            context_parts.append("")
            context_parts.append("IMPORTANTE: Los datos que te proporciono ya están FILTRADOS según estos criterios.")
            context_parts.append("Cuando respondas, habla específicamente de estos datos filtrados, no de datos generales.")
            context_parts.append("")

        context_parts.extend([
            "**DATOS DISPONIBLES (FILTRADOS):**",
            ""
        ])

        # Añadir información resumida de cada dataframe
        if 'produccion' in data_dict and not data_dict['produccion'].empty:
            prod = data_dict['produccion']
            total_ok = prod['piezas_ok'].sum()
            total_scrap = prod['piezas_scrap'].sum()
            total = total_ok + total_scrap
            scrap_pct = (total_scrap / total * 100) if total > 0 else 0

            context_parts.extend([
                "**PRODUCCIÓN:**",
                f"- Total piezas OK: {total_ok:,.0f}",
                f"- Total scrap: {total_scrap:,.0f}",
                f"- Scrap %: {scrap_pct:.2f}%",
                f"- Período: {prod['ts_ini'].min()} a {prod['ts_fin'].max()}",
                f"- Máquinas: {prod['machine_name'].nunique()}",
                f"- Referencias: {prod['ref_id_str'].nunique()}",
                ""
            ])

        if 'ordenes' in data_dict and not data_dict['ordenes'].empty:
            ord = data_dict['ordenes']
            context_parts.extend([
                "**ÓRDENES:**",
                f"- Total órdenes: {ord['work_order_id'].nunique()}",
                f"- Piezas planificadas: {ord['qty_plan'].sum():,.0f}",
                ""
            ])

        if 'rrhh' in data_dict and not data_dict['rrhh'].empty:
            rrhh = data_dict['rrhh']
            cols_disponibles = rrhh.columns.tolist()

            # Buscar columna de horas disponibles (puede ser horas_netas o horas_ajustadas)
            horas_disp = 0
            if 'horas_netas' in cols_disponibles:
                horas_disp = rrhh['horas_netas'].sum()
            elif 'horas_ajustadas' in cols_disponibles:
                horas_disp = rrhh['horas_ajustadas'].sum()

            horas_perdidas = 0
            for col in ['horas_enfermedad', 'horas_accidente', 'horas_permiso']:
                if col in cols_disponibles:
                    horas_perdidas += rrhh[col].sum()

            tasa_absentismo = (horas_perdidas / (horas_disp + horas_perdidas) * 100) if (horas_disp + horas_perdidas) > 0 else 0

            context_parts.extend([
                "**RECURSOS HUMANOS:**",
                f"- Horas netas: {horas_disp:,.0f}h",
                f"- Horas perdidas: {horas_perdidas:,.0f}h",
                f"- Tasa absentismo: {tasa_absentismo:.2f}%",
                ""
            ])

        if 'compras' in data_dict and not data_dict['compras'].empty:
            comp = data_dict['compras']
            cols = comp.columns.tolist()

            # La columna puede ser qty_recibida o cantidad
            total_mp = 0
            if 'qty_recibida' in cols:
                total_mp = comp['qty_recibida'].sum()
            elif 'cantidad' in cols:
                total_mp = comp['cantidad'].sum()

            num_refs = 0
            if 'ref_materia_str' in cols:
                num_refs = comp['ref_materia_str'].nunique()
            elif 'ref_materia' in cols:
                num_refs = comp['ref_materia'].nunique()

            context_parts.extend([
                "**ALMACÉN MP:**",
                f"- Total MP recibida: {total_mp:,.0f} kg",
                f"- Lotes recibidos: {len(comp)}",
                f"- Referencias MP: {num_refs}",
                ""
            ])

        # Añadir información sobre métricas clave
        context_parts.extend([
            "**MÉTRICAS CLAVE:**",
            "- OEE (Overall Equipment Effectiveness): Disponibilidad × Rendimiento × Calidad",
            "- Disponibilidad: % de tiempo que la máquina está produciendo",
            "- Rendimiento: Velocidad real vs velocidad teórica (UPH real vs ideal)",
            "- Calidad: % de piezas OK vs total",
            "- UPH (Units Per Hour): Piezas producidas por hora",
            "- Scrap %: Porcentaje de piezas defectuosas",
            "",
            "**INSTRUCCIONES:**",
            "- Responde en español de manera clara y concisa",
            "- Si detectas anomalías o patrones interesantes, menciónalo",
            "- Sugiere acciones concretas cuando sea apropiado",
            "- Explica las métricas en términos sencillos si el usuario lo necesita",
            "- Usa los datos proporcionados para respaldar tus respuestas",
            ""
        ])

        return "\n".join(context_parts)

    def chat(self, user_message: str, data_context: str) -> str:
        """
        Envía un mensaje al chatbot y obtiene la respuesta.

        Args:
            user_message: Mensaje del usuario
            data_context: Contexto de datos para el modelo

        Returns:
            Respuesta del chatbot
        """
        try:
            # Construir el prompt completo
            if not self.conversation_history:
                # Primera conversación, incluir contexto
                system_prompt = data_context
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            else:
                # Conversación continua
                messages = self.conversation_history + [
                    {"role": "user", "content": user_message}
                ]

            # Llamar a Ollama API
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                assistant_message = result['message']['content']

                # Actualizar historial
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": assistant_message})

                # Limitar historial a últimos 10 mensajes para no sobrecargar
                if len(self.conversation_history) > 10:
                    self.conversation_history = self.conversation_history[-10:]

                return assistant_message
            else:
                return f"Error al comunicarse con Ollama: {response.status_code}"

        except requests.exceptions.Timeout:
            return "La solicitud ha tardado demasiado tiempo. Por favor, intenta de nuevo."
        except Exception as e:
            return f"Error: {str(e)}"

    def reset_conversation(self):
        """Reinicia el historial de conversación."""
        self.conversation_history = []


def render_chatbot_bubble(data_dict: Dict[str, Any], current_page: str, filtros: Dict[str, Any] = None):
    """
    Renderiza el chatbot en el sidebar de Streamlit con carga diferida.

    Args:
        data_dict: Diccionario con los datos FILTRADOS del dashboard
        current_page: Nombre de la página actual
        filtros: Diccionario con los filtros activos (rango de fechas, máquina, etc.)
    """
    # Inicializar estados mínimos
    if 'chatbot_initialized' not in st.session_state:
        st.session_state.chatbot_initialized = False
        st.session_state.chatbot = None
        st.session_state.chat_messages = []
        st.session_state.chatbot_active = False
        st.session_state.initializing = False

    # Renderizar en el sidebar
    with st.sidebar:
        st.markdown("---")

        # Botón para activar/desactivar chatbot
        if not st.session_state.chatbot_active:
            if st.button("🤖 Activar Asistente IA", use_container_width=True, type="primary", key="activate_chatbot"):
                st.session_state.chatbot_active = True
                st.session_state.initializing = True
                st.rerun()
        else:
            if st.button("❌ Cerrar Asistente", use_container_width=True, key="deactivate_chatbot"):
                st.session_state.chatbot_active = False
                st.session_state.chatbot_initialized = False
                st.session_state.chatbot = None
                st.session_state.chat_messages = []
                st.rerun()

    # Si el chatbot está activo, inicializarlo y mostrar UI
    if st.session_state.chatbot_active:
        with st.sidebar:
            st.markdown("### 🤖 Asistente IA")

            # Inicializar chatbot si no está inicializado
            if not st.session_state.chatbot_initialized:
                if st.session_state.initializing:
                    with st.spinner("Inicializando asistente IA..."):
                        try:
                            st.session_state.chatbot = OllamaChatbot()

                            # Verificar si Ollama está disponible
                            if st.session_state.chatbot.check_ollama_available():
                                st.session_state.chatbot_initialized = True
                                st.session_state.initializing = False
                                st.success("✅ Asistente IA listo!")
                                st.rerun()
                            else:
                                st.session_state.initializing = False
                                st.error("⚠️ Ollama no disponible")
                                st.info("Ejecuta: `ollama serve`")

                                if st.button("🔄 Reintentar", key="retry_init"):
                                    st.session_state.initializing = True
                                    st.rerun()
                                return
                        except Exception as e:
                            st.session_state.initializing = False
                            st.error(f"Error: {str(e)}")
                            return
                return

            # Selector de modelo
            available_models = st.session_state.chatbot.get_available_models()
            if available_models:
                selected_model = st.selectbox(
                    "Modelo IA:",
                    available_models,
                    index=0 if st.session_state.chatbot.model not in available_models else available_models.index(st.session_state.chatbot.model),
                    key="model_selector"
                )
                if selected_model != st.session_state.chatbot.model:
                    st.session_state.chatbot.model = selected_model
                    st.session_state.chatbot.reset_conversation()
                    st.info(f"Modelo cambiado a: {selected_model}")

            st.markdown("---")

            # Mostrar historial de conversación
            if st.session_state.chat_messages:
                st.markdown("**Conversación:**")
                # Contenedor con altura fija y scroll
                with st.container():
                    for msg in st.session_state.chat_messages[-6:]:  # Últimos 6 mensajes
                        if msg["role"] == "user":
                            st.markdown(f"**🧑 Tú:**")
                            st.info(msg['content'])
                        else:
                            st.markdown(f"**🤖 Asistente:**")
                            st.success(msg['content'])

                if st.button("🗑️ Limpiar historial", use_container_width=True):
                    st.session_state.chat_messages = []
                    st.session_state.chatbot.reset_conversation()
                    st.rerun()

                st.markdown("---")

            # Input de usuario en sidebar
            user_input = st.text_area(
                "Tu pregunta:",
                placeholder="Ej: ¿Qué máquina tiene más scrap?",
                height=120,
                key="chat_input_sidebar"
            )

            if st.button("📤 Enviar pregunta", use_container_width=True, type="primary", key="send_chat"):
                if user_input:
                    # Añadir mensaje del usuario
                    st.session_state.chat_messages.append({
                        "role": "user",
                        "content": user_input
                    })

                    # Construir contexto
                    context = st.session_state.chatbot.build_context_prompt(data_dict, current_page, filtros)

                    # Obtener respuesta
                    with st.spinner("🤔 Pensando..."):
                        response = st.session_state.chatbot.chat(user_input, context)

                    # Añadir respuesta del bot
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response
                    })

                    st.rerun()
                else:
                    st.warning("Por favor, escribe una pregunta")

            # Sugerencias rápidas
            with st.expander("💡 Preguntas rápidas"):
                if st.button("Estado general de producción", use_container_width=True, key="q1"):
                    st.session_state.quick_question = "¿Cuál es el estado general de la producción?"
                if st.button("Máquina con peor rendimiento", use_container_width=True, key="q2"):
                    st.session_state.quick_question = "¿Qué máquina tiene el peor rendimiento?"
                if st.button("Explicar qué es el OEE", use_container_width=True, key="q3"):
                    st.session_state.quick_question = "Explícame qué es el OEE y cómo se calcula"

            # Procesar pregunta rápida si existe
            if 'quick_question' in st.session_state and st.session_state.quick_question:
                question = st.session_state.quick_question
                st.session_state.quick_question = None

                st.session_state.chat_messages.append({
                    "role": "user",
                    "content": question
                })

                context = st.session_state.chatbot.build_context_prompt(data_dict, current_page, filtros)

                with st.spinner("🤔 Pensando..."):
                    response = st.session_state.chatbot.chat(question, context)

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response
                })

                st.rerun()
