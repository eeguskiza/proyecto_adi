#!/bin/bash

# Script de instalación y configuración de Ollama para el chatbot del dashboard
# Compatible con Linux y macOS

set -e

echo "======================================"
echo "  Configuración de Ollama para       "
echo "  Dashboard Chatbot                  "
echo "======================================"
echo ""

# Función para detectar el sistema operativo
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)

# Verificar si Ollama ya está instalado
if command -v ollama &> /dev/null; then
    echo "✓ Ollama ya está instalado"
    ollama --version
else
    echo "✗ Ollama no encontrado. Procediendo a instalar..."
    echo ""

    if [ "$OS" == "linux" ]; then
        echo "Instalando Ollama en Linux..."
        curl -fsSL https://ollama.ai/install.sh | sh
    elif [ "$OS" == "macos" ]; then
        echo "Instalando Ollama en macOS..."
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "Homebrew no encontrado. Instalando manualmente..."
            curl -fsSL https://ollama.ai/install.sh | sh
        fi
    else
        echo "Sistema operativo no soportado automáticamente."
        echo "Por favor, instala Ollama manualmente desde: https://ollama.ai/download"
        exit 1
    fi
fi

echo ""
echo "======================================"
echo "  Iniciando Ollama                   "
echo "======================================"
echo ""

# Verificar si Ollama está ejecutándose
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama ya está ejecutándose en localhost:11434"
else
    echo "Iniciando servicio de Ollama..."

    if [ "$OS" == "linux" ]; then
        # En Linux, usar systemd si está disponible
        if command -v systemctl &> /dev/null; then
            sudo systemctl start ollama
            echo "✓ Ollama iniciado como servicio systemd"
        else
            # Si no hay systemd, iniciar en background
            nohup ollama serve > /tmp/ollama.log 2>&1 &
            echo "✓ Ollama iniciado en segundo plano (log: /tmp/ollama.log)"
        fi
    elif [ "$OS" == "macos" ]; then
        # En macOS, iniciar en background
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        echo "✓ Ollama iniciado en segundo plano (log: /tmp/ollama.log)"
    fi

    # Esperar a que Ollama esté listo
    echo "Esperando a que Ollama esté listo..."
    for i in {1..10}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✓ Ollama está listo"
            break
        fi
        echo "  Intentando conectar... ($i/10)"
        sleep 2
    done
fi

echo ""
echo "======================================"
echo "  Descargando modelo de IA           "
echo "======================================"
echo ""

# Verificar si ya hay modelos descargados
MODELS=$(ollama list 2>/dev/null | tail -n +2 | wc -l)

if [ "$MODELS" -gt 0 ]; then
    echo "Modelos ya descargados:"
    ollama list
    echo ""
    read -p "¿Deseas descargar un modelo adicional? (s/N): " download_more

    if [[ ! "$download_more" =~ ^[Ss]$ ]]; then
        echo "✓ Usando modelos existentes"
    else
        echo ""
        echo "Modelos recomendados:"
        echo "  1) llama3 (Recomendado, 4.7GB) - Mejor equilibrio calidad/velocidad"
        echo "  2) mistral (4.1GB) - Más rápido, buena calidad"
        echo "  3) llama2 (3.8GB) - Versión anterior, funcional"
        echo "  4) codellama (3.8GB) - Especializado en código"
        echo ""
        read -p "Selecciona un modelo (1-4): " model_choice

        case $model_choice in
            1) MODEL="llama3" ;;
            2) MODEL="mistral" ;;
            3) MODEL="llama2" ;;
            4) MODEL="codellama" ;;
            *) MODEL="llama3" ;;
        esac

        echo "Descargando $MODEL (esto puede tardar varios minutos)..."
        ollama pull $MODEL
        echo "✓ Modelo $MODEL descargado"
    fi
else
    echo "No se encontraron modelos. Descargando llama3 (recomendado)..."
    echo "Esto descargará aproximadamente 4.7GB. Puede tardar varios minutos."
    echo ""
    read -p "¿Continuar? (S/n): " continue_download

    if [[ "$continue_download" =~ ^[Nn]$ ]]; then
        echo "Descarga cancelada. Puedes descargar manualmente más tarde con:"
        echo "  ollama pull llama3"
    else
        ollama pull llama3
        echo "✓ Modelo llama3 descargado"
    fi
fi

echo ""
echo "======================================"
echo "  Configuración completada            "
echo "======================================"
echo ""
echo "✓ Todo listo para usar el chatbot del dashboard"
echo ""
echo "Próximos pasos:"
echo "  1. Ejecuta el dashboard: streamlit run app.py"
echo "  2. Haz clic en el botón 💬 para abrir el chatbot"
echo "  3. ¡Empieza a hacer preguntas sobre tus datos!"
echo ""
echo "Comandos útiles:"
echo "  - Ver modelos: ollama list"
echo "  - Descargar modelo: ollama pull <nombre>"
echo "  - Detener Ollama: killall ollama (o systemctl stop ollama en Linux)"
echo ""
echo "Documentación completa: CHATBOT_README.md"
echo "======================================"
