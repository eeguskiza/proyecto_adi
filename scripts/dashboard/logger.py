"""
Sistema de logging personalizado para el dashboard.
Proporciona logs coloridos y con formato en terminal.
"""

import logging
import sys
from datetime import datetime
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Formatter que añade colores a los logs en terminal."""

    # Códigos ANSI para colores
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Verde
        'WARNING': '\033[33m',    # Amarillo
        'ERROR': '\033[31m',      # Rojo
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m',       # Reset
        'BOLD': '\033[1m',        # Bold
    }

    def format(self, record):
        """Formatea el log con colores."""
        # Obtener el color según el nivel
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        bold = self.COLORS['BOLD']

        # Timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Formatear el mensaje
        log_message = f"{color}[{timestamp}] {bold}[{record.levelname}]{reset}{color} {record.getMessage()}{reset}"

        return log_message


class DashboardLogger:
    """Logger personalizado para el dashboard."""

    _instance: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls, name: str = "Dashboard") -> logging.Logger:
        """
        Obtiene o crea el logger del dashboard.

        Args:
            name: Nombre del logger

        Returns:
            Logger configurado
        """
        if cls._instance is None:
            cls._instance = logging.getLogger(name)
            cls._instance.setLevel(logging.INFO)

            # Evitar duplicar handlers
            if not cls._instance.handlers:
                # Handler para console
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.DEBUG)
                console_handler.setFormatter(ColoredFormatter())
                cls._instance.addHandler(console_handler)

        return cls._instance

    @classmethod
    def log_module_load(cls, module_name: str, success: bool = True):
        """
        Registra la carga de un módulo.

        Args:
            module_name: Nombre del módulo
            success: Si la carga fue exitosa
        """
        logger = cls.get_logger()
        if success:
            logger.info(f"Módulo '{module_name}' cargado correctamente")
        else:
            logger.error(f"Error al cargar módulo '{module_name}'")

    @classmethod
    def log_data_load(cls, data_name: str, rows: int, time_ms: Optional[float] = None):
        """
        Registra la carga de datos.

        Args:
            data_name: Nombre del dataset
            rows: Número de filas cargadas
            time_ms: Tiempo de carga en milisegundos
        """
        logger = cls.get_logger()
        time_str = f" ({time_ms:.1f}ms)" if time_ms else ""
        logger.info(f"Dataset '{data_name}': {rows:,} filas cargadas{time_str}")

    @classmethod
    def log_progress(cls, operation: str, current: int, total: int):
        """
        Registra el progreso de una operación.

        Args:
            operation: Nombre de la operación
            current: Progreso actual
            total: Total de pasos
        """
        logger = cls.get_logger()
        percentage = (current / total) * 100 if total > 0 else 0
        bar_length = 20
        filled = int(bar_length * current / total) if total > 0 else 0
        bar = '█' * filled + '░' * (bar_length - filled)
        logger.info(f"{operation}: [{bar}] {percentage:.0f}% ({current}/{total})")


# Instancia global del logger
logger = DashboardLogger.get_logger()
