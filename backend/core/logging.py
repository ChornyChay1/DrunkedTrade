import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from typing import Literal


class LogSettings:
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "text"
    LOG_TO_CONSOLE: bool = True
    LOG_TO_FILE: bool = True
    LOG_DIR: str = "logs"
    LOG_FILE_MAX_BYTES: int = 10 * 1024 * 1024  
    LOG_FILE_BACKUP_COUNT: int = 5


class JSONFormatter(logging.Formatter):
    """Форматтер для логов в JSON формате"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Форматтер для логов в текстовом формате"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record: logging.LogRecord) -> str:
        if record.levelname == 'DEBUG':
            color = self.COLORS['DEBUG']
        elif record.levelname == 'INFO':
            color = self.COLORS['INFO']
        elif record.levelname == 'WARNING':
            color = self.COLORS['WARNING']
        elif record.levelname == 'ERROR':
            color = self.COLORS['ERROR']
        elif record.levelname == 'CRITICAL':
            color = self.COLORS['CRITICAL']
        else:
            color = self.COLORS['RESET']
        
        reset = self.COLORS['RESET']
        
        if record.levelno >= logging.WARNING:
            format_str = f'{color}%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s{reset}'
        else:
            format_str = f'{color}%(asctime)s | %(levelname)-8s | %(message)s{reset}'
        
        formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)


def setup_logging() -> None:
    """Настройка логирования для всего приложения"""
    
    if LogSettings.LOG_TO_FILE:
        log_dir = Path(LogSettings.LOG_DIR)
        log_dir.mkdir(exist_ok=True)
    
    log_level = getattr(logging, LogSettings.LOG_LEVEL.upper(), logging.INFO)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    root_logger.handlers.clear()
    
    if LogSettings.LOG_FORMAT == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()
    
    if LogSettings.LOG_TO_CONSOLE:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    if LogSettings.LOG_TO_FILE:
        log_file = log_dir / "app.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=LogSettings.LOG_FILE_MAX_BYTES,
            backupCount=LogSettings.LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: level={LogSettings.LOG_LEVEL}, "
        f"format={LogSettings.LOG_FORMAT}, "
        f"console={LogSettings.LOG_TO_CONSOLE}, "
        f"file={LogSettings.LOG_TO_FILE}"
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)