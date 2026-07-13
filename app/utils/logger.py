"""
GUI log manager — singleton for categorized, timestamped logging.
All modules route log messages through this manager so the LogPanel
can display them in real time.
"""
import time
import threading
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class LogEntry:
    timestamp: str
    category: str
    level: LogLevel
    message: str

    def formatted(self) -> str:
        return f"[{self.timestamp}] [{self.level.value}] [{self.category}] {self.message}"


class LogManager:
    """Thread-safe singleton log manager."""

    _instance: Optional["LogManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._entries: List[LogEntry] = []
        self._max_entries = 5000
        self._listeners: List[callable] = []  # callable(LogEntry)

    def log(self, category: str, message: str, level: LogLevel = LogLevel.INFO):
        entry = LogEntry(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            category=category,
            level=level,
            message=message,
        )
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(entry)
            except Exception:
                pass

    def info(self, category: str, message: str):
        self.log(category, message, LogLevel.INFO)

    def warning(self, category: str, message: str):
        self.log(category, message, LogLevel.WARNING)

    def error(self, category: str, message: str):
        self.log(category, message, LogLevel.ERROR)

    def debug(self, category: str, message: str):
        self.log(category, message, LogLevel.DEBUG)

    def add_listener(self, callback: callable):
        """Register a callback that receives each new LogEntry."""
        self._listeners.append(callback)

    def remove_listener(self, callback: callable):
        if callback in self._listeners:
            self._listeners.remove(callback)

    def get_entries(self, categories: Optional[List[str]] = None,
                    levels: Optional[List[LogLevel]] = None,
                    search: str = "",
                    limit: int = 500) -> List[LogEntry]:
        """Query log entries with filtering."""
        with self._lock:
            result = list(self._entries)

        if categories:
            result = [e for e in result if e.category in categories]
        if levels:
            result = [e for e in result if e.level in levels]
        if search:
            result = [e for e in result if search.lower() in e.message.lower()]

        return result[-limit:]

    def clear(self):
        with self._lock:
            self._entries.clear()

    @property
    def entry_count(self) -> int:
        return len(self._entries)


# Convenience function
def log_info(category: str, message: str):
    LogManager().info(category, message)


def log_error(category: str, message: str):
    LogManager().error(category, message)


def log_warning(category: str, message: str):
    LogManager().warning(category, message)
