"""
Project data model — manages project state and JSON persistence.

The Project class holds all mutable application state and serializes
to/from a JSON project file (.vproj).
"""
import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class ModelVersion:
    """A saved model checkpoint entry."""
    name: str
    checkpoint_path: str
    config_snapshot: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    timestamp: str = ""
    phase: str = ""
    notes: str = ""
    active: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelVersion":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ExportRecord:
    """A completed firmware export entry."""
    timestamp: str
    model_name: str
    chip_name: str
    format: str
    output_path: str
    status: str  # 'success', 'failed'
    log: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExportRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProjectData:
    """Complete project state."""

    version: str = "1.0"
    project_name: str = "Untitled"
    created: str = ""
    modified: str = ""

    # Dataset config
    dataset_config: Dict[str, Any] = field(default_factory=dict)

    # Training config
    training_config: Dict[str, Any] = field(default_factory=dict)

    # Model versions (list of ModelVersion dicts)
    model_versions: List[Dict[str, Any]] = field(default_factory=list)
    active_version: Optional[str] = None  # name of active version

    # Chip selection
    selected_chip: Optional[str] = None

    # Export config
    export_config: Dict[str, Any] = field(default_factory=dict)

    # Export history
    export_history: List[Dict[str, Any]] = field(default_factory=list)


class Project:
    """High-level project manager with JSON persistence."""

    def __init__(self, project_path: Optional[str] = None):
        self._path: Optional[str] = project_path
        self._data = ProjectData()
        self._dirty = False

        if project_path and os.path.exists(project_path):
            self.load(project_path)
        else:
            self._data.created = datetime.now().isoformat()
            self._dirty = True

    # ================================================================
    # Persistence
    # ================================================================

    def load(self, path: str):
        """Load project from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._path = path
        self._data.version = raw.get("version", "1.0")
        self._data.project_name = raw.get("project_name", "Untitled")
        self._data.created = raw.get("created", "")
        self._data.modified = raw.get("modified", "")
        self._data.dataset_config = raw.get("dataset_config", {})
        self._data.training_config = raw.get("training_config", {})
        self._data.model_versions = raw.get("model_versions", [])
        self._data.active_version = raw.get("active_version")
        self._data.selected_chip = raw.get("selected_chip")
        self._data.export_config = raw.get("export_config", {})
        self._data.export_history = raw.get("export_history", [])
        self._dirty = False

    def save(self, path: Optional[str] = None):
        """Save project to JSON file."""
        if path:
            self._path = path
        if not self._path:
            raise ValueError("No project path set")

        self._data.modified = datetime.now().isoformat()

        data = {
            "version": self._data.version,
            "project_name": self._data.project_name,
            "created": self._data.created,
            "modified": self._data.modified,
            "dataset_config": self._data.dataset_config,
            "training_config": self._data.training_config,
            "model_versions": self._data.model_versions,
            "active_version": self._data.active_version,
            "selected_chip": self._data.selected_chip,
            "export_config": self._data.export_config,
            "export_history": self._data.export_history,
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self._dirty = False

    # ================================================================
    # Properties
    # ================================================================

    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def name(self) -> str:
        return self._data.project_name

    @name.setter
    def name(self, value: str):
        self._data.project_name = value
        self._dirty = True

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self):
        self._dirty = True

    # ================================================================
    # Dataset Config
    # ================================================================

    @property
    def dataset_config(self) -> dict:
        return self._data.dataset_config

    def update_dataset_config(self, **kwargs):
        self._data.dataset_config.update(kwargs)
        self._dirty = True

    # ================================================================
    # Training Config
    # ================================================================

    @property
    def training_config(self) -> dict:
        return self._data.training_config

    def update_training_config(self, config: dict):
        self._data.training_config = config
        self._dirty = True

    # ================================================================
    # Model Versions
    # ================================================================

    @property
    def model_versions(self) -> List[dict]:
        return self._data.model_versions

    def add_model_version(self, version: dict):
        self._data.model_versions.append(version)
        self._dirty = True

    def remove_model_version(self, name: str):
        self._data.model_versions = [
            v for v in self._data.model_versions if v.get("name") != name
        ]
        if self._data.active_version == name:
            self._data.active_version = None
        self._dirty = True

    def set_active_version(self, name: Optional[str]):
        for v in self._data.model_versions:
            v["active"] = (v.get("name") == name)
        self._data.active_version = name
        self._dirty = True

    @property
    def active_model_path(self) -> Optional[str]:
        if not self._data.active_version:
            return None
        for v in self._data.model_versions:
            if v.get("name") == self._data.active_version:
                return v.get("checkpoint_path")
        return None

    # ================================================================
    # Chip
    # ================================================================

    @property
    def selected_chip(self) -> Optional[str]:
        return self._data.selected_chip

    @selected_chip.setter
    def selected_chip(self, value: Optional[str]):
        self._data.selected_chip = value
        self._dirty = True

    # ================================================================
    # Export
    # ================================================================

    @property
    def export_config(self) -> dict:
        return self._data.export_config

    def update_export_config(self, **kwargs):
        self._data.export_config.update(kwargs)
        self._dirty = True

    def add_export_record(self, record: dict):
        self._data.export_history.append(record)
        self._dirty = True

    @property
    def export_history(self) -> List[dict]:
        return self._data.export_history
