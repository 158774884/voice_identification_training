"""
Dataset metadata model — tracks imported audio files and their annotations.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class AudioFileInfo:
    """Metadata for a single audio file."""
    path: str
    filename: str = ""
    duration: float = 0.0  # seconds
    sample_rate: int = 16000
    text: str = ""         # ground-truth transcription / command label
    speaker_id: str = ""   # parsed from filename or metadata
    dialect: str = ""      # dialect label if available
    gender: str = ""       # parsed from filename
    age: str = ""          # parsed from filename
    region: str = ""       # parsed from filename
    status: str = "valid"  # valid | short | long | silent | error
    split: str = "train"   # train | val | test

    @classmethod
    def from_path(cls, path: str) -> "AudioFileInfo":
        """Create from file path, parsing available metadata from filename."""
        info = cls(path=path, filename=os.path.basename(path))

        # Try to get duration using soundfile
        try:
            import soundfile as sf
            data = sf.info(path)
            info.duration = data.duration
            info.sample_rate = data.samplerate
        except Exception:
            pass

        # Parse filename for structured metadata
        # Format: "XXXX-性别-年龄-地区-序号.wav" (e.g., "0004-男-23-内蒙古-142.wav")
        stem = os.path.splitext(info.filename)[0]
        parts = stem.split("-")
        if len(parts) >= 4:
            info.speaker_id = parts[0]
            info.gender = parts[1] if len(parts) > 1 else ""
            info.age = parts[2] if len(parts) > 2 else ""
            info.region = parts[3] if len(parts) > 3 else ""
            if len(parts) > 4:
                # Last part could be an utterance index
                pass

        return info


@dataclass
class DatasetModel:
    """Collection of audio files with split configuration."""

    name: str = ""
    data_root: str = ""
    files: List[AudioFileInfo] = field(default_factory=list)
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    random_seed: int = 42
    min_duration: float = 0.5  # seconds
    max_duration: float = 15.0  # seconds

    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def train_files(self) -> List[AudioFileInfo]:
        return [f for f in self.files if f.split == "train"]

    @property
    def val_files(self) -> List[AudioFileInfo]:
        return [f for f in self.files if f.split == "val"]

    @property
    def test_files(self) -> List[AudioFileInfo]:
        return [f for f in self.files if f.split == "test"]

    def apply_split(self):
        """Assign train/val/test splits based on ratios."""
        import random
        random.seed(self.random_seed)
        valid_files = [
            f for f in self.files
            if f.status == "valid"
            and self.min_duration <= f.duration <= self.max_duration
        ]
        random.shuffle(valid_files)

        n = len(valid_files)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        for i, f in enumerate(valid_files):
            if i < n_train:
                f.split = "train"
            elif i < n_train + n_val:
                f.split = "val"
            else:
                f.split = "test"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "data_root": self.data_root,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "random_seed": self.random_seed,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "total_files": self.total_files,
        }

    def statistics(self) -> dict:
        """Compute dataset statistics."""
        stats = {
            "total": self.total_files,
            "train": len(self.train_files),
            "val": len(self.val_files),
            "test": len(self.test_files),
            "valid": sum(1 for f in self.files if f.status == "valid"),
            "short": sum(1 for f in self.files if f.status == "short"),
            "long": sum(1 for f in self.files if f.status == "long"),
            "silent": sum(1 for f in self.files if f.status == "silent"),
            "error": sum(1 for f in self.files if f.status == "error"),
            "avg_duration": 0.0,
            "total_duration": 0.0,
        }
        if self.files:
            durations = [f.duration for f in self.files if f.duration > 0]
            if durations:
                stats["avg_duration"] = sum(durations) / len(durations)
                stats["total_duration"] = sum(durations)
        return stats
