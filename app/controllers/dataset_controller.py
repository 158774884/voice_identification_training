"""
Dataset controller — manages dataset import, scanning, cleaning, and split operations.
Calls existing data/ modules for preprocessing and augmentation.
"""
import os
import random
from typing import List, Optional
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal

from app.models.dataset_model import DatasetModel, AudioFileInfo
from app.utils.logger import LogManager


@dataclass
class ImportProgress:
    current: int
    total: int
    current_file: str
    status: str  # 'scanning' | 'importing' | 'validating' | 'done'


class DatasetController(QObject):
    """Controls dataset operations."""

    # Signals
    import_progress = Signal(ImportProgress)
    import_finished = Signal(int)  # total files imported
    import_error = Signal(str)     # error message
    dataset_updated = Signal()     # emitted when dataset changes
    file_selected = Signal(AudioFileInfo)  # selected for preview

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dataset = DatasetModel()
        self._log = LogManager()

    @property
    def dataset(self) -> DatasetModel:
        return self._dataset

    @property
    def files(self) -> List[AudioFileInfo]:
        return self._dataset.files

    def import_from_directory(self, directory: str,
                              recursive: bool = True,
                              formats: Optional[List[str]] = None) -> int:
        """Scan and import audio files from a directory.

        Returns:
            Number of files found.
        """
        if formats is None:
            formats = {".wav", ".flac", ".mp3", ".ogg"}

        self._log.info("数据集", f"正在扫描目录: {directory}")

        # Find all audio files
        found = []
        if recursive:
            for root, dirs, filenames in os.walk(directory):
                for fn in filenames:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in formats:
                        found.append(os.path.join(root, fn))
        else:
            for fn in os.listdir(directory):
                ext = os.path.splitext(fn)[1].lower()
                if ext in formats:
                    found.append(os.path.join(directory, fn))

        total = len(found)
        self._log.info("数据集", f"发现 {total} 个音频文件")

        # Create AudioFileInfo entries
        new_files = []
        existing_paths = {f.path for f in self._dataset.files}

        for i, path in enumerate(found):
            if path in existing_paths:
                continue

            info = AudioFileInfo.from_path(path)

            # Validate
            if info.duration <= 0:
                info.status = "error"
            elif info.duration < self._dataset.min_duration:
                info.status = "short"
            elif info.duration > self._dataset.max_duration:
                info.status = "long"
            else:
                info.status = "valid"

            new_files.append(info)

            if i % 100 == 0 or i == total - 1:
                self.import_progress.emit(ImportProgress(
                    current=i + 1, total=total,
                    current_file=os.path.basename(path),
                    status="scanning"
                ))

        self._dataset.files.extend(new_files)
        self._dataset.data_root = directory

        # Auto-assign splits
        self._dataset.apply_split()

        counts = {
            "valid": sum(1 for f in new_files if f.status == "valid"),
            "short": sum(1 for f in new_files if f.status == "short"),
            "long": sum(1 for f in new_files if f.status == "long"),
            "error": sum(1 for f in new_files if f.status == "error"),
        }

        self._log.info(
            "数据集",
            f"导入完成: {len(new_files)} 新文件 "
            f"(有效: {counts['valid']}, "
            f"太短: {counts['short']}, "
            f"太长: {counts['long']}, "
            f"错误: {counts['error']})"
        )

        self.import_finished.emit(len(new_files))
        self.dataset_updated.emit()
        return len(new_files)

    def set_split_ratios(self, train: float, val: float, test: float):
        """Update train/val/test split ratios and re-split."""
        self._dataset.train_ratio = train
        self._dataset.val_ratio = val
        self._dataset.test_ratio = test
        self._dataset.apply_split()
        self._log.info(
            "数据集",
            f"数据集划分: 训练={train:.0%} 验证={val:.0%} 测试={test:.0%}"
        )
        self.dataset_updated.emit()

    def set_duration_filter(self, min_dur: float, max_dur: float):
        """Filter files by duration range."""
        self._dataset.min_duration = min_dur
        self._dataset.max_duration = max_dur

        changed = 0
        for f in self._dataset.files:
            old_status = f.status
            if f.duration <= 0:
                f.status = "error"
            elif f.duration < min_dur:
                f.status = "short"
            elif f.duration > max_dur:
                f.status = "long"
            else:
                f.status = "valid"
            if old_status != f.status:
                changed += 1

        self._dataset.apply_split()
        self._log.info(
            "数据集",
            f"时长过滤: {min_dur}s-{max_dur}s, {changed} 个文件状态变更"
        )
        self.dataset_updated.emit()

    def get_statistics(self) -> dict:
        """Return current dataset statistics."""
        stats = self._dataset.statistics()

        # Add dialect/gender/region breakdowns
        dialects = {}
        genders = {}
        regions = {}
        for f in self._dataset.files:
            if f.dialect:
                dialects[f.dialect] = dialects.get(f.dialect, 0) + 1
            if f.gender:
                genders[f.gender] = genders.get(f.gender, 0) + 1
            if f.region:
                regions[f.region] = regions.get(f.region, 0) + 1

        stats["dialects"] = dialects
        stats["genders"] = genders
        stats["regions"] = regions
        return stats

    def export_metadata_jsonl(self, output_path: str, split: Optional[str] = None):
        """Export dataset metadata to JSONL format for training.

        Args:
            output_path: Path to save .jsonl file
            split: 'train', 'val', 'test', or None for all
        """
        import json

        files = self._dataset.files
        if split:
            files = [f for f in files if f.split == split]

        with open(output_path, "w", encoding="utf-8") as f:
            for info in files:
                if info.status != "valid":
                    continue
                entry = {
                    "audio_path": info.path,
                    "text": info.text,
                    "dialect": info.dialect or "mandarin",
                    "speaker_id": info.speaker_id,
                    "duration": info.duration,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self._log.info("数据集", f"导出 {len(files)} 条记录到 {output_path}")

    def clear(self):
        """Clear all imported data."""
        self._dataset.files.clear()
        self._dataset.data_root = ""
        self.dataset_updated.emit()
        self._log.info("数据集", "已清空数据集")
