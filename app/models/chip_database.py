"""
Chip Database — SQLite-backed CRUD for embedded chip specifications.
"""
import os
import json
import sqlite3
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from app.app_config import CHIP_DB_DIR


@dataclass
class ChipSpec:
    """Chip specification dataclass."""
    id: int = 0
    name: str = ""
    manufacturer: str = ""
    architecture: str = "MCU"
    cpu_cores: int = 1
    cpu_freq_mhz: int = 100
    ram_kb: int = 128
    flash_kb: int = 1024
    npu_tops: float = 0.0
    dsp: bool = False
    supported_quant: List[str] = field(default_factory=list)
    max_model_size_kb: int = 1024
    power_consumption_mw: int = 500
    supported_ops: List[str] = field(default_factory=list)
    price_cny: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "manufacturer": self.manufacturer,
            "architecture": self.architecture,
            "cpu_cores": self.cpu_cores,
            "cpu_freq_mhz": self.cpu_freq_mhz,
            "ram_kb": self.ram_kb,
            "flash_kb": self.flash_kb,
            "npu_tops": self.npu_tops,
            "dsp": self.dsp,
            "supported_quant": self.supported_quant,
            "max_model_size_kb": self.max_model_size_kb,
            "power_consumption_mw": self.power_consumption_mw,
            "supported_ops": self.supported_ops,
            "price_cny": self.price_cny,
            "notes": self.notes,
        }

    @classmethod
    def from_row(cls, row: tuple) -> "ChipSpec":
        """Create from SQLite row tuple."""
        return cls(
            id=row[0], name=row[1], manufacturer=row[2], architecture=row[3],
            cpu_cores=row[4], cpu_freq_mhz=row[5], ram_kb=row[6], flash_kb=row[7],
            npu_tops=row[8], dsp=bool(row[9]),
            supported_quant=json.loads(row[10]) if row[10] else [],
            max_model_size_kb=row[11], power_consumption_mw=row[12],
            supported_ops=json.loads(row[13]) if row[13] else [],
            price_cny=row[14], notes=row[15] or "",
        )


class ChipDatabase:
    """SQLite CRUD for chip specifications."""

    DB_FILENAME = "chip_database.sqlite"

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(CHIP_DB_DIR, self.DB_FILENAME)

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema and seed data."""
        # Check if DB exists and has data
        needs_init = not os.path.exists(self._db_path)

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            if needs_init:
                # Run init SQL from file
                init_sql_path = os.path.join(CHIP_DB_DIR, "chip_db_init.sql")
                if os.path.exists(init_sql_path):
                    with open(init_sql_path, "r", encoding="utf-8") as f:
                        conn.executescript(f.read())
                else:
                    # Minimal fallback
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS chips (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL UNIQUE,
                            manufacturer TEXT NOT NULL,
                            architecture TEXT NOT NULL,
                            cpu_cores INTEGER DEFAULT 1,
                            cpu_freq_mhz INTEGER DEFAULT 100,
                            ram_kb INTEGER DEFAULT 128,
                            flash_kb INTEGER DEFAULT 1024,
                            npu_tops REAL DEFAULT 0.0,
                            dsp INTEGER DEFAULT 0,
                            supported_quant TEXT DEFAULT '[]',
                            max_model_size_kb INTEGER DEFAULT 1024,
                            power_consumption_mw INTEGER DEFAULT 500,
                            supported_ops TEXT DEFAULT '[]',
                            price_cny REAL DEFAULT 0.0,
                            notes TEXT DEFAULT ''
                        )
                    """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    # ================================================================
    # CRUD Operations
    # ================================================================

    def list_all(self) -> List[ChipSpec]:
        """Return all chips."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chips ORDER BY npu_tops DESC, cpu_freq_mhz DESC"
            ).fetchall()
        return [ChipSpec.from_row(r) for r in rows]

    def search(self, query: str = "",
               arch: Optional[str] = None,
               min_tops: float = 0.0,
               max_power: Optional[int] = None) -> List[ChipSpec]:
        """Search/filter chips."""
        with self._connect() as conn:
            sql = "SELECT * FROM chips WHERE 1=1"
            params = []

            if query:
                sql += " AND (name LIKE ? OR manufacturer LIKE ? OR notes LIKE ?)"
                q = f"%{query}%"
                params.extend([q, q, q])
            if arch:
                sql += " AND architecture = ?"
                params.append(arch)
            if min_tops > 0:
                sql += " AND npu_tops >= ?"
                params.append(min_tops)
            if max_power is not None:
                sql += " AND power_consumption_mw <= ?"
                params.append(max_power)

            sql += " ORDER BY npu_tops DESC"
            rows = conn.execute(sql, params).fetchall()

        return [ChipSpec.from_row(r) for r in rows]

    def get_by_id(self, chip_id: int) -> Optional[ChipSpec]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM chips WHERE id = ?", (chip_id,)).fetchone()
        return ChipSpec.from_row(row) if row else None

    def get_by_name(self, name: str) -> Optional[ChipSpec]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM chips WHERE name = ?", (name,)).fetchone()
        return ChipSpec.from_row(row) if row else None

    def add(self, chip: ChipSpec) -> int:
        """Add a new chip. Returns the new ID."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO chips (name, manufacturer, architecture, cpu_cores,
                cpu_freq_mhz, ram_kb, flash_kb, npu_tops, dsp, supported_quant,
                max_model_size_kb, power_consumption_mw, supported_ops, price_cny, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (chip.name, chip.manufacturer, chip.architecture,
                 chip.cpu_cores, chip.cpu_freq_mhz, chip.ram_kb, chip.flash_kb,
                 chip.npu_tops, int(chip.dsp), json.dumps(chip.supported_quant),
                 chip.max_model_size_kb, chip.power_consumption_mw,
                 json.dumps(chip.supported_ops), chip.price_cny, chip.notes)
            )
            return cursor.lastrowid

    def update(self, chip: ChipSpec):
        """Update an existing chip."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE chips SET name=?, manufacturer=?, architecture=?,
                cpu_cores=?, cpu_freq_mhz=?, ram_kb=?, flash_kb=?, npu_tops=?,
                dsp=?, supported_quant=?, max_model_size_kb=?,
                power_consumption_mw=?, supported_ops=?, price_cny=?, notes=?
                WHERE id=?""",
                (chip.name, chip.manufacturer, chip.architecture,
                 chip.cpu_cores, chip.cpu_freq_mhz, chip.ram_kb, chip.flash_kb,
                 chip.npu_tops, int(chip.dsp), json.dumps(chip.supported_quant),
                 chip.max_model_size_kb, chip.power_consumption_mw,
                 json.dumps(chip.supported_ops), chip.price_cny, chip.notes,
                 chip.id)
            )

    def delete(self, chip_id: int):
        with self._connect() as conn:
            conn.execute("DELETE FROM chips WHERE id = ?", (chip_id,))

    def get_architectures(self) -> List[str]:
        """List distinct architectures."""
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT architecture FROM chips").fetchall()
        return [r[0] for r in rows]

    def get_manufacturers(self) -> List[str]:
        """List distinct manufacturers."""
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT manufacturer FROM chips").fetchall()
        return [r[0] for r in rows]
