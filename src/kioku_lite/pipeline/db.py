"""KiokuDB — connection manager and schema bootstrap.

Creates a single SQLite connection, loads sqlite-vec, creates all tables,
then exposes MemoryStore and GraphStore as attributes.

Usage:
    db = KiokuDB(db_path, embed_dim=1024)
    db.memory   # MemoryStore instance
    db.graph    # GraphStore instance
    db.close()
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from kioku_lite.pipeline.graph_store import GraphStore
from kioku_lite.pipeline.memory_store import MemoryStore

log = logging.getLogger(__name__)


class KiokuDB:
    """Facade: one SQLite file, one connection, BothStores.

    All Kioku Lite data lives in a single `kioku.db`:
      - memories / memory_fts / memory_vec  (MemoryStore)
      - kg_nodes / kg_edges / kg_aliases    (GraphStore)
    """

    def __init__(self, db_path: Path, embed_dim: int = 1024):
        self.db_path = db_path
        self.embed_dim = embed_dim
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._vec_enabled = self._load_sqlite_vec()
        self._create_tables()

        self.memory = MemoryStore(self.conn, embed_dim=embed_dim, vec_enabled=self._vec_enabled)
        self.graph = GraphStore(self.conn)

    # ── sqlite-vec ─────────────────────────────────────────────────────────────

    def _load_sqlite_vec(self) -> bool:
        """Load sqlite-vec extension. Returns True if successful."""
        try:
            import sqlite_vec
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            log.info("sqlite-vec loaded")
            return True
        except Exception as e:
            log.warning("sqlite-vec unavailable (%s) — vector search disabled", e)
            return False

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _create_tables(self) -> None:
        """Create all tables (idempotent, safe to call on existing DB)."""
        cur = self.conn.cursor()
        self._create_memory_tables(cur)
        self._create_graph_tables(cur)
        self.conn.commit()

    def _create_memory_tables(self, cur: sqlite3.Cursor) -> None:
        # Source-of-truth table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                date TEXT NOT NULL,
                mood TEXT DEFAULT '',
                timestamp TEXT NOT NULL,
                content_hash TEXT UNIQUE NOT NULL,
                tags TEXT DEFAULT '[]',
                event_time TEXT DEFAULT ''
            )
        """)
        # Migrations for older DBs
        for col, default in [("tags", "'[]'"), ("event_time", "''")]:
            try:
                cur.execute(f"ALTER TABLE memories ADD COLUMN {col} TEXT DEFAULT {default}")
            except sqlite3.OperationalError:
                pass

        # FTS5 virtual table
        cur.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                content, date, mood,
                content='memories',
                content_rowid='id'
            )
        """)
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memory_fts(rowid, content, date, mood)
                VALUES (new.id, new.content, new.date, new.mood);
            END
        """)
        cur.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, content, date, mood)
                VALUES ('delete', old.id, old.content, old.date, old.mood);
            END
        """)

        # sqlite-vec virtual table (only if extension loaded)
        if self._vec_enabled:
            cur.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
                    content_hash TEXT PRIMARY KEY,
                    embedding FLOAT[{self.embed_dim}]
                )
            """)

    def _create_graph_tables(self, cur: sqlite3.Cursor) -> None:
        # Entity nodes
        cur.execute("""
            CREATE TABLE IF NOT EXISTS kg_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                type TEXT DEFAULT '',
                mention_count INTEGER DEFAULT 0,
                first_seen TEXT DEFAULT '',
                last_seen TEXT DEFAULT '',
                is_canonical INTEGER DEFAULT 0
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_kg_nodes_name ON kg_nodes(name COLLATE NOCASE)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_kg_nodes_cnt ON kg_nodes(mention_count DESC)")

        # Relationship edges
        cur.execute("""
            CREATE TABLE IF NOT EXISTS kg_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                rel_type TEXT DEFAULT '',
                weight REAL DEFAULT 0.5,
                evidence TEXT DEFAULT '',
                source_hash TEXT DEFAULT '',
                event_time TEXT DEFAULT '',
                valid_from TEXT DEFAULT '',
                valid_until TEXT DEFAULT NULL,
                UNIQUE(source, target, rel_type)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_kg_edges_src ON kg_edges(source COLLATE NOCASE)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_kg_edges_tgt ON kg_edges(target COLLATE NOCASE)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_kg_edges_valid ON kg_edges(valid_until)")
        # Migrations for older DBs
        for col, default in [("valid_from", "''"), ("valid_until", "NULL"), ("last_reinforced", "''")]:
            try:
                cur.execute(f"ALTER TABLE kg_edges ADD COLUMN {col} TEXT DEFAULT {default}")
            except sqlite3.OperationalError:
                pass

        # Migrations for older DBs
        try:
            cur.execute("ALTER TABLE kg_nodes ADD COLUMN confidence REAL DEFAULT 1.0")
        except sqlite3.OperationalError:
            pass

        # SAME_AS alias mapping
        cur.execute("""
            CREATE TABLE IF NOT EXISTS kg_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alias TEXT NOT NULL,
                canonical TEXT NOT NULL,
                UNIQUE(alias, canonical)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_kg_alias ON kg_aliases(alias COLLATE NOCASE)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_kg_canonical ON kg_aliases(canonical COLLATE NOCASE)")

        # Merge audit log
        cur.execute("""
            CREATE TABLE IF NOT EXISTS kg_merge_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                target_name TEXT NOT NULL,
                merge_type TEXT DEFAULT 'auto',
                vector_sim REAL DEFAULT 0.0,
                name_sim REAL DEFAULT 0.0,
                timestamp TEXT NOT NULL
            )
        """)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def close(self) -> None:
        self.conn.close()

    @property
    def vec_enabled(self) -> bool:
        return self._vec_enabled
