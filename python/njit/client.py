"""
NJIT (Neural JIT) Python Client & SQLite Binder
Maintained by RMAA Holdings
"""

import sqlite3
import struct
import time
from typing import Optional, Tuple, Dict, Any

NJIT_MAGIC = 0x54494A4E # 'NJIT'
NJIT_VERSION = 0x0100

class NJITDatabase:
    def __init__(self, db_path: str = "njit_knowledge.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.init_schema()

    def init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS njit_artifacts (
                    cid TEXT PRIMARY KEY,
                    tensor_type INTEGER,
                    num_layers INTEGER,
                    head_dim INTEGER,
                    seq_len INTEGER,
                    payload BLOB,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_njit_seq ON njit_artifacts (seq_len)
            """)

    def store_tensor_block(
        self,
        cid: str,
        tensor_type: int,
        num_layers: int,
        head_dim: int,
        seq_len: int,
        tensor_bytes: bytes,
        metadata_json: str = "{}"
    ) -> bool:
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO njit_artifacts
                (cid, tensor_type, num_layers, head_dim, seq_len, payload, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (cid, tensor_type, num_layers, head_dim, seq_len, tensor_bytes, metadata_json))
        return True

    def fetch_tensor_block(self, cid: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT cid, tensor_type, num_layers, head_dim, seq_len, payload, metadata_json
            FROM njit_artifacts WHERE cid = ?
        """, (cid,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "cid": row[0],
            "tensor_type": row[1],
            "num_layers": row[2],
            "head_dim": row[3],
            "seq_len": row[4],
            "payload": row[5],
            "metadata_json": row[6]
        }
